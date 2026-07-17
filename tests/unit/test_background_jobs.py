"""Tests for persistent background workflow execution."""

from __future__ import annotations

import time
from pathlib import Path
from threading import Event, Thread
from typing import Any

import pytest

from olympianim.database.models import ProjectCreate
from olympianim.database.repository import ProjectRepository
from olympianim.services.background_jobs import BackgroundJobService, BackgroundJobWorker
from olympianim.services.credential_service import CredentialStore, ResolvedKey
from olympianim.services.langgraph_workflow import (
    WorkflowCredentials,
    WorkflowModelSelection,
)


class FakeWorkflow:
    """Small workflow double that reports progress and returns a review phase."""

    def __init__(self, **kwargs: Any) -> None:
        self.progress_callback = kwargs["progress_callback"]
        self.cancellation_check = kwargs["cancellation_check"]

    def snapshot(self, project_id: str, **kwargs: Any) -> dict[str, Any]:
        return {}

    def start(self, project_id: str, **kwargs: Any) -> dict[str, Any]:
        self.cancellation_check()
        self.progress_callback("plan_presentation", 60)
        return {"phase": "review_plan_presentation"}

    def resume(self, project_id: str, decision: dict[str, Any], **kwargs: Any) -> dict[str, Any]:
        return {"phase": f"resumed_{decision['action']}"}

    def continue_run(self, project_id: str, **kwargs: Any) -> dict[str, Any]:
        return {"phase": "continued"}


class BlockingWorkflow(FakeWorkflow):
    """Workflow double used to observe work from a reloaded UI service."""

    entered = Event()
    release = Event()

    def resume(self, project_id: str, decision: dict[str, Any], **kwargs: Any) -> dict[str, Any]:
        phase = str(kwargs["expected_phase"])
        self.progress_callback(phase, 50)
        self.entered.set()
        assert self.release.wait(timeout=2)
        return {"phase": f"review_after_{phase}"}


class NoOpSolutionWorkflow(FakeWorkflow):
    """Simulate the old bug that returned success without leaving the presentation."""

    def resume(self, project_id: str, decision: dict[str, Any], **kwargs: Any) -> dict[str, Any]:
        _ = (project_id, decision, kwargs)
        return {"phase": "presentation_complete"}


class CountingWorkflow(FakeWorkflow):
    """Count invocations to prove a completed operation is not applied twice."""

    resume_calls = 0

    def resume(self, project_id: str, decision: dict[str, Any], **kwargs: Any) -> dict[str, Any]:
        _ = (project_id, decision, kwargs)
        type(self).resume_calls += 1
        return {"phase": "review_code_presentation"}


class MissingPhaseWorkflow(FakeWorkflow):
    """Return an invalid terminal snapshot to exercise lifecycle failure handling."""

    def start(self, project_id: str, **kwargs: Any) -> dict[str, Any]:
        _ = (project_id, kwargs)
        return {}


class RecoverAutomaticWorkflow(FakeWorkflow):
    """Expose an automatic checkpoint left between two review boundaries."""

    continue_calls = 0

    def snapshot(self, project_id: str, **kwargs: Any) -> dict[str, Any]:
        _ = (project_id, kwargs)
        return {"phase": "render_presentation"}

    def continue_run(self, project_id: str, **kwargs: Any) -> dict[str, Any]:
        _ = (project_id, kwargs)
        type(self).continue_calls += 1
        return {"phase": "presentation_complete"}


@pytest.fixture
def repository(tmp_path: Path) -> ProjectRepository:
    repository = ProjectRepository(tmp_path / "olympianim.db")
    repository.create_project(
        ProjectCreate(
            title="Projeto",
            problem_statement="Problema",
            llm_provider="OpenAI",
        ),
        project_id="project-id",
    )
    return repository


def _service(repository: ProjectRepository) -> BackgroundJobService:
    return BackgroundJobService(repository, workflow_factory=FakeWorkflow)


def test_job_lifecycle_persists_progress_and_result(repository: ProjectRepository) -> None:
    service = _service(repository)
    queued = service.enqueue(
        "project-id",
        "start",
        credentials=WorkflowCredentials(llm_api_key="secret"),
    )

    claimed = repository.claim_next_job()
    assert claimed is not None
    service.execute(claimed)

    completed = repository.get_job(queued.id)
    assert completed is not None
    assert completed.status == "completed"
    assert completed.current_step == "review_plan_presentation"
    assert completed.progress == 100
    assert completed.attempts == 1
    assert "review_plan_presentation" in completed.result
    project = repository.get_project("project-id")
    assert project is not None
    assert project.status == "review_plan_presentation"


def test_enqueue_rejects_unknown_action_and_project(repository: ProjectRepository) -> None:
    service = _service(repository)

    with pytest.raises(ValueError, match="não suportada"):
        service.enqueue("project-id", "unknown")
    with pytest.raises(ValueError, match="Projeto não encontrado"):
        service.enqueue("missing", "start")


def test_resume_infers_the_persisted_review_boundary(repository: ProjectRepository) -> None:
    repository.update_project_status("project-id", "review_code_presentation")

    job = _service(repository).enqueue(
        "project-id",
        "resume",
        decision={"action": "approve"},
        credentials=WorkflowCredentials(llm_api_key="secret"),
    )

    assert job.expected_phase == "review_code_presentation"


def test_enqueue_discards_transition_when_job_insert_fails(
    repository: ProjectRepository,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    service = _service(repository)

    def fail_to_create_job(*_args: Any, **_kwargs: Any) -> None:
        raise RuntimeError("queue failed")

    monkeypatch.setattr(repository, "create_job", fail_to_create_job)

    with pytest.raises(RuntimeError, match="queue failed"):
        service.enqueue("project-id", "start", operation_id="orphan-operation")

    assert repository.get_transition("orphan-operation") is None


def test_generate_solution_no_op_is_recorded_as_failure(
    repository: ProjectRepository,
) -> None:
    service = BackgroundJobService(repository, workflow_factory=NoOpSolutionWorkflow)
    queued = service.enqueue(
        "project-id",
        "resume",
        decision={"action": "generate_solution"},
        credentials=WorkflowCredentials(llm_api_key="secret"),
    )
    claimed = repository.claim_next_job()
    assert claimed is not None

    service.execute(claimed)

    failed = repository.get_job(queued.id)
    assert failed is not None
    assert failed.status == "failed"
    assert "não avançou" in failed.error_message
    assert any(
        event.event_type == "job.failed" for event in repository.list_workflow_events("project-id")
    )
    project = repository.get_project("project-id")
    assert project is not None
    assert project.status == "failed"


def test_only_one_active_job_is_allowed_per_project(repository: ProjectRepository) -> None:
    service = _service(repository)
    service.enqueue(
        "project-id",
        "start",
        credentials=WorkflowCredentials(llm_api_key="secret"),
    )

    with pytest.raises(ValueError, match="Já existe uma etapa"):
        service.enqueue(
            "project-id",
            "continue",
            credentials=WorkflowCredentials(llm_api_key="secret"),
        )


def test_cancellation_is_cooperative_and_job_can_be_resumed(
    repository: ProjectRepository,
) -> None:
    service = _service(repository)
    queued = service.enqueue(
        "project-id",
        "start",
        credentials=WorkflowCredentials(llm_api_key="secret"),
    )
    assert service.cancel(queued.id) is True

    claimed = repository.claim_next_job()
    assert claimed is not None
    service.execute(claimed)

    cancelled = repository.get_job(queued.id)
    assert cancelled is not None
    assert cancelled.status == "cancelled"
    assert any(
        event.event_type == "job.cancelled"
        for event in repository.list_workflow_events("project-id")
    )

    resumed = service.resume(
        cancelled.id,
        credentials=WorkflowCredentials(llm_api_key="secret"),
    )
    assert resumed.id != cancelled.id
    assert resumed.status == "pending"


def test_stale_running_job_returns_to_queue(repository: ProjectRepository) -> None:
    service = _service(repository)
    queued = service.enqueue(
        "project-id",
        "start",
        credentials=WorkflowCredentials(llm_api_key="secret"),
    )
    claimed = repository.claim_next_job()
    assert claimed is not None

    recovered = repository.recover_stale_jobs("9999-12-31T23:59:59+00:00")

    assert recovered == 1
    job = repository.get_job(queued.id)
    assert job is not None
    assert job.status == "pending"
    assert job.current_step == "recovering"


def test_service_computes_stale_cutoff_and_recovers_job(repository: ProjectRepository) -> None:
    service = _service(repository)
    queued = service.enqueue(
        "project-id",
        "start",
        credentials=WorkflowCredentials(llm_api_key="secret"),
    )
    assert repository.claim_next_job() is not None

    assert service.recover_stale_jobs() == 0
    assert repository.get_job(queued.id) is not None


def test_job_payload_never_contains_credentials(repository: ProjectRepository) -> None:
    service = _service(repository)
    job = service.enqueue(
        "project-id",
        "resume",
        decision={"action": "approve"},
        credentials=WorkflowCredentials(
            llm_api_key="llm-secret",
            voice_api_key="voice-secret",
        ),
        model_selection=WorkflowModelSelection("Google", "gemini-test"),
    )

    assert "llm-secret" not in job.payload
    assert "voice-secret" not in job.payload
    assert "approve" in job.payload
    assert "Google" in job.payload
    assert "gemini-test" in job.payload


def test_retry_can_override_the_failed_job_model(repository: ProjectRepository) -> None:
    service = _service(repository)
    source = service.enqueue(
        "project-id",
        "start",
        credentials=WorkflowCredentials(llm_api_key="secret"),
        model_selection=WorkflowModelSelection("OpenAI", "old-model"),
    )
    repository.update_job(source.id, status="failed", error_message="failure")

    retried = service.retry(
        source.id,
        credentials=WorkflowCredentials(llm_api_key="secret"),
        model_selection=WorkflowModelSelection("Google", "new-model"),
    )

    assert '"provider": "Google"' in retried.payload
    assert '"model": "new-model"' in retried.payload


def test_retry_and_wake_notifies_the_shared_worker(
    repository: ProjectRepository,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    service = _service(repository)
    source = service.enqueue(
        "project-id",
        "start",
        credentials=WorkflowCredentials(llm_api_key="secret"),
    )
    repository.update_job(source.id, status="failed", error_message="failure")
    woke: list[bool] = []
    monkeypatch.setattr(
        "olympianim.services.background_jobs.wake_background_worker",
        lambda: woke.append(True),
    )

    retried = service.retry_and_wake(
        source.id,
        credentials=WorkflowCredentials(llm_api_key="secret"),
    )

    assert retried.status == "pending"
    assert woke == [True]


def test_invalid_workflow_snapshot_is_persisted_as_failure(
    repository: ProjectRepository,
) -> None:
    service = BackgroundJobService(repository, workflow_factory=MissingPhaseWorkflow)
    queued = service.enqueue(
        "project-id",
        "start",
        credentials=WorkflowCredentials(llm_api_key="secret"),
    )
    claimed = repository.claim_next_job()
    assert claimed is not None

    service.execute(claimed)

    failed = repository.get_job(queued.id)
    assert failed is not None
    assert failed.status == "failed"
    assert "sem informar a fase" in failed.error_message


def test_start_recovers_an_automatic_checkpoint_without_replanning(
    repository: ProjectRepository,
) -> None:
    RecoverAutomaticWorkflow.continue_calls = 0
    service = BackgroundJobService(repository, workflow_factory=RecoverAutomaticWorkflow)
    queued = service.enqueue(
        "project-id",
        "start",
        credentials=WorkflowCredentials(llm_api_key="secret"),
    )
    claimed = repository.claim_next_job()
    assert claimed is not None

    service.execute(claimed)

    completed = repository.get_job(queued.id)
    assert completed is not None
    assert completed.current_step == "presentation_complete"
    assert RecoverAutomaticWorkflow.continue_calls == 1


def test_continue_job_uses_the_automatic_workflow_entrypoint(
    repository: ProjectRepository,
) -> None:
    service = _service(repository)
    queued = service.enqueue(
        "project-id",
        "continue",
        credentials=WorkflowCredentials(llm_api_key="secret"),
    )
    claimed = repository.claim_next_job()
    assert claimed is not None

    service.execute(claimed)

    completed = repository.get_job(queued.id)
    assert completed is not None
    assert completed.current_step == "continued"


def test_credentials_can_be_reloaded_from_environment_metadata(
    repository: ProjectRepository,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    service = _service(repository)
    job = service.enqueue("project-id", "start")
    monkeypatch.setattr(CredentialStore, "load_env", lambda *_args, **_kwargs: None)
    monkeypatch.setattr(
        CredentialStore,
        "resolve_llm",
        lambda _self, provider: ResolvedKey(provider, "env-secret", "env"),
    )

    credentials = service._resolve_credentials(job)

    assert credentials == WorkflowCredentials(llm_api_key="env-secret", voice_api_key="")


def test_missing_reloaded_llm_credential_is_explained(
    repository: ProjectRepository,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    service = _service(repository)
    job = service.enqueue("project-id", "start")
    monkeypatch.setattr(CredentialStore, "load_env", lambda *_args, **_kwargs: None)
    monkeypatch.setattr(
        CredentialStore,
        "resolve_llm",
        lambda _self, provider: ResolvedKey(provider, "", ""),
    )

    with pytest.raises(ValueError, match="chave da IA"):
        service._resolve_credentials(job)


def test_voice_credential_can_reuse_the_matching_llm_key(
    repository: ProjectRepository,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    repository.create_project(
        ProjectCreate(
            title="Voz",
            problem_statement="Problema",
            llm_provider="OpenAI",
            voice_provider="OpenAI",
            voiceover_enabled=True,
            reuse_llm_api_key=True,
        ),
        project_id="voice-project",
    )
    service = _service(repository)
    job = service.enqueue("voice-project", "start")
    monkeypatch.setattr(CredentialStore, "load_env", lambda *_args, **_kwargs: None)
    monkeypatch.setattr(
        CredentialStore,
        "resolve_llm",
        lambda _self, provider: ResolvedKey(provider, "shared-secret", "env"),
    )

    credentials = service._resolve_credentials(job)

    assert credentials.voice_api_key == "shared-secret"


def test_missing_separate_voice_credential_is_explained(
    repository: ProjectRepository,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    repository.create_project(
        ProjectCreate(
            title="Voz",
            problem_statement="Problema",
            llm_provider="OpenAI",
            voice_provider="Google",
            voiceover_enabled=True,
        ),
        project_id="voice-project",
    )
    service = _service(repository)
    job = service.enqueue("voice-project", "start")
    monkeypatch.setattr(CredentialStore, "load_env", lambda *_args, **_kwargs: None)
    monkeypatch.setattr(
        CredentialStore,
        "resolve_llm",
        lambda _self, provider: ResolvedKey(provider, "llm-secret", "env"),
    )
    monkeypatch.setattr(
        CredentialStore,
        "resolve_voice",
        lambda _self, provider: ResolvedKey(provider, "", ""),
    )

    with pytest.raises(ValueError, match="chave de voz"):
        service._resolve_credentials(job)


def test_model_selection_rejects_incomplete_payloads() -> None:
    assert BackgroundJobService._model_selection({}) is None
    assert BackgroundJobService._model_selection({"model_selection": "bad"}) is None
    assert (
        BackgroundJobService._model_selection(
            {"model_selection": {"provider": "OpenAI", "model": ""}}
        )
        is None
    )


def test_retry_rejects_missing_and_non_terminal_jobs(repository: ProjectRepository) -> None:
    service = _service(repository)

    with pytest.raises(ValueError, match="não encontrado"):
        service.retry("missing")
    active = service.enqueue(
        "project-id",
        "start",
        credentials=WorkflowCredentials(llm_api_key="secret"),
    )
    with pytest.raises(ValueError, match="falha ou canceladas"):
        service.retry(active.id)


@pytest.mark.parametrize(
    "phase",
    (
        "review_solution_basis",
        "review_plan_presentation",
        "review_code_presentation",
        "review_plan_solution",
    ),
)
def test_reloaded_interface_observes_running_job(
    repository: ProjectRepository, phase: str
) -> None:
    BlockingWorkflow.entered = Event()
    BlockingWorkflow.release = Event()
    service = BackgroundJobService(repository, workflow_factory=BlockingWorkflow)
    queued = service.enqueue(
        "project-id",
        "resume",
        decision={"action": "approve"},
        credentials=WorkflowCredentials(llm_api_key="secret"),
        expected_phase=phase,
    )
    claimed = repository.claim_next_job()
    assert claimed is not None
    execution = Thread(target=service.execute, args=(claimed,))
    execution.start()
    assert BlockingWorkflow.entered.wait(timeout=2)

    reloaded_service = BackgroundJobService(repository, workflow_factory=BlockingWorkflow)
    observed = reloaded_service.active_job("project-id")

    assert observed is not None
    assert observed.id == queued.id
    assert observed.status == "running"
    assert observed.current_step == phase
    BlockingWorkflow.release.set()
    execution.join(timeout=2)
    assert execution.is_alive() is False


def test_retry_reuses_completed_operation_without_reinvoking_workflow(
    repository: ProjectRepository,
) -> None:
    CountingWorkflow.resume_calls = 0
    repository.update_project_status("project-id", "review_plan_presentation")
    service = BackgroundJobService(repository, workflow_factory=CountingWorkflow)
    first = service.enqueue(
        "project-id",
        "resume",
        decision={"action": "approve"},
        credentials=WorkflowCredentials(llm_api_key="secret"),
        expected_phase="review_plan_presentation",
    )
    claimed = repository.claim_next_job()
    assert claimed is not None
    service.execute(claimed)
    assert CountingWorkflow.resume_calls == 1

    repository.update_job(first.id, status="failed", error_message="simulated UI retry")
    retried = service.retry(
        first.id,
        credentials=WorkflowCredentials(llm_api_key="secret"),
    )
    claimed_retry = repository.claim_next_job()
    assert claimed_retry is not None
    service.execute(claimed_retry)

    assert retried.operation_id == first.operation_id
    assert CountingWorkflow.resume_calls == 1
    completed_events = [
        event
        for event in repository.list_workflow_events("project-id")
        if event.event_type == "job.completed"
    ]
    assert len(completed_events) == 1


def test_local_worker_claims_and_finishes_a_queued_job(
    repository: ProjectRepository,
) -> None:
    service = _service(repository)
    queued = service.enqueue(
        "project-id",
        "start",
        credentials=WorkflowCredentials(llm_api_key="secret"),
    )
    worker = BackgroundJobWorker(service)

    worker.start()
    worker.wake()
    for _ in range(100):
        current = repository.get_job(queued.id)
        if current is not None and current.status == "completed":
            break
        time.sleep(0.01)
    worker.stop()

    completed = repository.get_job(queued.id)
    assert completed is not None
    assert completed.status == "completed"
