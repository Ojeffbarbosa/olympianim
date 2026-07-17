"""Persistent background execution for LangGraph workflow actions."""

from __future__ import annotations

import hashlib
import json
import threading
import uuid
from collections.abc import Callable
from datetime import UTC, datetime, timedelta
from typing import Any, Final

from olympianim.database.models import GenerationJobRecord
from olympianim.database.repository import ProjectRepository
from olympianim.schemas.workflow import (
    ResumeRequest,
    WorkflowJobAction,
    WorkflowPhase,
    is_review_phase,
)
from olympianim.services.credential_service import CredentialStore
from olympianim.services.langgraph_workflow import (
    LangGraphWorkflowService,
    WorkflowCredentials,
    WorkflowModelSelection,
)

POLL_INTERVAL_SECONDS: Final[float] = 0.5
HEARTBEAT_INTERVAL_SECONDS: Final[float] = 5.0
STALE_JOB_SECONDS: Final[int] = 30


class JobCancelledError(RuntimeError):
    """Signal cooperative cancellation between workflow nodes."""


class BackgroundJobService:
    """Queue workflow actions and keep credentials in process memory only."""

    def __init__(
        self,
        repository: ProjectRepository | None = None,
        *,
        workflow_factory: Callable[..., LangGraphWorkflowService] = LangGraphWorkflowService,
    ) -> None:
        self.repository = repository or ProjectRepository()
        self.workflow_factory = workflow_factory
        self._credentials: dict[str, WorkflowCredentials] = {}
        self._credentials_lock = threading.Lock()

    def enqueue(
        self,
        project_id: str,
        action: str,
        *,
        decision: dict[str, Any] | None = None,
        credentials: WorkflowCredentials | None = None,
        model_selection: WorkflowModelSelection | None = None,
        operation_id: str | None = None,
        expected_phase: str | None = None,
    ) -> GenerationJobRecord:
        """Persist a workflow action and retain its secrets only in memory."""
        if action not in {item.value for item in WorkflowJobAction}:
            raise ValueError(f"Ação de trabalho não suportada: {action!r}")
        project = self.repository.get_project(project_id)
        if project is None:
            raise ValueError("Projeto não encontrado.")
        if self.repository.get_active_job(project_id) is not None:
            raise ValueError("Já existe uma etapa em processamento para este projeto.")
        stable_operation_id = operation_id or str(uuid.uuid4())
        boundary = expected_phase if expected_phase is not None else ""
        if action == WorkflowJobAction.RESUME and expected_phase is None:
            requested_action = str((decision or {}).get("action", ""))
            if requested_action == "generate_solution":
                boundary = WorkflowPhase.PRESENTATION_COMPLETE
            elif is_review_phase(project.status):
                boundary = project.status
            else:
                boundary = WorkflowPhase.REVIEW_PLAN_PRESENTATION
        if action == WorkflowJobAction.RESUME:
            ResumeRequest(
                operation_id=stable_operation_id,
                expected_phase=boundary,
                decision=dict(decision or {}),
                workflow_revision=project.workflow_revision,
            ).validate()
        payload_data: dict[str, Any] = {"decision": decision or {}}
        if model_selection is not None:
            payload_data["model_selection"] = {
                "provider": model_selection.provider,
                "model": model_selection.model,
            }
        payload = json.dumps(payload_data, ensure_ascii=False)
        decision_sha256 = hashlib.sha256(
            json.dumps(decision or {}, ensure_ascii=False, sort_keys=True).encode("utf-8")
        ).hexdigest()
        self.repository.create_transition(
            project_id,
            operation_id=stable_operation_id,
            action=action,
            expected_phase=boundary,
            decision_sha256=decision_sha256,
            workflow_revision=project.workflow_revision,
        )
        try:
            job = self.repository.create_job(
                project_id,
                action=action,
                payload=payload,
                operation_id=stable_operation_id,
                expected_phase=boundary,
                workflow_revision=project.workflow_revision,
            )
        except Exception:
            self.repository.discard_orphan_transition(stable_operation_id)
            raise
        if credentials is not None:
            with self._credentials_lock:
                self._credentials[job.id] = credentials
        self.repository.update_project_status(project_id, "queued")
        self.repository.add_log(
            project_id,
            step="job.queued",
            message=f"Etapa adicionada à fila em segundo plano ({job.id}).",
        )
        self.repository.add_workflow_event(
            project_id,
            event_key=f"{stable_operation_id}:queued",
            event_type="job.queued",
            operation_id=stable_operation_id,
            job_id=job.id,
            phase=boundary,
            payload=json.dumps({"action": action}, ensure_ascii=False),
        )
        if model_selection is not None:
            self.repository.add_log(
                project_id,
                step="job.model_selected",
                message=(
                    "Modelo selecionado para a próxima etapa: "
                    f"{model_selection.provider}:{model_selection.model}."
                ),
            )
        return job

    def retry(
        self,
        job_id: str,
        *,
        credentials: WorkflowCredentials | None = None,
        model_selection: WorkflowModelSelection | None = None,
    ) -> GenerationJobRecord:
        """Queue a new attempt using the failed job's persisted action."""
        source = self._terminal_job(job_id)
        payload = json.loads(source.payload or "{}")
        selection = model_selection or self._model_selection(payload)
        return self.enqueue(
            source.project_id,
            source.action,
            decision=payload.get("decision", {}),
            credentials=credentials,
            model_selection=selection,
            operation_id=source.operation_id,
            expected_phase=source.expected_phase,
        )

    def resume(
        self, job_id: str, *, credentials: WorkflowCredentials | None = None
    ) -> GenerationJobRecord:
        """Resume a cancelled job as a new auditable attempt."""
        return self.retry(job_id, credentials=credentials)

    def retry_and_wake(
        self,
        job_id: str,
        *,
        credentials: WorkflowCredentials | None = None,
        model_selection: WorkflowModelSelection | None = None,
    ) -> GenerationJobRecord:
        """Queue a retry and immediately notify the shared worker."""
        job = self.retry(
            job_id,
            credentials=credentials,
            model_selection=model_selection,
        )
        wake_background_worker()
        return job

    def cancel(self, job_id: str) -> bool:
        """Request cancellation without terminating the worker process."""
        return self.repository.request_job_cancellation(job_id)

    def active_job(self, project_id: str) -> GenerationJobRecord | None:
        """Return the project's currently active job."""
        return self.repository.get_active_job(project_id)

    def latest_job(self, project_id: str) -> GenerationJobRecord | None:
        """Return the project's latest job."""
        return self.repository.latest_job(project_id)

    def execute(self, job: GenerationJobRecord) -> None:
        """Execute one claimed job and persist its complete lifecycle."""
        heartbeat_stop = threading.Event()
        heartbeat = threading.Thread(
            target=self._heartbeat,
            args=(job.id, heartbeat_stop),
            name=f"olympianim-heartbeat-{job.id}",
            daemon=True,
        )
        heartbeat.start()
        try:
            self._raise_if_cancelled(job.id)
            credentials = self._resolve_credentials(job)
            workflow = self.workflow_factory(
                repository=self.repository,
                cancellation_check=lambda: self._raise_if_cancelled(job.id),
                progress_callback=lambda step, progress: self.repository.update_job(
                    job.id,
                    current_step=step,
                    progress=progress,
                    heartbeat=True,
                ),
                execution_id=job.id,
                operation_id=job.operation_id,
            )
            transition = self.repository.get_transition(job.operation_id)
            if transition is None or transition.status != "completed":
                self.repository.update_transition(job.operation_id, status="running")
            self.repository.update_project_status(job.project_id, "processing")
            self.repository.update_job(
                job.id, current_step="workflow", progress=20, heartbeat=True
            )
            snapshot = self._run_workflow(workflow, job, credentials)
            self._raise_if_cancelled(job.id)
            phase = str(snapshot.get("phase", "")).strip()
            if not phase:
                raise RuntimeError("O workflow terminou sem informar a fase alcançada.")
            self._ensure_requested_transition_completed(job, snapshot)
            payload = json.loads(job.payload or "{}")
            decision = payload.get("decision", {})
            decision_action = str(decision.get("action", "")) if isinstance(decision, dict) else ""
            job_result = json.dumps(
                {
                    "phase": phase,
                    "workflow_action": job.action,
                    "decision_action": decision_action,
                },
                ensure_ascii=False,
            )
            snapshot_json = json.dumps(snapshot, ensure_ascii=False, default=str)
            self.repository.complete_job_and_transition(
                job_id=job.id,
                operation_id=job.operation_id,
                project_id=job.project_id,
                phase=phase,
                job_result=job_result,
                result_snapshot=snapshot_json,
            )
            self.repository.add_log(
                job.project_id,
                step="job.completed",
                message=f"Etapa em segundo plano concluída ({job.id}): {phase}.",
            )
            self.repository.add_workflow_event(
                job.project_id,
                event_key=f"{job.operation_id}:completed",
                event_type="job.completed",
                operation_id=job.operation_id,
                job_id=job.id,
                phase=phase,
                payload=json.dumps({"action": job.action}, ensure_ascii=False),
            )
        except JobCancelledError:
            self.repository.update_job(
                job.id,
                status="cancelled",
                current_step="cancelled",
                error_message="Etapa cancelada pelo usuário.",
                heartbeat=True,
            )
            self.repository.update_project_status(job.project_id, "cancelled")
            self.repository.update_transition(job.operation_id, status="cancelled")
            self.repository.add_log(
                job.project_id,
                level="warning",
                step="job.cancelled",
                message=f"Etapa cancelada ({job.id}).",
            )
            self.repository.add_workflow_event(
                job.project_id,
                event_key=f"{job.operation_id}:cancelled",
                event_type="job.cancelled",
                operation_id=job.operation_id,
                job_id=job.id,
                phase="cancelled",
            )
        except Exception as exc:
            message = str(exc) or exc.__class__.__name__
            self.repository.update_job(
                job.id,
                status="failed",
                current_step="failed",
                error_message=message,
                heartbeat=True,
            )
            self.repository.update_project_status(job.project_id, "failed")
            self.repository.update_transition(job.operation_id, status="failed")
            self.repository.add_log(
                job.project_id,
                level="error",
                step="job.failed",
                message=f"Falha na etapa em segundo plano ({job.id}): {message}",
            )
            self.repository.add_workflow_event(
                job.project_id,
                event_key=f"{job.operation_id}:failed",
                event_type="job.failed",
                operation_id=job.operation_id,
                job_id=job.id,
                phase="failed",
                payload=json.dumps(
                    {"error_type": exc.__class__.__name__},
                    ensure_ascii=False,
                ),
            )
        finally:
            heartbeat_stop.set()
            heartbeat.join(timeout=1)
            with self._credentials_lock:
                self._credentials.pop(job.id, None)

    def recover_stale_jobs(self) -> int:
        """Requeue jobs whose worker heartbeat expired."""
        stale_before = (datetime.now(UTC) - timedelta(seconds=STALE_JOB_SECONDS)).isoformat(
            timespec="seconds"
        )
        return self.repository.recover_stale_jobs(stale_before)

    def _run_workflow(
        self,
        workflow: LangGraphWorkflowService,
        job: GenerationJobRecord,
        credentials: WorkflowCredentials,
    ) -> dict[str, Any]:
        transition = self.repository.get_transition(job.operation_id)
        if transition is not None and transition.status == "completed":
            snapshot = json.loads(transition.result_snapshot or "{}")
            if isinstance(snapshot, dict) and snapshot:
                return dict(snapshot)
        payload = json.loads(job.payload or "{}")
        selection = self._model_selection(payload)
        if job.action == "start":
            existing = workflow.snapshot(job.project_id, credentials=credentials)
            if existing:
                phase = str(existing.get("phase", ""))
                if phase.startswith("review_") or phase in {
                    "presentation_complete",
                    "completed",
                    "failed",
                    "stopped",
                }:
                    return existing
                return workflow.continue_run(
                    job.project_id,
                    credentials=credentials,
                )
            return workflow.start(
                job.project_id,
                credentials=credentials,
                model_selection=selection,
            )
        if job.action == "resume":
            return workflow.resume(
                job.project_id,
                dict(payload.get("decision", {})),
                credentials=credentials,
                model_selection=selection,
                operation_id=job.operation_id,
                expected_phase=job.expected_phase,
            )
        return workflow.continue_run(
            job.project_id,
            credentials=credentials,
        )

    @staticmethod
    def _ensure_requested_transition_completed(
        job: GenerationJobRecord,
        snapshot: dict[str, Any],
    ) -> None:
        """Reject successful no-ops at explicit human-review boundaries."""
        if job.action != "resume":
            return
        payload = json.loads(job.payload or "{}")
        decision = payload.get("decision", {})
        action = decision.get("action") if isinstance(decision, dict) else None
        if action == "generate_solution" and snapshot.get("phase") == "presentation_complete":
            raise RuntimeError("A solicitação para gerar a resolução não avançou o workflow.")

    def _resolve_credentials(self, job: GenerationJobRecord) -> WorkflowCredentials:
        with self._credentials_lock:
            credentials = self._credentials.get(job.id)
        if credentials is not None and credentials.llm_api_key:
            return credentials

        project = self.repository.get_project(job.project_id)
        if project is None:
            raise ValueError("Projeto não encontrado.")
        CredentialStore.load_env()
        store = CredentialStore()
        payload = json.loads(job.payload or "{}")
        selection = self._model_selection(payload)
        llm_provider = selection.provider if selection else project.llm_provider
        llm_key = store.resolve_llm(llm_provider).value
        voice_key = ""
        if project.voiceover_enabled:
            if project.reuse_llm_api_key and project.voice_provider == llm_provider:
                voice_key = llm_key
            else:
                voice_key = store.resolve_voice(project.voice_provider).value
        if not llm_key:
            raise ValueError(
                "A chave da IA não está mais disponível em memória. "
                "Informe-a novamente e repita a etapa."
            )
        if project.voiceover_enabled and not voice_key:
            raise ValueError(
                "A chave de voz não está mais disponível em memória. "
                "Informe-a novamente e repita a etapa."
            )
        return WorkflowCredentials(llm_api_key=llm_key, voice_api_key=voice_key)

    @staticmethod
    def _model_selection(payload: dict[str, Any]) -> WorkflowModelSelection | None:
        value = payload.get("model_selection")
        if not isinstance(value, dict):
            return None
        provider = str(value.get("provider", "")).strip()
        model = str(value.get("model", "")).strip()
        if not provider or not model:
            return None
        return WorkflowModelSelection(provider=provider, model=model)

    def _raise_if_cancelled(self, job_id: str) -> None:
        if self.repository.is_job_cancellation_requested(job_id):
            raise JobCancelledError

    def _heartbeat(self, job_id: str, stop: threading.Event) -> None:
        while not stop.wait(HEARTBEAT_INTERVAL_SECONDS):
            self.repository.update_job(job_id, heartbeat=True)

    def _terminal_job(self, job_id: str) -> GenerationJobRecord:
        job = self.repository.get_job(job_id)
        if job is None:
            raise ValueError("Trabalho não encontrado.")
        if job.status not in {"failed", "cancelled"}:
            raise ValueError("Somente etapas com falha ou canceladas podem ser retomadas.")
        return job


class BackgroundJobWorker:
    """Single local worker that serializes Manim and workflow execution."""

    def __init__(self, service: BackgroundJobService) -> None:
        self.service = service
        self._stop = threading.Event()
        self._wake = threading.Event()
        self._thread = threading.Thread(
            target=self._run,
            name="olympianim-background-worker",
            daemon=True,
        )

    def start(self) -> None:
        """Start the worker once and recover abandoned work."""
        if self._thread.is_alive():
            self._wake.set()
            return
        self.service.recover_stale_jobs()
        self._thread.start()

    def wake(self) -> None:
        """Wake the worker after a new job is queued."""
        self._wake.set()

    def stop(self) -> None:
        """Stop polling after the current job reaches a cooperative boundary."""
        self._stop.set()
        self._wake.set()
        self._thread.join(timeout=2)

    def _run(self) -> None:
        while not self._stop.is_set():
            job = self.service.repository.claim_next_job()
            if job is None:
                self._wake.wait(POLL_INTERVAL_SECONDS)
                self._wake.clear()
                continue
            self.service.execute(job)


_WORKER_LOCK = threading.Lock()
_SHARED_SERVICE: BackgroundJobService | None = None
_SHARED_WORKER: BackgroundJobWorker | None = None


def get_background_job_service() -> BackgroundJobService:
    """Return the process-wide queue service and ensure its worker is running."""
    global _SHARED_SERVICE, _SHARED_WORKER
    with _WORKER_LOCK:
        if _SHARED_SERVICE is None:
            _SHARED_SERVICE = BackgroundJobService()
            _SHARED_WORKER = BackgroundJobWorker(_SHARED_SERVICE)
        worker = _SHARED_WORKER
    if worker is not None:
        worker.start()
    return _SHARED_SERVICE


def wake_background_worker() -> None:
    """Notify the process-wide worker that work is available."""
    with _WORKER_LOCK:
        worker = _SHARED_WORKER
    if worker is not None:
        worker.wake()
