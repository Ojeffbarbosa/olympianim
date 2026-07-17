"""Regression tests for the persistent LangGraph workflow facade."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from types import SimpleNamespace
from typing import Any, TypedDict

import pytest
from langgraph.checkpoint.memory import MemorySaver
from langgraph.graph import END, START, StateGraph
from langgraph.types import Command, interrupt

from olympianim.database.models import ProjectCreate
from olympianim.database.repository import ProjectRepository
from olympianim.services.artifact_service import ArtifactService
from olympianim.services.langgraph_workflow import (
    LangGraphWorkflowService,
    WorkflowCredentials,
    WorkflowModelSelection,
)


@dataclass
class _Connection:
    closed: bool = False

    def close(self) -> None:
        self.closed = True


def _task(name: str, *, interrupted: bool) -> SimpleNamespace:
    return SimpleNamespace(name=name, interrupts=(object(),) if interrupted else ())


def _snapshot(
    phase: str,
    *,
    interrupted: bool,
    checkpoint_id: str,
) -> SimpleNamespace:
    tasks = (_task("review_presentation_complete", interrupted=True),) if interrupted else ()
    return SimpleNamespace(
        values={"phase": phase},
        tasks=tasks,
        next=("review_presentation_complete",) if interrupted else (),
        config={
            "configurable": {
                "thread_id": "project-id",
                "checkpoint_id": checkpoint_id,
            }
        },
    )


class _Graph:
    def __init__(self, current: Any, *, after: Any, history: tuple[Any, ...] = ()) -> None:
        self.current = current
        self.after = after
        self.history = history
        self.invocations: list[tuple[Any, dict[str, Any]]] = []

    def get_state(self, config: dict[str, Any]) -> Any:
        _ = config
        return self.current

    def get_state_history(self, config: dict[str, Any]) -> tuple[Any, ...]:
        _ = config
        return self.history

    def invoke(self, command: Any, config: dict[str, Any]) -> None:
        self.invocations.append((command, config))
        self.current = self.after


class _RecoveryGraph:
    """Model the two invocations required to restart and resume a terminal graph."""

    def __init__(self) -> None:
        self.current = SimpleNamespace(
            values={"phase": "failed", "mode": "presentation", "render_error": "old"},
            tasks=(),
            next=(),
            config={"configurable": {"thread_id": "project-id"}},
        )
        self.invocations: list[tuple[Any, dict[str, Any]]] = []

    def get_state(self, config: dict[str, Any]) -> Any:
        _ = config
        return self.current

    def get_state_history(self, config: dict[str, Any]) -> tuple[Any, ...]:
        _ = config
        return ()

    def invoke(self, command: Any, config: dict[str, Any]) -> None:
        self.invocations.append((command, config))
        if isinstance(command, dict):
            self.current = _snapshot(
                "presentation_complete",
                interrupted=True,
                checkpoint_id="recovered",
            )
            self.current.values.update(command)
            return
        assert isinstance(command, Command)
        self.current = _snapshot(
            "review_plan_solution",
            interrupted=True,
            checkpoint_id="solution",
        )


class _WorkflowService(LangGraphWorkflowService):
    def __init__(self, repository: ProjectRepository, graph: Any) -> None:
        super().__init__(repository)
        self.graph = graph
        self.connections: list[_Connection] = []

    def _graph(
        self,
        project_id: str,
        credentials: WorkflowCredentials,
    ) -> tuple[Any, _Connection]:
        _ = (project_id, credentials)
        connection = _Connection()
        self.connections.append(connection)
        return self.graph, connection


@pytest.fixture
def repository(tmp_path: Path) -> ProjectRepository:
    repository = ProjectRepository(tmp_path / "olympianim.db")
    repository.create_project(
        ProjectCreate(title="Projeto", problem_statement="Problema"),
        project_id="project-id",
    )
    return repository


def test_resume_sends_model_selection_inside_the_resume_value(
    repository: ProjectRepository,
) -> None:
    graph = _Graph(
        _snapshot("review_plan_presentation", interrupted=True, checkpoint_id="live"),
        after=_snapshot("review_code_presentation", interrupted=True, checkpoint_id="after"),
    )
    service = _WorkflowService(repository, graph)

    result = service.resume(
        "project-id",
        {"action": "approve"},
        credentials=WorkflowCredentials(llm_api_key="secret"),
        model_selection=WorkflowModelSelection("OpenAI", "gpt-test"),
    )

    command, config = graph.invocations[0]
    assert isinstance(command, Command)
    assert command.resume == {
        "action": "approve",
        "model_selection": {"provider": "OpenAI", "model": "gpt-test"},
    }
    assert "checkpoint_id" not in config["configurable"]
    assert result["phase"] == "review_code_presentation"
    assert service.connections[0].closed is True


def test_snapshot_and_start_close_the_checkpoint_connection(
    repository: ProjectRepository,
) -> None:
    graph = _Graph(
        _snapshot("created", interrupted=False, checkpoint_id="before"),
        after=_snapshot(
            "review_plan_presentation",
            interrupted=True,
            checkpoint_id="after",
        ),
    )
    service = _WorkflowService(repository, graph)

    assert service.snapshot("project-id")["phase"] == "created"
    result = service.start(
        "project-id",
        credentials=WorkflowCredentials(llm_api_key="secret"),
        model_selection=WorkflowModelSelection("OpenAI", "planner-model"),
    )

    initial, _ = graph.invocations[0]
    assert initial["project_id"] == "project-id"
    assert initial["llm_model"] == "planner-model"
    assert result["phase"] == "review_plan_presentation"
    assert all(connection.closed for connection in service.connections)


def test_start_rejects_missing_project(repository: ProjectRepository) -> None:
    graph = _Graph(
        _snapshot("created", interrupted=False, checkpoint_id="before"),
        after=_snapshot("created", interrupted=False, checkpoint_id="after"),
    )
    service = _WorkflowService(repository, graph)

    with pytest.raises(ValueError, match="Projeto não encontrado"):
        service.start(
            "missing-project",
            credentials=WorkflowCredentials(llm_api_key="secret"),
        )


def test_resume_rejects_stale_decision_for_a_different_review_boundary(
    repository: ProjectRepository,
) -> None:
    graph = _Graph(
        _snapshot("review_code_presentation", interrupted=True, checkpoint_id="live"),
        after=_snapshot("presentation_complete", interrupted=True, checkpoint_id="after"),
    )
    service = _WorkflowService(repository, graph)

    with pytest.raises(RuntimeError, match="não corresponde à etapa"):
        service.resume(
            "project-id",
            {"action": "approve"},
            credentials=WorkflowCredentials(llm_api_key="secret"),
            operation_id="operation-old",
            expected_phase="review_plan_presentation",
        )

    assert graph.invocations == []


def test_resume_returns_current_state_when_operation_was_already_applied(
    repository: ProjectRepository,
) -> None:
    current = _snapshot(
        "review_code_presentation",
        interrupted=True,
        checkpoint_id="current",
    )
    current.values["last_applied_operation_id"] = "same-operation"
    graph = _Graph(
        current,
        after=_snapshot("presentation_complete", interrupted=True, checkpoint_id="after"),
    )
    service = _WorkflowService(repository, graph)

    result = service.resume(
        "project-id",
        {"action": "approve"},
        credentials=WorkflowCredentials(llm_api_key="secret"),
        operation_id="same-operation",
        expected_phase="review_plan_presentation",
    )

    assert result["phase"] == "review_code_presentation"
    assert graph.invocations == []


def test_applied_operation_finishes_its_pending_automatic_node(
    repository: ProjectRepository,
) -> None:
    current = SimpleNamespace(
        values={
            "phase": "build_presentation",
            "last_applied_operation_id": "same-operation",
        },
        tasks=(_task("build", interrupted=False),),
        next=("build",),
        config={"configurable": {"thread_id": "project-id"}},
    )
    graph = _Graph(
        current,
        after=_snapshot(
            "review_code_presentation",
            interrupted=True,
            checkpoint_id="after",
        ),
    )
    service = _WorkflowService(repository, graph)

    result = service.resume(
        "project-id",
        {"action": "approve"},
        credentials=WorkflowCredentials(llm_api_key="secret"),
        operation_id="same-operation",
        expected_phase="review_plan_presentation",
    )

    assert result["phase"] == "review_code_presentation"
    assert graph.invocations[0][0] is None


def test_resume_recovers_the_last_presentation_interrupt(
    repository: ProjectRepository,
) -> None:
    historical = _snapshot(
        "presentation_complete",
        interrupted=True,
        checkpoint_id="recoverable",
    )
    graph = _Graph(
        _snapshot("presentation_complete", interrupted=False, checkpoint_id="broken"),
        after=_snapshot("review_plan_solution", interrupted=True, checkpoint_id="after"),
        history=(historical,),
    )
    service = _WorkflowService(repository, graph)

    result = service.resume(
        "project-id",
        {"action": "generate_solution"},
        credentials=WorkflowCredentials(llm_api_key="secret"),
    )

    _, config = graph.invocations[0]
    assert config["configurable"]["checkpoint_id"] == "recoverable"
    assert result["phase"] == "review_plan_solution"
    logs = repository.list_logs("project-id")
    assert any(log.step == "workflow.checkpoint_recovered" for log in logs)


def test_resume_rejects_a_terminal_checkpoint_without_recoverable_interrupt(
    repository: ProjectRepository,
) -> None:
    graph = _Graph(
        _snapshot("completed", interrupted=False, checkpoint_id="terminal"),
        after=_snapshot("completed", interrupted=False, checkpoint_id="terminal"),
    )
    service = _WorkflowService(repository, graph)

    with pytest.raises(RuntimeError, match="não está aguardando uma decisão"):
        service.resume(
            "project-id",
            {"action": "approve"},
            credentials=WorkflowCredentials(llm_api_key="secret"),
        )

    assert graph.invocations == []
    assert service.connections[0].closed is True


def test_manual_presentation_recovery_requires_a_registered_nonempty_video(
    repository: ProjectRepository,
    tmp_path: Path,
) -> None:
    service = _WorkflowService(repository, _RecoveryGraph())
    snapshot = {"phase": "failed", "mode": "presentation"}
    projects_dir = tmp_path / "projects"
    artifacts = ArtifactService(repository=repository, projects_dir=projects_dir)
    video_path = artifacts.project_directory("project-id") / "presentation" / "manual.mp4"
    video_path.write_bytes(b"video")
    repository.update_project_artifacts(
        "project-id",
        presentation_video_path=str(video_path),
    )

    assert service.recoverable_manual_presentation_video("project-id", snapshot) == ""

    artifacts.register_video(
        "project-id",
        mode="presentation",
        video_path=video_path,
        version=7,
    )

    assert service.recoverable_manual_presentation_video(
        "project-id",
        snapshot,
    ) == str(video_path.resolve())
    assert (
        service.recoverable_manual_presentation_video(
            "project-id",
            {"phase": "failed", "mode": "solution"},
        )
        == ""
    )


def test_resume_rebuilds_review_boundary_from_a_registered_manual_presentation(
    repository: ProjectRepository,
    tmp_path: Path,
) -> None:
    artifacts = ArtifactService(repository=repository, projects_dir=tmp_path / "projects")
    video_path = artifacts.project_directory("project-id") / "presentation" / "manual.mp4"
    video_path.write_bytes(b"video")
    artifacts.register_video(
        "project-id",
        mode="presentation",
        video_path=video_path,
        version=7,
    )
    graph = _RecoveryGraph()
    service = _WorkflowService(repository, graph)

    result = service.resume(
        "project-id",
        {"action": "generate_solution"},
        credentials=WorkflowCredentials(llm_api_key="secret"),
        model_selection=WorkflowModelSelection("OpenAI", "solution-model"),
        operation_id="recover-operation",
        expected_phase="presentation_complete",
    )

    recovery_input, recovery_config = graph.invocations[0]
    assert recovery_input["workflow_entry"] == "recover_manual_presentation"
    assert recovery_input["presentation_render_path"] == str(video_path.resolve())
    assert recovery_input["render_error"] == ""
    assert "checkpoint_id" not in recovery_config["configurable"]
    resume_command, _ = graph.invocations[1]
    assert isinstance(resume_command, Command)
    assert resume_command.resume == {
        "action": "generate_solution",
        "model_selection": {"provider": "OpenAI", "model": "solution-model"},
        "operation_id": "recover-operation",
        "expected_phase": "presentation_complete",
    }
    assert result["phase"] == "review_plan_solution"
    assert any(
        log.step == "workflow.manual_presentation_recovered"
        for log in repository.list_logs("project-id")
    )


def test_resume_retries_an_automatic_node_without_replaying_human_input(
    repository: ProjectRepository,
) -> None:
    pending = SimpleNamespace(
        values={"phase": "plan_solution"},
        tasks=(_task("plan_solution", interrupted=False),),
        next=("plan_solution",),
        config={"configurable": {"thread_id": "project-id"}},
    )
    graph = _Graph(
        pending,
        after=_snapshot("review_plan_solution", interrupted=True, checkpoint_id="after"),
    )
    service = _WorkflowService(repository, graph)

    result = service.resume(
        "project-id",
        {"action": "generate_solution"},
        credentials=WorkflowCredentials(llm_api_key="secret"),
        model_selection=WorkflowModelSelection("OpenAI", "ignored-on-retry"),
    )

    command, config = graph.invocations[0]
    assert command is None
    assert "checkpoint_id" not in config["configurable"]
    assert result["phase"] == "review_plan_solution"
    logs = repository.list_logs("project-id")
    assert any(log.step == "workflow.retry_pending" for log in logs)


def test_continue_run_advances_only_an_automatic_checkpoint(
    repository: ProjectRepository,
) -> None:
    pending = SimpleNamespace(
        values={"phase": "render_presentation"},
        tasks=(_task("render", interrupted=False),),
        next=("render",),
        config={"configurable": {"thread_id": "project-id"}},
    )
    graph = _Graph(
        pending,
        after=_snapshot("presentation_complete", interrupted=True, checkpoint_id="after"),
    )
    service = _WorkflowService(repository, graph)

    result = service.continue_run(
        "project-id",
        credentials=WorkflowCredentials(llm_api_key="secret"),
    )

    assert result["phase"] == "presentation_complete"
    assert graph.invocations[0][0] is None
    assert service.connections[0].closed is True


def test_continue_run_rejects_review_and_terminal_checkpoints(
    repository: ProjectRepository,
) -> None:
    interrupted_graph = _Graph(
        _snapshot("review_plan_presentation", interrupted=True, checkpoint_id="review"),
        after=_snapshot("review_plan_presentation", interrupted=True, checkpoint_id="after"),
    )
    with pytest.raises(RuntimeError, match="aguarda uma decisão"):
        _WorkflowService(repository, interrupted_graph).continue_run(
            "project-id",
            credentials=WorkflowCredentials(llm_api_key="secret"),
        )

    terminal_graph = _Graph(
        _snapshot("completed", interrupted=False, checkpoint_id="done"),
        after=_snapshot("completed", interrupted=False, checkpoint_id="after"),
    )
    with pytest.raises(RuntimeError, match="Não há etapa automática"):
        _WorkflowService(repository, terminal_graph).continue_run(
            "project-id",
            credentials=WorkflowCredentials(llm_api_key="secret"),
        )


def test_google_voice_prompt_is_versioned_in_project_history(
    repository: ProjectRepository,
) -> None:
    graph = _Graph(
        _snapshot("created", interrupted=False, checkpoint_id="before"),
        after=_snapshot("created", interrupted=False, checkpoint_id="after"),
    )
    service = _WorkflowService(repository, graph)

    template = service._voice_prompt_template("project-id", "Google")

    assert "{transcript}" in template
    snapshots = repository.list_project_prompts("project-id")
    assert len(snapshots) == 1
    assert snapshots[0].agent_type == "gemini_tts"


class _LegacyState(TypedDict, total=False):
    phase: str
    llm_provider: str
    llm_model: str


def _legacy_graph(checkpointer: MemorySaver, *, fixed: bool) -> Any:
    def render_presentation(state: _LegacyState) -> Command[str]:
        _ = state
        return Command(
            update={"phase": "presentation_complete"},
            goto="review_presentation_complete",
        )

    def review_presentation_complete(state: _LegacyState) -> Command[str]:
        if fixed:
            decision = interrupt({"phase": state["phase"]})
            selection = decision["model_selection"]
            return Command(
                update={
                    "phase": "review_plan_solution",
                    "llm_provider": selection["provider"],
                    "llm_model": selection["model"],
                },
                goto=END,
            )
        interrupt({"phase": state["phase"]})
        return Command(update={"phase": "review_plan_solution"}, goto=END)

    builder = StateGraph(_LegacyState)
    builder.add_node("render_presentation", render_presentation)
    builder.add_node("review_presentation_complete", review_presentation_complete)
    builder.add_edge(START, "render_presentation")
    return builder.compile(checkpointer=checkpointer)


def test_real_langgraph_history_recovers_a_checkpoint_broken_by_the_old_resume_order(
    repository: ProjectRepository,
) -> None:
    checkpointer = MemorySaver()
    config = {"configurable": {"thread_id": "project-id"}}
    old_graph = _legacy_graph(checkpointer, fixed=False)
    old_graph.invoke({"phase": "presentation_complete"}, config)
    old_graph.update_state(
        config,
        {"llm_provider": "OpenAI", "llm_model": "old-model"},
    )
    old_graph.invoke(Command(resume={"action": "generate_solution"}), config)
    assert old_graph.get_state(config).next == ()

    fixed_graph = _legacy_graph(checkpointer, fixed=True)
    service = _WorkflowService(repository, fixed_graph)
    result = service.resume(
        "project-id",
        {"action": "generate_solution"},
        credentials=WorkflowCredentials(llm_api_key="secret"),
        model_selection=WorkflowModelSelection("OpenAI", "new-model"),
    )

    assert result == {
        "phase": "review_plan_solution",
        "llm_provider": "OpenAI",
        "llm_model": "new-model",
    }
