"""Small Streamlit-facing facade over the native LangGraph checkpoint."""

from __future__ import annotations

import sqlite3
from collections.abc import Callable, Mapping
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from langgraph.checkpoint.sqlite import SqliteSaver
from langgraph.types import Command

from olympianim.database.repository import ProjectRepository
from olympianim.graph.approved_workflow import (
    MANUAL_PRESENTATION_RECOVERY_ENTRY,
    build_approved_workflow,
)
from olympianim.manim.presentation import PresentationRenderer
from olympianim.prompts.service import PromptService
from olympianim.services.artifact_service import ArtifactService
from olympianim.services.llm_service import LLMService
from olympianim.services.usage_service import UsageService


@dataclass(frozen=True)
class WorkflowCredentials:
    """Sensitive credentials kept only in the active Streamlit invocation."""

    llm_api_key: str = ""
    voice_api_key: str = ""


@dataclass(frozen=True)
class WorkflowModelSelection:
    """Non-sensitive model choice captured for one workflow transition."""

    provider: str
    model: str


@dataclass(frozen=True)
class _PresentationRecoveryArtifacts:
    """Registered artifacts that can restore the presentation review boundary."""

    video_path: str
    subtitle_path: str
    version: int


class LangGraphWorkflowService:
    """Start and resume the only production workflow using one thread per project."""

    def __init__(
        self,
        repository: ProjectRepository | None = None,
        *,
        cancellation_check: Callable[[], None] | None = None,
        progress_callback: Callable[[str, int], None] | None = None,
        execution_id: str = "",
        operation_id: str = "",
    ) -> None:
        self.repository = repository or ProjectRepository()
        self.cancellation_check = cancellation_check or (lambda: None)
        self.progress_callback = progress_callback or (lambda _step, _progress: None)
        self.execution_id = execution_id
        self.operation_id = operation_id or execution_id

    def snapshot(
        self, project_id: str, *, credentials: WorkflowCredentials | None = None
    ) -> dict[str, Any]:
        graph, connection = self._graph(project_id, credentials or WorkflowCredentials())
        try:
            state = graph.get_state(self._config(project_id))
            return dict(state.values)
        finally:
            connection.close()

    def recoverable_manual_presentation_video(
        self,
        project_id: str,
        snapshot: Mapping[str, Any],
    ) -> str:
        """Return a valid registered video that can resume a failed presentation."""
        artifacts = self._presentation_recovery_artifacts(project_id, snapshot)
        return artifacts.video_path if artifacts is not None else ""

    def start(
        self,
        project_id: str,
        *,
        credentials: WorkflowCredentials,
        model_selection: WorkflowModelSelection | None = None,
    ) -> dict[str, Any]:
        project = self.repository.get_project(project_id)
        if project is None:
            raise ValueError("Projeto não encontrado.")
        selection = model_selection or WorkflowModelSelection(
            project.llm_provider, project.llm_model
        )
        voice_prompt_template = self._voice_prompt_template(project_id, project.voice_provider)
        self.repository.add_log(
            project_id,
            step="workflow.start",
            message="Fluxo de geração iniciado.",
        )
        graph, connection = self._graph(project_id, credentials)
        try:
            graph.invoke(
                {
                    "project_id": project.id,
                    "problem_statement": project.problem_statement,
                    "teacher_solution": project.teacher_solution,
                    "teacher_instructions": project.teacher_instructions,
                    "llm_provider": selection.provider,
                    "llm_model": selection.model,
                    "voice": {
                        "enabled": project.voiceover_enabled,
                        "provider": project.voice_provider,
                        "model": project.voice_model,
                        "voice": project.voice,
                        "language": project.voice_language,
                        "speed": project.voice_speed,
                    },
                    "voice_prompt_template": voice_prompt_template,
                    "color_palette_snapshot": project.color_palette_snapshot,
                    "retry_count": 0,
                },
                self._config(project_id),
            )
            return dict(graph.get_state(self._config(project_id)).values)
        finally:
            connection.close()

    def resume(
        self,
        project_id: str,
        decision: dict[str, Any],
        *,
        credentials: WorkflowCredentials,
        model_selection: WorkflowModelSelection | None = None,
        operation_id: str = "",
        expected_phase: str = "",
    ) -> dict[str, Any]:
        graph, connection = self._graph(project_id, credentials)
        try:
            thread_config = self._config(project_id)
            current = graph.get_state(thread_config)
            stable_operation_id = operation_id or self.operation_id
            if stable_operation_id and (
                current.values.get("last_applied_operation_id") == stable_operation_id
            ):
                if current.next and not self._has_pending_interrupt(current):
                    self.repository.add_log(
                        project_id,
                        step="workflow.operation_continue",
                        message=(
                            "A decisão já havia sido aplicada; a etapa automática "
                            "pendente será concluída."
                        ),
                    )
                    graph.invoke(None, thread_config)
                    return dict(graph.get_state(thread_config).values)
                self.repository.add_log(
                    project_id,
                    step="workflow.operation_reused",
                    message="A transição já havia sido aplicada; o estado foi reutilizado.",
                )
                return dict(current.values)
            resume_config = self._resume_config(
                graph,
                project_id,
                thread_config,
                decision,
                expected_phase=expected_phase,
            )
            if resume_config is None:
                graph.invoke(None, thread_config)
            else:
                resume_value = self._resume_value(
                    decision,
                    model_selection,
                    operation_id=stable_operation_id,
                    expected_phase=expected_phase,
                )
                graph.invoke(Command(resume=resume_value), resume_config)
            return dict(graph.get_state(thread_config).values)
        finally:
            connection.close()

    def continue_run(
        self,
        project_id: str,
        *,
        credentials: WorkflowCredentials,
    ) -> dict[str, Any]:
        """Continue a checkpoint that was interrupted outside a review boundary."""
        graph, connection = self._graph(project_id, credentials)
        try:
            thread_config = self._config(project_id)
            state = graph.get_state(thread_config)
            if self._has_pending_interrupt(state):
                raise RuntimeError(
                    "O fluxo aguarda uma decisão do usuário e não pode continuar automaticamente."
                )
            if not state.next:
                raise RuntimeError("Não há etapa automática pendente neste projeto.")
            graph.invoke(None, thread_config)
            return dict(graph.get_state(thread_config).values)
        finally:
            connection.close()

    def _voice_prompt_template(self, project_id: str, provider: str) -> str:
        if provider != "Google":
            return "{transcript}"
        service = PromptService(repository=self.repository)
        prompt = next(
            item
            for item in service.list_prompts("gemini_tts")
            if item.prompt.name == "Direção de narração Gemini - padrão"
        )
        self.repository.record_project_prompt(
            project_id,
            agent_type=prompt.prompt.agent_type,
            prompt_id=prompt.prompt.id,
            prompt_version=prompt.latest_version.version,
            rendered_prompt_snapshot=prompt.latest_version.template_text,
        )
        return prompt.latest_version.template_text

    def _graph(
        self,
        project_id: str,
        credentials: WorkflowCredentials,
    ) -> tuple[Any, sqlite3.Connection]:
        project = self.repository.get_project(project_id)
        connection = sqlite3.connect(
            self.repository.database_path,
            timeout=30,
            check_same_thread=False,
        )
        connection.execute("PRAGMA busy_timeout = 30000")
        connection.execute("PRAGMA journal_mode = WAL")
        saver = SqliteSaver(connection)
        saver.setup()
        return (
            build_approved_workflow(
                llm_api_key=credentials.llm_api_key,
                voice_api_key=credentials.voice_api_key,
                llm_service=LLMService(usage_service=UsageService(self.repository)),
                prompt_service=PromptService(repository=self.repository),
                artifact_service=ArtifactService(repository=self.repository),
                renderer=PresentationRenderer(cancellation_check=self.cancellation_check),
                repository=self.repository,
                project_id=project_id,
                checkpointer=saver,
                cancellation_check=self.cancellation_check,
                progress_callback=self.progress_callback,
                execution_id=self.execution_id,
                operation_id=self.operation_id,
                workflow_revision=project.workflow_revision if project is not None else 1,
            ),
            connection,
        )

    def _resume_config(
        self,
        graph: Any,
        project_id: str,
        thread_config: dict[str, Any],
        decision: dict[str, Any],
        *,
        expected_phase: str = "",
    ) -> dict[str, Any] | None:
        """Find a resumable interrupt or identify an automatic retry checkpoint."""
        current = graph.get_state(thread_config)
        if self._has_pending_interrupt(current):
            current_phase = str(current.values.get("phase", ""))
            if expected_phase and current_phase != expected_phase:
                raise RuntimeError(
                    "A decisão não corresponde à etapa atualmente aguardada "
                    f"(esperada: {expected_phase}; atual: {current_phase or 'desconhecida'})."
                )
            return thread_config

        if (
            current.values.get("phase") == "presentation_complete"
            and decision.get("action") == "generate_solution"
            and (not expected_phase or expected_phase == "presentation_complete")
        ):
            for historical in graph.get_state_history(thread_config):
                if (
                    historical.values.get("phase") == "presentation_complete"
                    and self._has_pending_interrupt(historical)
                    and any(
                        task.name == "review_presentation_complete" for task in historical.tasks
                    )
                ):
                    self.repository.add_log(
                        project_id,
                        level="warning",
                        step="workflow.checkpoint_recovered",
                        message=(
                            "Checkpoint de aprovação da apresentação recuperado "
                            "para iniciar a resolução."
                        ),
                    )
                    return dict(historical.config)

        if decision.get("action") == "generate_solution" and (
            not expected_phase or expected_phase == "presentation_complete"
        ):
            artifacts = self._presentation_recovery_artifacts(project_id, current.values)
            if artifacts is not None:
                graph.invoke(
                    {
                        "workflow_entry": MANUAL_PRESENTATION_RECOVERY_ENTRY,
                        "project_id": project_id,
                        "mode": "presentation",
                        "phase": "presentation_complete",
                        "render_path": artifacts.video_path,
                        "presentation_render_path": artifacts.video_path,
                        "presentation_subtitle_path": artifacts.subtitle_path,
                        "render_error": "",
                        "retry_count": 0,
                    },
                    thread_config,
                )
                recovered = graph.get_state(thread_config)
                if not (
                    recovered.values.get("phase") == "presentation_complete"
                    and self._has_pending_interrupt(recovered)
                    and any(
                        task.name == "review_presentation_complete" for task in recovered.tasks
                    )
                ):
                    raise RuntimeError(
                        "Não foi possível restaurar a etapa de conclusão da apresentação."
                    )
                self.repository.add_log(
                    project_id,
                    step="workflow.manual_presentation_recovered",
                    message=(
                        "Vídeo de apresentação renderizado no editor reconhecido; "
                        "o fluxo foi restaurado para iniciar a resolução."
                    ),
                )
                return thread_config

        if current.next:
            self.repository.add_log(
                project_id,
                step="workflow.retry_pending",
                message=(
                    "Etapa automática pendente retomada após falha: "
                    + ", ".join(str(node) for node in current.next)
                    + "."
                ),
            )
            return None

        raise RuntimeError(
            "O fluxo não está aguardando uma decisão nesta etapa. "
            "Recarregue o projeto antes de tentar novamente."
        )

    def _presentation_recovery_artifacts(
        self,
        project_id: str,
        snapshot: Mapping[str, Any],
    ) -> _PresentationRecoveryArtifacts | None:
        """Validate persisted evidence before recovering a failed presentation."""
        if snapshot.get("phase") != "failed" or snapshot.get("mode") != "presentation":
            return None
        project = self.repository.get_project(project_id)
        if project is None or not project.presentation_video_path:
            return None
        video = self._existing_nonempty_file(project.presentation_video_path)
        if video is None:
            return None

        records = self.repository.list_generated_files(project_id)
        video_record = next(
            (
                record
                for record in reversed(records)
                if record.file_type == "presentation_video" and self._same_path(record.path, video)
            ),
            None,
        )
        if video_record is None:
            return None

        subtitle_path = ""
        subtitle_record = next(
            (
                record
                for record in reversed(records)
                if record.file_type == "presentation_subtitle"
                and record.version == video_record.version
                and self._existing_nonempty_file(record.path) is not None
            ),
            None,
        )
        if subtitle_record is not None:
            subtitle_path = str(Path(subtitle_record.path).resolve())
        return _PresentationRecoveryArtifacts(
            video_path=str(video),
            subtitle_path=subtitle_path,
            version=video_record.version,
        )

    @staticmethod
    def _existing_nonempty_file(value: str) -> Path | None:
        try:
            path = Path(value).expanduser().resolve()
            return path if path.is_file() and path.stat().st_size > 0 else None
        except OSError:
            return None

    @staticmethod
    def _same_path(value: str, expected: Path) -> bool:
        try:
            return Path(value).expanduser().resolve() == expected
        except OSError:
            return False

    @staticmethod
    def _resume_value(
        decision: dict[str, Any],
        selection: WorkflowModelSelection | None,
        *,
        operation_id: str = "",
        expected_phase: str = "",
    ) -> dict[str, Any]:
        """Build one serializable resume value without mutating the checkpoint first."""
        value = dict(decision)
        if selection is not None:
            value["model_selection"] = {
                "provider": selection.provider,
                "model": selection.model,
            }
        if operation_id:
            value["operation_id"] = operation_id
        if expected_phase:
            value["expected_phase"] = expected_phase
        return value

    @staticmethod
    def _has_pending_interrupt(state: Any) -> bool:
        return any(bool(task.interrupts) for task in state.tasks)

    @staticmethod
    def _config(project_id: str) -> dict[str, Any]:
        return {"configurable": {"thread_id": project_id}}
