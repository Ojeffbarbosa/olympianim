"""Repository API for Olympianim's local SQLite database."""

from __future__ import annotations

import sqlite3
import uuid
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from olympianim.config import DATABASE_PATH
from olympianim.database.models import (
    AIUsageRecord,
    CodeEditorDraftRecord,
    ColorPaletteRecord,
    GeneratedFileRecord,
    GenerationJobRecord,
    GenerationLogRecord,
    LLMCallCacheRecord,
    ModelCatalogRecord,
    ProjectCreate,
    ProjectPromptRecord,
    ProjectRecord,
    PromptRecord,
    PromptVersionRecord,
    WorkflowEventRecord,
    WorkflowTransitionRecord,
)
from olympianim.database.sqlite import connect, initialize_database


def utc_now() -> str:
    """Return an ISO timestamp suitable for SQLite text columns."""
    return datetime.now(UTC).isoformat(timespec="seconds")


def new_id() -> str:
    """Return a random identifier for persisted records."""
    return str(uuid.uuid4())


class ProjectRepository:
    """SQLite-backed persistence operations for Olympianim."""

    def __init__(self, database_path: Path = DATABASE_PATH) -> None:
        self.database_path = database_path
        initialize_database(database_path)

    def create_project(
        self, data: ProjectCreate, *, project_id: str | None = None
    ) -> ProjectRecord:
        """Persist a project without storing any API key material."""
        now = utc_now()
        record = ProjectRecord(
            **data.__dict__,
            id=project_id or new_id(),
            created_at=now,
            updated_at=now,
        )
        with self._connect() as connection:
            connection.execute(
                """
                INSERT INTO projects (
                    id, title, problem_statement, problem_source,
                    problem_level, math_area, teacher_solution, teacher_instructions,
                    llm_provider, llm_model, llm_api_key_source, voice_provider, voice_model,
                    voice, voice_language, voice_speed, voice_api_key_source,
                    reuse_llm_api_key, voiceover_enabled,
                    color_palette_id, color_palette_snapshot,
                    presentation_video_path, solution_video_path, presentation_code_path,
                    solution_code_path, output_delivery_mode, final_video_path, status,
                    workflow_revision, created_at, updated_at
                )
                VALUES (
                    :id, :title, :problem_statement, :problem_source,
                    :problem_level, :math_area, :teacher_solution, :teacher_instructions,
                    :llm_provider, :llm_model, :llm_api_key_source, :voice_provider, :voice_model,
                    :voice, :voice_language, :voice_speed, :voice_api_key_source,
                    :reuse_llm_api_key, :voiceover_enabled,
                    :color_palette_id, :color_palette_snapshot,
                    :presentation_video_path, :solution_video_path, :presentation_code_path,
                    :solution_code_path, :output_delivery_mode, :final_video_path, :status,
                    :workflow_revision, :created_at, :updated_at
                )
                """,
                self._project_params(record),
            )
        return record

    def list_projects(self) -> list[ProjectRecord]:
        """Return projects ordered by most recent update first."""
        with self._connect() as connection:
            rows = connection.execute(
                "SELECT * FROM projects ORDER BY updated_at DESC, created_at DESC"
            ).fetchall()
        return [self._project_from_row(row) for row in rows]

    def get_project(self, project_id: str) -> ProjectRecord | None:
        """Return a project by id, or ``None`` when it does not exist."""
        with self._connect() as connection:
            row = connection.execute(
                "SELECT * FROM projects WHERE id = ?",
                (project_id,),
            ).fetchone()
        return self._project_from_row(row) if row is not None else None

    def update_project_status(self, project_id: str, status: str) -> None:
        """Update the generation status of a project."""
        self._update_project_fields(project_id, {"status": status})

    def update_project_artifacts(
        self,
        project_id: str,
        *,
        presentation_video_path: str | None = None,
        solution_video_path: str | None = None,
        presentation_code_path: str | None = None,
        solution_code_path: str | None = None,
        final_video_path: str | None = None,
        status: str | None = None,
    ) -> None:
        """Store generated video/code paths on the project record."""
        fields: dict[str, object] = {}
        if presentation_video_path is not None:
            fields["presentation_video_path"] = presentation_video_path
        if solution_video_path is not None:
            fields["solution_video_path"] = solution_video_path
        if presentation_code_path is not None:
            fields["presentation_code_path"] = presentation_code_path
        if solution_code_path is not None:
            fields["solution_code_path"] = solution_code_path
        if final_video_path is not None:
            fields["final_video_path"] = final_video_path
        if status is not None:
            fields["status"] = status
        self._update_project_fields(project_id, fields)

    def get_setting(self, key: str, default: str = "") -> str:
        """Read one application-wide non-sensitive setting."""
        with self._connect() as connection:
            row = connection.execute(
                "SELECT value FROM app_settings WHERE key = ?", (key,)
            ).fetchone()
        return str(row["value"]) if row is not None else default

    def set_setting(self, key: str, value: str) -> None:
        """Upsert one application-wide non-sensitive setting."""
        now = utc_now()
        with self._connect() as connection:
            connection.execute(
                """
                INSERT INTO app_settings (key, value, updated_at) VALUES (?, ?, ?)
                ON CONFLICT(key) DO UPDATE SET value = excluded.value, updated_at = excluded.updated_at
                """,
                (key, value, now),
            )

    def save_code_editor_draft(
        self,
        project_id: str,
        *,
        mode: str,
        code_content: str,
        source_code_sha256: str,
    ) -> CodeEditorDraftRecord:
        """Upsert one durable project/mode editor draft."""
        if mode not in {"presentation", "solution"}:
            raise ValueError(f"Modo de vídeo inválido: {mode!r}.")
        record = CodeEditorDraftRecord(
            project_id=project_id,
            mode=mode,
            code_content=code_content,
            source_code_sha256=source_code_sha256,
            updated_at=utc_now(),
        )
        with self._connect() as connection:
            connection.execute(
                """
                INSERT INTO code_editor_drafts (
                    project_id, mode, code_content, source_code_sha256, updated_at
                )
                VALUES (?, ?, ?, ?, ?)
                ON CONFLICT(project_id, mode) DO UPDATE SET
                    code_content = excluded.code_content,
                    source_code_sha256 = excluded.source_code_sha256,
                    updated_at = excluded.updated_at
                """,
                (
                    record.project_id,
                    record.mode,
                    record.code_content,
                    record.source_code_sha256,
                    record.updated_at,
                ),
            )
        return record

    def get_code_editor_draft(self, project_id: str, *, mode: str) -> CodeEditorDraftRecord | None:
        """Return the durable draft for one project/mode, when present."""
        with self._connect() as connection:
            row = connection.execute(
                """
                SELECT * FROM code_editor_drafts
                WHERE project_id = ? AND mode = ?
                """,
                (project_id, mode),
            ).fetchone()
        return CodeEditorDraftRecord(**dict(row)) if row is not None else None

    def delete_code_editor_draft(self, project_id: str, *, mode: str) -> None:
        """Delete only the selected project/mode draft."""
        with self._connect() as connection:
            connection.execute(
                "DELETE FROM code_editor_drafts WHERE project_id = ? AND mode = ?",
                (project_id, mode),
            )

    def list_catalog_models(
        self,
        *,
        provider: str | None = None,
        modality: str | None = None,
        enabled_only: bool = False,
    ) -> list[ModelCatalogRecord]:
        """Return catalog models in configured display order."""
        clauses: list[str] = []
        parameters: list[object] = []
        if provider is not None:
            clauses.append("provider = ?")
            parameters.append(provider)
        if modality is not None:
            clauses.append("modality = ?")
            parameters.append(modality)
        if enabled_only:
            clauses.append("enabled = 1")
        where = f"WHERE {' AND '.join(clauses)}" if clauses else ""
        with self._connect() as connection:
            rows = connection.execute(
                f"""
                SELECT * FROM model_catalog {where}
                ORDER BY provider, modality, is_default DESC, sort_order, model_id
                """,
                tuple(parameters),
            ).fetchall()
        return [self._catalog_model_from_row(row) for row in rows]

    def get_catalog_model(self, model_record_id: str) -> ModelCatalogRecord | None:
        """Return one model-catalog record by stable ID."""
        with self._connect() as connection:
            row = connection.execute(
                "SELECT * FROM model_catalog WHERE id = ?", (model_record_id,)
            ).fetchone()
        return self._catalog_model_from_row(row) if row is not None else None

    def save_catalog_model(self, record: ModelCatalogRecord) -> ModelCatalogRecord:
        """Insert or update a validated model-catalog record."""
        with self._connect() as connection:
            if record.is_default:
                connection.execute(
                    """
                    UPDATE model_catalog SET is_default = 0, updated_at = ?
                    WHERE provider = ? AND modality = ? AND id != ?
                    """,
                    (record.updated_at, record.provider, record.modality, record.id),
                )
            connection.execute(
                """
                INSERT INTO model_catalog (
                    id, provider, modality, model_id, display_name, enabled,
                    is_default, is_builtin, revision, sort_order, input_token_rate,
                    cached_input_token_rate, output_token_rate,
                    input_character_rate, audio_output_token_rate, created_at, updated_at
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                ON CONFLICT(id) DO UPDATE SET
                    provider = excluded.provider, modality = excluded.modality,
                    model_id = excluded.model_id, display_name = excluded.display_name,
                    enabled = excluded.enabled, is_default = excluded.is_default,
                    revision = excluded.revision, sort_order = excluded.sort_order,
                    input_token_rate = excluded.input_token_rate,
                    cached_input_token_rate = excluded.cached_input_token_rate,
                    output_token_rate = excluded.output_token_rate,
                    input_character_rate = excluded.input_character_rate,
                    audio_output_token_rate = excluded.audio_output_token_rate,
                    updated_at = excluded.updated_at
                """,
                (
                    record.id,
                    record.provider,
                    record.modality,
                    record.model_id,
                    record.display_name,
                    int(record.enabled),
                    int(record.is_default),
                    int(record.is_builtin),
                    record.revision,
                    record.sort_order,
                    record.input_token_rate,
                    record.cached_input_token_rate,
                    record.output_token_rate,
                    record.input_character_rate,
                    record.audio_output_token_rate,
                    record.created_at,
                    record.updated_at,
                ),
            )
        saved = self.get_catalog_model(record.id)
        if saved is None:  # pragma: no cover - insert/select invariant
            raise RuntimeError("O modelo salvo não foi encontrado no catálogo.")
        return saved

    def list_color_palettes(self, *, enabled_only: bool = False) -> list[ColorPaletteRecord]:
        """Return color palettes in configured display order."""
        where = "WHERE enabled = 1" if enabled_only else ""
        with self._connect() as connection:
            rows = connection.execute(f"""SELECT * FROM color_palettes {where}
                ORDER BY sort_order, name""").fetchall()
        return [self._color_palette_from_row(row) for row in rows]

    def get_color_palette(self, palette_id: str) -> ColorPaletteRecord | None:
        """Return one color palette by stable ID."""
        with self._connect() as connection:
            row = connection.execute(
                "SELECT * FROM color_palettes WHERE id = ?", (palette_id,)
            ).fetchone()
        return self._color_palette_from_row(row) if row is not None else None

    def save_color_palette(self, record: ColorPaletteRecord) -> ColorPaletteRecord:
        """Insert or update a validated semantic color palette."""
        with self._connect() as connection:
            connection.execute(
                """
                INSERT INTO color_palettes (
                    id, name, description, background, primary_text,
                    secondary_text, surface, primary_color, secondary_color, highlight,
                    stroke, enabled, is_builtin, revision, sort_order,
                    created_at, updated_at
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                ON CONFLICT(id) DO UPDATE SET
                    name = excluded.name, description = excluded.description,
                    background = excluded.background,
                    primary_text = excluded.primary_text,
                    secondary_text = excluded.secondary_text,
                    surface = excluded.surface, primary_color = excluded.primary_color,
                    secondary_color = excluded.secondary_color, highlight = excluded.highlight,
                    stroke = excluded.stroke, enabled = excluded.enabled,
                    revision = excluded.revision, sort_order = excluded.sort_order,
                    updated_at = excluded.updated_at
                """,
                (
                    record.id,
                    record.name,
                    record.description,
                    record.background,
                    record.primary_text,
                    record.secondary_text,
                    record.surface,
                    record.primary,
                    record.secondary,
                    record.highlight,
                    record.stroke,
                    int(record.enabled),
                    int(record.is_builtin),
                    record.revision,
                    record.sort_order,
                    record.created_at,
                    record.updated_at,
                ),
            )
        saved = self.get_color_palette(record.id)
        if saved is None:  # pragma: no cover
            raise RuntimeError("A paleta salva não foi encontrada no catálogo.")
        return saved

    def add_generated_file(
        self,
        project_id: str,
        *,
        file_type: str,
        path: str,
        version: int = 1,
        description: str = "",
        artifact_key: str = "",
        sha256: str = "",
        size_bytes: int = 0,
    ) -> GeneratedFileRecord:
        """Record or update one content-addressed generated artifact."""
        record = GeneratedFileRecord(
            id=new_id(),
            project_id=project_id,
            file_type=file_type,
            path=path,
            version=version,
            description=description,
            created_at=utc_now(),
            artifact_key=artifact_key,
            sha256=sha256,
            size_bytes=size_bytes,
        )
        with self._connect() as connection:
            if artifact_key:
                connection.execute(
                    """
                    INSERT INTO generated_files (
                        id, project_id, file_type, path, version, description, created_at,
                        artifact_key, sha256, size_bytes
                    ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                    ON CONFLICT(project_id, artifact_key) WHERE artifact_key <> '' DO UPDATE SET
                        file_type = excluded.file_type,
                        path = excluded.path,
                        version = excluded.version,
                        description = excluded.description,
                        sha256 = excluded.sha256,
                        size_bytes = excluded.size_bytes
                    """,
                    (
                        record.id,
                        record.project_id,
                        record.file_type,
                        record.path,
                        record.version,
                        record.description,
                        record.created_at,
                        record.artifact_key,
                        record.sha256,
                        record.size_bytes,
                    ),
                )
                row = connection.execute(
                    """
                    SELECT * FROM generated_files
                    WHERE project_id = ? AND artifact_key = ?
                    """,
                    (project_id, artifact_key),
                ).fetchone()
            else:
                connection.execute(
                    """
                    INSERT INTO generated_files (
                        id, project_id, file_type, path, version, description, created_at,
                        artifact_key, sha256, size_bytes
                    ) VALUES (?, ?, ?, ?, ?, ?, ?, '', ?, ?)
                    """,
                    (
                        record.id,
                        record.project_id,
                        record.file_type,
                        record.path,
                        record.version,
                        record.description,
                        record.created_at,
                        record.sha256,
                        record.size_bytes,
                    ),
                )
                row = connection.execute(
                    "SELECT * FROM generated_files WHERE id = ?", (record.id,)
                ).fetchone()
        assert row is not None
        return self._generated_file_from_row(row)

    def list_generated_files(self, project_id: str) -> list[GeneratedFileRecord]:
        """Return generated artifacts for a project."""
        with self._connect() as connection:
            rows = connection.execute(
                """
                SELECT * FROM generated_files
                WHERE project_id = ?
                ORDER BY created_at ASC
                """,
                (project_id,),
            ).fetchall()
        return [self._generated_file_from_row(row) for row in rows]

    def delete_generated_files(
        self,
        project_id: str,
        *,
        file_type: str,
    ) -> list[GeneratedFileRecord]:
        """Delete artifact metadata of one type and return the removed records."""
        if not file_type.strip():
            raise ValueError("O tipo do artefato não pode ficar vazio.")
        with self._connect() as connection:
            rows = connection.execute(
                """
                SELECT * FROM generated_files
                WHERE project_id = ? AND file_type = ?
                ORDER BY created_at ASC
                """,
                (project_id, file_type),
            ).fetchall()
            connection.execute(
                """
                DELETE FROM generated_files
                WHERE project_id = ? AND file_type = ?
                """,
                (project_id, file_type),
            )
        return [self._generated_file_from_row(row) for row in rows]

    def create_job(
        self,
        project_id: str,
        *,
        action: str,
        payload: str = "{}",
        operation_id: str = "",
        expected_phase: str = "",
        workflow_revision: int = 1,
    ) -> GenerationJobRecord:
        """Queue one workflow action, rejecting duplicate active work."""
        now = utc_now()
        job_id = new_id()
        try:
            with self._connect() as connection:
                connection.execute(
                    """
                    INSERT INTO generation_jobs (
                        id, project_id, action, payload, status, current_step,
                        progress, attempts, result, error_message, cancel_requested,
                        created_at, updated_at, started_at, finished_at, heartbeat_at,
                        operation_id, expected_phase, workflow_revision
                    ) VALUES (?, ?, ?, ?, 'pending', 'queued', 0, 0, '{}', '', 0,
                              ?, ?, '', '', '', ?, ?, ?)
                    """,
                    (
                        job_id,
                        project_id,
                        action,
                        payload,
                        now,
                        now,
                        operation_id,
                        expected_phase,
                        workflow_revision,
                    ),
                )
        except sqlite3.IntegrityError as exc:
            if self.get_active_job(project_id) is not None:
                raise ValueError(
                    "Já existe uma etapa em processamento para este projeto."
                ) from exc
            raise
        job = self.get_job(job_id)
        if job is None:  # pragma: no cover - SQLite insert invariant
            raise RuntimeError("O trabalho criado não foi encontrado.")
        return job

    def get_job(self, job_id: str) -> GenerationJobRecord | None:
        """Return one background job by identifier."""
        with self._connect() as connection:
            row = connection.execute(
                "SELECT * FROM generation_jobs WHERE id = ?", (job_id,)
            ).fetchone()
        return self._job_from_row(row) if row is not None else None

    def get_active_job(self, project_id: str) -> GenerationJobRecord | None:
        """Return the pending or running job for a project, if any."""
        with self._connect() as connection:
            row = connection.execute(
                """
                SELECT * FROM generation_jobs
                WHERE project_id = ? AND status IN ('pending', 'running')
                ORDER BY created_at DESC LIMIT 1
                """,
                (project_id,),
            ).fetchone()
        return self._job_from_row(row) if row is not None else None

    def latest_job(self, project_id: str) -> GenerationJobRecord | None:
        """Return the most recently updated job for a project."""
        with self._connect() as connection:
            row = connection.execute(
                """
                SELECT * FROM generation_jobs
                WHERE project_id = ? ORDER BY updated_at DESC, created_at DESC LIMIT 1
                """,
                (project_id,),
            ).fetchone()
        return self._job_from_row(row) if row is not None else None

    def list_jobs(self, project_id: str) -> list[GenerationJobRecord]:
        """Return all background jobs for audit and reproducibility exports."""
        with self._connect() as connection:
            rows = connection.execute(
                """
                SELECT * FROM generation_jobs WHERE project_id = ?
                ORDER BY created_at, id
                """,
                (project_id,),
            ).fetchall()
        return [self._job_from_row(row) for row in rows]

    def claim_next_job(self) -> GenerationJobRecord | None:
        """Atomically claim the oldest pending job."""
        now = utc_now()
        with self._connect() as connection:
            connection.execute("BEGIN IMMEDIATE")
            row = connection.execute(
                "SELECT id FROM generation_jobs WHERE status = 'pending' ORDER BY created_at LIMIT 1"
            ).fetchone()
            if row is None:
                return None
            cursor = connection.execute(
                """
                UPDATE generation_jobs
                SET status = 'running', current_step = 'starting', progress = 5,
                    attempts = attempts + 1, started_at = ?, heartbeat_at = ?, updated_at = ?
                WHERE id = ? AND status = 'pending'
                """,
                (now, now, now, row["id"]),
            )
            if cursor.rowcount != 1:
                return None
        return self.get_job(str(row["id"]))

    def update_job(
        self,
        job_id: str,
        *,
        status: str | None = None,
        current_step: str | None = None,
        progress: int | None = None,
        result: str | None = None,
        error_message: str | None = None,
        heartbeat: bool = False,
    ) -> None:
        """Update mutable job execution fields."""
        fields: dict[str, object] = {"updated_at": utc_now()}
        if status is not None:
            fields["status"] = status
            if status in {"completed", "cancelled", "failed"}:
                fields["finished_at"] = fields["updated_at"]
        if current_step is not None:
            fields["current_step"] = current_step
        if progress is not None:
            fields["progress"] = progress
        if result is not None:
            fields["result"] = result
        if error_message is not None:
            fields["error_message"] = error_message
        if heartbeat:
            fields["heartbeat_at"] = fields["updated_at"]
        self._update_job_fields(job_id, fields)

    def complete_job_and_transition(
        self,
        *,
        job_id: str,
        operation_id: str,
        project_id: str,
        phase: str,
        job_result: str,
        result_snapshot: str,
    ) -> None:
        """Commit job, operation and project state as one SQLite transaction."""
        now = utc_now()
        with self._connect() as connection:
            connection.execute("BEGIN IMMEDIATE")
            job_cursor = connection.execute(
                """
                UPDATE generation_jobs
                SET status = 'completed', current_step = ?, progress = 100,
                    result = ?, updated_at = ?, finished_at = ?, heartbeat_at = ?
                WHERE id = ? AND status = 'running'
                """,
                (phase, job_result, now, now, now, job_id),
            )
            transition_cursor = connection.execute(
                """
                UPDATE workflow_transitions
                SET status = 'completed', result_snapshot = ?, updated_at = ?
                WHERE operation_id = ?
                """,
                (result_snapshot, now, operation_id),
            )
            project_cursor = connection.execute(
                "UPDATE projects SET status = ?, updated_at = ? WHERE id = ?",
                (phase, now, project_id),
            )
            if (
                job_cursor.rowcount != 1
                or transition_cursor.rowcount != 1
                or project_cursor.rowcount != 1
            ):
                raise RuntimeError("Não foi possível concluir atomicamente a transição.")

    def request_job_cancellation(self, job_id: str) -> bool:
        """Request cooperative cancellation of an active job."""
        now = utc_now()
        with self._connect() as connection:
            cursor = connection.execute(
                """
                UPDATE generation_jobs SET cancel_requested = 1, updated_at = ?
                WHERE id = ? AND status IN ('pending', 'running')
                """,
                (now, job_id),
            )
        return cursor.rowcount == 1

    def is_job_cancellation_requested(self, job_id: str) -> bool:
        """Return whether cancellation was requested for a job."""
        job = self.get_job(job_id)
        return bool(job and job.cancel_requested)

    def recover_stale_jobs(self, stale_before: str) -> int:
        """Return abandoned running jobs to the queue after a worker restart."""
        now = utc_now()
        with self._connect() as connection:
            cursor = connection.execute(
                """
                UPDATE generation_jobs
                SET status = 'pending', current_step = 'recovering', progress = 0,
                    started_at = '', heartbeat_at = '', updated_at = ?
                WHERE status = 'running' AND heartbeat_at < ?
                """,
                (now, stale_before),
            )
        return cursor.rowcount

    def create_transition(
        self,
        project_id: str,
        *,
        operation_id: str,
        action: str,
        expected_phase: str,
        decision_sha256: str,
        workflow_revision: int,
    ) -> WorkflowTransitionRecord:
        """Create an idempotency record, or validate the existing operation."""
        now = utc_now()
        with self._connect() as connection:
            connection.execute(
                """
                INSERT OR IGNORE INTO workflow_transitions (
                    operation_id, project_id, action, expected_phase, decision_sha256,
                    workflow_revision, status, result_snapshot, created_at, updated_at
                ) VALUES (?, ?, ?, ?, ?, ?, 'queued', '{}', ?, ?)
                """,
                (
                    operation_id,
                    project_id,
                    action,
                    expected_phase,
                    decision_sha256,
                    workflow_revision,
                    now,
                    now,
                ),
            )
            row = connection.execute(
                "SELECT * FROM workflow_transitions WHERE operation_id = ?",
                (operation_id,),
            ).fetchone()
        assert row is not None
        record = self._transition_from_row(row)
        identity = (
            record.project_id,
            record.action,
            record.expected_phase,
            record.decision_sha256,
            record.workflow_revision,
        )
        requested = (
            project_id,
            action,
            expected_phase,
            decision_sha256,
            workflow_revision,
        )
        if identity != requested:
            raise ValueError("O identificador da operação já pertence a outra transição.")
        return record

    def get_transition(self, operation_id: str) -> WorkflowTransitionRecord | None:
        """Return an explicit workflow transition by stable operation id."""
        with self._connect() as connection:
            row = connection.execute(
                "SELECT * FROM workflow_transitions WHERE operation_id = ?",
                (operation_id,),
            ).fetchone()
        return self._transition_from_row(row) if row is not None else None

    def discard_orphan_transition(self, operation_id: str) -> None:
        """Remove an orphan transition when its background job could not be queued."""
        with self._connect() as connection:
            connection.execute(
                """
                DELETE FROM workflow_transitions
                WHERE operation_id = ? AND status = 'queued'
                  AND NOT EXISTS (
                      SELECT 1 FROM generation_jobs WHERE operation_id = ?
                  )
                """,
                (operation_id, operation_id),
            )

    def update_transition(
        self,
        operation_id: str,
        *,
        status: str,
        result_snapshot: str | None = None,
    ) -> None:
        """Update a transition lifecycle without changing its identity contract."""
        fields: dict[str, object] = {"status": status, "updated_at": utc_now()}
        if result_snapshot is not None:
            fields["result_snapshot"] = result_snapshot
        assignments = ", ".join(f"{field} = :{field}" for field in fields)
        with self._connect() as connection:
            connection.execute(
                f"UPDATE workflow_transitions SET {assignments} WHERE operation_id = :id",
                {**fields, "id": operation_id},
            )

    def get_llm_call_cache(self, cache_key: str) -> LLMCallCacheRecord | None:
        """Return a previously completed model call."""
        with self._connect() as connection:
            row = connection.execute(
                "SELECT * FROM llm_call_cache WHERE cache_key = ?", (cache_key,)
            ).fetchone()
        return self._llm_cache_from_row(row) if row is not None else None

    def save_llm_call_cache(
        self,
        *,
        cache_key: str,
        project_id: str,
        operation_id: str,
        role: str,
        mode: str,
        provider: str,
        requested_model: str,
        resolved_model: str,
        finish_reason: str,
        prompt_sha256: str,
        response_text: str,
    ) -> LLMCallCacheRecord:
        """Persist one successful response exactly once."""
        with self._connect() as connection:
            connection.execute(
                """
                INSERT OR IGNORE INTO llm_call_cache (
                    cache_key, project_id, operation_id, role, mode, provider,
                    requested_model, resolved_model, finish_reason, prompt_sha256,
                    response_text, created_at
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    cache_key,
                    project_id,
                    operation_id,
                    role,
                    mode,
                    provider,
                    requested_model,
                    resolved_model,
                    finish_reason,
                    prompt_sha256,
                    response_text,
                    utc_now(),
                ),
            )
            row = connection.execute(
                "SELECT * FROM llm_call_cache WHERE cache_key = ?", (cache_key,)
            ).fetchone()
        assert row is not None
        return self._llm_cache_from_row(row)

    def list_llm_call_cache(self, project_id: str) -> list[LLMCallCacheRecord]:
        """Return successful model outputs used by one project."""
        with self._connect() as connection:
            rows = connection.execute(
                """
                SELECT * FROM llm_call_cache WHERE project_id = ?
                ORDER BY created_at, cache_key
                """,
                (project_id,),
            ).fetchall()
        return [self._llm_cache_from_row(row) for row in rows]

    def add_workflow_event(
        self,
        project_id: str,
        *,
        event_key: str,
        event_type: str,
        operation_id: str = "",
        job_id: str = "",
        phase: str = "",
        payload: str = "{}",
    ) -> WorkflowEventRecord:
        """Persist a structured event once, even when a job is replayed."""
        with self._connect() as connection:
            connection.execute(
                """
                INSERT OR IGNORE INTO workflow_events (
                    id, event_key, project_id, operation_id, job_id,
                    event_type, phase, payload, created_at
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    new_id(),
                    event_key,
                    project_id,
                    operation_id,
                    job_id,
                    event_type,
                    phase,
                    payload,
                    utc_now(),
                ),
            )
            row = connection.execute(
                """
                SELECT * FROM workflow_events
                WHERE project_id = ? AND event_key = ?
                """,
                (project_id, event_key),
            ).fetchone()
        assert row is not None
        return self._workflow_event_from_row(row)

    def list_workflow_events(self, project_id: str) -> list[WorkflowEventRecord]:
        """Return structured workflow events in deterministic order."""
        with self._connect() as connection:
            rows = connection.execute(
                """
                SELECT * FROM workflow_events WHERE project_id = ?
                ORDER BY created_at, id
                """,
                (project_id,),
            ).fetchall()
        return [self._workflow_event_from_row(row) for row in rows]

    def add_ai_usage(
        self,
        project_id: str,
        *,
        execution_id: str,
        call_key: str,
        agent_type: str,
        stage: str,
        provider: str,
        model: str,
        status: str,
        attempt_type: str,
        input_tokens: int = 0,
        output_tokens: int = 0,
        total_tokens: int = 0,
        cache_read_tokens: int = 0,
        cache_creation_tokens: int = 0,
        reasoning_tokens: int = 0,
        modality: str = "text",
        input_characters: int = 0,
        audio_output_tokens: int = 0,
        audio_seconds: float = 0.0,
        estimated_cost_usd: float = 0.0,
        pricing_known: bool = False,
        usage_source: str = "provider",
        metadata_available: bool = False,
        error_type: str = "",
        error_code: str = "",
        error_status: str = "",
        error_message: str = "",
        error_transient: bool = False,
    ) -> AIUsageRecord:
        """Persist one idempotent provider attempt without sensitive content."""
        with self._connect() as connection:
            sequence_row = connection.execute(
                """
                SELECT COUNT(*) + 1 AS sequence FROM ai_usage
                WHERE project_id = ? AND agent_type = ? AND stage = ?
                """,
                (project_id, agent_type, stage),
            ).fetchone()
            record = AIUsageRecord(
                id=new_id(),
                project_id=project_id,
                execution_id=execution_id,
                call_key=call_key,
                agent_type=agent_type,
                stage=stage,
                provider=provider,
                model=model,
                status=status,
                attempt_type=attempt_type,
                sequence=int(sequence_row["sequence"]),
                input_tokens=input_tokens,
                output_tokens=output_tokens,
                total_tokens=total_tokens,
                cache_read_tokens=cache_read_tokens,
                cache_creation_tokens=cache_creation_tokens,
                reasoning_tokens=reasoning_tokens,
                modality=modality,
                input_characters=input_characters,
                audio_output_tokens=audio_output_tokens,
                audio_seconds=audio_seconds,
                estimated_cost_usd=estimated_cost_usd,
                pricing_known=pricing_known,
                usage_source=usage_source,
                metadata_available=metadata_available,
                error_type=error_type,
                error_code=error_code,
                error_status=error_status,
                error_message=error_message,
                error_transient=error_transient,
                created_at=utc_now(),
            )
            connection.execute(
                """
                INSERT INTO ai_usage (
                    id, project_id, execution_id, call_key, agent_type, stage,
                    provider, model, status, attempt_type, sequence,
                    input_tokens, output_tokens, total_tokens, cache_read_tokens,
                    cache_creation_tokens, reasoning_tokens, modality, input_characters,
                    audio_output_tokens, audio_seconds, estimated_cost_usd, pricing_known,
                    usage_source, metadata_available, error_type, error_code, error_status,
                    error_message, error_transient, created_at
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                ON CONFLICT(project_id, call_key, provider, model) DO NOTHING
                """,
                (
                    record.id,
                    record.project_id,
                    record.execution_id,
                    record.call_key,
                    record.agent_type,
                    record.stage,
                    record.provider,
                    record.model,
                    record.status,
                    record.attempt_type,
                    record.sequence,
                    record.input_tokens,
                    record.output_tokens,
                    record.total_tokens,
                    record.cache_read_tokens,
                    record.cache_creation_tokens,
                    record.reasoning_tokens,
                    record.modality,
                    record.input_characters,
                    record.audio_output_tokens,
                    record.audio_seconds,
                    record.estimated_cost_usd,
                    int(record.pricing_known),
                    record.usage_source,
                    int(record.metadata_available),
                    record.error_type,
                    record.error_code,
                    record.error_status,
                    record.error_message,
                    int(record.error_transient),
                    record.created_at,
                ),
            )
            row = connection.execute(
                """
                SELECT * FROM ai_usage
                WHERE project_id = ? AND call_key = ? AND provider = ? AND model = ?
                """,
                (project_id, call_key, provider, model),
            ).fetchone()
        if row is None:  # pragma: no cover - insert/select invariant
            raise RuntimeError("O registro de consumo não foi encontrado.")
        return self._ai_usage_from_row(row)

    def list_ai_usage(self, project_id: str | None = None) -> list[AIUsageRecord]:
        """Return usage attempts for one project or the whole application."""
        where_clause = "WHERE project_id = ?" if project_id is not None else ""
        parameters = (project_id,) if project_id is not None else ()
        with self._connect() as connection:
            rows = connection.execute(
                f"""
                SELECT * FROM ai_usage {where_clause}
                ORDER BY created_at ASC, sequence ASC
                """,
                parameters,
            ).fetchall()
        return [self._ai_usage_from_row(row) for row in rows]

    def create_prompt(
        self,
        *,
        name: str,
        agent_type: str,
        description: str = "",
        is_default: bool = False,
    ) -> PromptRecord:
        """Create prompt metadata; versions are stored separately."""
        now = utc_now()
        record = PromptRecord(
            id=new_id(),
            name=name,
            agent_type=agent_type,
            description=description,
            is_default=is_default,
            created_at=now,
            updated_at=now,
        )
        with self._connect() as connection:
            connection.execute(
                """
                INSERT INTO prompts (
                    id, name, agent_type, description, is_default, created_at, updated_at
                )
                VALUES (?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    record.id,
                    record.name,
                    record.agent_type,
                    record.description,
                    int(record.is_default),
                    record.created_at,
                    record.updated_at,
                ),
            )
        return record

    def add_prompt_version(
        self,
        prompt_id: str,
        *,
        template_text: str,
        version: int | None = None,
    ) -> PromptVersionRecord:
        """Add a template version to an existing prompt."""
        next_version = version if version is not None else self._next_prompt_version(prompt_id)
        record = PromptVersionRecord(
            id=new_id(),
            prompt_id=prompt_id,
            version=next_version,
            template_text=template_text,
            created_at=utc_now(),
        )
        with self._connect() as connection:
            connection.execute(
                """
                INSERT INTO prompt_versions (
                    id, prompt_id, version, template_text, created_at
                )
                VALUES (?, ?, ?, ?, ?)
                """,
                (
                    record.id,
                    record.prompt_id,
                    record.version,
                    record.template_text,
                    record.created_at,
                ),
            )
            connection.execute(
                "UPDATE prompts SET updated_at = ? WHERE id = ?",
                (record.created_at, prompt_id),
            )
        return record

    def list_prompts(self) -> list[PromptRecord]:
        """Return prompt metadata rows ordered by agent and name."""
        with self._connect() as connection:
            rows = connection.execute(
                "SELECT * FROM prompts ORDER BY agent_type ASC, name ASC"
            ).fetchall()
        return [self._prompt_from_row(row) for row in rows]

    def get_prompt(self, prompt_id: str) -> PromptRecord | None:
        """Return prompt metadata by id."""
        with self._connect() as connection:
            row = connection.execute(
                "SELECT * FROM prompts WHERE id = ?",
                (prompt_id,),
            ).fetchone()
        return self._prompt_from_row(row) if row is not None else None

    def get_latest_prompt_version(self, prompt_id: str) -> PromptVersionRecord | None:
        """Return the latest template version for a prompt."""
        with self._connect() as connection:
            row = connection.execute(
                """
                SELECT * FROM prompt_versions
                WHERE prompt_id = ?
                ORDER BY version DESC
                LIMIT 1
                """,
                (prompt_id,),
            ).fetchone()
        return self._prompt_version_from_row(row) if row is not None else None

    def record_project_prompt(
        self,
        project_id: str,
        *,
        agent_type: str,
        prompt_id: str,
        prompt_version: int,
        rendered_prompt_snapshot: str = "",
        rendered_system_snapshot: str = "",
        rendered_user_snapshot: str = "",
        prompt_sha256: str = "",
        operation_id: str = "",
    ) -> ProjectPromptRecord:
        """Persist which prompt version was used by a project."""
        record = ProjectPromptRecord(
            id=new_id(),
            project_id=project_id,
            agent_type=agent_type,
            prompt_id=prompt_id,
            prompt_version=prompt_version,
            rendered_prompt_snapshot=rendered_prompt_snapshot,
            created_at=utc_now(),
            rendered_system_snapshot=rendered_system_snapshot,
            rendered_user_snapshot=rendered_user_snapshot,
            prompt_sha256=prompt_sha256,
            operation_id=operation_id,
        )
        with self._connect() as connection:
            existing = None
            if operation_id and prompt_sha256:
                existing = connection.execute(
                    """
                    SELECT * FROM project_prompts
                    WHERE project_id = ? AND operation_id = ? AND agent_type = ?
                      AND prompt_sha256 = ?
                    LIMIT 1
                    """,
                    (project_id, operation_id, agent_type, prompt_sha256),
                ).fetchone()
            if existing is not None:
                return self._project_prompt_from_row(existing)
            connection.execute(
                """
                INSERT INTO project_prompts (
                    id, project_id, agent_type, prompt_id, prompt_version,
                    rendered_prompt_snapshot, created_at, rendered_system_snapshot,
                    rendered_user_snapshot, prompt_sha256, operation_id
                )
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    record.id,
                    record.project_id,
                    record.agent_type,
                    record.prompt_id,
                    record.prompt_version,
                    record.rendered_prompt_snapshot,
                    record.created_at,
                    record.rendered_system_snapshot,
                    record.rendered_user_snapshot,
                    record.prompt_sha256,
                    record.operation_id,
                ),
            )
        return record

    def list_project_prompts(self, project_id: str) -> list[ProjectPromptRecord]:
        """Return the exact prompt messages used by one project."""
        with self._connect() as connection:
            rows = connection.execute(
                """
                SELECT * FROM project_prompts WHERE project_id = ?
                ORDER BY created_at, id
                """,
                (project_id,),
            ).fetchall()
        return [self._project_prompt_from_row(row) for row in rows]

    def add_log(
        self,
        project_id: str,
        *,
        message: str,
        level: str = "info",
        step: str = "",
    ) -> GenerationLogRecord:
        """Persist a short project log line."""
        record = GenerationLogRecord(
            id=new_id(),
            project_id=project_id,
            level=level,
            step=step,
            message=message,
            created_at=utc_now(),
        )
        with self._connect() as connection:
            connection.execute(
                """
                INSERT INTO generation_logs (
                    id, project_id, level, step, message, created_at
                )
                VALUES (?, ?, ?, ?, ?, ?)
                """,
                (
                    record.id,
                    record.project_id,
                    record.level,
                    record.step,
                    record.message,
                    record.created_at,
                ),
            )
        return record

    def list_logs(self, project_id: str) -> list[GenerationLogRecord]:
        """Return persisted log lines for a project."""
        with self._connect() as connection:
            rows = connection.execute(
                """
                SELECT * FROM generation_logs
                WHERE project_id = ?
                ORDER BY created_at ASC
                """,
                (project_id,),
            ).fetchall()
        return [
            GenerationLogRecord(
                id=row["id"],
                project_id=row["project_id"],
                level=row["level"],
                step=row["step"],
                message=row["message"],
                created_at=row["created_at"],
            )
            for row in rows
        ]

    def _connect(self) -> sqlite3.Connection:
        return connect(self.database_path)

    def _next_prompt_version(self, prompt_id: str) -> int:
        with self._connect() as connection:
            row = connection.execute(
                "SELECT COALESCE(MAX(version), 0) + 1 AS next_version FROM prompt_versions WHERE prompt_id = ?",
                (prompt_id,),
            ).fetchone()
        return int(row["next_version"])

    def _update_project_fields(self, project_id: str, fields: dict[str, object]) -> None:
        if not fields:
            return
        fields["updated_at"] = utc_now()
        assignments = ", ".join(f"{field} = :{field}" for field in fields)
        params = {**fields, "id": project_id}
        with self._connect() as connection:
            connection.execute(
                f"UPDATE projects SET {assignments} WHERE id = :id",
                params,
            )

    def _update_job_fields(self, job_id: str, fields: dict[str, object]) -> None:
        assignments = ", ".join(f"{field} = :{field}" for field in fields)
        with self._connect() as connection:
            connection.execute(
                f"UPDATE generation_jobs SET {assignments} WHERE id = :id",
                {**fields, "id": job_id},
            )

    @staticmethod
    def _project_params(record: ProjectRecord) -> dict[str, object]:
        params = record.__dict__.copy()
        params["voiceover_enabled"] = int(record.voiceover_enabled)
        params["reuse_llm_api_key"] = int(record.reuse_llm_api_key)
        return params

    @staticmethod
    def _project_from_row(row: sqlite3.Row) -> ProjectRecord:
        data: dict[str, Any] = dict(row)
        data["voiceover_enabled"] = bool(data["voiceover_enabled"])
        data["reuse_llm_api_key"] = bool(data["reuse_llm_api_key"])
        data["voice_speed"] = float(data["voice_speed"])
        data["workflow_revision"] = int(data["workflow_revision"])
        return ProjectRecord(**data)

    @staticmethod
    def _generated_file_from_row(row: sqlite3.Row) -> GeneratedFileRecord:
        return GeneratedFileRecord(
            id=row["id"],
            project_id=row["project_id"],
            file_type=row["file_type"],
            path=row["path"],
            version=int(row["version"]),
            description=row["description"],
            created_at=row["created_at"],
            artifact_key=row["artifact_key"],
            sha256=row["sha256"],
            size_bytes=int(row["size_bytes"]),
        )

    @staticmethod
    def _prompt_from_row(row: sqlite3.Row) -> PromptRecord:
        return PromptRecord(
            id=row["id"],
            name=row["name"],
            agent_type=row["agent_type"],
            description=row["description"],
            is_default=bool(row["is_default"]),
            created_at=row["created_at"],
            updated_at=row["updated_at"],
        )

    @staticmethod
    def _job_from_row(row: sqlite3.Row) -> GenerationJobRecord:
        return GenerationJobRecord(
            id=row["id"],
            project_id=row["project_id"],
            action=row["action"],
            payload=row["payload"],
            status=row["status"],
            current_step=row["current_step"],
            progress=int(row["progress"]),
            attempts=int(row["attempts"]),
            result=row["result"],
            error_message=row["error_message"],
            cancel_requested=bool(row["cancel_requested"]),
            created_at=row["created_at"],
            updated_at=row["updated_at"],
            started_at=row["started_at"],
            finished_at=row["finished_at"],
            heartbeat_at=row["heartbeat_at"],
            operation_id=row["operation_id"],
            expected_phase=row["expected_phase"],
            workflow_revision=int(row["workflow_revision"]),
        )

    @staticmethod
    def _transition_from_row(row: sqlite3.Row) -> WorkflowTransitionRecord:
        return WorkflowTransitionRecord(
            operation_id=row["operation_id"],
            project_id=row["project_id"],
            action=row["action"],
            expected_phase=row["expected_phase"],
            decision_sha256=row["decision_sha256"],
            workflow_revision=int(row["workflow_revision"]),
            status=row["status"],
            result_snapshot=row["result_snapshot"],
            created_at=row["created_at"],
            updated_at=row["updated_at"],
        )

    @staticmethod
    def _llm_cache_from_row(row: sqlite3.Row) -> LLMCallCacheRecord:
        return LLMCallCacheRecord(
            cache_key=row["cache_key"],
            project_id=row["project_id"],
            operation_id=row["operation_id"],
            role=row["role"],
            mode=row["mode"],
            provider=row["provider"],
            requested_model=row["requested_model"],
            resolved_model=row["resolved_model"],
            finish_reason=row["finish_reason"],
            prompt_sha256=row["prompt_sha256"],
            response_text=row["response_text"],
            created_at=row["created_at"],
        )

    @staticmethod
    def _workflow_event_from_row(row: sqlite3.Row) -> WorkflowEventRecord:
        return WorkflowEventRecord(
            id=row["id"],
            event_key=row["event_key"],
            project_id=row["project_id"],
            operation_id=row["operation_id"],
            job_id=row["job_id"],
            event_type=row["event_type"],
            phase=row["phase"],
            payload=row["payload"],
            created_at=row["created_at"],
        )

    @staticmethod
    def _project_prompt_from_row(row: sqlite3.Row) -> ProjectPromptRecord:
        return ProjectPromptRecord(
            id=row["id"],
            project_id=row["project_id"],
            agent_type=row["agent_type"],
            prompt_id=row["prompt_id"],
            prompt_version=int(row["prompt_version"]),
            rendered_prompt_snapshot=row["rendered_prompt_snapshot"],
            created_at=row["created_at"],
            rendered_system_snapshot=row["rendered_system_snapshot"],
            rendered_user_snapshot=row["rendered_user_snapshot"],
            prompt_sha256=row["prompt_sha256"],
            operation_id=row["operation_id"],
        )

    @staticmethod
    def _ai_usage_from_row(row: sqlite3.Row) -> AIUsageRecord:
        return AIUsageRecord(
            id=row["id"],
            project_id=row["project_id"],
            execution_id=row["execution_id"],
            call_key=row["call_key"],
            agent_type=row["agent_type"],
            stage=row["stage"],
            provider=row["provider"],
            model=row["model"],
            status=row["status"],
            attempt_type=row["attempt_type"],
            sequence=int(row["sequence"]),
            input_tokens=int(row["input_tokens"]),
            output_tokens=int(row["output_tokens"]),
            total_tokens=int(row["total_tokens"]),
            cache_read_tokens=int(row["cache_read_tokens"]),
            cache_creation_tokens=int(row["cache_creation_tokens"]),
            reasoning_tokens=int(row["reasoning_tokens"]),
            modality=row["modality"],
            input_characters=int(row["input_characters"]),
            audio_output_tokens=int(row["audio_output_tokens"]),
            audio_seconds=float(row["audio_seconds"]),
            estimated_cost_usd=float(row["estimated_cost_usd"]),
            pricing_known=bool(row["pricing_known"]),
            usage_source=row["usage_source"],
            metadata_available=bool(row["metadata_available"]),
            error_type=row["error_type"],
            error_code=row["error_code"],
            error_status=row["error_status"],
            error_message=row["error_message"],
            error_transient=bool(row["error_transient"]),
            created_at=row["created_at"],
        )

    @staticmethod
    def _catalog_model_from_row(row: sqlite3.Row) -> ModelCatalogRecord:
        return ModelCatalogRecord(
            id=row["id"],
            provider=row["provider"],
            modality=row["modality"],
            model_id=row["model_id"],
            display_name=row["display_name"],
            enabled=bool(row["enabled"]),
            is_default=bool(row["is_default"]),
            is_builtin=bool(row["is_builtin"]),
            revision=int(row["revision"]),
            sort_order=int(row["sort_order"]),
            input_token_rate=float(row["input_token_rate"]),
            cached_input_token_rate=float(row["cached_input_token_rate"]),
            output_token_rate=float(row["output_token_rate"]),
            input_character_rate=float(row["input_character_rate"]),
            audio_output_token_rate=float(row["audio_output_token_rate"]),
            created_at=row["created_at"],
            updated_at=row["updated_at"],
        )

    @staticmethod
    def _color_palette_from_row(row: sqlite3.Row) -> ColorPaletteRecord:
        return ColorPaletteRecord(
            id=row["id"],
            name=row["name"],
            description=row["description"],
            background=row["background"],
            primary_text=row["primary_text"],
            secondary_text=row["secondary_text"],
            surface=row["surface"],
            primary=row["primary_color"],
            secondary=row["secondary_color"],
            highlight=row["highlight"],
            stroke=row["stroke"],
            enabled=bool(row["enabled"]),
            is_builtin=bool(row["is_builtin"]),
            revision=int(row["revision"]),
            sort_order=int(row["sort_order"]),
            created_at=row["created_at"],
            updated_at=row["updated_at"],
        )

    @staticmethod
    def _prompt_version_from_row(row: sqlite3.Row) -> PromptVersionRecord:
        return PromptVersionRecord(
            id=row["id"],
            prompt_id=row["prompt_id"],
            version=int(row["version"]),
            template_text=row["template_text"],
            created_at=row["created_at"],
        )
