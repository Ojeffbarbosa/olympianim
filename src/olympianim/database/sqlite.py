"""SQLite connection, integrity checks and ordered schema migrations."""

from __future__ import annotations

import sqlite3
from collections.abc import Callable
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path

from olympianim.config import DATABASE_PATH


@dataclass(frozen=True)
class SchemaMigration:
    """One immutable, ordered schema transition."""

    version: int
    name: str
    checksum: str
    apply: Callable[[sqlite3.Connection], None]


def connect(database_path: Path = DATABASE_PATH) -> sqlite3.Connection:
    """Open a SQLite connection with project defaults enabled."""
    database_path.parent.mkdir(parents=True, exist_ok=True)
    connection = sqlite3.connect(database_path, timeout=30)
    connection.row_factory = sqlite3.Row
    connection.execute("PRAGMA busy_timeout = 30000")
    connection.execute("PRAGMA journal_mode = WAL")
    connection.execute("PRAGMA foreign_keys = ON")
    return connection


def initialize_database(database_path: Path = DATABASE_PATH) -> None:
    """Validate, back up when needed, and migrate the local database."""
    database_path.parent.mkdir(parents=True, exist_ok=True)
    existed = database_path.is_file() and database_path.stat().st_size > 0
    with connect(database_path) as connection:
        current = _current_version(connection)
        if existed and current < LATEST_SCHEMA_VERSION and _has_application_tables(connection):
            assert_database_integrity(connection)
            _backup_database(connection, database_path, current, LATEST_SCHEMA_VERSION)
        migrate(connection)


def migrate(connection: sqlite3.Connection) -> None:
    """Apply every pending migration exactly once, each in its own transaction."""
    connection.execute("""
        CREATE TABLE IF NOT EXISTS schema_migrations (
            version INTEGER PRIMARY KEY,
            name TEXT NOT NULL,
            checksum TEXT NOT NULL,
            applied_at TEXT NOT NULL
        )
        """)
    connection.commit()
    applied = {
        int(row["version"]): str(row["checksum"])
        for row in connection.execute("SELECT version, checksum FROM schema_migrations")
    }
    for migration in MIGRATIONS:
        checksum = _migration_checksum(migration)
        if migration.version in applied:
            if applied[migration.version] != checksum:
                raise RuntimeError(
                    f"A migração SQLite {migration.version} foi alterada após ser aplicada."
                )
            continue
        connection.execute("BEGIN IMMEDIATE")
        try:
            migration.apply(connection)
            connection.execute(
                """
                INSERT INTO schema_migrations (version, name, checksum, applied_at)
                VALUES (?, ?, ?, ?)
                """,
                (
                    migration.version,
                    migration.name,
                    checksum,
                    datetime.now(UTC).isoformat(timespec="seconds"),
                ),
            )
            connection.commit()
        except Exception:
            connection.rollback()
            raise
        assert_database_integrity(connection)


def assert_database_integrity(connection: sqlite3.Connection) -> None:
    """Raise before continuing when SQLite reports structural corruption."""
    result = str(connection.execute("PRAGMA quick_check").fetchone()[0])
    if result.casefold() != "ok":
        raise RuntimeError(f"Falha na verificação de integridade do SQLite: {result}")


def _migration_001_initial_schema(connection: sqlite3.Connection) -> None:
    _execute_statements(
        connection,
        """
        CREATE TABLE IF NOT EXISTS projects (
            id TEXT PRIMARY KEY,
            title TEXT NOT NULL,
            problem_statement TEXT NOT NULL,
            problem_source TEXT NOT NULL DEFAULT '',
            problem_level TEXT NOT NULL DEFAULT '',
            math_area TEXT NOT NULL DEFAULT '',
            teacher_solution TEXT NOT NULL DEFAULT '',
            teacher_instructions TEXT NOT NULL DEFAULT '',
            llm_provider TEXT NOT NULL DEFAULT '',
            llm_model TEXT NOT NULL DEFAULT '',
            llm_api_key_source TEXT NOT NULL DEFAULT '',
            voice_provider TEXT NOT NULL DEFAULT '',
            voice_model TEXT NOT NULL DEFAULT '',
            voice TEXT NOT NULL DEFAULT '',
            voice_language TEXT NOT NULL DEFAULT '',
            voice_speed REAL NOT NULL DEFAULT 1.0,
            voice_api_key_source TEXT NOT NULL DEFAULT '',
            reuse_llm_api_key INTEGER NOT NULL DEFAULT 0,
            voiceover_enabled INTEGER NOT NULL DEFAULT 0,
            color_palette_id TEXT NOT NULL DEFAULT '',
            color_palette_snapshot TEXT NOT NULL DEFAULT '',
            presentation_video_path TEXT NOT NULL DEFAULT '',
            solution_video_path TEXT NOT NULL DEFAULT '',
            presentation_code_path TEXT NOT NULL DEFAULT '',
            solution_code_path TEXT NOT NULL DEFAULT '',
            output_delivery_mode TEXT NOT NULL DEFAULT 'separate',
            final_video_path TEXT NOT NULL DEFAULT '',
            status TEXT NOT NULL DEFAULT 'created',
            created_at TEXT NOT NULL,
            updated_at TEXT NOT NULL
        );
        CREATE TABLE IF NOT EXISTS generated_files (
            id TEXT PRIMARY KEY,
            project_id TEXT NOT NULL,
            file_type TEXT NOT NULL,
            path TEXT NOT NULL,
            version INTEGER NOT NULL DEFAULT 1,
            description TEXT NOT NULL DEFAULT '',
            created_at TEXT NOT NULL,
            FOREIGN KEY (project_id) REFERENCES projects(id) ON DELETE CASCADE
        );
        CREATE INDEX IF NOT EXISTS idx_generated_files_project_id
            ON generated_files(project_id);
        CREATE TABLE IF NOT EXISTS prompts (
            id TEXT PRIMARY KEY,
            name TEXT NOT NULL,
            agent_type TEXT NOT NULL,
            description TEXT NOT NULL DEFAULT '',
            is_default INTEGER NOT NULL DEFAULT 0,
            created_at TEXT NOT NULL,
            updated_at TEXT NOT NULL
        );
        CREATE INDEX IF NOT EXISTS idx_prompts_agent_type ON prompts(agent_type);
        CREATE TABLE IF NOT EXISTS prompt_versions (
            id TEXT PRIMARY KEY,
            prompt_id TEXT NOT NULL,
            version INTEGER NOT NULL,
            template_text TEXT NOT NULL,
            created_at TEXT NOT NULL,
            UNIQUE (prompt_id, version),
            FOREIGN KEY (prompt_id) REFERENCES prompts(id) ON DELETE CASCADE
        );
        CREATE INDEX IF NOT EXISTS idx_prompt_versions_prompt_id
            ON prompt_versions(prompt_id);
        CREATE TABLE IF NOT EXISTS project_prompts (
            id TEXT PRIMARY KEY,
            project_id TEXT NOT NULL,
            agent_type TEXT NOT NULL,
            prompt_id TEXT NOT NULL,
            prompt_version INTEGER NOT NULL,
            rendered_prompt_snapshot TEXT NOT NULL DEFAULT '',
            created_at TEXT NOT NULL,
            FOREIGN KEY (project_id) REFERENCES projects(id) ON DELETE CASCADE,
            FOREIGN KEY (prompt_id) REFERENCES prompts(id) ON DELETE RESTRICT
        );
        CREATE INDEX IF NOT EXISTS idx_project_prompts_project_id
            ON project_prompts(project_id);
        CREATE TABLE IF NOT EXISTS generation_logs (
            id TEXT PRIMARY KEY,
            project_id TEXT NOT NULL,
            level TEXT NOT NULL DEFAULT 'info',
            step TEXT NOT NULL DEFAULT '',
            message TEXT NOT NULL,
            created_at TEXT NOT NULL,
            FOREIGN KEY (project_id) REFERENCES projects(id) ON DELETE CASCADE
        );
        CREATE INDEX IF NOT EXISTS idx_generation_logs_project_id
            ON generation_logs(project_id);
        CREATE TABLE IF NOT EXISTS generation_jobs (
            id TEXT PRIMARY KEY,
            project_id TEXT NOT NULL,
            action TEXT NOT NULL,
            payload TEXT NOT NULL DEFAULT '{}',
            status TEXT NOT NULL DEFAULT 'pending',
            current_step TEXT NOT NULL DEFAULT '',
            progress INTEGER NOT NULL DEFAULT 0,
            attempts INTEGER NOT NULL DEFAULT 0,
            result TEXT NOT NULL DEFAULT '{}',
            error_message TEXT NOT NULL DEFAULT '',
            cancel_requested INTEGER NOT NULL DEFAULT 0,
            created_at TEXT NOT NULL,
            updated_at TEXT NOT NULL,
            started_at TEXT NOT NULL DEFAULT '',
            finished_at TEXT NOT NULL DEFAULT '',
            heartbeat_at TEXT NOT NULL DEFAULT '',
            FOREIGN KEY (project_id) REFERENCES projects(id) ON DELETE CASCADE,
            CHECK (status IN ('pending', 'running', 'completed', 'cancelled', 'failed')),
            CHECK (progress BETWEEN 0 AND 100)
        );
        CREATE INDEX IF NOT EXISTS idx_generation_jobs_project_status
            ON generation_jobs(project_id, status, created_at);
        CREATE UNIQUE INDEX IF NOT EXISTS idx_generation_jobs_one_active_project
            ON generation_jobs(project_id) WHERE status IN ('pending', 'running');
        CREATE TABLE IF NOT EXISTS ai_usage (
            id TEXT PRIMARY KEY,
            project_id TEXT NOT NULL,
            execution_id TEXT NOT NULL DEFAULT '',
            call_key TEXT NOT NULL,
            agent_type TEXT NOT NULL DEFAULT '',
            stage TEXT NOT NULL DEFAULT '',
            provider TEXT NOT NULL,
            model TEXT NOT NULL,
            status TEXT NOT NULL,
            attempt_type TEXT NOT NULL DEFAULT 'primary',
            sequence INTEGER NOT NULL DEFAULT 1,
            input_tokens INTEGER NOT NULL DEFAULT 0,
            output_tokens INTEGER NOT NULL DEFAULT 0,
            total_tokens INTEGER NOT NULL DEFAULT 0,
            cache_read_tokens INTEGER NOT NULL DEFAULT 0,
            cache_creation_tokens INTEGER NOT NULL DEFAULT 0,
            reasoning_tokens INTEGER NOT NULL DEFAULT 0,
            modality TEXT NOT NULL DEFAULT 'text',
            input_characters INTEGER NOT NULL DEFAULT 0,
            audio_output_tokens INTEGER NOT NULL DEFAULT 0,
            audio_seconds REAL NOT NULL DEFAULT 0,
            estimated_cost_usd REAL NOT NULL DEFAULT 0,
            pricing_known INTEGER NOT NULL DEFAULT 0,
            usage_source TEXT NOT NULL DEFAULT 'provider',
            metadata_available INTEGER NOT NULL DEFAULT 0,
            error_type TEXT NOT NULL DEFAULT '',
            error_code TEXT NOT NULL DEFAULT '',
            error_status TEXT NOT NULL DEFAULT '',
            error_message TEXT NOT NULL DEFAULT '',
            error_transient INTEGER NOT NULL DEFAULT 0,
            created_at TEXT NOT NULL,
            FOREIGN KEY (project_id) REFERENCES projects(id) ON DELETE CASCADE,
            UNIQUE (project_id, call_key, provider, model),
            CHECK (status IN ('completed', 'failed')),
            CHECK (attempt_type IN ('primary', 'fallback'))
        );
        CREATE INDEX IF NOT EXISTS idx_ai_usage_project_created
            ON ai_usage(project_id, created_at);
        CREATE INDEX IF NOT EXISTS idx_ai_usage_provider_model
            ON ai_usage(provider, model);
        CREATE TABLE IF NOT EXISTS model_catalog (
            id TEXT PRIMARY KEY,
            provider TEXT NOT NULL,
            modality TEXT NOT NULL,
            model_id TEXT NOT NULL,
            display_name TEXT NOT NULL DEFAULT '',
            enabled INTEGER NOT NULL DEFAULT 1,
            is_default INTEGER NOT NULL DEFAULT 0,
            is_builtin INTEGER NOT NULL DEFAULT 0,
            revision INTEGER NOT NULL DEFAULT 1,
            sort_order INTEGER NOT NULL DEFAULT 0,
            input_token_rate REAL NOT NULL DEFAULT 0,
            cached_input_token_rate REAL NOT NULL DEFAULT 0,
            output_token_rate REAL NOT NULL DEFAULT 0,
            input_character_rate REAL NOT NULL DEFAULT 0,
            audio_output_token_rate REAL NOT NULL DEFAULT 0,
            created_at TEXT NOT NULL,
            updated_at TEXT NOT NULL,
            UNIQUE (provider, modality, model_id),
            CHECK (modality IN ('text', 'speech'))
        );
        CREATE INDEX IF NOT EXISTS idx_model_catalog_provider_modality
            ON model_catalog(provider, modality, enabled, sort_order);
        CREATE UNIQUE INDEX IF NOT EXISTS idx_model_catalog_default
            ON model_catalog(provider, modality) WHERE is_default = 1;
        CREATE TABLE IF NOT EXISTS color_palettes (
            id TEXT PRIMARY KEY,
            name TEXT NOT NULL,
            description TEXT NOT NULL DEFAULT '',
            background TEXT NOT NULL,
            primary_text TEXT NOT NULL,
            secondary_text TEXT NOT NULL,
            surface TEXT NOT NULL,
            primary_color TEXT NOT NULL,
            secondary_color TEXT NOT NULL,
            highlight TEXT NOT NULL,
            stroke TEXT NOT NULL,
            enabled INTEGER NOT NULL DEFAULT 1,
            is_builtin INTEGER NOT NULL DEFAULT 0,
            revision INTEGER NOT NULL DEFAULT 1,
            sort_order INTEGER NOT NULL DEFAULT 0,
            created_at TEXT NOT NULL,
            updated_at TEXT NOT NULL,
            UNIQUE (name)
        );
        CREATE INDEX IF NOT EXISTS idx_color_palettes_enabled_order
            ON color_palettes(enabled, sort_order, name);
        CREATE TABLE IF NOT EXISTS app_settings (
            key TEXT PRIMARY KEY,
            value TEXT NOT NULL,
            updated_at TEXT NOT NULL
        );
        """,
    )


def _migration_002_legacy_columns(connection: sqlite3.Connection) -> None:
    """Normalize databases created before the last unversioned schema revisions."""
    _add_column_in_migration(
        connection, "projects", "color_palette_id", "TEXT NOT NULL DEFAULT ''"
    )
    _add_column_in_migration(
        connection, "projects", "color_palette_snapshot", "TEXT NOT NULL DEFAULT ''"
    )
    for column, definition in (
        ("error_type", "TEXT NOT NULL DEFAULT ''"),
        ("error_code", "TEXT NOT NULL DEFAULT ''"),
        ("error_status", "TEXT NOT NULL DEFAULT ''"),
        ("error_message", "TEXT NOT NULL DEFAULT ''"),
        ("error_transient", "INTEGER NOT NULL DEFAULT 0"),
    ):
        _add_column_in_migration(connection, "ai_usage", column, definition)


def _migration_003_reliability_contracts(connection: sqlite3.Connection) -> None:
    for table, column, definition in (
        ("projects", "workflow_revision", "INTEGER NOT NULL DEFAULT 1"),
        ("generated_files", "artifact_key", "TEXT NOT NULL DEFAULT ''"),
        ("generated_files", "sha256", "TEXT NOT NULL DEFAULT ''"),
        ("generated_files", "size_bytes", "INTEGER NOT NULL DEFAULT 0"),
        ("generation_jobs", "operation_id", "TEXT NOT NULL DEFAULT ''"),
        ("generation_jobs", "expected_phase", "TEXT NOT NULL DEFAULT ''"),
        ("generation_jobs", "workflow_revision", "INTEGER NOT NULL DEFAULT 1"),
        ("project_prompts", "rendered_system_snapshot", "TEXT NOT NULL DEFAULT ''"),
        ("project_prompts", "rendered_user_snapshot", "TEXT NOT NULL DEFAULT ''"),
        ("project_prompts", "prompt_sha256", "TEXT NOT NULL DEFAULT ''"),
        ("project_prompts", "operation_id", "TEXT NOT NULL DEFAULT ''"),
    ):
        _add_column_in_migration(connection, table, column, definition)
    _execute_statements(
        connection,
        """
        CREATE UNIQUE INDEX IF NOT EXISTS idx_generated_files_artifact_key
            ON generated_files(project_id, artifact_key) WHERE artifact_key <> '';
        CREATE TABLE IF NOT EXISTS workflow_transitions (
            operation_id TEXT PRIMARY KEY,
            project_id TEXT NOT NULL,
            action TEXT NOT NULL,
            expected_phase TEXT NOT NULL DEFAULT '',
            decision_sha256 TEXT NOT NULL,
            workflow_revision INTEGER NOT NULL DEFAULT 1,
            status TEXT NOT NULL DEFAULT 'queued',
            result_snapshot TEXT NOT NULL DEFAULT '{}',
            created_at TEXT NOT NULL,
            updated_at TEXT NOT NULL,
            FOREIGN KEY (project_id) REFERENCES projects(id) ON DELETE CASCADE,
            CHECK (status IN ('queued', 'running', 'completed', 'failed', 'cancelled'))
        );
        CREATE INDEX IF NOT EXISTS idx_workflow_transitions_project
            ON workflow_transitions(project_id, created_at);
        CREATE TABLE IF NOT EXISTS llm_call_cache (
            cache_key TEXT PRIMARY KEY,
            project_id TEXT NOT NULL,
            operation_id TEXT NOT NULL,
            role TEXT NOT NULL,
            mode TEXT NOT NULL,
            provider TEXT NOT NULL,
            requested_model TEXT NOT NULL,
            resolved_model TEXT NOT NULL DEFAULT '',
            finish_reason TEXT NOT NULL DEFAULT '',
            prompt_sha256 TEXT NOT NULL,
            response_text TEXT NOT NULL,
            created_at TEXT NOT NULL,
            FOREIGN KEY (project_id) REFERENCES projects(id) ON DELETE CASCADE
        );
        CREATE INDEX IF NOT EXISTS idx_llm_call_cache_project
            ON llm_call_cache(project_id, created_at);
        CREATE TABLE IF NOT EXISTS workflow_events (
            id TEXT PRIMARY KEY,
            event_key TEXT NOT NULL,
            project_id TEXT NOT NULL,
            operation_id TEXT NOT NULL DEFAULT '',
            job_id TEXT NOT NULL DEFAULT '',
            event_type TEXT NOT NULL,
            phase TEXT NOT NULL DEFAULT '',
            payload TEXT NOT NULL DEFAULT '{}',
            created_at TEXT NOT NULL,
            FOREIGN KEY (project_id) REFERENCES projects(id) ON DELETE CASCADE,
            UNIQUE (project_id, event_key)
        );
        CREATE INDEX IF NOT EXISTS idx_workflow_events_project
            ON workflow_events(project_id, created_at);
        """,
    )


def _migration_004_code_editor_drafts(connection: sqlite3.Connection) -> None:
    _execute_statements(
        connection,
        """
        CREATE TABLE IF NOT EXISTS code_editor_drafts (
            project_id TEXT NOT NULL,
            mode TEXT NOT NULL,
            code_content TEXT NOT NULL,
            source_code_sha256 TEXT NOT NULL,
            updated_at TEXT NOT NULL,
            PRIMARY KEY (project_id, mode),
            FOREIGN KEY (project_id) REFERENCES projects(id) ON DELETE CASCADE,
            CHECK (mode IN ('presentation', 'solution'))
        );
        CREATE INDEX IF NOT EXISTS idx_code_editor_drafts_project
            ON code_editor_drafts(project_id, updated_at);
        """,
    )


MIGRATIONS: tuple[SchemaMigration, ...] = (
    SchemaMigration(
        1,
        "initial_schema",
        "782af4529945f643298c903745ea29714f2afabc7836e3d9c7c379607bf97dc5",
        _migration_001_initial_schema,
    ),
    SchemaMigration(
        2,
        "normalize_legacy_columns",
        "2431c742b5f26b8d0a9ebf10ed7ed44963d324820d1442da16afdbd9d6949e73",
        _migration_002_legacy_columns,
    ),
    SchemaMigration(
        3,
        "reliability_contracts",
        "213d2957aea6dd98cde46b4bab0424d56505a3170c6c20751cf160137d9693b2",
        _migration_003_reliability_contracts,
    ),
    SchemaMigration(
        4,
        "code_editor_drafts",
        "55c1e234b85ecf86a450615835858cc27794a51d719bef08458768ed6fe4ae53",
        _migration_004_code_editor_drafts,
    ),
)
LATEST_SCHEMA_VERSION = MIGRATIONS[-1].version


def _execute_statements(connection: sqlite3.Connection, script: str) -> None:
    for statement in script.split(";"):
        if statement.strip():
            connection.execute(statement)


def _add_column_in_migration(
    connection: sqlite3.Connection,
    table: str,
    column: str,
    definition: str,
) -> None:
    columns = {str(row[1]) for row in connection.execute(f"PRAGMA table_info({table})")}
    if column not in columns:
        connection.execute(f"ALTER TABLE {table} ADD COLUMN {column} {definition}")


def _current_version(connection: sqlite3.Connection) -> int:
    table = connection.execute(
        "SELECT 1 FROM sqlite_master WHERE type = 'table' AND name = 'schema_migrations'"
    ).fetchone()
    if table is None:
        return 0
    row = connection.execute("SELECT COALESCE(MAX(version), 0) FROM schema_migrations").fetchone()
    return int(row[0])


def _has_application_tables(connection: sqlite3.Connection) -> bool:
    return (
        connection.execute(
            "SELECT 1 FROM sqlite_master WHERE type = 'table' AND name = 'projects'"
        ).fetchone()
        is not None
    )


def _migration_checksum(migration: SchemaMigration) -> str:
    return migration.checksum


def _backup_database(
    connection: sqlite3.Connection,
    database_path: Path,
    current_version: int,
    target_version: int,
) -> Path:
    backup_dir = database_path.parent / ".backups"
    backup_dir.mkdir(parents=True, exist_ok=True)
    stamp = datetime.now(UTC).strftime("%Y%m%dT%H%M%SZ")
    backup_path = backup_dir / (
        f"{database_path.stem}.v{current_version}-to-v{target_version}.{stamp}.sqlite3"
    )
    with sqlite3.connect(backup_path) as destination:
        connection.backup(destination)
    return backup_path
