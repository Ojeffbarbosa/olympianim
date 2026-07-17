"""Unit tests for SQLite persistence."""

from __future__ import annotations

import sqlite3
from pathlib import Path

from olympianim.database.models import ProjectCreate
from olympianim.database.repository import ProjectRepository
from olympianim.database.sqlite import connect, initialize_database


def _table_names(database_path: Path) -> set[str]:
    with connect(database_path) as connection:
        rows = connection.execute("SELECT name FROM sqlite_master WHERE type = 'table'").fetchall()
    return {row["name"] for row in rows}


def _column_names(database_path: Path, table: str) -> set[str]:
    with connect(database_path) as connection:
        rows = connection.execute(f"PRAGMA table_info({table})").fetchall()
    return {row["name"] for row in rows}


def test_initialize_database_creates_expected_tables(tmp_path: Path) -> None:
    database_path = tmp_path / "olympianim.db"
    initialize_database(database_path)

    assert {
        "projects",
        "generated_files",
        "prompts",
        "prompt_versions",
        "project_prompts",
        "generation_logs",
        "generation_jobs",
        "ai_usage",
        "model_catalog",
        "color_palettes",
        "code_editor_drafts",
    } <= _table_names(database_path)


def test_project_table_does_not_contain_api_key_column(tmp_path: Path) -> None:
    database_path = tmp_path / "olympianim.db"
    initialize_database(database_path)

    columns = _column_names(database_path, "projects")
    assert "api_key" not in columns
    assert "llm_api_key" not in columns
    assert "llm_api_key_source" in columns
    assert "voiceover_enabled" in columns
    assert "voice_model" in columns
    assert "voice_api_key_source" in columns
    assert "reuse_llm_api_key" in columns
    assert "color_palette_id" in columns
    assert "color_palette_snapshot" in columns


def test_create_list_and_open_project(tmp_path: Path) -> None:
    repository = ProjectRepository(tmp_path / "olympianim.db")
    created = repository.create_project(
        ProjectCreate(
            title="Problema OBMEP",
            problem_statement="Enunciado completo",
            teacher_solution="Uma solução",
            teacher_instructions="Use geometria",
            llm_provider="OpenAI",
            llm_model="gpt-5.5",
            llm_api_key_source="session",
            voice_provider="OpenAI",
            voice="alloy",
            voice_language="Português (Brasil)",
            voiceover_enabled=True,
        )
    )

    opened = repository.get_project(created.id)
    assert opened is not None
    assert opened.problem_statement == "Enunciado completo"
    assert opened.teacher_solution == "Uma solução"
    assert opened.llm_api_key_source == "session"
    assert opened.voiceover_enabled is True
    assert repository.list_projects()[0].id == created.id


def test_generated_files_logs_and_project_artifacts(tmp_path: Path) -> None:
    repository = ProjectRepository(tmp_path / "olympianim.db")
    project = repository.create_project(
        ProjectCreate(title="Projeto", problem_statement="Problema")
    )

    repository.update_project_artifacts(
        project.id,
        presentation_video_path="/video/presentation.mp4",
        presentation_code_path="/code/presentation.py",
        status="rendered",
    )
    repository.add_generated_file(
        project.id,
        file_type="presentation_video",
        path="/video/presentation.mp4",
        description="Vídeo de apresentação",
    )
    repository.add_log(project.id, step="render", message="Renderização concluída")

    opened = repository.get_project(project.id)
    assert opened is not None
    assert opened.presentation_video_path == "/video/presentation.mp4"
    assert opened.presentation_code_path == "/code/presentation.py"
    assert opened.status == "rendered"
    assert repository.list_generated_files(project.id)[0].file_type == "presentation_video"
    assert repository.list_logs(project.id)[0].message == "Renderização concluída"


def test_delete_generated_files_is_scoped_by_project_and_type(tmp_path: Path) -> None:
    repository = ProjectRepository(tmp_path / "olympianim.db")
    first = repository.create_project(ProjectCreate(title="Primeiro", problem_statement="P"))
    second = repository.create_project(ProjectCreate(title="Segundo", problem_statement="P"))
    repository.add_generated_file(
        first.id,
        file_type="presentation_captioned_video",
        path="/first-captioned.mp4",
    )
    repository.add_generated_file(
        first.id,
        file_type="presentation_video",
        path="/first.mp4",
    )
    repository.add_generated_file(
        second.id,
        file_type="presentation_captioned_video",
        path="/second-captioned.mp4",
    )

    removed = repository.delete_generated_files(
        first.id,
        file_type="presentation_captioned_video",
    )

    assert [record.path for record in removed] == ["/first-captioned.mp4"]
    assert [record.path for record in repository.list_generated_files(first.id)] == ["/first.mp4"]
    assert [record.path for record in repository.list_generated_files(second.id)] == [
        "/second-captioned.mp4"
    ]


def test_code_editor_draft_is_scoped_updated_and_deleted(tmp_path: Path) -> None:
    repository = ProjectRepository(tmp_path / "olympianim.db")
    project = repository.create_project(
        ProjectCreate(title="Projeto", problem_statement="Problema")
    )

    first = repository.save_code_editor_draft(
        project.id,
        mode="solution",
        code_content="print('primeiro')",
        source_code_sha256="source-v1",
    )
    second = repository.save_code_editor_draft(
        project.id,
        mode="solution",
        code_content="print('segundo')",
        source_code_sha256="source-v1",
    )

    loaded = repository.get_code_editor_draft(project.id, mode="solution")
    assert first.project_id == project.id
    assert second.code_content == "print('segundo')"
    assert loaded == second
    assert repository.get_code_editor_draft(project.id, mode="presentation") is None

    repository.delete_code_editor_draft(project.id, mode="solution")
    assert repository.get_code_editor_draft(project.id, mode="solution") is None


def test_prompt_tables_support_versions_and_project_usage(tmp_path: Path) -> None:
    repository = ProjectRepository(tmp_path / "olympianim.db")
    project = repository.create_project(
        ProjectCreate(title="Projeto", problem_statement="Problema")
    )
    prompt = repository.create_prompt(
        name="Analista padrão",
        agent_type="workflow_planner",
        description="Prompt inicial",
        is_default=True,
    )
    v1 = repository.add_prompt_version(prompt.id, template_text="Analise {problem_statement}")
    v2 = repository.add_prompt_version(prompt.id, template_text="Analise sem resolver")
    used = repository.record_project_prompt(
        project.id,
        agent_type="workflow_planner",
        prompt_id=prompt.id,
        prompt_version=v2.version,
        rendered_prompt_snapshot="Analise sem resolver",
    )

    assert repository.list_prompts()[0].is_default is True
    assert v1.version == 1
    assert v2.version == 2
    assert used.project_id == project.id
    assert used.prompt_version == 2


def test_foreign_keys_are_enabled(tmp_path: Path) -> None:
    repository = ProjectRepository(tmp_path / "olympianim.db")
    with repository._connect() as connection:
        try:
            connection.execute("""
                INSERT INTO generated_files (
                    id, project_id, file_type, path, version, description, created_at
                )
                VALUES ('file-id', 'missing-project', 'video', '/x.mp4', 1, '', 'now')
                """)
        except sqlite3.IntegrityError:
            pass
        else:
            raise AssertionError("Foreign key constraint should reject missing projects")
