"""Tests for project-scoped workflow and LangChain tool logs."""

from __future__ import annotations

from uuid import uuid4

from olympianim.database.models import ProjectCreate
from olympianim.database.repository import ProjectRepository
from olympianim.services.project_logging import ProjectLogger, ProjectToolCallback


def test_project_logger_preserves_complete_errors_and_redacts_secrets(tmp_path) -> None:
    repository = ProjectRepository(tmp_path / "olympianim.db")
    project = repository.create_project(ProjectCreate(title="Projeto", problem_statement="P"))
    logger = ProjectLogger(repository, project.id, secrets=("sk-secret",))

    logger.error(
        "agent.builder.presentation",
        f"Falha sk-secret\n{'x' * 800}\nTypeError: VGroup only accepts VMobject",
    )

    entry = repository.list_logs(project.id)[0]
    assert entry.level == "error"
    assert "sk-secret" not in entry.message
    assert "\n" in entry.message
    assert len(entry.message) > 800
    assert entry.message.endswith("TypeError: VGroup only accepts VMobject")


def test_tool_callback_records_query_and_outcome(tmp_path) -> None:
    repository = ProjectRepository(tmp_path / "olympianim.db")
    project = repository.create_project(ProjectCreate(title="Projeto", problem_statement="P"))
    callback = ProjectToolCallback(
        ProjectLogger(repository, project.id),
        role="builder",
        mode="presentation",
    )
    run_id = uuid4()

    callback.on_tool_start(
        {"name": "search_manim_reference"},
        "",
        run_id=run_id,
        inputs={"query": "Scene.play", "limit": 3},
    )
    callback.on_tool_end("resultado", run_id=run_id)

    logs = repository.list_logs(project.id)
    assert len(logs) == 2
    assert all(entry.step == "tool.builder.presentation.search_manim_reference" for entry in logs)
    assert "Scene.play" in logs[0].message
    assert "sucesso" in logs[1].message
