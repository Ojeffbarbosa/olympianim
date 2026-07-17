"""Unit tests for project persistence use cases."""

from __future__ import annotations

from pathlib import Path

from olympianim.database.repository import ProjectRepository
from olympianim.services.project_service import ProjectImageInput, ProjectInput, ProjectService


def test_create_project_saves_metadata_and_problem_image(tmp_path: Path) -> None:
    repository = ProjectRepository(tmp_path / "olympianim.db")
    service = ProjectService(repository=repository, projects_dir=tmp_path / "projects")

    project = service.create_project(
        ProjectInput(
            title="Tabuleiro com somas iguais",
            problem_statement="Determine o valor de x.\nCom detalhes.",
            problem_images=(
                ProjectImageInput("../figura prova.png", b"image-bytes"),
                ProjectImageInput("pagina 2.jpg", b"second-image"),
            ),
            problem_source="OBMEP",
            problem_level="Nível 2",
            math_area="Álgebra",
            teacher_solution="x = 1",
            solution_images=(ProjectImageInput("caderno.png", b"solution-image"),),
            teacher_instructions="Não revele a ideia na apresentação.",
            llm_provider="OpenAI",
            llm_model="gpt-5.5",
            llm_api_key_source="env",
            voice_provider="OpenAI",
            voice_model="tts-1",
            voice="alloy",
            voice_language="Português (Brasil)",
            voiceover_enabled=True,
            voice_api_key_source="session",
            reuse_llm_api_key=True,
            color_palette_id="builtin:manim-dark",
            color_palette_snapshot='{"background":"#000000"}',
        )
    )

    opened = service.open_project(project.id)
    assert opened is not None
    assert opened.title == "Tabuleiro com somas iguais"
    assert opened.problem_source == "OBMEP"
    assert opened.teacher_solution == "x = 1"
    assert opened.llm_api_key_source == "env"
    assert opened.voiceover_enabled is True
    assert opened.voice_model == "tts-1"
    assert opened.voice_api_key_source == "session"
    assert opened.reuse_llm_api_key is True
    assert opened.color_palette_id == "builtin:manim-dark"
    assert opened.color_palette_snapshot == '{"background":"#000000"}'
    assert "api_key" not in opened.__dict__

    files = repository.list_generated_files(project.id)
    assert [(item.file_type, item.version, Path(item.path).name) for item in files[:3]] == [
        ("problem_image", 1, "01_figura_prova.png"),
        ("problem_image", 2, "02_pagina_2.jpg"),
        ("solution_image", 1, "01_caderno.png"),
    ]


def test_project_service_creates_expected_workspace_tree(tmp_path: Path) -> None:
    repository = ProjectRepository(tmp_path / "olympianim.db")
    service = ProjectService(repository=repository, projects_dir=tmp_path / "projects")

    project = service.create_project(ProjectInput(problem_statement="Problema"))
    project_dir = tmp_path / "projects" / project.id

    for relative in (
        "input",
        "analysis",
        "prompts",
        "teacher",
        "presentation/versions",
        "solution/versions",
        "logs",
    ):
        assert (project_dir / relative).is_dir()


def test_list_and_open_projects_use_repository(tmp_path: Path) -> None:
    repository = ProjectRepository(tmp_path / "olympianim.db")
    service = ProjectService(repository=repository, projects_dir=tmp_path / "projects")

    project = service.create_project(ProjectInput(problem_statement="Problema"))

    assert service.list_projects()[0].id == project.id
    assert service.open_project(project.id) == project
    assert service.open_project("missing") is None
