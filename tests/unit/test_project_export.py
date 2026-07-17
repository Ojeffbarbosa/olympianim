"""Tests for sanitized, rights-aware project evidence ZIP files."""

from __future__ import annotations

import io
import zipfile
from pathlib import Path, PurePosixPath

import pytest

from olympianim.database.models import ProjectCreate
from olympianim.database.repository import ProjectRepository
from olympianim.services.artifact_service import ArtifactService
from olympianim.services.project_export import ProjectExportService


def _fake_provider_key() -> str:
    return "sk-proj-" + "abcdefghijklmnop"


@pytest.fixture
def export_case(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> tuple[ProjectExportService, ProjectRepository, Path]:
    database_path = tmp_path / "olympianim.db"
    projects_dir = tmp_path / "projects"
    repository = ProjectRepository(database_path)
    repository.create_project(
        ProjectCreate(
            title="Caso",
            problem_statement="Mostre que x=1.",
            llm_provider="OpenAI",
            llm_model="test-model",
        ),
        project_id="project-id",
    )
    artifacts = ArtifactService(repository=repository, projects_dir=projects_dir)
    artifacts.save_manim_code(
        "project-id",
        mode="presentation",
        code="from manim import Scene\n",
        version=1,
    )
    project_root = artifacts.project_directory("project-id")
    video = project_root / "presentation/presentation.mp4"
    video.write_bytes(b"video")
    artifacts.register_video("project-id", mode="presentation", video_path=video)
    image = project_root / "input/problem.png"
    image.parent.mkdir(parents=True, exist_ok=True)
    image.write_bytes(b"image")
    repository.add_generated_file(
        "project-id",
        file_type="problem_image",
        path=str(image),
        artifact_key="problem_image:v1",
    )
    repository.record_project_prompt(
        "project-id",
        agent_type="workflow_planner",
        prompt_id=repository.create_prompt(
            name="Teste",
            agent_type="workflow_planner",
            is_default=False,
        ).id,
        prompt_version=1,
        rendered_system_snapshot="Siga as regras.",
        rendered_user_snapshot=(f"segredo {_fake_provider_key()} e caminho {project_root}"),
        prompt_sha256="hash",
        operation_id="operation",
    )
    outside = tmp_path / "outside.txt"
    outside.write_text("outside", encoding="utf-8")
    repository.add_generated_file(
        "project-id",
        file_type="presentation_code_diff",
        path=str(outside),
        artifact_key="outside:v1",
    )
    monkeypatch.setattr(
        "olympianim.services.project_export._command_version",
        lambda _command: "test-version",
    )
    return (
        ProjectExportService(repository=repository, projects_dir=projects_dir),
        repository,
        project_root,
    )


def test_default_export_is_sanitized_and_omits_original_media(
    export_case: tuple[ProjectExportService, ProjectRepository, Path],
) -> None:
    service, repository, project_root = export_case

    payload = service.build_zip("project-id")

    with zipfile.ZipFile(io.BytesIO(payload)) as archive:
        names = archive.namelist()
        combined_text = "\n".join(
            archive.read(name).decode("utf-8", errors="ignore") for name in names
        )
    assert "manifest.json" in names
    assert any(name.endswith("presentation_v1.py") for name in names)
    assert not any(name.endswith(".mp4") or name.endswith("problem.png") for name in names)
    assert not any("outside.txt" in name for name in names)
    assert str(project_root) not in combined_text
    assert _fake_provider_key() not in combined_text
    assert str(repository.database_path) not in combined_text
    assert all(
        not PurePosixPath(name).is_absolute() and ".." not in PurePosixPath(name).parts
        for name in names
    )


def test_original_media_requires_rights_confirmation(
    export_case: tuple[ProjectExportService, ProjectRepository, Path],
) -> None:
    service, _, _ = export_case

    with pytest.raises(ValueError, match="direitos de uso"):
        service.build_zip("project-id", include_originals=True)

    payload = service.build_zip(
        "project-id",
        include_originals=True,
        rights_confirmed=True,
    )
    with zipfile.ZipFile(io.BytesIO(payload)) as archive:
        names = archive.namelist()
    assert any(name.endswith(".mp4") for name in names)
    assert any(name.endswith("problem.png") for name in names)
