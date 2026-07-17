"""Tests for atomic and idempotent project artifacts."""

from __future__ import annotations

import hashlib
from pathlib import Path

import pytest

from olympianim.database.models import ProjectCreate
from olympianim.database.repository import ProjectRepository
from olympianim.services.artifact_service import ArtifactService


@pytest.fixture
def artifacts(tmp_path: Path) -> tuple[ArtifactService, ProjectRepository]:
    repository = ProjectRepository(tmp_path / "olympianim.db")
    repository.create_project(
        ProjectCreate(title="Projeto", problem_statement="Problema"),
        project_id="project-id",
    )
    return (
        ArtifactService(repository=repository, projects_dir=tmp_path / "projects"),
        repository,
    )


def test_same_artifact_identity_is_updated_without_duplicate_metadata(
    artifacts: tuple[ArtifactService, ProjectRepository],
) -> None:
    service, repository = artifacts
    first = service.save_text(
        "project-id",
        relative_path="analysis/plan.md",
        content="primeiro",
        file_type="plan",
        description="Plano",
        artifact_key="plan:v1",
    )
    second = service.save_text(
        "project-id",
        relative_path="analysis/plan.md",
        content="segundo",
        file_type="plan",
        description="Plano",
        artifact_key="plan:v1",
    )

    records = repository.list_generated_files("project-id")
    assert first == second
    assert second.read_text(encoding="utf-8") == "segundo"
    assert len(records) == 1
    assert records[0].sha256 == hashlib.sha256(b"segundo").hexdigest()
    assert not list(second.parent.glob(f".{second.name}.*.tmp"))


def test_repair_source_creates_a_unified_diff(
    artifacts: tuple[ArtifactService, ProjectRepository],
) -> None:
    service, repository = artifacts
    service.save_manim_code("project-id", mode="presentation", code="x = 1\n", version=1)
    service.save_manim_code("project-id", mode="presentation", code="x = 2\n", version=2)

    diff = (
        service.project_directory("project-id")
        / "presentation/versions/presentation_v1_to_v2.diff"
    )
    assert "-x = 1" in diff.read_text(encoding="utf-8")
    assert "+x = 2" in diff.read_text(encoding="utf-8")
    assert {item.artifact_key for item in repository.list_generated_files("project-id")} >= {
        "presentation_code:v1",
        "presentation_code:v2",
        "presentation_code_diff:v2",
    }


def test_artifact_path_traversal_is_rejected(
    artifacts: tuple[ArtifactService, ProjectRepository],
) -> None:
    service, _ = artifacts

    with pytest.raises(ValueError, match="fora do projeto"):
        service.save_text(
            "project-id",
            relative_path="../../escape.txt",
            content="no",
            file_type="test",
            description="test",
        )
