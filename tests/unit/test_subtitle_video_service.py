"""Tests for reversible FFmpeg hard-subtitle derivatives."""

from __future__ import annotations

import subprocess
from pathlib import Path

import pytest

from olympianim.database.models import ProjectCreate
from olympianim.database.repository import ProjectRepository
from olympianim.services.artifact_service import ArtifactService
from olympianim.services.subtitle_style import SubtitleStyle
from olympianim.services.subtitle_video_service import SubtitleVideoMode, SubtitleVideoService

_SRT = """1
00:00:00,000 --> 00:00:01,000
Olá, estudante!
"""


def _service(
    tmp_path: Path,
    runner,
    *,
    mode: SubtitleVideoMode = "presentation",
) -> tuple[SubtitleVideoService, ProjectRepository, str, Path]:
    repository = ProjectRepository(tmp_path / "olympianim.db")
    project = repository.create_project(ProjectCreate(title="Projeto", problem_statement="P"))
    artifacts = ArtifactService(repository=repository, projects_dir=tmp_path / "projects")
    video = artifacts.project_directory(project.id) / mode / f"{mode}.mp4"
    video.write_bytes(b"original-video")
    artifacts.register_existing(
        project.id,
        file_type=f"{mode}_video",
        path=video,
        description="Vídeo original",
        artifact_key=f"{mode}_video:v1",
    )
    artifacts.save_text(
        project.id,
        relative_path=f"{mode}/{mode}.srt",
        content=_SRT,
        file_type=f"{mode}_subtitle",
        description="Legendas",
        artifact_key=f"{mode}_subtitle:v1",
    )
    return (
        SubtitleVideoService(
            repository=repository,
            artifact_service=artifacts,
            command_runner=runner,
        ),
        repository,
        project.id,
        video,
    )


def _successful_runner(command: list[str], **_kwargs: object) -> subprocess.CompletedProcess[str]:
    Path(command[-1]).write_bytes(b"captioned-video")
    return subprocess.CompletedProcess(command, 0, "", "")


def test_add_registers_matching_derivative_and_preserves_original(tmp_path: Path) -> None:
    commands: list[list[str]] = []

    def runner(command: list[str], **kwargs: object) -> subprocess.CompletedProcess[str]:
        commands.append(command)
        return _successful_runner(command, **kwargs)

    service, repository, project_id, original = _service(tmp_path, runner)

    result = service.add(project_id, "presentation", original)
    state = service.state(project_id, "presentation", original)

    assert result.success
    assert original.read_bytes() == b"original-video"
    assert Path(result.video_path).read_bytes() == b"captioned-video"
    assert state.captioned
    assert state.display_video_path == result.video_path
    assert "libx264" in commands[0]
    video_filter = commands[0][commands[0].index("-vf") + 1]
    assert "subtitles=filename=" in video_filter
    assert "PrimaryColour=&H0000FFFF" in video_filter
    records = repository.list_generated_files(project_id)
    assert sum(record.file_type == "presentation_captioned_video" for record in records) == 1


def test_remove_deletes_only_derivative_and_restores_original(tmp_path: Path) -> None:
    service, repository, project_id, original = _service(tmp_path, _successful_runner)
    result = service.add(project_id, "presentation", original)

    service.remove(project_id, "presentation")
    state = service.state(project_id, "presentation", original)

    assert original.is_file()
    assert not Path(result.video_path).exists()
    assert state.display_video_path == str(original.resolve())
    assert not any(
        record.file_type == "presentation_captioned_video"
        for record in repository.list_generated_files(project_id)
    )


def test_ffmpeg_failure_leaves_original_and_no_partial_derivative(tmp_path: Path) -> None:
    def runner(command: list[str], **_kwargs: object) -> subprocess.CompletedProcess[str]:
        return subprocess.CompletedProcess(command, 1, "", "codec failure")

    service, repository, project_id, original = _service(tmp_path, runner)

    result = service.add(project_id, "presentation", original)

    assert not result.success
    assert "codec failure" in result.error_message
    assert original.read_bytes() == b"original-video"
    assert not any(
        record.file_type == "presentation_captioned_video"
        for record in repository.list_generated_files(project_id)
    )
    assert not list(original.parent.glob(".captioned_*.mp4"))
    assert not list(original.parent.glob(".subtitle_*.srt"))


def test_source_change_invalidates_and_replaces_previous_derivative(tmp_path: Path) -> None:
    service, repository, project_id, original = _service(tmp_path, _successful_runner)
    first = service.add(project_id, "presentation", original)
    first_key = next(
        record.artifact_key
        for record in repository.list_generated_files(project_id)
        if record.file_type == "presentation_captioned_video"
    )
    original.write_bytes(b"new-original-video")

    stale = service.state(project_id, "presentation", original)
    second = service.add(project_id, "presentation", original)
    records = [
        record
        for record in repository.list_generated_files(project_id)
        if record.file_type == "presentation_captioned_video"
    ]

    assert first.success and second.success
    assert not stale.captioned
    assert len(records) == 1
    assert records[0].artifact_key != first_key


def test_style_change_invalidates_and_replaces_previous_derivative(tmp_path: Path) -> None:
    commands: list[list[str]] = []

    def runner(command: list[str], **kwargs: object) -> subprocess.CompletedProcess[str]:
        commands.append(command)
        return _successful_runner(command, **kwargs)

    service, repository, project_id, original = _service(tmp_path, runner)
    first = service.add(project_id, "presentation", original)
    custom_style = SubtitleStyle("#123456")

    stale = service.state(
        project_id,
        "presentation",
        original,
        style=custom_style,
    )
    second = service.add(
        project_id,
        "presentation",
        original,
        style=custom_style,
    )
    records = [
        record
        for record in repository.list_generated_files(project_id)
        if record.file_type == "presentation_captioned_video"
    ]

    assert first.success and second.success
    assert not stale.captioned
    assert "PrimaryColour=&H00563412" in commands[-1][commands[-1].index("-vf") + 1]
    assert len(records) == 1
    assert records[0].artifact_key.endswith(custom_style.fingerprint)


@pytest.mark.parametrize("mode", ("presentation", "solution", "final"))
def test_modes_are_independent(tmp_path: Path, mode: SubtitleVideoMode) -> None:
    service, repository, project_id, original = _service(
        tmp_path,
        _successful_runner,
        mode=mode,
    )

    result = service.add(project_id, mode, original)
    service.remove(project_id, mode)

    assert result.success
    assert original.is_file()
    assert not any(
        record.file_type == f"{mode}_captioned_video"
        for record in repository.list_generated_files(project_id)
    )
