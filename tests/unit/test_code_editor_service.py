"""Tests for manual Manim versions and non-destructive restoration."""

from __future__ import annotations

from pathlib import Path

from olympianim.database.models import ProjectCreate
from olympianim.database.repository import ProjectRepository
from olympianim.schemas.render import RenderResult, VoiceConfig
from olympianim.services.artifact_service import ArtifactService
from olympianim.services.code_editor_service import CodeEditorService

_CODE = """from manim import *

class Demo(Scene):
    def construct(self):
        self.add(Text('Original'))
"""


class _SuccessfulRenderer:
    def __init__(self) -> None:
        self.last_kwargs = {}

    def render(self, code, *, project_directory: Path, mode: str, **kwargs) -> RenderResult:
        self.last_kwargs = kwargs
        output = project_directory / mode / f"{mode}.mp4"
        subtitle = project_directory / mode / f"{mode}.srt"
        output.parent.mkdir(parents=True, exist_ok=True)
        output.write_bytes(code.code.encode("utf-8"))
        subtitle.write_text(
            "1\n00:00:00,000 --> 00:00:01,000\nNarração editada.\n",
            encoding="utf-8",
        )
        return RenderResult(
            mode=mode,
            success=True,
            return_code=0,
            video_path=str(output),
            subtitle_path=str(subtitle),
            code_path=code.code_path,
            attempts=1,
        )


class _FailingRenderer:
    def render(self, code, *, project_directory: Path, mode: str, **kwargs) -> RenderResult:
        _ = (project_directory, kwargs)
        return RenderResult(
            mode=mode,
            success=False,
            return_code=1,
            code_path=code.code_path,
            stderr="Falha de teste",
            attempts=1,
        )


def _service(tmp_path: Path) -> tuple[CodeEditorService, ProjectRepository, str]:
    repository = ProjectRepository(tmp_path / "olympianim.db")
    project = repository.create_project(ProjectCreate(title="Projeto", problem_statement="P"))
    artifacts = ArtifactService(repository=repository, projects_dir=tmp_path / "projects")
    code_path = artifacts.save_manim_code(
        project.id,
        mode="presentation",
        code=_CODE,
        version=1,
    )
    current_video = artifacts.project_directory(project.id) / "presentation/presentation.mp4"
    current_video.write_bytes(b"original-video")
    artifacts.register_video(
        project.id,
        mode="presentation",
        video_path=current_video,
        version=1,
    )
    assert code_path.is_file()
    return (
        CodeEditorService(
            repository=repository,
            artifact_service=artifacts,
            renderer=_SuccessfulRenderer(),
        ),
        repository,
        project.id,
    )


def test_manual_render_preserves_original_and_creates_new_version(tmp_path: Path) -> None:
    service, repository, project_id = _service(tmp_path)
    edited = _CODE.replace("Original", "Editado")

    result = service.render_new_version(project_id, "presentation", edited)

    assert result.success
    versions = service.list_versions(project_id, "presentation")
    assert [item.version for item in versions] == [2, 1]
    assert Path(versions[0].video_path).read_bytes() == edited.strip().encode("utf-8")
    assert Path(versions[1].video_path).read_bytes() == b"original-video"
    project = repository.get_project(project_id)
    assert project is not None
    assert project.presentation_code_path == versions[0].code_path
    assert project.presentation_video_path == versions[0].video_path


def test_manual_render_preserves_versioned_subtitle_and_transcript(tmp_path: Path) -> None:
    service, repository, project_id = _service(tmp_path)

    result = service.render_new_version(
        project_id,
        "presentation",
        _CODE.replace("Original", "Editado"),
    )

    subtitle = Path(result.subtitle_path)
    assert subtitle.name == "presentation_v2.srt"
    assert subtitle.is_file()
    records = repository.list_generated_files(project_id)
    subtitle_record = next(
        record for record in records if record.artifact_key == "presentation_subtitle:v2"
    )
    transcript_record = next(
        record for record in records if record.artifact_key == "presentation_transcript:v2"
    )
    assert Path(subtitle_record.path) == subtitle
    assert Path(transcript_record.path).read_text(encoding="utf-8") == "Narração editada.\n"


def test_manual_render_allows_unchanged_code_and_creates_new_version(
    tmp_path: Path,
) -> None:
    service, _, project_id = _service(tmp_path)

    result = service.render_new_version(project_id, "presentation", _CODE)

    assert result.success
    versions = service.list_versions(project_id, "presentation")
    assert [item.version for item in versions] == [2, 1]
    assert Path(versions[0].code_path).read_text(encoding="utf-8") == _CODE


def test_accepted_draft_survives_service_restart_and_is_cleared_after_render(
    tmp_path: Path,
) -> None:
    service, repository, project_id = _service(tmp_path)
    edited = _CODE.replace("Original", "Editado")
    source_hash = "source-v1"

    saved = service.save_draft(
        project_id,
        "presentation",
        edited,
        source_code_sha256=source_hash,
    )
    reopened = CodeEditorService(
        repository=repository,
        artifact_service=service.artifacts,
        renderer=service.renderer,
    )

    assert reopened.load_draft(project_id, "presentation") == saved
    reopened.render_new_version(project_id, "presentation", edited)
    assert reopened.load_draft(project_id, "presentation") is None


def test_explicit_draft_save_is_durable_idempotent_and_clears_when_reverted(
    tmp_path: Path,
) -> None:
    service, repository, project_id = _service(tmp_path)
    edited = _CODE.replace("Original", "Editado")

    saved = service.save_or_discard_draft(
        project_id,
        "presentation",
        edited,
        source_code=_CODE,
        source_code_sha256="source-v1",
    )

    assert saved is not None
    assert [item.version for item in service.list_versions(project_id, "presentation")] == [1]
    saved_again = service.save_or_discard_draft(
        project_id,
        "presentation",
        edited,
        source_code=_CODE,
        source_code_sha256="source-v1",
    )
    assert saved_again == saved
    draft_logs = [
        item
        for item in repository.list_logs(project_id)
        if item.step == "code_editor.draft.presentation"
    ]
    assert len(draft_logs) == 1
    reopened = CodeEditorService(
        repository=repository,
        artifact_service=service.artifacts,
        renderer=service.renderer,
    )
    assert reopened.load_draft(project_id, "presentation") == saved

    reopened.save_or_discard_draft(
        project_id,
        "presentation",
        _CODE,
        source_code=_CODE,
        source_code_sha256="source-v1",
    )
    assert reopened.load_draft(project_id, "presentation") is None


def test_explicit_render_attempt_preserves_draft_after_failure(tmp_path: Path) -> None:
    service, repository, project_id = _service(tmp_path)
    edited = _CODE.replace("Original", "Editado")
    source_hash = "source-v1"
    service.renderer = _FailingRenderer()

    service.save_or_discard_draft(
        project_id,
        "presentation",
        edited,
        source_code=_CODE,
        source_code_sha256=source_hash,
    )
    result = service.render_new_version(project_id, "presentation", edited)

    assert result.success is False
    draft = repository.get_code_editor_draft(project_id, mode="presentation")
    assert draft is not None
    assert draft.code_content == edited


def test_restore_historical_code_creates_third_version(tmp_path: Path) -> None:
    service, _, project_id = _service(tmp_path)
    service.render_new_version(
        project_id,
        "presentation",
        _CODE.replace("Original", "Editado"),
    )

    result = service.restore_version(project_id, "presentation", 1)

    assert result.success
    versions = service.list_versions(project_id, "presentation")
    assert [item.version for item in versions] == [3, 2, 1]
    assert Path(versions[2].code_path).read_text(encoding="utf-8") == Path(
        versions[0].code_path
    ).read_text(encoding="utf-8")


def test_manual_render_uses_temporary_voice_override_without_mutating_project(
    tmp_path: Path,
) -> None:
    service, repository, project_id = _service(tmp_path)
    selected = VoiceConfig(
        enabled=False,
        provider="OpenAI",
        model="tts-1",
        voice="nova",
        language="Inglês (EUA)",
        speed=1.4,
    )

    service.render_new_version(
        project_id,
        "presentation",
        _CODE.replace("Original", "Editado"),
        voice_api_key="temporary-key",
        voice_config=selected,
    )

    renderer = service.renderer
    assert isinstance(renderer, _SuccessfulRenderer)
    assert renderer.last_kwargs["voice_provider"] == "OpenAI"
    assert renderer.last_kwargs["voice_model"] == "tts-1"
    assert renderer.last_kwargs["voice"] == "nova"
    assert renderer.last_kwargs["voice_language"] == "Inglês (EUA)"
    assert renderer.last_kwargs["voice_speed"] == 1.4
    project = repository.get_project(project_id)
    assert project is not None
    assert project.voice_provider == ""
    assert project.voice_model == ""
