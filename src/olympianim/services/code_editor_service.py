"""Manual Manim editing, immutable versions, and re-rendering."""

from __future__ import annotations

import re
import shutil
from dataclasses import dataclass
from pathlib import Path
from typing import Literal

from olympianim.database.models import (
    CodeEditorDraftRecord,
    GeneratedFileRecord,
    ProjectRecord,
)
from olympianim.database.repository import ProjectRepository
from olympianim.manim.presentation import (
    PresentationRenderer,
    check_generated_code_safety,
    prepare_source_watermark_code,
    prepare_voiceover_code,
    presentation_scene_name,
)
from olympianim.prompts.service import PromptService
from olympianim.schemas.render import ManimCodeResult, RenderResult, VoiceConfig
from olympianim.services.artifact_service import ArtifactService
from olympianim.services.image_asset_service import ImageAssetService
from olympianim.services.project_logging import ProjectLogger
from olympianim.services.subtitle_service import SubtitleService
from olympianim.services.usage_service import UsageService

VideoMode = Literal["presentation", "solution"]
_VERSION_PATTERN = re.compile(r"_v(?P<version>\d+)\.(?:py|mp4)$")


@dataclass(frozen=True)
class CodeVersion:
    """One immutable code version and its optional rendered video."""

    version: int
    code_path: str
    video_path: str = ""
    created_at: str = ""


class CodeEditorService:
    """Own post-generation edits without mutating prior artifacts."""

    def __init__(
        self,
        repository: ProjectRepository | None = None,
        artifact_service: ArtifactService | None = None,
        renderer: PresentationRenderer | None = None,
        prompt_service: PromptService | None = None,
    ) -> None:
        self.repository = repository or ProjectRepository()
        self.artifacts = artifact_service or ArtifactService(repository=self.repository)
        self.renderer = renderer or PresentationRenderer()
        self.prompt_service = prompt_service or PromptService(repository=self.repository)

    def list_versions(self, project_id: str, mode: VideoMode) -> list[CodeVersion]:
        """Return code versions newest first, paired with their videos."""
        records = self.repository.list_generated_files(project_id)
        codes = self._records_by_version(records, f"{mode}_code")
        videos = self._records_by_version(records, f"{mode}_video")
        return [
            CodeVersion(
                version=version,
                code_path=codes[version].path,
                video_path=videos[version].path if version in videos else "",
                created_at=codes[version].created_at,
            )
            for version in sorted(codes, reverse=True)
            if Path(codes[version].path).is_file()
        ]

    def current_code(self, project_id: str, mode: VideoMode) -> str:
        """Load the active code, falling back to the newest preserved version."""
        project = self._project(project_id)
        path = Path(getattr(project, f"{mode}_code_path"))
        if not path.is_file():
            versions = self.list_versions(project_id, mode)
            if not versions:
                return ""
            path = Path(versions[0].code_path)
        return path.read_text(encoding="utf-8")

    def load_draft(self, project_id: str, mode: VideoMode) -> CodeEditorDraftRecord | None:
        """Load a durable draft without changing the active code version."""
        self._project(project_id)
        return self.repository.get_code_editor_draft(project_id, mode=mode)

    def save_draft(
        self,
        project_id: str,
        mode: VideoMode,
        code: str,
        *,
        source_code_sha256: str,
    ) -> CodeEditorDraftRecord:
        """Persist edited code as a recoverable draft."""
        self._project(project_id)
        if not code.strip():
            raise ValueError("O rascunho de código não pode ficar vazio.")
        existing = self.repository.get_code_editor_draft(project_id, mode=mode)
        if (
            existing is not None
            and existing.source_code_sha256 == source_code_sha256
            and existing.code_content == code
        ):
            return existing
        record = self.repository.save_code_editor_draft(
            project_id,
            mode=mode,
            code_content=code,
            source_code_sha256=source_code_sha256,
        )
        self.repository.add_log(
            project_id,
            step=f"code_editor.draft.{mode}",
            message="Rascunho de edição salvo.",
        )
        return record

    def save_or_discard_draft(
        self,
        project_id: str,
        mode: VideoMode,
        code: str,
        *,
        source_code: str,
        source_code_sha256: str,
    ) -> CodeEditorDraftRecord | None:
        """Persist one explicit edit, or discard it when reverted to its source."""
        self._project(project_id)
        existing = self.repository.get_code_editor_draft(project_id, mode=mode)
        belongs_to_source = bool(
            existing is not None and existing.source_code_sha256 == source_code_sha256
        )
        if code == source_code:
            if belongs_to_source:
                self.repository.delete_code_editor_draft(project_id, mode=mode)
            return None
        if not code.strip():
            raise ValueError("O rascunho de código não pode ficar vazio.")
        if belongs_to_source and existing is not None and existing.code_content == code:
            return existing
        return self.save_draft(
            project_id,
            mode,
            code,
            source_code_sha256=source_code_sha256,
        )

    def discard_draft(self, project_id: str, mode: VideoMode) -> None:
        """Discard only the selected project's draft."""
        self.repository.delete_code_editor_draft(project_id, mode=mode)

    def render_new_version(
        self,
        project_id: str,
        mode: VideoMode,
        code: str,
        *,
        voice_api_key: str = "",
        voice_prompt_template: str = "{transcript}",
        voice_config: VoiceConfig | None = None,
    ) -> RenderResult:
        """Save edited code, render it, and preserve both artifacts by version."""
        project = self._project(project_id)
        voice = voice_config or VoiceConfig(
            enabled=project.voiceover_enabled,
            provider=project.voice_provider,
            model=project.voice_model,
            voice=project.voice,
            language=project.voice_language,
            speed=project.voice_speed,
        )
        normalized = prepare_voiceover_code(code, require_voiceover=voice.enabled)
        normalized = prepare_source_watermark_code(normalized, project.problem_source)
        errors = check_generated_code_safety(
            normalized,
            require_voiceover=voice.enabled,
            allowed_image_paths={
                asset.manim_path
                for asset in ImageAssetService(
                    repository=self.repository,
                    projects_dir=self.artifacts.projects_dir,
                ).list_assets(project_id)
            },
        )
        if errors:
            raise ValueError(" ".join(errors))
        self.artifacts.clear_final_video(project_id)
        self._preserve_current_video(project_id, mode)
        version = self._next_version(project_id, mode)
        code_path = self.artifacts.save_manim_code(
            project_id,
            mode=mode,
            code=normalized,
            version=version,
        )
        logger = ProjectLogger(
            self.repository,
            project_id,
            secrets=(voice_api_key,),
        )
        logger.info(f"manual_render.{mode}", f"Renderização manual v{version} iniciada.")
        if voice.enabled:
            logger.info(
                f"manual_render.{mode}.voice",
                f"Voz: {voice.provider}:{voice.model}, {voice.voice}, "
                f"{voice.language}, velocidade {voice.speed:.1f}.",
            )
        result = self.renderer.render(
            ManimCodeResult(
                mode=mode,
                scene_name=presentation_scene_name(
                    normalized,
                    require_voiceover=voice.enabled,
                ),
                code=normalized,
                code_path=str(code_path),
            ),
            project_directory=self.artifacts.project_directory(project_id),
            mode=mode,
            api_key=voice_api_key,
            voice_provider=voice.provider,
            voice_model=voice.model,
            voice=voice.voice,
            voice_language=voice.language,
            voice_speed=voice.speed,
            voice_prompt_template=self._voice_prompt_template(
                voice.provider,
                voice_prompt_template,
            ),
            voiceover_enabled=voice.enabled,
            quality=self.repository.get_setting("render_quality", "low_quality"),
        )
        usage_service = UsageService(self.repository)
        usage_service.record_speech_events(
            result.usage_events,
            project_id=project_id,
            execution_id=f"manual-render-{version}",
            stage=mode,
            render_key=f"manual-render-{version}:voice:{mode}",
        )
        if not result.success:
            logger.error(
                f"manual_render.{mode}",
                result.stderr or result.error_traceback or "Falha sem diagnóstico.",
            )
            return result

        versioned_video = code_path.with_suffix(".mp4")
        shutil.copy2(result.video_path, versioned_video)
        self.artifacts.register_video(
            project_id,
            mode=mode,
            video_path=versioned_video,
            version=version,
        )
        versioned_subtitle = ""
        if result.subtitle_path and Path(result.subtitle_path).is_file():
            subtitle_path = code_path.with_suffix(".srt")
            shutil.copy2(result.subtitle_path, subtitle_path)
            transcript_path = self.artifacts.save_text(
                project_id,
                relative_path=f"{mode}/versions/{mode}_v{version}.txt",
                content=SubtitleService.transcript_file(subtitle_path),
                file_type=f"{mode}_transcript",
                description=f"Transcrição textual do vídeo de {mode}.",
                version=version,
                artifact_key=f"{mode}_transcript:v{version}",
            )
            self.artifacts.register_subtitle(
                project_id,
                mode=mode,
                subtitle_path=subtitle_path,
                transcript_path=transcript_path,
                version=version,
            )
            versioned_subtitle = str(subtitle_path)
        self.discard_draft(project_id, mode)
        logger.info(f"manual_render.{mode}", f"Versão v{version} concluída.")
        return result.model_copy(
            update={
                "video_path": str(versioned_video),
                "subtitle_path": versioned_subtitle,
            }
        )

    def restore_version(
        self,
        project_id: str,
        mode: VideoMode,
        version: int,
        *,
        voice_api_key: str = "",
        voice_prompt_template: str = "{transcript}",
        voice_config: VoiceConfig | None = None,
    ) -> RenderResult:
        """Restore historical code as a new auditable version."""
        selected = next(
            (item for item in self.list_versions(project_id, mode) if item.version == version),
            None,
        )
        if selected is None:
            raise ValueError(f"Versão v{version} não encontrada.")
        code = Path(selected.code_path).read_text(encoding="utf-8")
        return self.render_new_version(
            project_id,
            mode,
            code,
            voice_api_key=voice_api_key,
            voice_prompt_template=voice_prompt_template,
            voice_config=voice_config,
        )

    def _preserve_current_video(self, project_id: str, mode: VideoMode) -> None:
        project = self._project(project_id)
        current = Path(getattr(project, f"{mode}_video_path"))
        versions = self.list_versions(project_id, mode)
        if not current.is_file() or not versions:
            return
        latest = versions[0]
        if (
            latest.video_path
            and Path(latest.video_path).is_file()
            and Path(latest.video_path) != current
        ):
            return
        target = Path(latest.code_path).with_suffix(".mp4")
        if current == target:
            return
        shutil.copy2(current, target)
        self.artifacts.register_video(
            project_id,
            mode=mode,
            video_path=target,
            version=latest.version,
        )

    def _next_version(self, project_id: str, mode: VideoMode) -> int:
        versions = self.list_versions(project_id, mode)
        return (versions[0].version if versions else 0) + 1

    @staticmethod
    def _records_by_version(
        records: list[GeneratedFileRecord],
        file_type: str,
    ) -> dict[int, GeneratedFileRecord]:
        selected: dict[int, GeneratedFileRecord] = {}
        for record in records:
            if record.file_type != file_type:
                continue
            match = _VERSION_PATTERN.search(record.path)
            version = int(match.group("version")) if match else record.version
            selected[version] = record
        return selected

    def _project(self, project_id: str) -> ProjectRecord:
        project = self.repository.get_project(project_id)
        if project is None:
            raise ValueError("Projeto não encontrado.")
        return project

    def _voice_prompt_template(self, provider: str, requested: str) -> str:
        if provider != "Google" or requested != "{transcript}":
            return requested
        prompt = next(
            item
            for item in self.prompt_service.list_prompts("gemini_tts")
            if item.prompt.name == "Direção de narração Gemini - padrão"
        )
        return prompt.latest_version.template_text
