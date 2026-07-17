"""Reversible hard-subtitle derivatives built from native Manim SRT files."""

from __future__ import annotations

import hashlib
import os
import shutil
import subprocess
import uuid
from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path
from typing import Literal

from olympianim.config import PROJECTS_DIR
from olympianim.database.models import GeneratedFileRecord
from olympianim.database.repository import ProjectRepository
from olympianim.services.artifact_service import ArtifactService
from olympianim.services.subtitle_style import SubtitleStyle

SubtitleVideoMode = Literal["presentation", "solution", "final"]
CommandRunner = Callable[..., subprocess.CompletedProcess[str]]


@dataclass(frozen=True)
class SubtitleVideoState:
    """Current original, native subtitle and matching derived-video paths."""

    mode: SubtitleVideoMode
    original_video_path: str
    subtitle_path: str = ""
    captioned_video_path: str = ""
    subtitle_version: int = 1
    artifact_key: str = ""

    @property
    def display_video_path(self) -> str:
        """Prefer the matching hard-subtitle derivative when it exists."""
        return self.captioned_video_path or self.original_video_path

    @property
    def subtitle_available(self) -> bool:
        return bool(self.subtitle_path)

    @property
    def captioned(self) -> bool:
        return bool(self.captioned_video_path)


@dataclass(frozen=True)
class SubtitleVideoResult:
    """Outcome of a local FFmpeg subtitle operation."""

    success: bool
    video_path: str = ""
    error_message: str = ""


class SubtitleVideoService:
    """Create and remove captioned MP4 derivatives without replacing originals."""

    def __init__(
        self,
        repository: ProjectRepository | None = None,
        artifact_service: ArtifactService | None = None,
        *,
        projects_dir: Path = PROJECTS_DIR,
        command_runner: CommandRunner = subprocess.run,
    ) -> None:
        self.repository = repository or ProjectRepository()
        self.artifacts = artifact_service or ArtifactService(
            repository=self.repository,
            projects_dir=projects_dir,
        )
        self.command_runner = command_runner

    def state(
        self,
        project_id: str,
        mode: SubtitleVideoMode,
        original_video_path: str | Path,
        *,
        style: SubtitleStyle | None = None,
    ) -> SubtitleVideoState:
        """Resolve a derivative only when it matches the current MP4 and SRT."""
        selected_style = style or SubtitleStyle()
        original = self._project_file(project_id, Path(original_video_path))
        records = self.repository.list_generated_files(project_id)
        subtitle_record = self._latest_existing(
            records,
            file_type=f"{mode}_subtitle",
            project_id=project_id,
        )
        if subtitle_record is None:
            return SubtitleVideoState(mode=mode, original_video_path=str(original))

        subtitle = self._project_file(project_id, Path(subtitle_record.path))
        original_record = next(
            (
                record
                for record in reversed(records)
                if record.file_type == f"{mode}_video"
                and self._same_path(Path(record.path), original)
            ),
            None,
        )
        expected_key = self._artifact_key(
            mode,
            self._recorded_sha256(original_record, original),
            self._recorded_sha256(subtitle_record, subtitle),
            selected_style,
        )
        captioned_record = next(
            (
                record
                for record in reversed(records)
                if record.file_type == f"{mode}_captioned_video"
                and record.artifact_key == expected_key
                and self._is_project_file(project_id, Path(record.path))
            ),
            None,
        )
        return SubtitleVideoState(
            mode=mode,
            original_video_path=str(original),
            subtitle_path=str(subtitle),
            captioned_video_path=(captioned_record.path if captioned_record is not None else ""),
            subtitle_version=subtitle_record.version,
            artifact_key=expected_key,
        )

    def add(
        self,
        project_id: str,
        mode: SubtitleVideoMode,
        original_video_path: str | Path,
        *,
        style: SubtitleStyle | None = None,
    ) -> SubtitleVideoResult:
        """Burn the current native SRT into a separate MP4 using local FFmpeg."""
        selected_style = style or SubtitleStyle()
        current = self.state(
            project_id,
            mode,
            original_video_path,
            style=selected_style,
        )
        if current.captioned:
            return SubtitleVideoResult(success=True, video_path=current.captioned_video_path)
        if not current.subtitle_available:
            raise ValueError("Este vídeo não possui legendas SRT disponíveis.")

        original = Path(current.original_video_path)
        subtitle = Path(current.subtitle_path)
        output = self.artifacts.project_directory(project_id) / mode / f"{mode}_captioned.mp4"
        token = uuid.uuid4().hex
        temporary_subtitle = output.parent / f".subtitle_{token}.srt"
        temporary_output = output.parent / f".captioned_{token}.mp4"
        shutil.copy2(subtitle, temporary_subtitle)
        command = [
            "ffmpeg",
            "-y",
            "-i",
            str(original),
            "-map",
            "0:v:0",
            "-map",
            "0:a?",
            "-sn",
            "-vf",
            (
                f"subtitles=filename='{temporary_subtitle.name}':charenc=UTF-8:"
                f"force_style='{selected_style.force_style}'"
            ),
            "-c:v",
            "libx264",
            "-crf",
            "18",
            "-preset",
            "medium",
            "-c:a",
            "aac",
            "-b:a",
            "192k",
            "-movflags",
            "+faststart",
            str(temporary_output),
        ]
        try:
            completed = self.command_runner(
                command,
                cwd=output.parent,
                capture_output=True,
                text=True,
                check=False,
            )
        except OSError as exc:
            return SubtitleVideoResult(
                success=False,
                error_message=f"Não foi possível iniciar o FFmpeg: {exc}.",
            )
        finally:
            temporary_subtitle.unlink(missing_ok=True)

        if completed.returncode != 0 or not temporary_output.is_file():
            temporary_output.unlink(missing_ok=True)
            detail = (completed.stderr or completed.stdout).strip()
            return SubtitleVideoResult(
                success=False,
                error_message=(
                    f"Falha ao incorporar as legendas com FFmpeg. {detail}"
                    if detail
                    else "Falha ao incorporar as legendas com FFmpeg."
                ),
            )

        try:
            os.replace(temporary_output, output)
            self._drop_derivatives(project_id, mode, preserve=output)
            self.artifacts.register_existing(
                project_id,
                file_type=f"{mode}_captioned_video",
                path=output,
                description=(
                    f"Vídeo de {mode} com legendas incorporadas "
                    f"na cor {selected_style.text_color}."
                ),
                version=current.subtitle_version,
                artifact_key=current.artifact_key,
            )
            self.repository.add_log(
                project_id,
                step=f"subtitle.embed.{mode}",
                message=(
                    f"Legendas {selected_style.text_color} incorporadas ao vídeo "
                    "sem alterar o original."
                ),
            )
        finally:
            temporary_output.unlink(missing_ok=True)
        return SubtitleVideoResult(success=True, video_path=str(output))

    def remove(self, project_id: str, mode: SubtitleVideoMode) -> None:
        """Remove only hard-subtitle derivatives for the selected video mode."""
        removed = self._drop_derivatives(project_id, mode)
        if removed:
            self.repository.add_log(
                project_id,
                step=f"subtitle.remove.{mode}",
                message="Versão com legendas removida; o vídeo original foi preservado.",
            )

    def _drop_derivatives(
        self,
        project_id: str,
        mode: SubtitleVideoMode,
        *,
        preserve: Path | None = None,
    ) -> bool:
        removed = self.repository.delete_generated_files(
            project_id,
            file_type=f"{mode}_captioned_video",
        )
        preserved = preserve.resolve() if preserve is not None else None
        for record in removed:
            path = Path(record.path)
            if not self._is_project_file(project_id, path):
                continue
            if preserved is not None and path.resolve() == preserved:
                continue
            path.unlink(missing_ok=True)
        return bool(removed)

    def _latest_existing(
        self,
        records: list[GeneratedFileRecord],
        *,
        file_type: str,
        project_id: str,
    ) -> GeneratedFileRecord | None:
        return next(
            (
                record
                for record in reversed(records)
                if record.file_type == file_type
                and self._is_project_file(project_id, Path(record.path))
            ),
            None,
        )

    def _project_file(self, project_id: str, path: Path) -> Path:
        if not self._is_project_file(project_id, path):
            raise ValueError("O arquivo de vídeo ou legenda não pertence ao projeto.")
        return path.resolve()

    def _is_project_file(self, project_id: str, path: Path) -> bool:
        try:
            resolved = path.resolve(strict=True)
        except (FileNotFoundError, OSError):
            return False
        root = self.artifacts.project_directory(project_id).resolve()
        return resolved.is_relative_to(root) and resolved.is_file()

    @staticmethod
    def _same_path(first: Path, second: Path) -> bool:
        try:
            return first.resolve(strict=True) == second.resolve(strict=True)
        except (FileNotFoundError, OSError):
            return False

    @staticmethod
    def _recorded_sha256(record: GeneratedFileRecord | None, path: Path) -> str:
        if record is not None and record.sha256 and record.size_bytes == path.stat().st_size:
            return record.sha256
        return _sha256(path)

    @staticmethod
    def _artifact_key(
        mode: SubtitleVideoMode,
        video_sha256: str,
        subtitle_sha256: str,
        style: SubtitleStyle,
    ) -> str:
        return (
            f"{mode}_captioned_video:"
            f"{video_sha256[:16]}:{subtitle_sha256[:16]}:{style.fingerprint}"
        )


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()
