"""Final-video assembly using the local FFmpeg executable."""

from __future__ import annotations

import subprocess
from collections.abc import Callable, Sequence
from dataclasses import dataclass
from pathlib import Path


@dataclass(frozen=True)
class VideoAssemblyResult:
    """Result of combining the approved presentation and solution videos."""

    success: bool
    video_path: str = ""
    error_message: str = ""


CommandRunner = Callable[..., subprocess.CompletedProcess[str]]


class VideoAssembler:
    """Combine ordered MP4 files into one portable H.264/AAC video."""

    def __init__(self, command_runner: CommandRunner = subprocess.run) -> None:
        self.command_runner = command_runner

    def combine(self, videos: Sequence[Path], output_path: Path) -> VideoAssemblyResult:
        """Create ``output_path`` from ``videos`` without modifying the inputs."""
        if len(videos) < 2:
            return VideoAssemblyResult(
                success=False,
                error_message="São necessários os vídeos de apresentação e resolução.",
            )
        missing = next((path for path in videos if not path.is_file()), None)
        if missing is not None:
            return VideoAssemblyResult(
                success=False,
                error_message=f"Vídeo necessário para a montagem não foi encontrado: {missing.name}.",
            )

        output_path.parent.mkdir(parents=True, exist_ok=True)
        manifest_path = output_path.with_suffix(".concat.txt")
        manifest_path.write_text(
            "".join(f"file {self._quote(path.resolve())}\n" for path in videos),
            encoding="utf-8",
        )
        command = [
            "ffmpeg",
            "-y",
            "-f",
            "concat",
            "-safe",
            "0",
            "-i",
            str(manifest_path),
            "-c:v",
            "libx264",
            "-c:a",
            "aac",
            "-movflags",
            "+faststart",
            str(output_path),
        ]
        try:
            completed = self.command_runner(command, capture_output=True, text=True, check=False)
        except OSError as exc:
            return VideoAssemblyResult(
                success=False,
                error_message=f"Não foi possível iniciar o FFmpeg: {exc}.",
            )
        finally:
            manifest_path.unlink(missing_ok=True)

        if completed.returncode != 0 or not output_path.is_file():
            detail = (completed.stderr or completed.stdout).strip()
            return VideoAssemblyResult(
                success=False,
                error_message=(
                    f"Falha ao montar o vídeo único com FFmpeg. {detail}"
                    if detail
                    else "Falha ao montar o vídeo único com FFmpeg."
                ),
            )
        return VideoAssemblyResult(success=True, video_path=str(output_path))

    def probe_duration(self, video_path: Path) -> float | None:
        """Return exact media duration for subtitle offsets using FFprobe."""
        try:
            completed = self.command_runner(
                [
                    "ffprobe",
                    "-v",
                    "error",
                    "-show_entries",
                    "format=duration",
                    "-of",
                    "default=noprint_wrappers=1:nokey=1",
                    str(video_path),
                ],
                capture_output=True,
                text=True,
                check=False,
            )
        except OSError:
            return None
        if completed.returncode != 0:
            return None
        try:
            duration = float(completed.stdout.strip())
        except ValueError:
            return None
        return duration if duration >= 0 else None

    @staticmethod
    def _quote(path: Path) -> str:
        return "'" + str(path).replace("'", "'\\''") + "'"
