"""Real FFmpeg validation for reversible hard subtitles."""

from __future__ import annotations

import shutil
import subprocess
from pathlib import Path

import pytest

from olympianim.database.models import ProjectCreate
from olympianim.database.repository import ProjectRepository
from olympianim.services.artifact_service import ArtifactService
from olympianim.services.subtitle_video_service import SubtitleVideoService


@pytest.mark.integration
def test_ffmpeg_burns_utf8_subtitles_and_preserves_playable_audio_video(
    tmp_path: Path,
) -> None:
    if shutil.which("ffmpeg") is None or shutil.which("ffprobe") is None:
        pytest.skip("FFmpeg/FFprobe não estão disponíveis.")
    repository = ProjectRepository(tmp_path / "olympianim.db")
    project = repository.create_project(ProjectCreate(title="Projeto", problem_statement="P"))
    artifacts = ArtifactService(
        repository=repository,
        projects_dir=tmp_path / "projects with spaces",
    )
    original = artifacts.project_directory(project.id) / "presentation/presentation.mp4"
    created = subprocess.run(
        [
            "ffmpeg",
            "-y",
            "-f",
            "lavfi",
            "-i",
            "color=c=0x1b263b:s=640x360:r=24:d=1.2",
            "-f",
            "lavfi",
            "-i",
            "sine=frequency=440:sample_rate=44100:duration=1.2",
            "-c:v",
            "libx264",
            "-pix_fmt",
            "yuv420p",
            "-c:a",
            "aac",
            "-shortest",
            str(original),
        ],
        capture_output=True,
        text=True,
        check=False,
    )
    assert created.returncode == 0, created.stderr
    artifacts.register_video(project.id, mode="presentation", video_path=original)
    artifacts.save_text(
        project.id,
        relative_path="presentation/presentation.srt",
        content=("1\n00:00:00,100 --> 00:00:01,000\n" "Qual é a próxima razão?\n"),
        file_type="presentation_subtitle",
        description="Legendas",
        artifact_key="presentation_subtitle:v1",
    )

    result = SubtitleVideoService(
        repository=repository,
        artifact_service=artifacts,
    ).add(project.id, "presentation", original)

    assert result.success, result.error_message
    captioned = Path(result.video_path)
    probe = subprocess.run(
        [
            "ffprobe",
            "-v",
            "error",
            "-show_entries",
            "stream=codec_type,codec_name",
            "-of",
            "csv=p=0",
            str(captioned),
        ],
        capture_output=True,
        text=True,
        check=False,
    )
    assert probe.returncode == 0, probe.stderr
    assert "h264,video" in probe.stdout
    assert "aac,audio" in probe.stdout

    original_frame = tmp_path / "original.png"
    captioned_frame = tmp_path / "captioned.png"
    for video, frame in ((original, original_frame), (captioned, captioned_frame)):
        extracted = subprocess.run(
            [
                "ffmpeg",
                "-y",
                "-ss",
                "0.5",
                "-i",
                str(video),
                "-frames:v",
                "1",
                str(frame),
            ],
            capture_output=True,
            text=True,
            check=False,
        )
        assert extracted.returncode == 0, extracted.stderr
    assert original_frame.read_bytes() != captioned_frame.read_bytes()
