"""Tests for final video assembly with FFmpeg."""

from __future__ import annotations

import subprocess
from pathlib import Path

from olympianim.services.video_assembler import VideoAssembler


def test_combines_presentation_and_solution_in_order(tmp_path: Path) -> None:
    presentation = tmp_path / "presentation.mp4"
    solution = tmp_path / "solution.mp4"
    presentation.write_bytes(b"presentation")
    solution.write_bytes(b"solution")
    commands: list[list[str]] = []

    def runner(command: list[str], **_kwargs: object) -> subprocess.CompletedProcess[str]:
        commands.append(command)
        Path(command[-1]).write_bytes(b"final")
        return subprocess.CompletedProcess(command, 0, "", "")

    output = tmp_path / "final" / "olympianim_final.mp4"
    result = VideoAssembler(command_runner=runner).combine((presentation, solution), output)

    assert result.success
    assert result.video_path == str(output)
    assert commands[0][:5] == ["ffmpeg", "-y", "-f", "concat", "-safe"]
    assert output.read_bytes() == b"final"


def test_keeps_sources_when_ffmpeg_fails(tmp_path: Path) -> None:
    presentation = tmp_path / "presentation.mp4"
    solution = tmp_path / "solution.mp4"
    presentation.write_bytes(b"presentation")
    solution.write_bytes(b"solution")

    def runner(command: list[str], **_kwargs: object) -> subprocess.CompletedProcess[str]:
        return subprocess.CompletedProcess(command, 1, "", "codec mismatch")

    result = VideoAssembler(command_runner=runner).combine(
        (presentation, solution), tmp_path / "final.mp4"
    )

    assert not result.success
    assert "codec mismatch" in result.error_message
    assert presentation.is_file()
    assert solution.is_file()
