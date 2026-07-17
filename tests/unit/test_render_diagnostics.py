"""Render isolation and diagnostic classification tests."""

from __future__ import annotations

import subprocess
import sys
from pathlib import Path

import pytest

from olympianim.manim.presentation import (
    PresentationRenderer,
    _run_cancellable_process,
    build_render_environment,
    classify_render_error,
)
from olympianim.schemas.render import AIUsageEvent, ManimCodeResult


@pytest.mark.parametrize(
    ("traceback_text", "category"),
    (
        ("SyntaxError: invalid syntax", "Erro de sintaxe"),
        ("ModuleNotFoundError: No module named 'foo'", "Erro de importação"),
        ("AttributeError: Circle has no attribute foo", "Erro de API do Manim"),
        ("FileNotFoundError: No such file or directory", "Erro de caminho de arquivo"),
    ),
)
def test_classify_render_error(traceback_text: str, category: str) -> None:
    assert classify_render_error(traceback_text).startswith(category)


def test_render_environment_uses_an_allowlist_and_removes_all_provider_keys() -> None:
    environment = build_render_environment(
        {
            "PATH": "/usr/bin",
            "HOME": "/tmp/home",
            "OPENAI_API_KEY": "openai-secret",
            "GOOGLE_API_KEY": "google-secret",
            "ANTHROPIC_API_KEY": "anthropic-secret",
            "UNRELATED_PRIVATE_VALUE": "private",
        }
    )

    assert environment == {"PATH": "/usr/bin", "HOME": "/tmp/home"}


def test_renderer_uses_requested_quality_and_captures_process_result(
    tmp_path: Path,
) -> None:
    code_path = tmp_path / "scene.py"
    code_path.write_text("from manim import *", encoding="utf-8")
    captured: dict[str, object] = {}

    def fake_run(command, **kwargs):
        captured["command"] = command
        captured["kwargs"] = kwargs
        return subprocess.CompletedProcess(
            command,
            returncode=1,
            stdout="stdout capturado",
            stderr="AttributeError: Scene has no attribute animate_wrong",
        )

    result = PresentationRenderer(command_runner=fake_run).render(
        ManimCodeResult(
            mode="solution",
            scene_name="SolutionScene",
            code="from manim import *",
            code_path=str(code_path),
        ),
        project_directory=tmp_path,
        mode="solution",
        quality="high_quality",
    )

    assert not result.success
    assert result.return_code == 1
    assert result.stdout == "stdout capturado"
    assert result.stderr.startswith("Erro de API do Manim")
    assert result.quality == "high_quality"
    assert "-qh" in captured["command"]
    assert callable(captured["kwargs"]["cancellation_check"])


def test_renderer_collects_voice_usage_even_when_manim_fails(
    tmp_path: Path,
) -> None:
    code_path = tmp_path / "scene.py"
    code_path.write_text("from manim import *", encoding="utf-8")

    def fake_run(command, **kwargs):
        usage_path = Path(kwargs["env"]["OLYMPIANIM_USAGE_EVENTS_PATH"])
        usage_path.parent.mkdir(parents=True, exist_ok=True)
        usage_path.write_text(
            AIUsageEvent(
                provider="OpenAI",
                model="tts-1",
                status="completed",
                attempt_number=1,
                call_index=1,
                input_characters=25,
            ).model_dump_json()
            + "\n",
            encoding="utf-8",
        )
        return subprocess.CompletedProcess(command, 1, "", "render failed")

    result = PresentationRenderer(command_runner=fake_run).render(
        ManimCodeResult(
            mode="presentation",
            scene_name="Demo",
            code="from manim import *",
            code_path=str(code_path),
        ),
        project_directory=tmp_path,
        voiceover_enabled=True,
        voice_provider="OpenAI",
        voice_model="tts-1",
        voice="alloy",
        api_key="secret",
    )

    assert not result.success
    assert len(result.usage_events) == 1
    assert result.usage_events[0].input_characters == 25


def test_raw_render_log_redacts_the_selected_voice_key(tmp_path: Path) -> None:
    code_path = tmp_path / "scene.py"
    code_path.write_text("from manim import *", encoding="utf-8")

    def fake_run(command, **kwargs):
        return subprocess.CompletedProcess(command, 1, "", "failed with voice-secret")

    result = PresentationRenderer(command_runner=fake_run).render(
        ManimCodeResult(
            mode="presentation",
            scene_name="Demo",
            code="from manim import *",
            code_path=str(code_path),
        ),
        project_directory=tmp_path,
        voiceover_enabled=True,
        voice_provider="OpenAI",
        voice_model="tts-1",
        voice="alloy",
        api_key="voice-secret",
    )

    raw_log = Path(result.raw_log_path).read_text(encoding="utf-8")
    assert "voice-secret" not in raw_log
    assert "voice-secret" not in result.stderr
    assert "REDACTED" in raw_log


def test_process_runner_terminates_when_cancellation_is_requested(tmp_path: Path) -> None:
    def cancel() -> None:
        raise RuntimeError("cancelled")

    with pytest.raises(RuntimeError, match="cancelled"):
        _run_cancellable_process(
            [sys.executable, "-c", "import time; time.sleep(10)"],
            cwd=tmp_path,
            env={},
            cancellation_check=cancel,
            timeout_seconds=10,
        )


def test_process_runner_enforces_timeout(tmp_path: Path) -> None:
    result = _run_cancellable_process(
        [sys.executable, "-c", "import time; time.sleep(10)"],
        cwd=tmp_path,
        env={},
        cancellation_check=lambda: None,
        timeout_seconds=0,
    )

    assert result.returncode == 124
    assert "excedeu o limite" in result.stderr
