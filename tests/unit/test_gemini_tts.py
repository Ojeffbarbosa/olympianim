"""Tests for the Gemini adapter exposed through Manim Voiceover."""

from __future__ import annotations

import base64
import wave
from pathlib import Path
from types import SimpleNamespace

import httpx
import pytest

from olympianim.manim.gemini_tts import GeminiSpeechService
from olympianim.manim.usage_events import read_usage_events


class _Interactions:
    def __init__(self, *, failures: int = 0) -> None:
        self.failures = failures
        self.calls: list[dict[str, object]] = []

    def create(self, **kwargs: object) -> object:
        self.calls.append(kwargs)
        if len(self.calls) <= self.failures:
            raise httpx.ConnectError("temporary outage")
        pcm = b"\x00\x00" * 240
        return SimpleNamespace(
            output_audio=SimpleNamespace(data=base64.b64encode(pcm).decode("ascii")),
            usage=SimpleNamespace(total_input_tokens=12, total_output_tokens=240),
        )


def _service(tmp_path: Path, interactions: _Interactions) -> GeminiSpeechService:
    return GeminiSpeechService(
        api_key="google-secret",
        model="gemini-3.1-flash-tts-preview",
        voice="Kore",
        language="Português (Brasil)",
        video_mode="presentation",
        prompt_template="Idioma: {language}\nModo: {video_mode}\n### TRANSCRIPT\n{transcript}",
        cache_dir=tmp_path,
        retry_delay=0,
        client=SimpleNamespace(interactions=interactions),
    )


def test_gemini_service_generates_valid_wave_and_preserves_text(tmp_path: Path) -> None:
    interactions = _Interactions()
    service = _service(tmp_path, interactions)

    result = service._wrap_generate_from_text("Leia dois mais dois.")

    with wave.open(str(tmp_path / result["final_audio"]), "rb") as audio:
        assert audio.getnchannels() == 1
        assert audio.getframerate() == 24_000
        assert audio.getsampwidth() == 2
    assert result["input_text"] == "Leia dois mais dois."
    call = interactions.calls[0]
    assert call["model"] == "gemini-3.1-flash-tts-preview"
    assert call["generation_config"] == {"speech_config": [{"voice": "Kore"}]}
    assert "### TRANSCRIPT\nLeia dois mais dois." in str(call["input"])


def test_gemini_service_uses_manim_local_cache(tmp_path: Path) -> None:
    interactions = _Interactions()
    service = _service(tmp_path, interactions)

    service._wrap_generate_from_text("Mesmo texto.")
    service._wrap_generate_from_text("Mesmo texto.")

    assert len(interactions.calls) == 1


def test_gemini_service_retries_transient_transport_errors(tmp_path: Path) -> None:
    interactions = _Interactions(failures=2)
    service = _service(tmp_path, interactions)

    service._wrap_generate_from_text("Tente novamente.")

    assert len(interactions.calls) == 3


def test_gemini_service_emits_each_real_attempt(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    usage_path = tmp_path / "usage.jsonl"
    monkeypatch.setenv("OLYMPIANIM_USAGE_EVENTS_PATH", str(usage_path))
    interactions = _Interactions(failures=2)

    _service(tmp_path, interactions)._wrap_generate_from_text("Tente novamente.")

    events = read_usage_events(usage_path)
    assert [event.status for event in events] == ["failed", "failed", "completed"]
    assert events[-1].input_tokens == 12
    assert events[-1].audio_output_tokens == 240
    assert events[-1].audio_seconds == pytest.approx(0.01)


def test_gemini_service_does_not_retry_permanent_errors(tmp_path: Path) -> None:
    class _PermanentInteractions:
        calls = 0

        def create(self, **kwargs: object) -> object:
            _ = kwargs
            self.calls += 1
            raise ValueError("invalid voice")

    interactions = _PermanentInteractions()
    service = _service(tmp_path, interactions)  # type: ignore[arg-type]

    with pytest.raises(RuntimeError, match="invalid voice"):
        service._wrap_generate_from_text("Falha permanente.")

    assert interactions.calls == 1


def test_gemini_service_records_sanitized_error_details(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    usage_path = tmp_path / "usage.jsonl"
    monkeypatch.setenv("OLYMPIANIM_USAGE_EVENTS_PATH", str(usage_path))

    class InvalidVoiceError(ValueError):
        code = 400
        status = "INVALID_ARGUMENT"
        message = "invalid voice"

    class InvalidInteractions:
        def create(self, **kwargs: object) -> object:
            _ = kwargs
            raise InvalidVoiceError()

    service = _service(tmp_path, InvalidInteractions())  # type: ignore[arg-type]
    with pytest.raises(RuntimeError, match="400: invalid voice"):
        service._wrap_generate_from_text("Falha permanente.")

    event = read_usage_events(usage_path)[0]
    assert event.error_type == "InvalidVoiceError"
    assert event.error_code == "400"
    assert event.error_status == "INVALID_ARGUMENT"
    assert event.error_message == "invalid voice"
    assert event.error_transient is False
