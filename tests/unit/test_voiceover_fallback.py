"""Tests for strict handling of failures from the selected TTS provider."""

from __future__ import annotations

import wave
from pathlib import Path

import pytest
from manim import Scene
from manim_voiceover.services.base import PathLike, SpeechService
from manim_voiceover.tracker import VoiceoverTracker

from olympianim.manim.voiceover import (
    SelectedSpeechService,
    VoiceoverProviderError,
    selected_speech_service_from_environment,
)


class _FailingSpeechService(SpeechService):
    """TTS double that always fails before an audio file is generated."""

    def generate_from_text(
        self,
        text: str,
        cache_dir: PathLike | None = None,
        path: PathLike | None = None,
        **kwargs: object,
    ) -> dict[str, object]:
        _ = (text, cache_dir, path, kwargs)
        raise RuntimeError("Synthetic TTS is unavailable")


class _CachedSpeechService(SpeechService):
    """Return provider metadata whose cached text has no current bookmarks."""

    def generate_from_text(
        self,
        text: str,
        cache_dir: PathLike | None = None,
        path: PathLike | None = None,
        **kwargs: object,
    ) -> dict[str, object]:
        _ = (text, cache_dir, path, kwargs)
        return {
            "input_text": "Texto em cache sem marcador.",
            "input_data": {"cache": True},
            "original_audio": "voice.wav",
        }


def test_selected_voiceover_service_surfaces_tts_failure(tmp_path: Path) -> None:
    service = SelectedSpeechService(
        _FailingSpeechService(cache_dir=tmp_path),
        provider_name="OpenAI",
        cache_dir=tmp_path,
    )

    with pytest.raises(VoiceoverProviderError, match="Narração via OpenAI falhou"):
        service.generate_from_text("Observe os dois dados destacados.")


def test_selected_voiceover_service_preserves_bookmarks_and_enables_fallback(
    tmp_path: Path,
) -> None:
    service = SelectedSpeechService(
        _CachedSpeechService(cache_dir=tmp_path),
        provider_name="Google",
        cache_dir=tmp_path,
    )
    narration = "Observe. <bookmark mark='mostrar_resultado'/> Agora aparece o resultado."

    generated = service.generate_from_text(narration)

    assert generated["input_text"] == narration
    assert generated["word_boundaries"] == []


def test_selected_voiceover_service_does_not_add_boundaries_without_bookmark(
    tmp_path: Path,
) -> None:
    service = SelectedSpeechService(
        _CachedSpeechService(cache_dir=tmp_path),
        provider_name="Google",
        cache_dir=tmp_path,
    )

    generated = service.generate_from_text("Uma fala curta sem marco interno.")

    assert "word_boundaries" not in generated


def test_native_bookmark_timing_uses_proportional_fallback(tmp_path: Path) -> None:
    audio_path = tmp_path / "voice.wav"
    with wave.open(str(audio_path), "wb") as audio:
        audio.setnchannels(1)
        audio.setsampwidth(2)
        audio.setframerate(8_000)
        audio.writeframes(b"\0\0" * 8_000)
    tracker = VoiceoverTracker(
        Scene(),
        {
            "input_text": "Antes. <bookmark mark='mostrar'/> Agora.",
            "final_audio": audio_path.name,
            "word_boundaries": [],
        },
        str(tmp_path),
    )

    assert tracker.time_until_bookmark("mostrar") == pytest.approx(0.5)


def test_openai_voiceover_reports_missing_selected_api_key(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("OLYMPIANIM_VOICE_PROVIDER", "OpenAI")
    monkeypatch.setenv("OLYMPIANIM_VOICE", "alloy")
    monkeypatch.delenv("OPENAI_API_KEY", raising=False)

    with pytest.raises(VoiceoverProviderError, match="chave da API da OpenAI"):
        selected_speech_service_from_environment()
