"""Strict Manim Voiceover services for local presentation rendering."""

from __future__ import annotations

import os
from typing import cast

from manim_voiceover import VoiceoverScene
from manim_voiceover._typing import VoiceoverData
from manim_voiceover.services.base import PathLike, SpeechService


class VoiceoverProviderError(RuntimeError):
    """Raised when the provider selected for narration cannot generate audio."""


class SelectedSpeechService(SpeechService):
    """Delegate only to the speech provider explicitly selected by the user."""

    def __init__(
        self,
        speech_service: SpeechService,
        *,
        provider_name: str,
        global_speed: float = 1.0,
        cache_dir: PathLike | None = None,
    ) -> None:
        super().__init__(
            global_speed=global_speed,
            cache_dir=cache_dir,
            transcription_model=None,
        )
        self.speech_service = speech_service
        self.provider_name = provider_name

    def generate_from_text(
        self,
        text: str,
        cache_dir: PathLike | None = None,
        path: PathLike | None = None,
        **kwargs: object,
    ) -> VoiceoverData:
        """Generate audio or surface the selected provider's failure."""
        try:
            generated = dict(
                self.speech_service.generate_from_text(
                    text,
                    cache_dir=cache_dir,
                    path=path,
                    **kwargs,
                )
            )
            # Provider caches intentionally ignore bookmark markup because it does not
            # change the spoken audio. Restore the current markup for the tracker and
            # opt into Manim Voiceover's proportional fallback when no provider-native
            # word boundaries are available.
            generated["input_text"] = text
            if "<bookmark" in text:
                generated.setdefault("word_boundaries", [])
            return cast(VoiceoverData, generated)
        except (Exception, SystemExit) as exc:
            raise VoiceoverProviderError(
                f"Narração via {self.provider_name} falhou: {exc}"
            ) from exc


class ConfiguredVoiceoverScene(VoiceoverScene):
    """Configure the selected speech provider before the scene is constructed."""

    def setup(self) -> None:
        """Install the user-selected provider before any voiceover block runs."""
        super().setup()
        self.set_speech_service(selected_speech_service_from_environment())


def selected_speech_service_from_environment() -> SelectedSpeechService:
    """Build only the provider, voice and speed selected for this render."""
    provider = os.environ.get("OLYMPIANIM_VOICE_PROVIDER", "").strip()
    voice = os.environ.get("OLYMPIANIM_VOICE", "").strip()
    model = os.environ.get("OLYMPIANIM_VOICE_MODEL", "").strip()
    language = os.environ.get("OLYMPIANIM_VOICE_LANGUAGE", "").strip()
    video_mode = os.environ.get("OLYMPIANIM_VIDEO_MODE", "presentation").strip()
    prompt_template = os.environ.get("OLYMPIANIM_VOICE_PROMPT_TEMPLATE", "{transcript}")
    speed = _voice_speed_from_environment()
    speech_service: SpeechService

    if provider == "Google":
        from olympianim.manim.gemini_tts import (
            DEFAULT_GEMINI_TTS_MODEL,
            DEFAULT_GEMINI_VOICE,
            GeminiSpeechService,
        )

        if not os.environ.get("GOOGLE_API_KEY", "").strip():
            raise VoiceoverProviderError(
                "A chave da API do Google não foi informada para a narração."
            )
        try:
            speech_service = GeminiSpeechService(
                api_key=os.environ["GOOGLE_API_KEY"],
                voice=voice or DEFAULT_GEMINI_VOICE,
                model=model or DEFAULT_GEMINI_TTS_MODEL,
                language=language,
                video_mode=video_mode,
                prompt_template=prompt_template,
            )
        except Exception as exc:
            raise VoiceoverProviderError(
                f"Não foi possível configurar a narração via Google: {exc}"
            ) from exc
    elif provider == "OpenAI":
        from olympianim.manim.openai_tts import OpenAISpeechService

        if not voice:
            raise VoiceoverProviderError("A voz da narração OpenAI não foi informada.")
        if not os.environ.get("OPENAI_API_KEY", "").strip():
            raise VoiceoverProviderError(
                "A chave da API da OpenAI não foi informada para a narração."
            )
        try:
            speech_service = OpenAISpeechService(
                api_key=os.environ["OPENAI_API_KEY"],
                voice=voice,
                model=model or "tts-1",
            )
        except Exception as exc:
            raise VoiceoverProviderError(
                f"Não foi possível configurar a narração via OpenAI: {exc}"
            ) from exc
    else:
        raise VoiceoverProviderError(
            f"Provedor de narração inválido: {provider or 'não informado'}."
        )

    return SelectedSpeechService(
        speech_service,
        provider_name=provider,
        global_speed=speed,
    )


def _voice_speed_from_environment() -> float:
    """Validate the selected voice speed before constructing the provider."""
    raw_speed = os.environ.get("OLYMPIANIM_VOICE_SPEED", "1.0")
    try:
        speed = float(raw_speed)
    except ValueError as exc:
        raise VoiceoverProviderError(f"Velocidade de narração inválida: {raw_speed!r}.") from exc
    if not 0.5 <= speed <= 2.0:
        raise VoiceoverProviderError("A velocidade de narração deve estar entre 0.5 e 2.0.")
    return speed
