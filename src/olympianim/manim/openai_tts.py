"""Metered OpenAI TTS adapter for Manim Voiceover."""

from __future__ import annotations

import time
from pathlib import Path
from typing import Any

from manim_voiceover._typing import JsonValue, VoiceoverData
from manim_voiceover.helper import remove_bookmarks
from manim_voiceover.modify_audio import get_duration
from manim_voiceover.services.base import PathLike, SpeechService, path_to_string
from openai import OpenAI

from olympianim.manim.tts_errors import describe_tts_error, is_transient_tts_error
from olympianim.manim.usage_events import emit_usage_event, next_call_index
from olympianim.schemas.render import AIUsageEvent


class OpenAISpeechService(SpeechService):
    """Generate OpenAI speech and emit one event per real endpoint call."""

    def __init__(
        self,
        *,
        api_key: str,
        voice: str,
        model: str = "tts-1",
        client: Any | None = None,
        max_attempts: int = 3,
        retry_delay: float = 1.0,
        **kwargs: Any,
    ) -> None:
        super().__init__(transcription_model=None, **kwargs)
        if not api_key:
            raise ValueError("A chave da API da OpenAI não foi informada para a narração.")
        self.client = client or OpenAI(api_key=api_key)
        self.voice = voice
        self.model = model or "tts-1"
        self.max_attempts = max_attempts
        self.retry_delay = retry_delay

    def generate_from_text(
        self,
        text: str,
        cache_dir: PathLike | None = None,
        path: PathLike | None = None,
        **kwargs: object,
    ) -> VoiceoverData:
        """Synthesize a block, excluding Manim cache hits from consumption."""
        resolved_cache_dir = Path(cache_dir or self.cache_dir)
        resolved_cache_dir.mkdir(parents=True, exist_ok=True)
        raw_speed = kwargs.get("speed", 1.0)
        if not isinstance(raw_speed, int | float):
            raise TypeError("A velocidade da voz deve ser numérica.")
        speed = float(raw_speed)
        transcript = remove_bookmarks(text)
        input_data: dict[str, JsonValue] = {
            "input_text": transcript,
            "service": "openai",
            "config": {"voice": self.voice, "model": self.model, "speed": speed},
        }
        if cached := self.get_cached_result(input_data, resolved_cache_dir):
            return cached

        audio_path = (
            self.get_audio_basename(input_data) + ".mp3" if path is None else path_to_string(path)
        )
        destination = resolved_cache_dir / audio_path
        call_index = next_call_index()
        for attempt in range(1, self.max_attempts + 1):
            try:
                with self.client.audio.speech.with_streaming_response.create(
                    model=self.model,
                    voice=self.voice,
                    input=transcript,
                    speed=speed,
                ) as response:
                    response.stream_to_file(destination)
                duration = get_duration(destination)
                break
            except Exception as exc:
                transient = is_transient_tts_error(exc)
                details = describe_tts_error(exc, transient=transient)
                emit_usage_event(
                    AIUsageEvent(
                        provider="OpenAI",
                        model=self.model,
                        status="failed",
                        attempt_number=attempt,
                        call_index=call_index,
                        input_characters=len(transcript),
                        usage_source="openai_speech",
                        error_type=details.exception_type,
                        error_code=details.code,
                        error_status=details.status,
                        error_message=details.message,
                        error_transient=details.transient,
                    )
                )
                destination.unlink(missing_ok=True)
                if not transient or attempt == self.max_attempts:
                    raise RuntimeError(
                        f"OpenAI TTS ({self.model}) falhou: {details.summary()}"
                    ) from exc
                time.sleep(self.retry_delay * (2 ** (attempt - 1)))

        emit_usage_event(
            AIUsageEvent(
                provider="OpenAI",
                model=self.model,
                status="completed",
                attempt_number=1,
                call_index=call_index,
                input_characters=len(transcript),
                audio_seconds=duration,
                metadata_available=True,
                usage_source="openai_characters",
            )
        )
        return {
            "input_text": text,
            "input_data": input_data,
            "original_audio": audio_path,
        }
