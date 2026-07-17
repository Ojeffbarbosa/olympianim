"""Gemini TTS adapter for Manim Voiceover's public SpeechService contract."""

from __future__ import annotations

import base64
import time
import wave
from hashlib import sha256
from pathlib import Path
from typing import Any, Protocol, cast

import httpx
from google import genai
from google.genai import errors
from manim_voiceover._typing import JsonValue, VoiceoverData
from manim_voiceover.helper import remove_bookmarks
from manim_voiceover.services.base import PathLike, SpeechService, path_to_string

from olympianim.manim.tts_errors import TTSErrorDetails, describe_tts_error
from olympianim.manim.usage_events import emit_usage_event, next_call_index
from olympianim.prompts.validator import render_prompt_template
from olympianim.schemas.render import AIUsageEvent

DEFAULT_GEMINI_TTS_MODEL = "gemini-3.1-flash-tts-preview"
DEFAULT_GEMINI_VOICE = "Kore"
_SAMPLE_RATE = 24_000
_SAMPLE_WIDTH = 2
_CHANNELS = 1


class _InteractionsClient(Protocol):
    def create(self, **kwargs: object) -> object: ...


class _GeminiClient(Protocol):
    interactions: _InteractionsClient


class GeminiSpeechService(SpeechService):
    """Generate Gemini speech while preserving Manim Voiceover metadata."""

    def __init__(
        self,
        *,
        api_key: str,
        voice: str = DEFAULT_GEMINI_VOICE,
        model: str = DEFAULT_GEMINI_TTS_MODEL,
        language: str = "Português (Brasil)",
        video_mode: str = "presentation",
        prompt_template: str = "{transcript}",
        max_attempts: int = 3,
        retry_delay: float = 1.0,
        client: _GeminiClient | None = None,
        **kwargs: Any,
    ) -> None:
        super().__init__(transcription_model=None, **kwargs)
        if not api_key:
            raise ValueError("A chave da API do Google não foi informada para a narração.")
        self.client = client or cast(_GeminiClient, genai.Client(api_key=api_key))
        self.voice = voice or DEFAULT_GEMINI_VOICE
        self.model = model or DEFAULT_GEMINI_TTS_MODEL
        self.language = language or "Português (Brasil)"
        self.video_mode = video_mode
        self.prompt_template = prompt_template
        self.max_attempts = max_attempts
        self.retry_delay = retry_delay

    def generate_from_text(
        self,
        text: str,
        cache_dir: PathLike | None = None,
        path: PathLike | None = None,
        **kwargs: object,
    ) -> VoiceoverData:
        """Synthesize one voiceover block and save it as a WAV file."""
        _ = kwargs
        resolved_cache_dir = Path(cache_dir or self.cache_dir)
        resolved_cache_dir.mkdir(parents=True, exist_ok=True)
        transcript = remove_bookmarks(text)
        prompt = render_prompt_template(
            self.prompt_template,
            {
                "transcript": transcript,
                "language": self.language,
                "video_mode": self.video_mode,
            },
        )
        input_data = self._input_data(transcript, prompt)
        if cached := self.get_cached_result(input_data, resolved_cache_dir):
            return cached

        audio_path = (
            self.get_audio_basename(input_data) + ".wav" if path is None else path_to_string(path)
        )
        pcm_audio = self._generate_pcm(prompt, call_index=next_call_index())
        self._write_wave(resolved_cache_dir / audio_path, pcm_audio)
        return {
            "input_text": text,
            "input_data": input_data,
            "original_audio": audio_path,
        }

    def _generate_pcm(self, prompt: str, *, call_index: int) -> bytes:
        last_error: Exception | None = None
        for attempt in range(1, self.max_attempts + 1):
            try:
                interaction = self.client.interactions.create(
                    model=self.model,
                    input=prompt,
                    response_format={"type": "audio"},
                    generation_config={"speech_config": [{"voice": self.voice}]},
                )
                audio = self._decode_audio(interaction)
                usage = getattr(interaction, "usage", None)
                input_tokens = _usage_integer(usage, "total_input_tokens")
                audio_tokens = _usage_integer(usage, "total_output_tokens")
                emit_usage_event(
                    AIUsageEvent(
                        provider="Google",
                        model=self.model,
                        status="completed",
                        attempt_number=attempt,
                        call_index=call_index,
                        input_tokens=input_tokens,
                        input_characters=len(prompt),
                        audio_output_tokens=audio_tokens,
                        audio_seconds=len(audio) / (_SAMPLE_RATE * _SAMPLE_WIDTH * _CHANNELS),
                        metadata_available=usage is not None,
                        usage_source="gemini_interaction",
                    )
                )
                return audio
            except Exception as exc:
                transient = self._is_transient(exc)
                details = describe_tts_error(exc, transient=transient)
                emit_usage_event(
                    AIUsageEvent(
                        provider="Google",
                        model=self.model,
                        status="failed",
                        attempt_number=attempt,
                        call_index=call_index,
                        input_characters=len(prompt),
                        usage_source="gemini_interaction",
                        error_type=details.exception_type,
                        error_code=details.code,
                        error_status=details.status,
                        error_message=details.message,
                        error_transient=details.transient,
                    )
                )
                if not transient or attempt == self.max_attempts:
                    raise RuntimeError(self._error_message(details)) from exc
                last_error = exc
                time.sleep(self.retry_delay * (2 ** (attempt - 1)))
        details = describe_tts_error(last_error or RuntimeError("falha desconhecida"))
        raise RuntimeError(self._error_message(details))

    def _input_data(self, transcript: str, prompt: str) -> dict[str, JsonValue]:
        return {
            "input_text": transcript,
            "service": "gemini_tts",
            "config": {
                "model": self.model,
                "voice": self.voice,
                "language": self.language,
                "video_mode": self.video_mode,
                "prompt_sha256": sha256(prompt.encode("utf-8")).hexdigest(),
            },
        }

    @staticmethod
    def _decode_audio(interaction: object) -> bytes:
        output_audio = getattr(interaction, "output_audio", None)
        data = getattr(output_audio, "data", None)
        if isinstance(data, bytes):
            return data
        if isinstance(data, str) and data:
            return base64.b64decode(data, validate=True)
        raise ValueError("A resposta do Gemini TTS não contém áudio.")

    @staticmethod
    def _write_wave(path: Path, pcm_audio: bytes) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        with wave.open(str(path), "wb") as wave_file:
            wave_file.setnchannels(_CHANNELS)
            wave_file.setsampwidth(_SAMPLE_WIDTH)
            wave_file.setframerate(_SAMPLE_RATE)
            wave_file.writeframes(pcm_audio)

    @staticmethod
    def _is_transient(exc: Exception) -> bool:
        if isinstance(exc, errors.ServerError | httpx.TransportError):
            return True
        code = getattr(exc, "code", None) or getattr(
            getattr(exc, "response", None), "status_code", None
        )
        return code in {408, 429} or (isinstance(code, int) and code >= 500)

    def _error_message(self, details: TTSErrorDetails) -> str:
        return f"Gemini TTS ({self.model}, voz {self.voice}) falhou: {details.summary()}"


def _usage_integer(usage: object, attribute: str) -> int:
    value = getattr(usage, attribute, 0)
    return int(value) if isinstance(value, int | float) else 0
