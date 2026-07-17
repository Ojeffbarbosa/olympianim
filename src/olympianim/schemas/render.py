"""Manim code and render-result schemas."""

from __future__ import annotations

from typing import Literal

from pydantic import Field

from olympianim.config import DEFAULT_RENDER_QUALITY
from olympianim.schemas.base import OlympianimModel

VideoMode = Literal["presentation", "solution"]


class AIUsageEvent(OlympianimModel):
    """Sanitized provider attempt emitted by the isolated render process."""

    provider: str
    model: str
    status: Literal["completed", "failed"]
    attempt_number: int = Field(ge=1)
    call_index: int = Field(ge=1)
    input_tokens: int = Field(default=0, ge=0)
    output_tokens: int = Field(default=0, ge=0)
    input_characters: int = Field(default=0, ge=0)
    audio_output_tokens: int = Field(default=0, ge=0)
    audio_seconds: float = Field(default=0.0, ge=0)
    metadata_available: bool = False
    usage_source: str = "provider"
    error_type: str = ""
    error_code: str = ""
    error_status: str = ""
    error_message: str = ""
    error_transient: bool = False


class VoiceConfig(OlympianimModel):
    """Non-sensitive optional voiceover configuration for a render."""

    enabled: bool = False
    provider: str = ""
    model: str = ""
    voice: str = ""
    language: str = ""
    speed: float = Field(default=1.0, gt=0)


class ManimCodeResult(OlympianimModel):
    """Generated Manim code and narration for one video mode."""

    mode: VideoMode
    scene_name: str = Field(min_length=1)
    code: str = Field(min_length=1)
    code_path: str = ""
    notes: list[str] = Field(default_factory=list)


class RenderResult(OlympianimModel):
    """Result of a Manim render attempt."""

    mode: VideoMode
    success: bool
    return_code: int
    video_path: str = ""
    code_path: str = ""
    stdout: str = ""
    stderr: str = ""
    error_traceback: str = ""
    subtitle_path: str = ""
    raw_log_path: str = ""
    attempts: int = Field(ge=1)
    quality: str = DEFAULT_RENDER_QUALITY
    usage_events: list[AIUsageEvent] = Field(default_factory=list)

    @property
    def failed(self) -> bool:
        """Whether rendering failed."""
        return not self.success
