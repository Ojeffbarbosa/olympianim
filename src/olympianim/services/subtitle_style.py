"""Validated, persistent styling for FFmpeg/libass subtitles."""

from __future__ import annotations

import hashlib
import re
from dataclasses import dataclass
from typing import Literal

from olympianim.database.repository import ProjectRepository

SubtitleStyleMode = Literal["presentation", "solution", "final"]
DEFAULT_SUBTITLE_TEXT_COLOR = "#FFFF00"
_HEX_COLOR = re.compile(r"#[0-9A-Fa-f]{6}")
_SETTING_PREFIX = "subtitle.text_color"


@dataclass(frozen=True)
class SubtitleStyle:
    """Safe subtitle style with conversion from RGB to ASS BGR notation."""

    text_color: str = DEFAULT_SUBTITLE_TEXT_COLOR

    def __post_init__(self) -> None:
        normalized = self.text_color.strip().upper()
        if _HEX_COLOR.fullmatch(normalized) is None:
            raise ValueError("A cor da legenda deve usar o formato hexadecimal #RRGGBB.")
        object.__setattr__(self, "text_color", normalized)

    @property
    def ass_primary_color(self) -> str:
        """Return opaque ASS colour in ``&HAABBGGRR`` notation."""
        red = self.text_color[1:3]
        green = self.text_color[3:5]
        blue = self.text_color[5:7]
        return f"&H00{blue}{green}{red}"

    @property
    def force_style(self) -> str:
        """Return the complete, validated libass style override."""
        return ",".join(
            (
                "FontName=Noto Sans",
                "FontSize=20",
                f"PrimaryColour={self.ass_primary_color}",
                "BackColour=&H80000000",
                "BorderStyle=3",
                "Outline=1",
                "Shadow=0",
                "Alignment=2",
                "MarginV=30",
            )
        )

    @property
    def fingerprint(self) -> str:
        """Identify every rendering-relevant style field."""
        return hashlib.sha256(self.force_style.encode("utf-8")).hexdigest()[:16]


class SubtitleStyleService:
    """Persist one subtitle preference for each project video mode."""

    def __init__(self, repository: ProjectRepository | None = None) -> None:
        self.repository = repository or ProjectRepository()

    def load(self, project_id: str, mode: SubtitleStyleMode) -> SubtitleStyle:
        """Load a style, falling back safely when a stored value is invalid."""
        raw = self.repository.get_setting(
            self._setting_key(project_id, mode),
            DEFAULT_SUBTITLE_TEXT_COLOR,
        )
        try:
            return SubtitleStyle(text_color=raw)
        except ValueError:
            return SubtitleStyle()

    def save(
        self,
        project_id: str,
        mode: SubtitleStyleMode,
        text_color: str,
    ) -> SubtitleStyle:
        """Validate and persist a subtitle colour."""
        style = SubtitleStyle(text_color=text_color)
        self.repository.set_setting(
            self._setting_key(project_id, mode),
            style.text_color,
        )
        return style

    @staticmethod
    def _setting_key(project_id: str, mode: SubtitleStyleMode) -> str:
        return f"{_SETTING_PREFIX}.{project_id}.{mode}"
