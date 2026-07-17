"""Tests for validated and persistent subtitle styling."""

from __future__ import annotations

from pathlib import Path

import pytest

from olympianim.database.repository import ProjectRepository
from olympianim.services.subtitle_style import (
    DEFAULT_SUBTITLE_TEXT_COLOR,
    SubtitleStyle,
    SubtitleStyleService,
)


def test_default_style_is_yellow_and_uses_ass_bgr_order() -> None:
    style = SubtitleStyle()

    assert style.text_color == "#FFFF00"
    assert style.text_color == DEFAULT_SUBTITLE_TEXT_COLOR
    assert style.ass_primary_color == "&H0000FFFF"
    assert "PrimaryColour=&H0000FFFF" in style.force_style


def test_style_normalizes_hex_and_rejects_unsafe_values() -> None:
    assert SubtitleStyle("  #a1b2c3 ").text_color == "#A1B2C3"

    with pytest.raises(ValueError, match="#RRGGBB"):
        SubtitleStyle("yellow,FontSize=100")


def test_preferences_are_persistent_and_scoped_by_video(tmp_path: Path) -> None:
    repository = ProjectRepository(tmp_path / "olympianim.db")
    service = SubtitleStyleService(repository)

    saved = service.save("project-1", "solution", "#123456")
    reopened = SubtitleStyleService(repository)

    assert saved.text_color == "#123456"
    assert reopened.load("project-1", "solution") == saved
    assert reopened.load("project-1", "presentation") == SubtitleStyle()


def test_invalid_stored_preference_falls_back_to_default(tmp_path: Path) -> None:
    repository = ProjectRepository(tmp_path / "olympianim.db")
    repository.set_setting("subtitle.text_color.project-1.final", "invalid")

    assert SubtitleStyleService(repository).load("project-1", "final") == SubtitleStyle()
