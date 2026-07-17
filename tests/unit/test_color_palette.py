"""Tests for semantic color palette configuration."""

from __future__ import annotations

import json

import pytest

from olympianim.database.repository import ProjectRepository
from olympianim.services.color_palette import (
    ColorPaletteInput,
    ColorPaletteService,
)


def test_builtin_palettes_are_seeded_and_protected(tmp_path) -> None:
    service = ColorPaletteService(ProjectRepository(tmp_path / "palettes.sqlite3"))

    palettes = service.list_palettes(enabled_only=True)

    assert [palette.name for palette in palettes] == [
        "Manim escura",
        "Manim clara",
        "Okabe-Ito escura",
        "Okabe-Ito clara",
    ]
    assert all(palette.is_builtin for palette in palettes)
    with pytest.raises(ValueError, match="Duplique"):
        service.save(
            ColorPaletteInput(name="Alterada"),
            record_id=palettes[0].id,
        )


def test_custom_palette_round_trip_and_snapshot(tmp_path) -> None:
    service = ColorPaletteService(ProjectRepository(tmp_path / "palettes.sqlite3"))
    palette = service.save(ColorPaletteInput(name="Aula personalizada"))

    snapshot = service.snapshot(palette.id)
    data = json.loads(snapshot)

    assert data["name"] == "Aula personalizada"
    assert data["primary"] == "#58C4DD"
    context = service.prompt_context(snapshot)
    assert "fundo #000000" in context
    assert "texto principal #FFFFFF" in context
    assert ColorPaletteService.prompt_context("") == ""


def test_palette_rejects_insufficient_text_contrast(tmp_path) -> None:
    service = ColorPaletteService(ProjectRepository(tmp_path / "palettes.sqlite3"))

    with pytest.raises(ValueError, match=r"4\.5:1"):
        service.save(
            ColorPaletteInput(
                name="Sem contraste",
                background="#FFFFFF",
                primary_text="#FFFFFF",
                secondary_text="#FFFFFF",
                surface="#FFFFFF",
                stroke="#000000",
            )
        )


def test_duplicate_is_editable_and_has_unique_name(tmp_path) -> None:
    service = ColorPaletteService(ProjectRepository(tmp_path / "palettes.sqlite3"))
    source = service.list_palettes()[0]

    first = service.duplicate(source.id)
    second = service.duplicate(source.id)

    assert first.is_builtin is False
    assert first.name == "Manim escura - cópia"
    assert second.name == "Manim escura - cópia 2"
