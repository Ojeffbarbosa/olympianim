"""Tests for persistent non-sensitive generation preferences."""

from __future__ import annotations

import json

from olympianim.database.repository import ProjectRepository
from olympianim.services.generation_preferences import (
    GenerationPreferences,
    GenerationPreferencesService,
)


def test_preferences_round_trip_without_credentials(tmp_path) -> None:
    repository = ProjectRepository(tmp_path / "preferences.sqlite3")
    service = GenerationPreferencesService(repository)
    preferences = GenerationPreferences(
        llm_provider="Google",
        llm_model="gemini-3.5-flash",
        voiceover_enabled=True,
        voice_provider="Google",
        voice_model="gemini-3.1-flash-tts-preview",
        voice="Kore",
        voice_language="Português (Brasil)",
        voice_speed=1.2,
        reuse_llm_api_key=True,
        color_palette_id="builtin:manim-dark",
    )

    service.save(preferences)

    stored = repository.get_setting("generation_preferences")
    assert service.load() == preferences
    payload = json.loads(stored)
    assert {"llm_api_key", "voice_api_key"}.isdisjoint(payload)
    assert payload["llm_model"] == "gemini-3.5-flash"
    assert payload["color_palette_id"] == "builtin:manim-dark"


def test_malformed_preferences_are_ignored(tmp_path) -> None:
    repository = ProjectRepository(tmp_path / "preferences.sqlite3")
    repository.set_setting("generation_preferences", "not-json")

    assert GenerationPreferencesService(repository).load() is None


def test_out_of_range_voice_speed_uses_safe_default(tmp_path) -> None:
    repository = ProjectRepository(tmp_path / "preferences.sqlite3")
    repository.set_setting(
        "generation_preferences",
        json.dumps({"voice_speed": 9, "voiceover_enabled": True}),
    )

    preferences = GenerationPreferencesService(repository).load()
    assert preferences is not None
    assert preferences.voice_speed == 1.0
    assert preferences.voiceover_enabled is True
