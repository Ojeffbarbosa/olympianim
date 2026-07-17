"""Tests for the global non-sensitive Manim assistant model preference."""

from __future__ import annotations

import json

import pytest

from olympianim.database.repository import ProjectRepository
from olympianim.services.code_assistant_preferences import (
    CodeAssistantPreferences,
    CodeAssistantPreferencesService,
)
from olympianim.services.model_catalog import ModelCatalogService


def test_preferences_round_trip_without_credentials(tmp_path) -> None:
    repository = ProjectRepository(tmp_path / "assistant.sqlite3")
    service = CodeAssistantPreferencesService(repository)
    service.catalog.ensure_defaults()

    service.save(CodeAssistantPreferences("Google", "gemini-3.5-flash"))

    assert service.load() == CodeAssistantPreferences("Google", "gemini-3.5-flash")
    payload = json.loads(repository.get_setting("code_assistant_preferences"))
    assert payload == {"model": "gemini-3.5-flash", "provider": "Google"}


def test_resolve_falls_back_when_saved_model_is_disabled(tmp_path) -> None:
    repository = ProjectRepository(tmp_path / "assistant.sqlite3")
    service = CodeAssistantPreferencesService(repository)
    service.catalog.ensure_defaults()
    service.save(CodeAssistantPreferences("Google", "gemini-3.5-flash"))
    record = service.catalog.find("Google", "text", "gemini-3.5-flash")
    assert record is not None
    service.catalog.deactivate(record.id)

    resolved = service.resolve()

    assert resolved != CodeAssistantPreferences("Google", "gemini-3.5-flash")
    assert resolved.model in service.catalog.model_ids(resolved.provider, "text")


def test_save_rejects_inactive_or_unknown_model(tmp_path) -> None:
    repository = ProjectRepository(tmp_path / "assistant.sqlite3")
    service = CodeAssistantPreferencesService(repository)

    with pytest.raises(ValueError, match="modelo de IA ativo"):
        service.save(CodeAssistantPreferences("OpenAI", "modelo-inexistente"))


def test_resolve_uses_catalog_default_without_saved_preferences(tmp_path) -> None:
    repository = ProjectRepository(tmp_path / "assistant.sqlite3")
    service = CodeAssistantPreferencesService(repository)
    catalog = ModelCatalogService(repository)

    resolved = service.resolve()

    assert resolved.model == catalog.default_model_id(resolved.provider, "text")


def test_resolve_for_provider_never_uses_another_providers_model(tmp_path) -> None:
    repository = ProjectRepository(tmp_path / "assistant.sqlite3")
    service = CodeAssistantPreferencesService(repository)
    service.catalog.ensure_defaults()
    service.save(CodeAssistantPreferences("OpenAI", "gpt-5.4-mini"))

    resolved = service.resolve_for_provider("Google")

    assert resolved.provider == "Google"
    assert resolved.model == service.catalog.default_model_id("Google", "text")


def test_resolve_for_provider_preserves_saved_model_for_same_provider(tmp_path) -> None:
    repository = ProjectRepository(tmp_path / "assistant.sqlite3")
    service = CodeAssistantPreferencesService(repository)
    service.catalog.ensure_defaults()
    saved = CodeAssistantPreferences("Google", "gemini-3.5-flash")
    service.save(saved)

    assert service.resolve_for_provider("Google") == saved
