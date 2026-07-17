"""Persistent non-sensitive model preference for the Manim assistant."""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass

from olympianim.database.repository import ProjectRepository
from olympianim.services.model_catalog import ModelCatalogService

_SETTING_KEY = "code_assistant_preferences"


@dataclass(frozen=True)
class CodeAssistantPreferences:
    """Provider and model shared by conversation and edit modes."""

    provider: str = ""
    model: str = ""


class CodeAssistantPreferencesService:
    """Store and resolve the global default model for the code assistant."""

    def __init__(self, repository: ProjectRepository | None = None) -> None:
        self.repository = repository or ProjectRepository()
        self.catalog = ModelCatalogService(self.repository)

    def load(self) -> CodeAssistantPreferences | None:
        raw_value = self.repository.get_setting(_SETTING_KEY)
        if not raw_value:
            return None
        try:
            payload = json.loads(raw_value)
        except json.JSONDecodeError:
            return None
        if not isinstance(payload, dict):
            return None
        provider = payload.get("provider", "")
        model = payload.get("model", "")
        if not isinstance(provider, str) or not isinstance(model, str):
            return None
        return CodeAssistantPreferences(provider=provider, model=model)

    def save(self, preferences: CodeAssistantPreferences) -> None:
        active_models = self.catalog.model_ids(preferences.provider, "text")
        if preferences.model not in active_models:
            raise ValueError("Selecione um modelo de IA ativo para o assistente.")
        self.repository.set_setting(
            _SETTING_KEY,
            json.dumps(asdict(preferences), ensure_ascii=True, sort_keys=True),
        )

    def resolve(self) -> CodeAssistantPreferences:
        """Return the saved active model or the current catalog default."""
        providers = self.catalog.providers("text")
        if not providers:
            return CodeAssistantPreferences()
        saved = self.load()
        if saved is not None and saved.provider in providers:
            models = self.catalog.model_ids(saved.provider, "text")
            if saved.model in models:
                return saved
        provider = providers[0]
        return CodeAssistantPreferences(
            provider=provider,
            model=self.catalog.default_model_id(provider, "text"),
        )

    def resolve_for_provider(self, provider: str) -> CodeAssistantPreferences:
        """Return the saved model or the catalog default for one provider."""
        models = self.catalog.model_ids(provider, "text")
        if not models:
            return CodeAssistantPreferences()
        saved = self.load()
        if saved is not None and saved.provider == provider and saved.model in models:
            return saved
        return CodeAssistantPreferences(
            provider=provider,
            model=self.catalog.default_model_id(provider, "text"),
        )
