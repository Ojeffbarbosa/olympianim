"""Configurable provider-model catalog shared by UI and usage accounting."""

from __future__ import annotations

import sqlite3
from dataclasses import dataclass, replace

from olympianim.database.models import ModelCatalogRecord
from olympianim.database.repository import ProjectRepository, new_id, utc_now

SUPPORTED_PROVIDERS = ("OpenAI", "Google", "Anthropic")
SUPPORTED_MODALITIES = ("text", "speech")
_DEFAULTS_VERSION = "4"


@dataclass(frozen=True)
class CatalogModelInput:
    """Editable values accepted by the model-catalog service."""

    provider: str
    modality: str
    model_id: str
    display_name: str = ""
    enabled: bool = True
    is_default: bool = False
    sort_order: int = 0
    input_token_rate: float = 0.0
    cached_input_token_rate: float = 0.0
    output_token_rate: float = 0.0
    input_character_rate: float = 0.0
    audio_output_token_rate: float = 0.0


@dataclass(frozen=True)
class _BuiltinModel:
    provider: str
    modality: str
    model_id: str
    sort_order: int
    input_token_rate: float = 0.0
    cached_input_token_rate: float = 0.0
    output_token_rate: float = 0.0
    input_character_rate: float = 0.0
    audio_output_token_rate: float = 0.0

    @property
    def stable_id(self) -> str:
        return f"builtin:{self.provider}:{self.modality}:{self.model_id}"


_BUILTINS = (
    _BuiltinModel("OpenAI", "text", "gpt-5.6-sol", 0, 5.00, 0.50, 30.00),
    _BuiltinModel("OpenAI", "text", "gpt-5.6-terra", 1, 2.50, 0.25, 15.00),
    _BuiltinModel("OpenAI", "text", "gpt-5.6-luna", 2, 1.00, 0.10, 6.00),
    _BuiltinModel("OpenAI", "text", "gpt-5.5", 3, 5.00, 0.50, 30.00),
    _BuiltinModel("OpenAI", "text", "gpt-5.5-pro", 4, 30.00, 0.00, 180.00),
    _BuiltinModel("OpenAI", "text", "gpt-5.4", 5, 2.50, 0.25, 15.00),
    _BuiltinModel("OpenAI", "text", "gpt-5.4-pro", 6, 30.00, 0.00, 180.00),
    _BuiltinModel("OpenAI", "text", "gpt-5.4-mini", 7, 0.75, 0.075, 4.50),
    _BuiltinModel("OpenAI", "text", "gpt-5.4-nano", 8, 0.20, 0.02, 1.25),
    _BuiltinModel("Google", "text", "gemini-3.1-pro-preview", 0, 2.00, 0.20, 12.00),
    _BuiltinModel("Google", "text", "gemini-3.5-flash", 1, 1.50, 0.15, 9.00),
    _BuiltinModel("Google", "text", "gemini-3.1-flash-lite", 2, 0.25, 0.025, 1.50),
    # Sonnet 5 uses its introductory price through 2026-08-31.
    _BuiltinModel("Anthropic", "text", "claude-sonnet-5", 0, 2.00, 0.20, 10.00),
    _BuiltinModel("Anthropic", "text", "claude-opus-4-8", 1, 5.00, 0.50, 25.00),
    _BuiltinModel("Anthropic", "text", "claude-haiku-4-5", 2, 1.00, 0.10, 5.00),
    _BuiltinModel("Anthropic", "text", "claude-fable-5", 3, 10.00, 1.00, 50.00),
    _BuiltinModel("OpenAI", "speech", "tts-1", 0, input_character_rate=15.00),
    _BuiltinModel("Google", "speech", "gemini-3.1-flash-tts-preview", 0, 1.00, 0, 0, 0, 20.00),
    _BuiltinModel("Google", "speech", "gemini-2.5-flash-preview-tts", 1, 0.50, 0, 0, 0, 10.00),
    _BuiltinModel("Google", "speech", "gemini-2.5-pro-preview-tts", 2, 1.00, 0, 0, 0, 20.00),
)
_BUILTINS_BY_ID = {item.stable_id: item for item in _BUILTINS}


class ModelCatalogService:
    """Own catalog seeding, validation, selection and price labels."""

    def __init__(self, repository: ProjectRepository | None = None) -> None:
        self.repository = repository or ProjectRepository()

    def ensure_defaults(self) -> None:
        """Seed missing built-ins without changing existing records."""
        existing_ids = {item.id for item in self.repository.list_catalog_models()}
        now = utc_now()
        for item in _BUILTINS:
            if item.stable_id in existing_ids:
                continue
            has_default = any(
                record.is_default
                for record in self.repository.list_catalog_models(
                    provider=item.provider, modality=item.modality
                )
            )
            self.repository.save_catalog_model(
                ModelCatalogRecord(
                    id=item.stable_id,
                    provider=item.provider,
                    modality=item.modality,
                    model_id=item.model_id,
                    display_name="",
                    enabled=True,
                    is_default=not has_default,
                    is_builtin=True,
                    revision=1,
                    sort_order=item.sort_order,
                    input_token_rate=item.input_token_rate,
                    cached_input_token_rate=item.cached_input_token_rate,
                    output_token_rate=item.output_token_rate,
                    input_character_rate=item.input_character_rate,
                    audio_output_token_rate=item.audio_output_token_rate,
                    created_at=now,
                    updated_at=now,
                )
            )
        if self.repository.get_setting("model_catalog_defaults_version") != _DEFAULTS_VERSION:
            self._apply_builtin_defaults_revision()
            self.repository.set_setting("model_catalog_defaults_version", _DEFAULTS_VERSION)

    def list_models(
        self,
        *,
        provider: str | None = None,
        modality: str | None = None,
        enabled_only: bool = False,
    ) -> list[ModelCatalogRecord]:
        self.ensure_defaults()
        return self.repository.list_catalog_models(
            provider=provider,
            modality=modality,
            enabled_only=enabled_only,
        )

    def providers(self, modality: str) -> tuple[str, ...]:
        """Return providers that currently expose at least one active model."""
        records = self.list_models(modality=modality, enabled_only=True)
        return tuple(
            provider
            for provider in SUPPORTED_PROVIDERS
            if any(item.provider == provider for item in records)
        )

    def model_ids(self, provider: str, modality: str) -> tuple[str, ...]:
        return tuple(
            item.model_id
            for item in self.list_models(provider=provider, modality=modality, enabled_only=True)
        )

    def default_model_id(self, provider: str, modality: str) -> str:
        models = self.list_models(provider=provider, modality=modality, enabled_only=True)
        default = next((item for item in models if item.is_default), None)
        if default is not None:
            return default.model_id
        return models[0].model_id if models else ""

    def find(self, provider: str, modality: str, model_id: str) -> ModelCatalogRecord | None:
        return next(
            (
                item
                for item in self.list_models(provider=provider, modality=modality)
                if item.model_id == model_id
            ),
            None,
        )

    def save(
        self,
        data: CatalogModelInput,
        *,
        record_id: str | None = None,
    ) -> ModelCatalogRecord:
        """Create or update one model after validating all editable values."""
        self._validate(data)
        current = self.repository.get_catalog_model(record_id) if record_id else None
        if (
            current is not None
            and current.modality == "text"
            and current.enabled
            and not data.enabled
            and len(self.list_models(modality="text", enabled_only=True)) == 1
        ):
            raise ValueError("Mantenha ao menos um modelo de IA ativo no aplicativo.")
        now = utc_now()
        record = ModelCatalogRecord(
            id=current.id if current else new_id(),
            provider=data.provider,
            modality=data.modality,
            model_id=data.model_id.strip(),
            display_name=data.display_name.strip(),
            enabled=data.enabled,
            is_default=data.is_default and data.enabled,
            is_builtin=current.is_builtin if current else False,
            revision=current.revision + 1 if current else 1,
            sort_order=data.sort_order,
            input_token_rate=data.input_token_rate,
            cached_input_token_rate=data.cached_input_token_rate,
            output_token_rate=data.output_token_rate,
            input_character_rate=data.input_character_rate,
            audio_output_token_rate=data.audio_output_token_rate,
            created_at=current.created_at if current else now,
            updated_at=now,
        )
        try:
            saved = self.repository.save_catalog_model(record)
        except sqlite3.IntegrityError as exc:
            raise ValueError("Já existe esse modelo para o provedor e modalidade.") from exc
        self._ensure_active_default(saved.provider, saved.modality)
        return self.repository.get_catalog_model(saved.id) or saved

    def deactivate(self, record_id: str) -> ModelCatalogRecord:
        current = self._required(record_id)
        if (
            current.modality == "text"
            and current.enabled
            and len(self.list_models(modality="text", enabled_only=True)) == 1
        ):
            raise ValueError("Mantenha ao menos um modelo de IA ativo no aplicativo.")
        saved = self.repository.save_catalog_model(
            replace(current, enabled=False, is_default=False, updated_at=utc_now())
        )
        self._ensure_active_default(saved.provider, saved.modality)
        return saved

    def restore_builtin(self, record_id: str) -> ModelCatalogRecord:
        current = self._required(record_id)
        default = _BUILTINS_BY_ID.get(current.id)
        if default is None:
            raise ValueError("Somente modelos fornecidos pelo app podem ser restaurados.")
        return self.save(
            CatalogModelInput(
                provider=default.provider,
                modality=default.modality,
                model_id=default.model_id,
                enabled=True,
                is_default=current.is_default,
                sort_order=default.sort_order,
                input_token_rate=default.input_token_rate,
                cached_input_token_rate=default.cached_input_token_rate,
                output_token_rate=default.output_token_rate,
                input_character_rate=default.input_character_rate,
                audio_output_token_rate=default.audio_output_token_rate,
            ),
            record_id=current.id,
        )

    @staticmethod
    def can_restore_builtin(record: ModelCatalogRecord) -> bool:
        """Return whether a record belongs to the current built-in catalog."""
        return record.id in _BUILTINS_BY_ID

    def label(self, provider: str, modality: str, model_id: str) -> str:
        record = self.find(provider, modality, model_id)
        if record is None:
            return model_id
        name = record.display_name or record.model_id
        return f"{name} · {self.price_label(record)}"

    @staticmethod
    def price_label(record: ModelCatalogRecord) -> str:
        """Return only the compact native-unit price summary."""
        if record.modality == "text" and (
            record.input_token_rate or record.cached_input_token_rate or record.output_token_rate
        ):
            return (
                f"entrada ${record.input_token_rate:g} / "
                f"cache ${record.cached_input_token_rate:g} / "
                f"saída ${record.output_token_rate:g} por 1M tokens"
            )
        if record.input_character_rate:
            return f"${record.input_character_rate:g} por 1M caracteres"
        if record.input_token_rate or record.audio_output_token_rate:
            return (
                f"entrada ${record.input_token_rate:g} / "
                f"áudio ${record.audio_output_token_rate:g} por 1M tokens"
            )
        return "preço não informado"

    def _ensure_active_default(self, provider: str, modality: str) -> None:
        models = self.repository.list_catalog_models(
            provider=provider, modality=modality, enabled_only=True
        )
        if not models or any(item.is_default for item in models):
            return
        first = models[0]
        self.repository.save_catalog_model(replace(first, is_default=True, updated_at=utc_now()))

    def _apply_builtin_defaults_revision(self) -> None:
        """Apply the current built-in catalog while preserving historical rows."""
        now = utc_now()
        for default in _BUILTINS:
            current = self.repository.get_catalog_model(default.stable_id)
            if current is None:
                continue
            self.repository.save_catalog_model(
                replace(
                    current,
                    input_token_rate=default.input_token_rate,
                    cached_input_token_rate=default.cached_input_token_rate,
                    output_token_rate=default.output_token_rate,
                    input_character_rate=default.input_character_rate,
                    audio_output_token_rate=default.audio_output_token_rate,
                    updated_at=now,
                )
            )

    def _required(self, record_id: str) -> ModelCatalogRecord:
        record = self.repository.get_catalog_model(record_id)
        if record is None:
            raise ValueError("Modelo não encontrado no catálogo.")
        return record

    @staticmethod
    def _validate(data: CatalogModelInput) -> None:
        if data.provider not in SUPPORTED_PROVIDERS:
            raise ValueError("Provedor de modelo inválido.")
        if data.modality not in SUPPORTED_MODALITIES:
            raise ValueError("Modalidade de modelo inválida.")
        if not data.model_id.strip() or any(char.isspace() for char in data.model_id):
            raise ValueError("O identificador da API é obrigatório e não pode conter espaços.")
        rates = (
            data.input_token_rate,
            data.cached_input_token_rate,
            data.output_token_rate,
            data.input_character_rate,
            data.audio_output_token_rate,
        )
        if any(rate < 0 for rate in rates):
            raise ValueError("Preços não podem ser negativos.")
