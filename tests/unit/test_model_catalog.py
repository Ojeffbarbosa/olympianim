"""Tests for configurable provider models and prices."""

from __future__ import annotations

from dataclasses import replace
from pathlib import Path

import pytest

from olympianim.database.models import ProjectCreate
from olympianim.database.repository import ProjectRepository
from olympianim.services.model_catalog import CatalogModelInput, ModelCatalogService
from olympianim.services.pricing import estimate_cost_usd
from olympianim.services.usage_service import UsageContext, UsageService


def _service(tmp_path: Path) -> tuple[ModelCatalogService, ProjectRepository]:
    repository = ProjectRepository(tmp_path / "olympianim.db")
    return ModelCatalogService(repository), repository


def test_seeds_defaults_idempotently_with_prices(tmp_path: Path) -> None:
    service, _ = _service(tmp_path)

    service.ensure_defaults()
    service.ensure_defaults()

    assert len(service.list_models(modality="text")) == 16
    assert len(service.list_models(modality="speech")) == 4
    assert service.default_model_id("OpenAI", "text") == "gpt-5.6-sol"
    priced = service.find("OpenAI", "text", "gpt-5.4")
    assert priced is not None
    assert priced.input_token_rate == 2.5
    assert priced.cached_input_token_rate == 0.25
    assert priced.output_token_rate == 15.0
    assert service.model_ids("Anthropic", "text") == (
        "claude-sonnet-5",
        "claude-opus-4-8",
        "claude-haiku-4-5",
        "claude-fable-5",
    )


def test_adds_custom_model_and_formats_its_price(tmp_path: Path) -> None:
    service, _ = _service(tmp_path)
    created = service.save(
        CatalogModelInput(
            provider="Anthropic",
            modality="text",
            model_id="claude-custom",
            display_name="Claude personalizado",
            input_token_rate=1.25,
            cached_input_token_rate=0.25,
            output_token_rate=5.0,
        )
    )

    assert created.is_builtin is False
    assert "claude-custom" in service.model_ids("Anthropic", "text")
    assert service.label("Anthropic", "text", "claude-custom").startswith(
        "Claude personalizado · entrada $1.25"
    )


def test_default_is_unique_and_deactivation_promotes_next(tmp_path: Path) -> None:
    service, _ = _service(tmp_path)
    models = service.list_models(provider="Google", modality="text")
    second = models[1]
    service.save(
        CatalogModelInput(
            provider=second.provider,
            modality=second.modality,
            model_id=second.model_id,
            enabled=True,
            is_default=True,
            sort_order=second.sort_order,
            input_token_rate=second.input_token_rate,
            cached_input_token_rate=second.cached_input_token_rate,
            output_token_rate=second.output_token_rate,
        ),
        record_id=second.id,
    )

    assert service.default_model_id("Google", "text") == second.model_id
    service.deactivate(second.id)
    assert service.default_model_id("Google", "text") == models[0].model_id


def test_cannot_deactivate_last_text_model(tmp_path: Path) -> None:
    service, repository = _service(tmp_path)
    models = service.list_models(modality="text", enabled_only=True)
    for model in models[1:]:
        repository.save_catalog_model(replace(model, enabled=False, is_default=False))

    with pytest.raises(ValueError, match="ao menos um modelo"):
        service.deactivate(models[0].id)


def test_restores_edited_builtin(tmp_path: Path) -> None:
    service, repository = _service(tmp_path)
    original = service.find("OpenAI", "speech", "tts-1")
    assert original is not None
    repository.save_catalog_model(
        replace(original, model_id="tts-edited", input_character_rate=99.0)
    )

    restored = service.restore_builtin(original.id)

    assert restored.model_id == "tts-1"
    assert restored.input_character_rate == 15.0


def test_price_change_only_affects_future_usage(tmp_path: Path) -> None:
    service, repository = _service(tmp_path)
    project = repository.create_project(
        ProjectCreate(title="Projeto", problem_statement="Problema")
    )
    usage = UsageService(repository)
    context = UsageContext(project.id, "render", "voice:block-1", "voice", "presentation")
    first = usage.record_speech_attempt(
        context,
        provider="OpenAI",
        model="tts-1",
        completed=True,
        attempt_number=1,
        input_characters=1_000,
        metadata_available=True,
    )
    model = service.find("OpenAI", "speech", "tts-1")
    assert model is not None
    service.save(
        CatalogModelInput(
            provider=model.provider,
            modality=model.modality,
            model_id=model.model_id,
            enabled=True,
            is_default=True,
            input_character_rate=30.0,
        ),
        record_id=model.id,
    )
    second = usage.record_speech_attempt(
        replace(context, call_key="voice:block-2"),
        provider="OpenAI",
        model="tts-1",
        completed=True,
        attempt_number=1,
        input_characters=1_000,
        metadata_available=True,
    )

    assert first.estimated_cost_usd == pytest.approx(0.015)
    assert second.estimated_cost_usd == pytest.approx(0.03)
    assert ":r1" in first.usage_source
    assert ":r2" in second.usage_source
    assert usage.list_project_usage(project.id)[0].estimated_cost_usd == pytest.approx(0.015)


def test_cached_tokens_use_their_own_rate(tmp_path: Path) -> None:
    _, repository = _service(tmp_path)
    cost, known, _ = estimate_cost_usd(
        provider="OpenAI",
        model="gpt-5.4",
        modality="text",
        input_tokens=1_000,
        cache_read_tokens=400,
        output_tokens=100,
        repository=repository,
    )

    assert known
    assert cost == pytest.approx((600 * 2.5 + 400 * 0.25 + 100 * 15) / 1_000_000)
