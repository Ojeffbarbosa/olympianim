"""Cost estimation backed by the configurable model catalog."""

from __future__ import annotations

from olympianim.database.repository import ProjectRepository
from olympianim.services.model_catalog import ModelCatalogService


def estimate_cost_usd(
    *,
    provider: str,
    model: str,
    modality: str,
    input_tokens: int = 0,
    output_tokens: int = 0,
    cache_read_tokens: int = 0,
    input_characters: int = 0,
    audio_output_tokens: int = 0,
    repository: ProjectRepository | None = None,
) -> tuple[float, bool, str]:
    """Return cost, availability and the catalog revision used."""
    record = ModelCatalogService(repository).find(provider, modality, model)
    if record is None:
        return 0.0, False, ""
    applicable_rates = (
        record.input_token_rate,
        record.cached_input_token_rate,
        record.output_token_rate,
        record.input_character_rate,
        record.audio_output_token_rate,
    )
    if not any(applicable_rates):
        return 0.0, False, f"{record.id}:r{record.revision}"
    uncached_input = max(input_tokens - cache_read_tokens, 0)
    cost = (
        uncached_input * record.input_token_rate
        + cache_read_tokens * record.cached_input_token_rate
        + output_tokens * record.output_token_rate
        + input_characters * record.input_character_rate
        + audio_output_tokens * record.audio_output_token_rate
    ) / 1_000_000
    return cost, True, f"{record.id}:r{record.revision}"
