"""Provider-neutral LLM usage normalization and persistence."""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass
from datetime import date
from typing import Any, Literal

from langchain_core.callbacks import UsageMetadataCallbackHandler

from olympianim.database.models import AIUsageRecord
from olympianim.database.repository import ProjectRepository
from olympianim.schemas.render import AIUsageEvent
from olympianim.services.pricing import estimate_cost_usd


@dataclass(frozen=True)
class UsageContext:
    """Non-sensitive identity metadata for one logical LLM call."""

    project_id: str
    execution_id: str
    call_key: str
    agent_type: str
    stage: str


@dataclass(frozen=True)
class UsageFilters:
    """Optional filters applied by the consumption dashboard."""

    project_ids: frozenset[str] = frozenset()
    providers: frozenset[str] = frozenset()
    models: frozenset[str] = frozenset()
    agents: frozenset[str] = frozenset()
    stages: frozenset[str] = frozenset()
    modalities: frozenset[str] = frozenset()
    start_date: date | None = None
    end_date: date | None = None


@dataclass(frozen=True)
class UsageTotals:
    """Token and status totals for a group of attempts."""

    input_tokens: int = 0
    output_tokens: int = 0
    total_tokens: int = 0
    cache_read_tokens: int = 0
    cache_creation_tokens: int = 0
    reasoning_tokens: int = 0
    completed_calls: int = 0
    failed_calls: int = 0
    repeated_calls: int = 0
    missing_metadata_calls: int = 0
    input_characters: int = 0
    audio_output_tokens: int = 0
    audio_seconds: float = 0.0
    estimated_cost_usd: float = 0.0
    unknown_cost_calls: int = 0


class UsageService:
    """Create native callbacks and persist their normalized totals."""

    def __init__(self, repository: ProjectRepository | None = None) -> None:
        self.repository = repository or ProjectRepository()

    @staticmethod
    def callback() -> UsageMetadataCallbackHandler:
        """Return LangChain's native aggregate usage callback."""
        return UsageMetadataCallbackHandler()

    def record_attempt(
        self,
        context: UsageContext,
        callback: UsageMetadataCallbackHandler,
        *,
        provider: str,
        model: str,
        completed: bool,
        attempt_type: str,
        attempt_number: int,
    ) -> AIUsageRecord:
        """Normalize all model messages emitted by one provider attempt."""
        usage = _aggregate_usage(callback.usage_metadata)
        cost, pricing_known, pricing_revision = estimate_cost_usd(
            provider=provider,
            model=model,
            modality="text",
            input_tokens=usage["input_tokens"],
            output_tokens=usage["output_tokens"],
            cache_read_tokens=usage["cache_read_tokens"],
            repository=self.repository,
        )
        return self.repository.add_ai_usage(
            context.project_id,
            execution_id=context.execution_id,
            call_key=f"{context.call_key}:attempt-{attempt_number}",
            agent_type=context.agent_type,
            stage=context.stage,
            provider=provider,
            model=model,
            status="completed" if completed else "failed",
            attempt_type=attempt_type,
            input_tokens=usage["input_tokens"],
            output_tokens=usage["output_tokens"],
            total_tokens=usage["total_tokens"],
            cache_read_tokens=usage["cache_read_tokens"],
            cache_creation_tokens=usage["cache_creation_tokens"],
            reasoning_tokens=usage["reasoning_tokens"],
            modality="text",
            estimated_cost_usd=cost,
            pricing_known=pricing_known,
            usage_source=f"langchain:pricing-{pricing_revision or 'unknown'}",
            metadata_available=bool(callback.usage_metadata),
        )

    def record_speech_attempt(
        self,
        context: UsageContext,
        *,
        provider: str,
        model: str,
        completed: bool,
        attempt_number: int,
        input_tokens: int = 0,
        output_tokens: int = 0,
        input_characters: int = 0,
        audio_output_tokens: int = 0,
        audio_seconds: float = 0.0,
        metadata_available: bool = False,
        usage_source: str = "provider",
        error_type: str = "",
        error_code: str = "",
        error_status: str = "",
        error_message: str = "",
        error_transient: bool = False,
    ) -> AIUsageRecord:
        """Persist one real speech endpoint attempt using native billing units."""
        cost, pricing_known, pricing_revision = estimate_cost_usd(
            provider=provider,
            model=model,
            modality="speech",
            input_tokens=input_tokens,
            output_tokens=output_tokens,
            input_characters=input_characters,
            audio_output_tokens=audio_output_tokens,
            repository=self.repository,
        )
        if not completed:
            cost, pricing_known = 0.0, False
        return self.repository.add_ai_usage(
            context.project_id,
            execution_id=context.execution_id,
            call_key=f"{context.call_key}:attempt-{attempt_number}",
            agent_type=context.agent_type,
            stage=context.stage,
            provider=provider,
            model=model,
            status="completed" if completed else "failed",
            attempt_type="primary" if attempt_number == 1 else "fallback",
            input_tokens=input_tokens,
            output_tokens=output_tokens,
            total_tokens=input_tokens + output_tokens,
            modality="speech",
            input_characters=input_characters,
            audio_output_tokens=audio_output_tokens,
            audio_seconds=audio_seconds,
            estimated_cost_usd=cost,
            pricing_known=pricing_known,
            usage_source=f"{usage_source}:pricing-{pricing_revision or 'unknown'}",
            metadata_available=metadata_available,
            error_type=error_type,
            error_code=error_code,
            error_status=error_status,
            error_message=error_message,
            error_transient=error_transient,
        )

    def record_speech_events(
        self,
        events: Sequence[AIUsageEvent],
        *,
        project_id: str,
        execution_id: str,
        stage: str,
        render_key: str,
    ) -> list[AIUsageRecord]:
        """Persist all sanitized speech attempts returned by one render."""
        records: list[AIUsageRecord] = []
        for event in events:
            records.append(
                self.record_speech_attempt(
                    UsageContext(
                        project_id=project_id,
                        execution_id=execution_id,
                        call_key=f"{render_key}:block-{event.call_index}",
                        agent_type="voice",
                        stage=stage,
                    ),
                    provider=event.provider,
                    model=event.model,
                    completed=event.status == "completed",
                    attempt_number=event.attempt_number,
                    input_tokens=event.input_tokens,
                    output_tokens=event.output_tokens,
                    input_characters=event.input_characters,
                    audio_output_tokens=event.audio_output_tokens,
                    audio_seconds=event.audio_seconds,
                    metadata_available=event.metadata_available,
                    usage_source=event.usage_source,
                    error_type=event.error_type,
                    error_code=event.error_code,
                    error_status=event.error_status,
                    error_message=event.error_message,
                    error_transient=event.error_transient,
                )
            )
        return records

    def list_project_usage(self, project_id: str) -> list[AIUsageRecord]:
        """Return persisted usage for one project."""
        return self.repository.list_ai_usage(project_id)

    def list_all_usage(self) -> list[AIUsageRecord]:
        """Return usage across every project."""
        return self.repository.list_ai_usage()

    def filtered_project_usage(
        self, project_id: str | None, filters: UsageFilters
    ) -> list[AIUsageRecord]:
        """Return usage in the selected scope matching every filter."""
        return [
            record
            for record in self.repository.list_ai_usage(project_id)
            if _matches_filters(record, filters)
        ]

    @staticmethod
    def totals(records: list[AIUsageRecord]) -> UsageTotals:
        """Aggregate token and call counters without estimating missing data."""
        return UsageTotals(
            input_tokens=sum(record.input_tokens for record in records),
            output_tokens=sum(record.output_tokens for record in records),
            total_tokens=sum(record.total_tokens for record in records),
            cache_read_tokens=sum(record.cache_read_tokens for record in records),
            cache_creation_tokens=sum(record.cache_creation_tokens for record in records),
            reasoning_tokens=sum(record.reasoning_tokens for record in records),
            completed_calls=sum(record.status == "completed" for record in records),
            failed_calls=sum(record.status == "failed" for record in records),
            repeated_calls=sum(record.sequence > 1 for record in records),
            missing_metadata_calls=sum(not record.metadata_available for record in records),
            input_characters=sum(record.input_characters for record in records),
            audio_output_tokens=sum(record.audio_output_tokens for record in records),
            audio_seconds=sum(record.audio_seconds for record in records),
            estimated_cost_usd=sum(record.estimated_cost_usd for record in records),
            unknown_cost_calls=sum(not record.pricing_known for record in records),
        )

    @classmethod
    def grouped_totals(
        cls,
        records: list[AIUsageRecord],
        dimension: Literal[
            "project_id", "provider", "model", "agent_type", "stage", "modality", "day"
        ],
    ) -> dict[str, UsageTotals]:
        """Aggregate records by a dashboard dimension."""
        groups: dict[str, list[AIUsageRecord]] = {}
        for record in records:
            key = record.created_at[:10] if dimension == "day" else str(getattr(record, dimension))
            groups.setdefault(key, []).append(record)
        return {key: cls.totals(values) for key, values in sorted(groups.items())}


def _aggregate_usage(metadata: dict[str, Any]) -> dict[str, int]:
    totals = {
        "input_tokens": 0,
        "output_tokens": 0,
        "total_tokens": 0,
        "cache_read_tokens": 0,
        "cache_creation_tokens": 0,
        "reasoning_tokens": 0,
    }
    for usage in metadata.values():
        if not isinstance(usage, dict):
            continue
        totals["input_tokens"] += _integer(usage.get("input_tokens"))
        totals["output_tokens"] += _integer(usage.get("output_tokens"))
        totals["total_tokens"] += _integer(usage.get("total_tokens"))
        input_details = usage.get("input_token_details", {})
        if isinstance(input_details, dict):
            totals["cache_read_tokens"] += _integer(input_details.get("cache_read"))
            totals["cache_creation_tokens"] += _integer(input_details.get("cache_creation"))
        output_details = usage.get("output_token_details", {})
        if isinstance(output_details, dict):
            totals["reasoning_tokens"] += _integer(output_details.get("reasoning"))
    return totals


def _integer(value: object) -> int:
    return int(value) if isinstance(value, int | float) else 0


def _matches_filters(record: AIUsageRecord, filters: UsageFilters) -> bool:
    created_date = date.fromisoformat(record.created_at[:10])
    return (
        (not filters.project_ids or record.project_id in filters.project_ids)
        and (not filters.providers or record.provider in filters.providers)
        and (not filters.models or record.model in filters.models)
        and (not filters.agents or record.agent_type in filters.agents)
        and (not filters.stages or record.stage in filters.stages)
        and (not filters.modalities or record.modality in filters.modalities)
        and (filters.start_date is None or created_date >= filters.start_date)
        and (filters.end_date is None or created_date <= filters.end_date)
    )
