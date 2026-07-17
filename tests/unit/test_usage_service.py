"""Tests for provider-neutral LLM usage tracking."""

from __future__ import annotations

from datetime import date
from pathlib import Path

import pytest
from langchain_core.messages import AIMessage
from langchain_core.outputs import ChatGeneration, LLMResult

from olympianim.database.models import ProjectCreate
from olympianim.database.repository import ProjectRepository
from olympianim.schemas.render import AIUsageEvent
from olympianim.services.usage_service import UsageContext, UsageFilters, UsageService


def _service(tmp_path: Path) -> tuple[UsageService, ProjectRepository]:
    repository = ProjectRepository(tmp_path / "olympianim.db")
    repository.create_project(
        ProjectCreate(title="Projeto", problem_statement="Problema"),
        project_id="project-id",
    )
    return UsageService(repository), repository


def _callback(
    service: UsageService,
    *,
    model: str,
    input_tokens: int,
    output_tokens: int,
    cache_read: int = 0,
    cache_creation: int = 0,
    reasoning: int = 0,
):
    callback = service.callback()
    message = AIMessage(
        content="ok",
        response_metadata={"model_name": model},
        usage_metadata={
            "input_tokens": input_tokens,
            "output_tokens": output_tokens,
            "total_tokens": input_tokens + output_tokens,
            "input_token_details": {
                "cache_read": cache_read,
                "cache_creation": cache_creation,
            },
            "output_token_details": {"reasoning": reasoning},
        },
    )
    callback.on_llm_end(LLMResult(generations=[[ChatGeneration(message=message)]]))
    return callback


def test_records_native_usage_metadata_for_each_provider(tmp_path: Path) -> None:
    service, _ = _service(tmp_path)
    context = UsageContext(
        project_id="project-id",
        execution_id="job-id",
        call_key="job-id:planner:presentation:1",
        agent_type="planner",
        stage="presentation",
    )

    for attempt, (provider, model) in enumerate(
        (
            ("OpenAI", "gpt-5.4-mini"),
            ("Google", "gemini-3.5-flash"),
            ("Anthropic", "claude-sonnet-5"),
        ),
        start=1,
    ):
        service.record_attempt(
            context,
            _callback(
                service,
                model=model,
                input_tokens=10 * attempt,
                output_tokens=5 * attempt,
                cache_read=attempt,
                cache_creation=attempt + 1,
                reasoning=attempt + 2,
            ),
            provider=provider,
            model=model,
            completed=True,
            attempt_type="primary" if attempt == 1 else "fallback",
            attempt_number=attempt,
        )

    records = service.list_project_usage("project-id")
    totals = service.totals(records)
    assert {record.provider for record in records} == {"OpenAI", "Google", "Anthropic"}
    assert totals.input_tokens == 60
    assert totals.output_tokens == 30
    assert totals.total_tokens == 90
    assert totals.cache_read_tokens == 6
    assert totals.cache_creation_tokens == 9
    assert totals.reasoning_tokens == 12


def test_failed_call_without_metadata_is_not_estimated(tmp_path: Path) -> None:
    service, _ = _service(tmp_path)
    context = UsageContext(
        project_id="project-id",
        execution_id="job-id",
        call_key="failure",
        agent_type="builder",
        stage="solution",
    )

    record = service.record_attempt(
        context,
        service.callback(),
        provider="OpenAI",
        model="missing-model",
        completed=False,
        attempt_type="primary",
        attempt_number=1,
    )

    assert record.status == "failed"
    assert record.metadata_available is False
    assert record.total_tokens == 0
    assert service.totals([record]).missing_metadata_calls == 1


def test_records_speech_native_units_and_official_cost(tmp_path: Path) -> None:
    service, _ = _service(tmp_path)
    record = service.record_speech_attempt(
        UsageContext(
            project_id="project-id",
            execution_id="render-id",
            call_key="voice:presentation:block-1",
            agent_type="voice",
            stage="presentation",
        ),
        provider="OpenAI",
        model="tts-1",
        completed=True,
        attempt_number=1,
        input_characters=1_000,
        audio_seconds=12.5,
        metadata_available=True,
        usage_source="openai_characters",
    )

    assert record.modality == "speech"
    assert record.estimated_cost_usd == pytest.approx(0.015)
    assert record.pricing_known is True
    totals = service.totals([record])
    assert totals.input_characters == 1_000
    assert totals.audio_seconds == pytest.approx(12.5)


def test_persists_structured_speech_error_details(tmp_path: Path) -> None:
    service, repository = _service(tmp_path)
    service.record_speech_events(
        [
            AIUsageEvent(
                provider="Google",
                model="gemini-tts",
                status="failed",
                attempt_number=1,
                call_index=2,
                error_type="ClientError",
                error_code="429",
                error_status="RESOURCE_EXHAUSTED",
                error_message="Quota temporariamente excedida.",
                error_transient=True,
            )
        ],
        project_id="project-id",
        execution_id="render-id",
        stage="presentation",
        render_key="voice-render",
    )

    record = repository.list_ai_usage("project-id")[0]
    assert record.error_type == "ClientError"
    assert record.error_code == "429"
    assert record.error_status == "RESOURCE_EXHAUSTED"
    assert record.error_message == "Quota temporariamente excedida."
    assert record.error_transient is True


def test_call_key_prevents_duplicates_during_resume(tmp_path: Path) -> None:
    service, _ = _service(tmp_path)
    context = UsageContext(
        project_id="project-id",
        execution_id="job-id",
        call_key="stable-call",
        agent_type="debugger",
        stage="presentation",
    )
    callback = _callback(service, model="gpt-5.4-mini", input_tokens=10, output_tokens=2)

    for _ in range(2):
        service.record_attempt(
            context,
            callback,
            provider="OpenAI",
            model="gpt-5.4-mini",
            completed=True,
            attempt_type="primary",
            attempt_number=1,
        )

    assert len(service.list_project_usage("project-id")) == 1


def test_filters_and_grouping_keep_provider_and_stage_separate(tmp_path: Path) -> None:
    service, repository = _service(tmp_path)
    for index, provider in enumerate(("OpenAI", "Google"), start=1):
        repository.add_ai_usage(
            "project-id",
            execution_id=f"job-{index}",
            call_key=f"call-{index}",
            agent_type="planner",
            stage="presentation" if index == 1 else "solution",
            provider=provider,
            model=f"model-{index}",
            status="completed",
            attempt_type="primary",
            input_tokens=index * 10,
            output_tokens=index,
            total_tokens=index * 11,
            metadata_available=True,
        )

    records = service.filtered_project_usage(
        "project-id",
        UsageFilters(
            providers=frozenset({"Google"}),
            stages=frozenset({"solution"}),
            start_date=date(2000, 1, 1),
            end_date=date(2100, 1, 1),
        ),
    )

    assert len(records) == 1
    assert records[0].provider == "Google"
    assert service.grouped_totals(records, "provider")["Google"].total_tokens == 22


def test_general_scope_combines_and_filters_projects(tmp_path: Path) -> None:
    service, repository = _service(tmp_path)
    repository.create_project(
        ProjectCreate(title="Segundo", problem_statement="Outro"),
        project_id="second-project",
    )
    for project_id, tokens in (("project-id", 10), ("second-project", 20)):
        repository.add_ai_usage(
            project_id,
            execution_id=f"job-{project_id}",
            call_key=f"call-{project_id}",
            agent_type="planner",
            stage="presentation",
            provider="OpenAI",
            model="gpt-test",
            status="completed",
            attempt_type="primary",
            input_tokens=tokens,
            output_tokens=0,
            total_tokens=tokens,
            metadata_available=True,
        )

    all_records = service.list_all_usage()
    selected = service.filtered_project_usage(
        None,
        UsageFilters(project_ids=frozenset({"second-project"})),
    )

    assert service.totals(all_records).total_tokens == 30
    assert len(selected) == 1
    assert selected[0].project_id == "second-project"
    assert service.grouped_totals(all_records, "project_id")["project-id"].total_tokens == 10
