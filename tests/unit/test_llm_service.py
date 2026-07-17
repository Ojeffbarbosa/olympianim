"""Unit tests for the active LLM service."""

from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace
from typing import Any

import pytest
from langchain.agents.structured_output import (
    ProviderStrategy,
    StructuredOutputValidationError,
)
from langchain_core.messages import AIMessage
from langchain_core.outputs import ChatGeneration, LLMResult

from olympianim.database.models import ProjectCreate
from olympianim.database.repository import ProjectRepository
from olympianim.prompts.service import PromptService
from olympianim.providers.llm.base import LLMCallResult
from olympianim.schemas.llm import ManimCodeOutput
from olympianim.services.llm_service import LLMFallback, LLMImage, LLMRequest, LLMService
from olympianim.services.usage_service import UsageContext, UsageService
from olympianim.tools import search_manim_reference


class _FakeTextProvider:
    name = "Fake"

    def __init__(self, *, ok: bool, content: str = "ok", message: str = "") -> None:
        self.ok = ok
        self.content = content
        self.message = message or ("sucesso" if ok else "falha sk-secret")

    def invoke(
        self,
        prompt: str,
        *,
        model: str,
        system_prompt: str,
        temperature: float,
        max_tokens: int | None,
    ) -> LLMCallResult:
        _ = (system_prompt, temperature, max_tokens)
        return LLMCallResult(
            ok=self.ok,
            provider=self.name,
            model=model,
            content=f"{self.content}: {prompt}",
            message=self.message,
        )


def _prompt_service(tmp_path: Path) -> PromptService:
    return PromptService(repository=ProjectRepository(tmp_path / "olympianim.db"))


def test_llm_service_renders_editable_prompt_and_calls_provider(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    service = LLMService(prompt_service=_prompt_service(tmp_path))
    prompt = service.prompt_service.list_prompts("workflow_planner")[0]

    monkeypatch.setattr(
        "olympianim.services.llm_service.get_provider",
        lambda _provider, _api_key: _FakeTextProvider(ok=True),
    )

    result = service.call_text(
        LLMRequest(
            provider="OpenAI",
            model="gpt-5.5",
            api_key="sk-secret",
            prompt_id=prompt.prompt.id,
            prompt_values={"problem_statement": "Mostre que x=1."},
        )
    )

    assert result.result.ok
    assert "Mostre que x=1." in result.result.content
    assert result.prompt.prompt_id == prompt.prompt.id
    assert result.prompt.prompt_version == 1
    assert result.attempted == ("OpenAI:gpt-5.5",)


def test_prompt_context_is_sent_as_user_message_and_instructions_as_system(
    tmp_path: Path,
) -> None:
    service = LLMService(prompt_service=_prompt_service(tmp_path))

    rendered = service.render_prompt(
        LLMRequest(
            provider="OpenAI",
            model="test",
            api_key="secret",
            template_text=(
                "# Identidade\nSiga as regras.\n\n# Contexto\n"
                "<enunciado>{problem_statement}</enunciado>"
            ),
            prompt_values={"problem_statement": "Mostre que x=1."},
        )
    )

    assert rendered.system_prompt == "# Identidade\nSiga as regras."
    assert rendered.user_prompt == "<enunciado>Mostre que x=1.</enunciado>"
    assert "Mostre que x=1." not in rendered.system_prompt
    assert len(rendered.prompt_sha256) == 64


def test_custom_prompt_without_context_marker_keeps_legacy_routing(tmp_path: Path) -> None:
    service = LLMService(prompt_service=_prompt_service(tmp_path))

    rendered = service.render_prompt(
        LLMRequest(
            provider="OpenAI",
            model="test",
            api_key="secret",
            template_text="Prompt livre: {value}",
            prompt_values={"value": "conteúdo"},
        )
    )

    assert rendered.user_prompt == "Prompt livre: conteúdo"
    assert rendered.rendered_prompt == rendered.user_prompt


def test_llm_service_falls_back_when_primary_provider_fails(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    service = LLMService(prompt_service=_prompt_service(tmp_path))
    providers = iter([_FakeTextProvider(ok=False), _FakeTextProvider(ok=True, content="fallback")])

    monkeypatch.setattr(
        "olympianim.services.llm_service.get_provider",
        lambda _provider, _api_key: next(providers),
    )

    result = service.call_text(
        LLMRequest(
            provider="OpenAI",
            model="gpt-5.5",
            api_key="sk-secret",
            template_text="Diga {word}",
            prompt_values={"word": "ok"},
            fallbacks=(LLMFallback(provider="Anthropic", model="claude-sonnet-5", api_key="a"),),
        )
    )

    assert result.result.ok
    assert result.result.content.startswith("fallback")
    assert result.attempted == ("OpenAI:gpt-5.5", "Anthropic:claude-sonnet-5")


def test_llm_service_returns_legible_error_when_all_fallbacks_fail(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    service = LLMService(prompt_service=_prompt_service(tmp_path))

    monkeypatch.setattr(
        "olympianim.services.llm_service.get_provider",
        lambda _provider, _api_key: _FakeTextProvider(ok=False),
    )

    result = service.call_text(
        LLMRequest(
            provider="OpenAI",
            model="gpt-5.5",
            api_key="sk-secret",
            template_text="Diga ok",
            fallbacks=(
                LLMFallback(provider="Google", model="gemini-3.1-flash-lite", api_key="g"),
            ),
        )
    )

    assert not result.result.ok
    assert "Todos os provedores configurados falharam" in result.result.message
    assert "sk-secret" not in result.result.message
    assert result.attempted == ("OpenAI:gpt-5.5", "Google:gemini-3.1-flash-lite")


def test_llm_service_calls_native_agent_with_bounded_tool_loop(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    service = LLMService(prompt_service=_prompt_service(tmp_path))
    captured: dict[str, Any] = {}

    class _AgentProvider:
        def create_chat_model(self, **kwargs: object) -> object:
            captured["model_kwargs"] = kwargs
            return object()

    class _Agent:
        def invoke(
            self,
            state: dict[str, object],
            config: dict[str, object],
        ) -> dict[str, object]:
            captured["state"] = state
            captured["config"] = config
            return {"messages": [SimpleNamespace(content="from manim import *")]}

    def fake_create_agent(**kwargs: object) -> _Agent:
        captured["agent_kwargs"] = kwargs
        return _Agent()

    monkeypatch.setattr(
        "olympianim.services.llm_service.get_provider",
        lambda _provider, _api_key: _AgentProvider(),
    )
    monkeypatch.setattr("olympianim.services.llm_service.create_agent", fake_create_agent)

    result = service.call_agent(
        LLMRequest(
            provider="OpenAI",
            model="gpt-5.4-mini",
            api_key="sk-secret",
            template_text="Crie código para {subject}",
            prompt_values={"subject": "um círculo"},
            temperature=0.0,
        ),
        tools=(search_manim_reference,),
    )

    assert result.result.ok
    assert result.result.content == "from manim import *"
    agent_kwargs = captured["agent_kwargs"]
    assert agent_kwargs["tools"] == (search_manim_reference,)
    assert len(agent_kwargs["middleware"]) == 2
    tool_limit = agent_kwargs["middleware"][0]
    assert tool_limit.tool_name == search_manim_reference.name
    assert tool_limit.run_limit == 10
    assert tool_limit.exit_behavior == "error"
    assert agent_kwargs["checkpointer"] is False
    assert "response_format" not in agent_kwargs
    assert "um círculo" in str(captured["state"])
    assert captured["config"] == {"callbacks": []}


def test_llm_service_uses_structured_code_contract_when_requested(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    service = LLMService(prompt_service=_prompt_service(tmp_path))
    captured: dict[str, Any] = {}

    class _AgentProvider:
        def create_chat_model(self, **kwargs: object) -> object:
            captured["model_kwargs"] = kwargs
            return object()

        def supports_native_structured_output(self, _model: str) -> bool:
            return True

        def code_generation_max_tokens(self, _model: str) -> int:
            return 32_768

    class _Agent:
        def invoke(self, *_args: object, **_kwargs: object) -> dict[str, object]:
            return {"structured_response": {"code": "from manim import *"}}

    def fake_create_agent(**kwargs: object) -> _Agent:
        captured["agent_kwargs"] = kwargs
        return _Agent()

    monkeypatch.setattr(
        "olympianim.services.llm_service.get_provider",
        lambda _provider, _api_key: _AgentProvider(),
    )
    monkeypatch.setattr("olympianim.services.llm_service.create_agent", fake_create_agent)

    result = service.call_agent(
        LLMRequest(
            provider="Anthropic",
            model="claude-sonnet-5",
            api_key="secret",
            template_text="Crie uma animação.",
        ),
        tools=(search_manim_reference,),
        response_schema=ManimCodeOutput,
    )

    assert result.result.ok
    assert result.result.content == "from manim import *"
    strategy = captured["agent_kwargs"]["response_format"]
    assert isinstance(strategy, ProviderStrategy)
    assert strategy.schema is ManimCodeOutput
    assert strategy.schema_spec.strict is True
    assert captured["agent_kwargs"]["checkpointer"] is False
    assert captured["model_kwargs"]["max_tokens"] == 32_768
    tool_limit = captured["agent_kwargs"]["middleware"][0]
    assert tool_limit.tool_name == search_manim_reference.name


def test_llm_service_rejects_truncated_structured_response(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    service = LLMService(prompt_service=_prompt_service(tmp_path))

    class _AgentProvider:
        def create_chat_model(self, **_kwargs: object) -> object:
            return object()

        def supports_native_structured_output(self, _model: str) -> bool:
            return True

        def code_generation_max_tokens(self, _model: str) -> int:
            return 32_768

    class _Agent:
        def invoke(self, *_args: object, **_kwargs: object) -> dict[str, object]:
            return {
                "structured_response": {"code": "from manim import *"},
                "messages": [
                    AIMessage(
                        content='{"code":"from manim import *"}',
                        response_metadata={"stop_reason": "max_tokens"},
                    )
                ],
            }

    monkeypatch.setattr(
        "olympianim.services.llm_service.get_provider",
        lambda _provider, _api_key: _AgentProvider(),
    )
    monkeypatch.setattr(
        "olympianim.services.llm_service.create_agent",
        lambda **_kwargs: _Agent(),
    )

    result = service.call_agent(
        LLMRequest(
            provider="Anthropic",
            model="claude-sonnet-5",
            api_key="secret",
            template_text="Crie uma animação.",
        ),
        tools=(search_manim_reference,),
        response_schema=ManimCodeOutput,
    )

    assert not result.result.ok
    assert result.result.content == ""
    assert result.result.finish_reason == "max_tokens"
    assert "interrompida antes de terminar" in result.result.message


def test_llm_service_classifies_truncation_from_structured_output_error(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    service = LLMService(prompt_service=_prompt_service(tmp_path))

    class _AgentProvider:
        def create_chat_model(self, **_kwargs: object) -> object:
            return object()

        def supports_native_structured_output(self, _model: str) -> bool:
            return True

        def code_generation_max_tokens(self, _model: str) -> int:
            return 32_768

    class _Agent:
        def invoke(self, *_args: object, **_kwargs: object) -> dict[str, object]:
            message = AIMessage(
                content='{"code":"from manim import',
                response_metadata={"stop_reason": "max_tokens"},
            )
            raise StructuredOutputValidationError(
                "ManimCodeOutput",
                ValueError("JSON incompleto"),
                message,
            )

    monkeypatch.setattr(
        "olympianim.services.llm_service.get_provider",
        lambda _provider, _api_key: _AgentProvider(),
    )
    monkeypatch.setattr(
        "olympianim.services.llm_service.create_agent",
        lambda **_kwargs: _Agent(),
    )

    result = service.call_agent(
        LLMRequest(
            provider="Anthropic",
            model="claude-sonnet-5",
            api_key="secret",
            template_text="Crie uma animação.",
        ),
        tools=(search_manim_reference,),
        response_schema=ManimCodeOutput,
    )

    assert not result.result.ok
    assert result.result.finish_reason == "max_tokens"
    assert "JSON incompleto" not in result.result.message
    assert "conteúdo incompleto não foi armazenado" in result.result.message


def test_llm_service_rejects_missing_structured_agent_response(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    service = LLMService(prompt_service=_prompt_service(tmp_path))

    class _AgentProvider:
        def create_chat_model(self, **_kwargs: object) -> object:
            return object()

    class _Agent:
        def invoke(self, *_args: object, **_kwargs: object) -> dict[str, object]:
            return {"messages": [SimpleNamespace(content="from manim import *")]}

    monkeypatch.setattr(
        "olympianim.services.llm_service.get_provider",
        lambda _provider, _api_key: _AgentProvider(),
    )
    monkeypatch.setattr(
        "olympianim.services.llm_service.create_agent",
        lambda **_kwargs: _Agent(),
    )

    result = service.call_agent(
        LLMRequest(
            provider="Anthropic",
            model="claude-sonnet-5",
            api_key="secret",
            template_text="Crie uma animação.",
        ),
        tools=(search_manim_reference,),
        response_schema=ManimCodeOutput,
    )

    assert not result.result.ok
    assert "não retornou a saída estruturada" in result.result.message


def test_native_agent_timeout_reports_the_local_limit(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    service = LLMService(prompt_service=_prompt_service(tmp_path))

    class _AgentProvider:
        def create_chat_model(self, **_kwargs: object) -> object:
            return object()

    class _Agent:
        def invoke(self, *_args: object, **_kwargs: object) -> object:
            raise TimeoutError("Request timed out")

    monkeypatch.setenv("OLYMPIANIM_LLM_TIMEOUT_SECONDS", "900")
    monkeypatch.setattr(
        "olympianim.services.llm_service.get_provider",
        lambda _provider, _api_key: _AgentProvider(),
    )
    monkeypatch.setattr(
        "olympianim.services.llm_service.create_agent",
        lambda **_kwargs: _Agent(),
    )

    result = service.call_agent(
        LLMRequest(
            provider="OpenAI",
            model="gpt-test",
            api_key="sk-secret",
            template_text="Crie uma animação.",
        ),
        tools=(search_manim_reference,),
    )

    assert not result.result.ok
    assert "limite local de 900 segundos" in result.result.message
    assert "sk-secret" not in result.result.message


def test_llm_service_sends_problem_image_as_multimodal_content(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    service = LLMService(prompt_service=_prompt_service(tmp_path))
    captured: dict[str, Any] = {}

    class _AgentProvider:
        def create_chat_model(self, **_kwargs: object) -> object:
            return object()

    class _Agent:
        def invoke(
            self,
            state: dict[str, object],
            config: dict[str, object],
        ) -> dict[str, object]:
            captured["state"] = state
            return {"messages": [SimpleNamespace(content="resultado")]}

    monkeypatch.setattr(
        "olympianim.services.llm_service.get_provider",
        lambda _provider, _api_key: _AgentProvider(),
    )
    monkeypatch.setattr(
        "olympianim.services.llm_service.create_agent",
        lambda **_kwargs: _Agent(),
    )

    result = service.call_agent(
        LLMRequest(
            provider="OpenAI",
            model="gpt-5.4-mini",
            api_key="secret",
            template_text="Analise a imagem da questão.",
            images=(
                LLMImage(
                    data=b"image-bytes",
                    mime_type="image/png",
                    label="Imagem do enunciado 1",
                ),
            ),
        ),
        tools=(search_manim_reference,),
    )

    assert result.result.ok
    messages = captured["state"]["messages"]
    content = messages[0]["content"]
    assert content[0] == {"type": "text", "text": "Analise a imagem da questão."}
    assert content[1] == {"type": "text", "text": "Imagem do enunciado 1"}
    assert content[2] == {
        "type": "image",
        "base64": "aW1hZ2UtYnl0ZXM=",
        "mime_type": "image/png",
    }


def test_llm_image_loads_supported_file(tmp_path: Path) -> None:
    image_path = tmp_path / "questao.jpg"
    image_path.write_bytes(b"jpeg-bytes")

    image = LLMImage.from_path(image_path)

    assert image.data == b"jpeg-bytes"
    assert image.mime_type == "image/jpeg"


def test_llm_service_persists_native_usage_callback(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    repository = ProjectRepository(tmp_path / "olympianim.db")
    project = repository.create_project(
        ProjectCreate(title="Projeto", problem_statement="Problema")
    )
    usage_service = UsageService(repository)
    service = LLMService(
        prompt_service=PromptService(repository=repository),
        usage_service=usage_service,
    )

    class _UsageProvider:
        def invoke(self, prompt: str, **kwargs: Any) -> LLMCallResult:
            for callback in kwargs["callbacks"]:
                message = AIMessage(
                    content="ok",
                    response_metadata={"model_name": kwargs["model"]},
                    usage_metadata={
                        "input_tokens": 8,
                        "output_tokens": 3,
                        "total_tokens": 11,
                    },
                )
                callback.on_llm_end(LLMResult(generations=[[ChatGeneration(message=message)]]))
            return LLMCallResult(
                ok=True,
                provider="OpenAI",
                model=kwargs["model"],
                content="ok",
            )

    monkeypatch.setattr(
        "olympianim.services.llm_service.get_provider",
        lambda _provider, _api_key: _UsageProvider(),
    )
    result = service.call_text(
        LLMRequest(
            provider="OpenAI",
            model="gpt-test",
            api_key="secret",
            template_text="Diga ok",
            usage_context=UsageContext(
                project_id=project.id,
                execution_id="job-id",
                call_key="job-id:planner:presentation:1",
                agent_type="planner",
                stage="presentation",
            ),
        )
    )

    records = usage_service.list_project_usage(project.id)
    assert result.result.ok
    assert len(records) == 1
    assert records[0].total_tokens == 11
