"""Unit tests for the LLM provider abstraction and connection probes.

The concrete providers are exercised with stubbed SDK clients so no
real network call is made. The goal is to verify:

* the base contract is honoured;
* a missing key yields a clear Portuguese error;
* an authentication error is converted into a teacher-friendly message;
* a successful probe returns ``ok=True``;
* the API key never appears in the resulting message.
"""

from __future__ import annotations

import sys
import types
from typing import ClassVar

import pytest

from olympianim.config import DEFAULT_LLM_MAX_RETRIES, DEFAULT_LLM_REQUEST_TIMEOUT_SECONDS
from olympianim.providers.llm import get_provider, supported_providers
from olympianim.providers.llm.anthropic_provider import AnthropicProvider
from olympianim.providers.llm.base import ConnectionResult, LLMProvider
from olympianim.providers.llm.google_provider import GoogleProvider
from olympianim.providers.llm.openai_provider import OpenAIProvider


def test_supported_providers_match_prd() -> None:
    assert supported_providers() == ("OpenAI", "Google", "Anthropic")


def test_get_provider_returns_correct_class() -> None:
    assert isinstance(get_provider("OpenAI", "k"), OpenAIProvider)
    assert isinstance(get_provider("Google", "k"), GoogleProvider)
    assert isinstance(get_provider("Anthropic", "k"), AnthropicProvider)


def test_get_provider_unknown_raises() -> None:
    with pytest.raises(ValueError, match="Provedor"):
        get_provider("Unknown", "k")


@pytest.mark.parametrize(
    ("model", "expected"),
    [
        ("gpt-5.4-mini", 128_000),
        ("gpt-5.5", 128_000),
        ("gpt-5.6-sol", 128_000),
    ],
)
def test_openai_code_generation_uses_model_output_ceiling(
    model: str,
    expected: int,
) -> None:
    assert OpenAIProvider("key").code_generation_max_tokens(model) == expected


@pytest.mark.parametrize(
    ("model", "expected"),
    [
        ("claude-fable-5", 128_000),
        ("claude-opus-4-8", 128_000),
        ("claude-sonnet-5", 128_000),
        ("claude-haiku-4-5", 64_000),
    ],
)
def test_anthropic_code_generation_uses_model_output_ceiling(
    model: str,
    expected: int,
) -> None:
    assert AnthropicProvider("key").code_generation_max_tokens(model) == expected


@pytest.mark.parametrize(
    "model",
    ["gemini-3.1-flash-lite", "gemini-3.1-pro-preview", "gemini-3.5-flash"],
)
def test_google_code_generation_uses_model_output_ceiling(model: str) -> None:
    assert GoogleProvider("key").code_generation_max_tokens(model) == 65_536


@pytest.mark.parametrize(
    "provider",
    [OpenAIProvider("key"), GoogleProvider("key"), AnthropicProvider("key")],
)
def test_unknown_model_does_not_receive_an_unsafe_assumed_ceiling(
    provider: LLMProvider,
) -> None:
    assert provider.code_generation_max_tokens("future-model") is None


def test_connection_result_truthiness() -> None:
    assert ConnectionResult(ok=True, provider="x", message="y")
    assert not ConnectionResult(ok=False, provider="x", message="y")


# --- missing key -----------------------------------------------------------
def test_openai_missing_key_reports_error() -> None:
    result = OpenAIProvider(api_key="").test_connection()
    assert not result.ok
    assert "OpenAI" in result.message


def test_google_missing_key_reports_error() -> None:
    result = GoogleProvider(api_key="").test_connection()
    assert not result.ok
    assert "Google" in result.message


def test_anthropic_missing_key_reports_error() -> None:
    result = AnthropicProvider(api_key="").test_connection()
    assert not result.ok
    assert "Anthropic" in result.message


# --- successful probe (stubbed SDK) ----------------------------------------
def test_openai_success(monkeypatch: pytest.MonkeyPatch) -> None:
    import openai

    class _FakeModels:
        def list(self) -> object:
            return object()

    class _FakeClient:
        def __init__(self, **_kwargs: object) -> None:
            self.models = _FakeModels()

    monkeypatch.setattr(openai, "OpenAI", _FakeClient)
    result = OpenAIProvider(api_key="sk-success").test_connection()
    assert result.ok
    assert "bem-sucedida" in result.message


def test_google_success(monkeypatch: pytest.MonkeyPatch) -> None:
    import google.genai as genai

    class _FakeModels:
        def list(self) -> list[str]:
            return ["gemini-test"]

    class _FakeClient:
        def __init__(self, **_kwargs: object) -> None:
            self.models = _FakeModels()

    monkeypatch.setattr(genai, "Client", _FakeClient)
    result = GoogleProvider(api_key="g-success").test_connection()
    assert result.ok
    assert "bem-sucedida" in result.message


def test_anthropic_success(monkeypatch: pytest.MonkeyPatch) -> None:
    import anthropic

    class _FakeModels:
        def list(self) -> object:
            return object()

    class _FakeClient:
        def __init__(self, **_kwargs: object) -> None:
            self.models = _FakeModels()

    monkeypatch.setattr(anthropic, "Anthropic", _FakeClient)
    result = AnthropicProvider(api_key="sk-ant-success").test_connection()
    assert result.ok
    assert "bem-sucedida" in result.message


# --- auth failure (stubbed SDK) -------------------------------------------
def test_openai_auth_failure_does_not_leak_key(monkeypatch: pytest.MonkeyPatch) -> None:
    import openai

    class _AuthErr(openai.AuthenticationError):
        def __init__(self) -> None:
            super().__init__(message="invalid key sk-LEAK-123", response=None, body=None)

    class _FakeClient:
        def __init__(self, **_kwargs: object) -> None:
            raise RuntimeError("not used")

        @property
        def models(self) -> object:
            raise _AuthErr()

    monkeypatch.setattr(openai, "OpenAI", _FakeClient)
    result = OpenAIProvider(api_key="sk-LEAK-123").test_connection()
    assert not result.ok
    assert "sk-LEAK-123" not in result.message
    assert "OpenAI" in result.message


# --- base helpers ----------------------------------------------------------
def test_format_error_redacts_env_key(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("OPENAI_API_KEY", "sk-ENV-LEAK")
    result = LLMProvider._format_error("OpenAI", ValueError("boom sk-ENV-LEAK boom"))
    assert not result.ok
    assert "sk-ENV-LEAK" not in result.message


# --- LangChain calls (stubbed integrations) --------------------------------
class _FakeMessage:
    def __init__(self, content: object) -> None:
        self.content = content
        self.response_metadata = {
            "model_name": "resolved-test-model",
            "finish_reason": "stop",
        }


class _FakeChatModel:
    kwargs_seen: ClassVar[dict[str, object]] = {}
    messages_seen: ClassVar[object] = None

    def __init__(self, **kwargs: object) -> None:
        type(self).kwargs_seen = kwargs

    def invoke(self, messages: object, config: object = None) -> _FakeMessage:
        _ = config
        type(self).messages_seen = messages
        return _FakeMessage("ok")


def _install_fake_module(
    monkeypatch: pytest.MonkeyPatch,
    module_name: str,
    class_name: str,
) -> None:
    module = types.ModuleType(module_name)
    setattr(module, class_name, _FakeChatModel)
    monkeypatch.setitem(sys.modules, module_name, module)


def test_openai_langchain_text_call(monkeypatch: pytest.MonkeyPatch) -> None:
    _install_fake_module(monkeypatch, "langchain_openai", "ChatOpenAI")

    result = OpenAIProvider(api_key="sk-test").invoke("Diga ok", model="gpt-5.5")

    assert result.ok
    assert result.content == "ok"
    assert _FakeChatModel.kwargs_seen["model"] == "gpt-5.5"
    assert _FakeChatModel.kwargs_seen["api_key"] == "sk-test"
    assert _FakeChatModel.kwargs_seen["timeout"] == float(DEFAULT_LLM_REQUEST_TIMEOUT_SECONDS)
    assert _FakeChatModel.kwargs_seen["max_retries"] == DEFAULT_LLM_MAX_RETRIES
    assert result.resolved_model == "resolved-test-model"
    assert result.finish_reason == "stop"


@pytest.mark.parametrize("model", ["gpt-5.6-sol", "gpt-5.6-terra", "gpt-5.6-luna"])
def test_openai_gpt_56_uses_responses_api(
    monkeypatch: pytest.MonkeyPatch,
    model: str,
) -> None:
    _install_fake_module(monkeypatch, "langchain_openai", "ChatOpenAI")

    result = OpenAIProvider(api_key="sk-test").invoke("Diga ok", model=model)

    assert result.ok
    assert _FakeChatModel.kwargs_seen["use_responses_api"] is True


def test_openai_older_models_keep_langchain_default_endpoint(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _install_fake_module(monkeypatch, "langchain_openai", "ChatOpenAI")

    result = OpenAIProvider(api_key="sk-test").invoke("Diga ok", model="gpt-5.4-mini")

    assert result.ok
    assert "use_responses_api" not in _FakeChatModel.kwargs_seen


def test_google_langchain_text_call(monkeypatch: pytest.MonkeyPatch) -> None:
    _install_fake_module(monkeypatch, "langchain_google_genai", "ChatGoogleGenerativeAI")

    result = GoogleProvider(api_key="g-test").invoke("Diga ok", model="gemini-3.1-flash-lite")

    assert result.ok
    assert result.content == "ok"
    assert _FakeChatModel.kwargs_seen["model"] == "gemini-3.1-flash-lite"
    assert _FakeChatModel.kwargs_seen["google_api_key"] == "g-test"
    assert _FakeChatModel.kwargs_seen["request_timeout"] == float(
        DEFAULT_LLM_REQUEST_TIMEOUT_SECONDS
    )
    assert _FakeChatModel.kwargs_seen["retries"] == DEFAULT_LLM_MAX_RETRIES


def test_anthropic_langchain_text_call(monkeypatch: pytest.MonkeyPatch) -> None:
    _install_fake_module(monkeypatch, "langchain_anthropic", "ChatAnthropic")

    result = AnthropicProvider(api_key="a-test").invoke(
        "Diga ok",
        model="claude-sonnet-5",
        temperature=0.2,
    )

    assert result.ok
    assert result.content == "ok"
    assert _FakeChatModel.kwargs_seen["model"] == "claude-sonnet-5"
    assert _FakeChatModel.kwargs_seen["api_key"] == "a-test"
    assert _FakeChatModel.kwargs_seen["timeout"] == float(DEFAULT_LLM_REQUEST_TIMEOUT_SECONDS)
    assert _FakeChatModel.kwargs_seen["max_retries"] == DEFAULT_LLM_MAX_RETRIES
    assert "temperature" not in _FakeChatModel.kwargs_seen


def test_provider_rejects_truncated_text_response(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    class _TruncatedChatModel(_FakeChatModel):
        def invoke(self, messages: object, config: object = None) -> _FakeMessage:
            _ = (messages, config)
            response = _FakeMessage("resposta parcial")
            response.response_metadata["finish_reason"] = "max_tokens"
            return response

    module = types.ModuleType("langchain_anthropic")
    module.ChatAnthropic = _TruncatedChatModel
    monkeypatch.setitem(sys.modules, "langchain_anthropic", module)

    result = AnthropicProvider(api_key="a-test").invoke(
        "Gere um arquivo completo.",
        model="claude-sonnet-5",
    )

    assert not result.ok
    assert result.content == ""
    assert result.finish_reason == "max_tokens"
    assert "não foi armazenado" in result.message


def test_missing_langchain_integration_reports_teacher_friendly_error(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setitem(sys.modules, "langchain_openai", None)

    result = OpenAIProvider(api_key="sk-test").invoke("Diga ok", model="gpt-5.5")

    assert not result.ok
    assert "langchain-openai" in result.message
