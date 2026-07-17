"""Language model providers: OpenAI, Google and Anthropic behind a common
abstraction so the pedagogical flow does not depend on a single vendor."""

from olympianim.providers.llm.anthropic_provider import AnthropicProvider
from olympianim.providers.llm.base import (
    DEFAULT_SYSTEM_PROMPT,
    ConnectionResult,
    LLMCallResult,
    LLMProvider,
    LLMProviderError,
    is_truncated_finish_reason,
    truncated_response_message,
)
from olympianim.providers.llm.google_provider import GoogleProvider
from olympianim.providers.llm.openai_provider import OpenAIProvider

__all__ = [
    "DEFAULT_SYSTEM_PROMPT",
    "AnthropicProvider",
    "ConnectionResult",
    "GoogleProvider",
    "LLMCallResult",
    "LLMProvider",
    "LLMProviderError",
    "OpenAIProvider",
    "get_provider",
    "is_truncated_finish_reason",
    "truncated_response_message",
]

_PROVIDERS: dict[str, type[LLMProvider]] = {
    "OpenAI": OpenAIProvider,
    "Google": GoogleProvider,
    "Anthropic": AnthropicProvider,
}


def get_provider(name: str, api_key: str) -> LLMProvider:
    """Instantiate the provider identified by ``name`` with ``api_key``.

    Raises ``ValueError`` when the name is not recognised so the caller
    can surface a teacher-friendly error.
    """
    cls = _PROVIDERS.get(name)
    if cls is None:
        raise ValueError(f"Provedor de IA desconhecido: {name!r}")
    return cls(api_key=api_key)


def supported_providers() -> tuple[str, ...]:
    """Return the names of the registered providers."""
    return tuple(_PROVIDERS)
