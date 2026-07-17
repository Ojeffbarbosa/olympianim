"""Anthropic (Claude) language-model provider.

Implements ``test_connection`` via the cheapest authenticated call
(``models.list``).
"""

from __future__ import annotations

from typing import Any

from olympianim.config import DEFAULT_LLM_MAX_RETRIES, llm_request_timeout_seconds
from olympianim.providers.llm.base import ConnectionResult, LLMProvider

_CODE_GENERATION_MAX_TOKENS = {
    "claude-fable-5": 128_000,
    "claude-haiku-4-5": 64_000,
    "claude-opus-4-8": 128_000,
    "claude-sonnet-5": 128_000,
}


class AnthropicProvider(LLMProvider):
    """Anthropic Claude provider using the official ``anthropic`` SDK."""

    name = "Anthropic"

    def supports_native_structured_output(self, model: str) -> bool:
        """Use Claude JSON outputs instead of representing the final answer as a tool."""
        _ = model
        return True

    def code_generation_max_tokens(self, model: str) -> int | None:
        """Return the synchronous Messages API ceiling for a configured model."""
        return _CODE_GENERATION_MAX_TOKENS.get(model)

    def _create_chat_model(
        self,
        *,
        model: str,
        temperature: float,
        max_tokens: int | None,
    ) -> Any:
        """Create a LangChain Anthropic chat model."""
        try:
            from langchain_anthropic import ChatAnthropic
        except ImportError as exc:
            raise RuntimeError(
                "Integração LangChain da Anthropic não instalada. "
                "Instale o extra `.[anthropic]` ou `langchain-anthropic`."
            ) from exc

        # Anthropic deprecated sampling parameters for current models. Omitting
        # temperature is backward-compatible and lets the selected model apply
        # its supported default instead of returning an invalid-request error.
        _ = temperature
        kwargs: dict[str, Any] = {
            "model": model,
            "api_key": self.api_key,
            "timeout": float(llm_request_timeout_seconds()),
            "max_retries": DEFAULT_LLM_MAX_RETRIES,
        }
        if max_tokens is not None:
            kwargs["max_tokens"] = max_tokens
        return ChatAnthropic(**kwargs)

    def test_connection(self) -> ConnectionResult:
        if not self.api_key:
            return ConnectionResult(
                ok=False,
                provider=self.name,
                message="Nenhuma chave de API fornecida para a Anthropic.",
            )
        try:
            import anthropic
        except ImportError as exc:
            return ConnectionResult(
                ok=False,
                provider=self.name,
                message=f"SDK da Anthropic não instalado: {exc}",
            )

        try:
            client = anthropic.Anthropic(api_key=self.api_key)
            client.models.list()
        except anthropic.AuthenticationError as exc:
            return self._format_error(self.name, exc)
        except anthropic.APIError as exc:
            return self._format_error(self.name, exc)
        except Exception as exc:
            return self._format_error(self.name, exc)

        return ConnectionResult(
            ok=True,
            provider=self.name,
            message="Conexão com a Anthropic bem-sucedida.",
        )
