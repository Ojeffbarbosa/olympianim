"""OpenAI language-model provider.

Implements ``test_connection`` via the cheapest authenticated call
(``models.list``).
"""

from __future__ import annotations

from typing import Any

from olympianim.config import DEFAULT_LLM_MAX_RETRIES, llm_request_timeout_seconds
from olympianim.providers.llm.base import ConnectionResult, LLMProvider

_CODE_GENERATION_MAX_TOKENS = {
    "gpt-5.4": 128_000,
    "gpt-5.4-mini": 128_000,
    "gpt-5.4-nano": 128_000,
    "gpt-5.4-pro": 128_000,
    "gpt-5.5": 128_000,
    "gpt-5.5-pro": 128_000,
    "gpt-5.6": 128_000,
    "gpt-5.6-luna": 128_000,
    "gpt-5.6-sol": 128_000,
    "gpt-5.6-terra": 128_000,
}


class OpenAIProvider(LLMProvider):
    """OpenAI provider using the official ``openai`` SDK."""

    name = "OpenAI"

    def supports_native_structured_output(self, model: str) -> bool:
        """Use the structured-output API supported by the configured GPT catalog."""
        _ = model
        return True

    def code_generation_max_tokens(self, model: str) -> int | None:
        """Return the documented output ceiling for a configured GPT model."""
        return _CODE_GENERATION_MAX_TOKENS.get(model)

    def _create_chat_model(
        self,
        *,
        model: str,
        temperature: float,
        max_tokens: int | None,
    ) -> Any:
        """Create a LangChain OpenAI chat model."""
        try:
            from langchain_openai import ChatOpenAI
        except ImportError as exc:
            raise RuntimeError(
                "Integração LangChain da OpenAI não instalada. "
                "Instale o extra `.[openai]` ou `langchain-openai`."
            ) from exc

        kwargs: dict[str, Any] = {
            "model": model,
            "api_key": self.api_key,
            "temperature": temperature,
            "timeout": float(llm_request_timeout_seconds()),
            "max_retries": DEFAULT_LLM_MAX_RETRIES,
        }
        if max_tokens is not None:
            kwargs["max_tokens"] = max_tokens
        if model.startswith("gpt-5.6"):
            kwargs["use_responses_api"] = True
        return ChatOpenAI(**kwargs)

    def test_connection(self) -> ConnectionResult:
        if not self.api_key:
            return ConnectionResult(
                ok=False,
                provider=self.name,
                message="Nenhuma chave de API fornecida para a OpenAI.",
            )
        try:
            from openai import APIError, AuthenticationError, OpenAI
        except ImportError as exc:
            return ConnectionResult(
                ok=False,
                provider=self.name,
                message=f"SDK da OpenAI não instalado: {exc}",
            )

        try:
            client = OpenAI(api_key=self.api_key)
            client.models.list()
        except AuthenticationError as exc:
            return self._format_error(self.name, exc)
        except APIError as exc:
            return self._format_error(self.name, exc)
        except Exception as exc:
            return self._format_error(self.name, exc)

        return ConnectionResult(
            ok=True,
            provider=self.name,
            message="Conexão com a OpenAI bem-sucedida.",
        )
