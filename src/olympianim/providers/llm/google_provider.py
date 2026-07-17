"""Google (Gemini) language-model provider.

Implements ``test_connection`` via the cheapest authenticated call
(``models.list``).
"""

from __future__ import annotations

from typing import Any

from olympianim.config import DEFAULT_LLM_MAX_RETRIES, llm_request_timeout_seconds
from olympianim.providers.llm.base import ConnectionResult, LLMProvider

_CODE_GENERATION_MAX_TOKENS = {
    "gemini-3.1-flash-lite": 65_536,
    "gemini-3.1-pro-preview": 65_536,
    "gemini-3.5-flash": 65_536,
}


class GoogleProvider(LLMProvider):
    """Google Gemini provider using the ``google-genai`` SDK."""

    name = "Google"

    def supports_native_structured_output(self, model: str) -> bool:
        """Use Gemini's JSON-schema response format for the current model catalog."""
        _ = model
        return True

    def code_generation_max_tokens(self, model: str) -> int | None:
        """Return the documented output ceiling for a configured Gemini model."""
        return _CODE_GENERATION_MAX_TOKENS.get(model)

    def _create_chat_model(
        self,
        *,
        model: str,
        temperature: float,
        max_tokens: int | None,
    ) -> Any:
        """Create a LangChain Google Gemini chat model."""
        try:
            from langchain_google_genai import ChatGoogleGenerativeAI
        except ImportError as exc:
            raise RuntimeError(
                "Integração LangChain do Google não instalada. "
                "Instale o extra `.[google]` ou `langchain-google-genai`."
            ) from exc

        kwargs: dict[str, Any] = {
            "model": model,
            "google_api_key": self.api_key,
            "temperature": temperature,
            "request_timeout": float(llm_request_timeout_seconds()),
            "retries": DEFAULT_LLM_MAX_RETRIES,
        }
        if max_tokens is not None:
            kwargs["max_tokens"] = max_tokens
        return ChatGoogleGenerativeAI(**kwargs)

    def test_connection(self) -> ConnectionResult:
        if not self.api_key:
            return ConnectionResult(
                ok=False,
                provider=self.name,
                message="Nenhuma chave de API fornecida para o Google.",
            )
        try:
            from google import genai
            from google.genai.errors import ClientError
        except ImportError as exc:
            return ConnectionResult(
                ok=False,
                provider=self.name,
                message=f"SDK do Google (google-genai) não instalado: {exc}",
            )

        try:
            client = genai.Client(api_key=self.api_key)
            list(client.models.list())
        except ClientError as exc:
            return self._format_error(self.name, exc)
        except Exception as exc:
            return self._format_error(self.name, exc)

        return ConnectionResult(
            ok=True,
            provider=self.name,
            message="Conexão com o Google bem-sucedida.",
        )
