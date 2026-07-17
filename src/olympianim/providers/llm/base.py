"""Common interface for language-model providers.

Each concrete provider (OpenAI, Google, Anthropic) implements
``test_connection`` so the teacher can verify a key before starting a
generation. It also supplies the provider-neutral
text and multimodal invocation contract used by the workflow.

``test_connection`` must:

* never log the API key (the active ``SecretFilter`` redacts it);
* return a ``ConnectionResult`` with a human-readable Portuguese
  message safe to show in the interface.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from typing import Any

from langchain_core.callbacks import BaseCallbackHandler

from olympianim.config import llm_request_timeout_seconds
from olympianim.utils.logging import redact

DEFAULT_SYSTEM_PROMPT = (
    "Voce e um componente interno do Olympianim. Responda de forma objetiva, "
    "seguindo exatamente o formato solicitado."
)


@dataclass(frozen=True)
class ConnectionResult:
    """Outcome of a connection probe against a provider."""

    ok: bool
    message: str
    provider: str

    def __bool__(self) -> bool:
        """Truthy when the connection succeeded."""
        return self.ok


@dataclass(frozen=True)
class LLMCallResult:
    """Result of a plain text call to a language model."""

    ok: bool
    provider: str
    model: str
    content: str
    message: str = ""
    resolved_model: str = ""
    finish_reason: str = ""

    def __bool__(self) -> bool:
        """Truthy when the call succeeded."""
        return self.ok


class LLMProviderError(RuntimeError):
    """Provider failure safe to surface in Portuguese UI messages."""


_TRUNCATED_FINISH_REASONS = frozenset(
    {
        "length",
        "max_output_tokens",
        "max_tokens",
        "model_context_window_exceeded",
    }
)


def is_truncated_finish_reason(value: str) -> bool:
    """Return whether a provider explicitly reported an incomplete response."""
    return value.strip().casefold() in _TRUNCATED_FINISH_REASONS


def truncated_response_message(provider: str, finish_reason: str) -> str:
    """Build the stable user-facing diagnostic for incomplete model output."""
    return (
        f"A resposta de {provider} foi interrompida antes de terminar "
        f"({finish_reason or 'limite de saída'}). Tente novamente; o conteúdo "
        "incompleto não foi armazenado nem enviado para execução."
    )


def safe_provider_call_error(provider: str, error: Exception, api_key: str) -> str:
    """Describe a provider failure without leaking credentials or hiding timeouts."""
    if _is_timeout_error(error):
        timeout = llm_request_timeout_seconds()
        return (
            f"Falha ao chamar {provider}: o limite local de {timeout} segundos "
            "por tentativa foi atingido."
        )
    return f"Falha ao chamar {provider}: {redact(str(error), [api_key])}"


def _is_timeout_error(error: Exception) -> bool:
    """Recognize timeout wrappers used by the supported SDKs and HTTP clients."""
    current: BaseException | None = error
    visited: set[int] = set()
    while current is not None and id(current) not in visited:
        visited.add(id(current))
        error_name = type(current).__name__.casefold()
        error_text = str(current).casefold()
        if (
            isinstance(current, TimeoutError)
            or "timeout" in error_name
            or "timed out" in error_text
            or "deadline exceeded" in error_text
        ):
            return True
        current = current.__cause__ or current.__context__
    return False


class LLMProvider(ABC):
    """Abstract base for language-model providers.

    Concrete subclasses are constructed with a resolved API key (the
    credential service is responsible for obtaining it). The key is
    kept only in the instance, never persisted, and registered with the
    ``SecretFilter`` so it is redacted from logs.
    """

    name: str = ""

    def __init__(self, api_key: str) -> None:
        self.api_key = api_key

    @abstractmethod
    def test_connection(self) -> ConnectionResult:
        """Probe the provider with a minimal request.

        Implementations should perform the cheapest authenticated call
        possible (e.g. list models or send a one-token prompt) and
        translate any error into a teacher-friendly Portuguese message.
        """
        raise NotImplementedError

    def invoke(
        self,
        prompt: str,
        *,
        model: str,
        system_prompt: str = DEFAULT_SYSTEM_PROMPT,
        temperature: float = 0.2,
        max_tokens: int | None = None,
        callbacks: Sequence[BaseCallbackHandler] = (),
        images: Sequence[Any] = (),
    ) -> LLMCallResult:
        """Call a chat model and return text content."""
        if not self.api_key:
            return LLMCallResult(
                ok=False,
                provider=self.name,
                model=model,
                content="",
                message=f"Nenhuma chave de API fornecida para {self.name}.",
            )
        try:
            chat_model = self._create_chat_model(
                model=model,
                temperature=temperature,
                max_tokens=max_tokens,
            )
            response = chat_model.invoke(
                self._messages(system_prompt, prompt, images),
                config={"callbacks": list(callbacks)},
            )
            content = self._extract_text(response)
        except Exception as exc:
            return LLMCallResult(
                ok=False,
                provider=self.name,
                model=model,
                content="",
                message=self._safe_error_message(exc),
            )

        metadata = getattr(response, "response_metadata", {})
        if not isinstance(metadata, Mapping):
            metadata = {}
        resolved_model = str(metadata.get("model_name") or metadata.get("model") or model)
        finish_reason = str(
            metadata.get("finish_reason")
            or metadata.get("stop_reason")
            or metadata.get("stop_sequence")
            or ""
        )
        if is_truncated_finish_reason(finish_reason):
            return LLMCallResult(
                ok=False,
                provider=self.name,
                model=model,
                content="",
                message=truncated_response_message(self.name, finish_reason),
                resolved_model=resolved_model,
                finish_reason=finish_reason,
            )
        return LLMCallResult(
            ok=True,
            provider=self.name,
            model=model,
            content=content,
            message=f"Chamada com {self.name} concluída.",
            resolved_model=resolved_model,
            finish_reason=finish_reason,
        )

    def create_chat_model(
        self,
        *,
        model: str,
        temperature: float = 0.2,
        max_tokens: int | None = None,
    ) -> Any:
        """Create the provider-neutral LangChain model used by native agents."""
        if not self.api_key:
            raise LLMProviderError(f"Nenhuma chave de API fornecida para {self.name}.")
        return self._create_chat_model(
            model=model,
            temperature=temperature,
            max_tokens=max_tokens,
        )

    def supports_native_structured_output(self, model: str) -> bool:
        """Return whether this adapter can enforce schemas at provider level."""
        _ = model
        return False

    def code_generation_max_tokens(self, model: str) -> int | None:
        """Return the provider-safe ceiling for a complete generated source file."""
        _ = model
        return None

    @abstractmethod
    def _create_chat_model(
        self,
        *,
        model: str,
        temperature: float,
        max_tokens: int | None,
    ) -> Any:
        """Create the provider-specific LangChain chat model."""
        raise NotImplementedError

    @staticmethod
    def _format_error(provider: str, error: Exception) -> ConnectionResult:
        """Build a failure result from an exception, without leaking the key."""
        text = str(error)
        # Defensive: scrub the key if it slipped into the message.
        import os

        from olympianim.services.credential_service import PROVIDER_ENV_VARS
        from olympianim.utils.logging import redact

        env_var = PROVIDER_ENV_VARS.get(provider, "")
        secret = os.environ.get(env_var, "") if env_var else ""
        redacted_text = redact(text, [secret]) if secret else text
        return ConnectionResult(
            ok=False,
            provider=provider,
            message=f"Falha ao conectar com {provider}: {redacted_text}",
        )

    def _safe_error_message(self, error: Exception) -> str:
        """Return a user-facing error message with the API key redacted."""
        return safe_provider_call_error(self.name, error, self.api_key)

    @staticmethod
    def _messages(
        system_prompt: str,
        prompt: str,
        images: Sequence[Any] = (),
    ) -> list[dict[str, Any]]:
        """Return messages in LangChain's provider-neutral dict format."""
        user_content: str | list[dict[str, str]] = prompt
        if images:
            blocks = [{"type": "text", "text": prompt}]
            for image in images:
                blocks.extend(
                    (
                        {"type": "text", "text": image.label},
                        image.content_block(),
                    )
                )
            user_content = blocks
        return [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_content},
        ]

    @staticmethod
    def _extract_text(response: Any) -> str:
        """Extract text content from a LangChain message-like object."""
        content = getattr(response, "content", response)
        if isinstance(content, str):
            return content
        if isinstance(content, list):
            parts: list[str] = []
            for block in content:
                if isinstance(block, str):
                    parts.append(block)
                elif isinstance(block, Mapping):
                    text = block.get("text") or block.get("content")
                    if isinstance(text, str):
                        parts.append(text)
            return "\n".join(parts)
        return str(content)
