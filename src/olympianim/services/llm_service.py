"""High-level LLM calls using editable prompts and provider fallback."""

from __future__ import annotations

import base64
import hashlib
import mimetypes
from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from langchain.agents import create_agent
from langchain.agents.middleware import ToolCallLimitMiddleware, ToolRetryMiddleware
from langchain.agents.structured_output import ProviderStrategy, ToolStrategy
from langchain_core.callbacks import BaseCallbackHandler, UsageMetadataCallbackHandler
from langchain_core.tools import BaseTool

from olympianim.config import MANIM_REFERENCE_TOOL_CALL_LIMIT
from olympianim.prompts.service import PromptService
from olympianim.prompts.validator import render_prompt_template
from olympianim.providers.llm import (
    DEFAULT_SYSTEM_PROMPT,
    LLMCallResult,
    get_provider,
    is_truncated_finish_reason,
    truncated_response_message,
)
from olympianim.providers.llm.base import safe_provider_call_error
from olympianim.schemas.llm import ManimCodeOutput
from olympianim.services.usage_service import UsageContext, UsageService
from olympianim.utils.logging import redact


def split_prompt_template(template_text: str) -> tuple[str, str | None]:
    """Split editable defaults into instruction and user-context messages.

    Templates without an exact ``# Contexto`` heading retain the legacy behavior,
    which keeps every custom prompt editable and backwards compatible.
    """
    lines = template_text.splitlines()
    marker = next(
        (index for index, line in enumerate(lines) if line.strip() == "# Contexto"),
        None,
    )
    if marker is None:
        return template_text, None
    instructions = "\n".join(lines[:marker]).strip()
    context = "\n".join(lines[marker + 1 :]).strip()
    return instructions, context


@dataclass(frozen=True)
class LLMFallback:
    """Fallback provider/model/key triple used when the primary call fails."""

    provider: str
    model: str
    api_key: str


@dataclass(frozen=True)
class LLMImage:
    """Validated image input sent through LangChain's multimodal content blocks."""

    data: bytes
    mime_type: str
    label: str = "Imagem anexada"

    @classmethod
    def from_path(cls, path: str | Path, *, label: str = "Imagem anexada") -> LLMImage:
        """Load a supported local image without exposing its path to the model."""
        image_path = Path(path)
        mime_type, _ = mimetypes.guess_type(image_path.name)
        if mime_type not in {"image/jpeg", "image/png", "image/webp"}:
            raise ValueError(
                f"Formato de imagem não suportado: {image_path.suffix or 'desconhecido'}"
            )
        return cls(data=image_path.read_bytes(), mime_type=mime_type, label=label)

    def content_block(self) -> dict[str, str]:
        """Return the provider-neutral LangChain image block."""
        return {
            "type": "image",
            "base64": base64.b64encode(self.data).decode("ascii"),
            "mime_type": self.mime_type,
        }


@dataclass(frozen=True)
class LLMRequest:
    """Request for a plain or structured LLM call."""

    provider: str
    model: str
    api_key: str
    prompt_values: dict[str, object] = field(default_factory=dict)
    prompt_id: str = ""
    template_text: str = ""
    system_prompt: str = ""
    temperature: float = 0.2
    max_tokens: int | None = None
    fallbacks: tuple[LLMFallback, ...] = ()
    callbacks: tuple[BaseCallbackHandler, ...] = ()
    usage_context: UsageContext | None = None
    images: tuple[LLMImage, ...] = ()


@dataclass(frozen=True)
class PromptRenderResult:
    """Rendered prompt metadata used for reproducibility."""

    prompt_id: str
    prompt_version: int | None
    agent_type: str
    rendered_prompt: str
    system_prompt: str
    user_prompt: str
    prompt_sha256: str


@dataclass(frozen=True)
class TextLLMServiceResult:
    """Result returned by ``LLMService.call_text``."""

    result: LLMCallResult
    prompt: PromptRenderResult
    attempted: tuple[str, ...]


class LLMService:
    """Single use-case entry point for language-model calls."""

    def __init__(
        self,
        prompt_service: PromptService | None = None,
        usage_service: UsageService | None = None,
    ) -> None:
        self.prompt_service = prompt_service or PromptService()
        self.usage_service = usage_service

    def render_prompt(self, request: LLMRequest) -> PromptRenderResult:
        """Render either a saved prompt template or an inline template."""
        if request.prompt_id:
            prompt = self.prompt_service.get_prompt(request.prompt_id)
            if prompt is None:
                raise ValueError(f"Prompt não encontrado: {request.prompt_id!r}")
            return self._render_messages(
                prompt.latest_version.template_text,
                request,
                prompt_id=prompt.prompt.id,
                prompt_version=prompt.latest_version.version,
                agent_type=prompt.prompt.agent_type,
            )

        if not request.template_text:
            raise ValueError("Informe `prompt_id` ou `template_text` para chamar o modelo.")

        return self._render_messages(
            request.template_text,
            request,
            prompt_id="",
            prompt_version=None,
            agent_type="",
        )

    @staticmethod
    def _render_messages(
        template_text: str,
        request: LLMRequest,
        *,
        prompt_id: str,
        prompt_version: int | None,
        agent_type: str,
    ) -> PromptRenderResult:
        instruction_template, context_template = split_prompt_template(template_text)
        if context_template is None:
            system_prompt = request.system_prompt or DEFAULT_SYSTEM_PROMPT
            user_prompt = render_prompt_template(template_text, request.prompt_values)
            rendered_prompt = user_prompt
        else:
            built_in_system = render_prompt_template(
                instruction_template, request.prompt_values
            ).strip()
            system_prompt = "\n\n".join(
                part for part in (request.system_prompt.strip(), built_in_system) if part
            )
            user_prompt = render_prompt_template(context_template, request.prompt_values).strip()
            rendered_prompt = f"{system_prompt}\n\n# Contexto\n{user_prompt}".strip()
        digest = hashlib.sha256(f"{system_prompt}\0{user_prompt}".encode()).hexdigest()
        return PromptRenderResult(
            prompt_id=prompt_id,
            prompt_version=prompt_version,
            agent_type=agent_type,
            rendered_prompt=rendered_prompt,
            system_prompt=system_prompt,
            user_prompt=user_prompt,
            prompt_sha256=digest,
        )

    def call_text(self, request: LLMRequest) -> TextLLMServiceResult:
        """Render a prompt and call the selected model with text output."""
        prompt = self.render_prompt(request)
        attempted: list[str] = []
        last_result: LLMCallResult | None = None

        for attempt_number, (provider_name, model_name, api_key) in enumerate(
            self._attempts(request), start=1
        ):
            attempted.append(f"{provider_name}:{model_name}")
            provider = get_provider(provider_name, api_key)
            usage_callback = self._usage_callback(request)
            invoke_kwargs: dict[str, Any] = {}
            callbacks = self._callbacks(request, usage_callback)
            if callbacks:
                invoke_kwargs["callbacks"] = callbacks
            if request.images:
                invoke_kwargs["images"] = request.images
            result = provider.invoke(
                prompt.user_prompt,
                model=model_name,
                system_prompt=prompt.system_prompt,
                temperature=request.temperature,
                max_tokens=request.max_tokens,
                **invoke_kwargs,
            )
            self._record_usage(
                request,
                usage_callback,
                provider=provider_name,
                model=model_name,
                completed=result.ok,
                attempt_number=attempt_number,
            )
            last_result = result
            if result.ok:
                return TextLLMServiceResult(
                    result=result,
                    prompt=prompt,
                    attempted=tuple(attempted),
                )

        assert last_result is not None
        return TextLLMServiceResult(
            result=self._with_fallback_message(last_result, attempted, request),
            prompt=prompt,
            attempted=tuple(attempted),
        )

    def call_agent(
        self,
        request: LLMRequest,
        *,
        tools: Sequence[BaseTool],
        response_schema: type[ManimCodeOutput] | None = None,
    ) -> TextLLMServiceResult:
        """Call a LangChain agent with bounded tools and optional typed output."""
        prompt = self.render_prompt(request)
        attempted: list[str] = []
        last_result: LLMCallResult | None = None

        for attempt_number, (provider_name, model_name, api_key) in enumerate(
            self._attempts(request), start=1
        ):
            attempted.append(f"{provider_name}:{model_name}")
            provider = get_provider(provider_name, api_key)
            usage_callback = self._usage_callback(request)
            try:
                max_tokens = request.max_tokens
                if max_tokens is None and response_schema is not None:
                    budget_resolver = getattr(provider, "code_generation_max_tokens", None)
                    if callable(budget_resolver):
                        max_tokens = budget_resolver(model_name)
                model = provider.create_chat_model(
                    model=model_name,
                    temperature=request.temperature,
                    max_tokens=max_tokens,
                )
                agent_kwargs: dict[str, Any] = {
                    "model": model,
                    "tools": tools,
                    "system_prompt": prompt.system_prompt,
                    "middleware": (
                        *(
                            ToolCallLimitMiddleware(
                                tool_name=tool.name,
                                run_limit=MANIM_REFERENCE_TOOL_CALL_LIMIT,
                                exit_behavior="error",
                            )
                            for tool in tools
                        ),
                        ToolRetryMiddleware(
                            max_retries=2,
                            tools=[tool.name for tool in tools],
                            on_failure="continue",
                        ),
                    ),
                    "name": "olympianim_manim_agent",
                    # Builder/debugger calls are one-shot subgraphs. Persisting their
                    # internal tool messages in the parent workflow can replay a
                    # partially failed provider conversation on the next job.
                    "checkpointer": False,
                }
                if response_schema is not None:
                    native_support = getattr(
                        provider, "supports_native_structured_output", lambda _model: False
                    )
                    agent_kwargs["response_format"] = (
                        ProviderStrategy(response_schema, strict=True)
                        if native_support(model_name)
                        else ToolStrategy(response_schema, handle_errors=False)
                    )
                agent = create_agent(
                    **agent_kwargs,
                )
                state = agent.invoke(
                    {
                        "messages": [
                            {
                                "role": "user",
                                "content": build_user_content(
                                    prompt.user_prompt,
                                    request.images,
                                ),
                            }
                        ]
                    },
                    config={"callbacks": list(self._callbacks(request, usage_callback))},
                )
                content = (
                    _structured_manim_code(state, response_schema)
                    if response_schema is not None
                    else _last_message_text(state)
                )
                if not content:
                    raise ValueError("O agente terminou sem produzir conteúdo.")
                resolved_model, finish_reason = _agent_response_details(state, model_name)
            except Exception as exc:
                resolved_model, finish_reason = _exception_response_details(exc, model_name)
                message = (
                    truncated_response_message(provider_name, finish_reason)
                    if is_truncated_finish_reason(finish_reason)
                    else safe_provider_call_error(provider_name, exc, api_key)
                )
                last_result = LLMCallResult(
                    ok=False,
                    provider=provider_name,
                    model=model_name,
                    content="",
                    message=message,
                    resolved_model=resolved_model,
                    finish_reason=finish_reason,
                )
                self._record_usage(
                    request,
                    usage_callback,
                    provider=provider_name,
                    model=model_name,
                    completed=False,
                    attempt_number=attempt_number,
                )
                continue

            if is_truncated_finish_reason(finish_reason):
                last_result = LLMCallResult(
                    ok=False,
                    provider=provider_name,
                    model=model_name,
                    content="",
                    message=truncated_response_message(provider_name, finish_reason),
                    resolved_model=resolved_model,
                    finish_reason=finish_reason,
                )
                self._record_usage(
                    request,
                    usage_callback,
                    provider=provider_name,
                    model=model_name,
                    completed=False,
                    attempt_number=attempt_number,
                )
                continue

            result = LLMCallResult(
                ok=True,
                provider=provider_name,
                model=model_name,
                content=content,
                message=f"Agente LangChain com {provider_name} concluído.",
                resolved_model=resolved_model,
                finish_reason=finish_reason,
            )
            self._record_usage(
                request,
                usage_callback,
                provider=provider_name,
                model=model_name,
                completed=True,
                attempt_number=attempt_number,
            )
            return TextLLMServiceResult(
                result=result,
                prompt=prompt,
                attempted=tuple(attempted),
            )

        assert last_result is not None
        return TextLLMServiceResult(
            result=self._with_fallback_message(last_result, attempted, request),
            prompt=prompt,
            attempted=tuple(attempted),
        )

    @staticmethod
    def _attempts(request: LLMRequest) -> tuple[tuple[str, str, str], ...]:
        return (
            (request.provider, request.model, request.api_key),
            *(
                (fallback.provider, fallback.model, fallback.api_key)
                for fallback in request.fallbacks
            ),
        )

    @staticmethod
    def _fallback_failure_message(
        message: str,
        attempted: list[str],
        request: LLMRequest,
    ) -> str:
        safe_message = redact(
            message,
            [request.api_key, *(fallback.api_key for fallback in request.fallbacks)],
        )
        return (
            "Todos os provedores configurados falharam. "
            f"Tentativas: {', '.join(attempted)}. Último erro: {safe_message}"
        )

    def _with_fallback_message(
        self,
        result: LLMCallResult,
        attempted: list[str],
        request: LLMRequest,
    ) -> LLMCallResult:
        return LLMCallResult(
            ok=False,
            provider=result.provider,
            model=result.model,
            content="",
            message=self._fallback_failure_message(result.message, attempted, request),
            resolved_model=result.resolved_model,
            finish_reason=result.finish_reason,
        )

    def _usage_callback(self, request: LLMRequest) -> UsageMetadataCallbackHandler | None:
        if self.usage_service is None or request.usage_context is None:
            return None
        return self.usage_service.callback()

    @staticmethod
    def _callbacks(
        request: LLMRequest,
        usage_callback: UsageMetadataCallbackHandler | None,
    ) -> tuple[BaseCallbackHandler, ...]:
        if usage_callback is None:
            return request.callbacks
        return (*request.callbacks, usage_callback)

    def _record_usage(
        self,
        request: LLMRequest,
        usage_callback: UsageMetadataCallbackHandler | None,
        *,
        provider: str,
        model: str,
        completed: bool,
        attempt_number: int,
    ) -> None:
        if self.usage_service is None or request.usage_context is None or usage_callback is None:
            return
        self.usage_service.record_attempt(
            request.usage_context,
            usage_callback,
            provider=provider,
            model=model,
            completed=completed,
            attempt_type="primary" if attempt_number == 1 else "fallback",
            attempt_number=attempt_number,
        )


def _last_message_text(state: Any) -> str:
    """Extract the final AI response from a native LangChain agent state."""
    if not isinstance(state, Mapping):
        return ""
    messages = state.get("messages")
    if not isinstance(messages, Sequence) or isinstance(messages, (str, bytes)) or not messages:
        return ""
    message = messages[-1]
    text = getattr(message, "text", None)
    if isinstance(text, str) and text:
        return text
    content = getattr(message, "content", message)
    if isinstance(content, str):
        return content
    if not isinstance(content, Sequence) or isinstance(content, (str, bytes)):
        return str(content) if content else ""
    parts: list[str] = []
    for block in content:
        if isinstance(block, str):
            parts.append(block)
        elif isinstance(block, Mapping):
            value = block.get("text") or block.get("content")
            if isinstance(value, str):
                parts.append(value)
    return "\n".join(parts)


def _agent_response_details(state: Any, requested_model: str) -> tuple[str, str]:
    """Read the final provider metadata from an agent state without provider coupling."""
    if not isinstance(state, Mapping):
        return requested_model, ""
    messages = state.get("messages")
    if not isinstance(messages, Sequence) or isinstance(messages, (str, bytes)):
        return requested_model, ""
    for message in reversed(messages):
        metadata = getattr(message, "response_metadata", None)
        if not isinstance(metadata, Mapping) or not metadata:
            continue
        resolved_model = str(
            metadata.get("model_name") or metadata.get("model") or requested_model
        )
        finish_reason = str(
            metadata.get("finish_reason")
            or metadata.get("stop_reason")
            or _incomplete_reason(metadata)
            or ""
        )
        return resolved_model, finish_reason
    return requested_model, ""


def _exception_response_details(error: Exception, requested_model: str) -> tuple[str, str]:
    """Recover stop metadata carried by structured-output validation exceptions."""
    pending: list[BaseException] = [error]
    visited: set[int] = set()
    while pending:
        current = pending.pop()
        if id(current) in visited:
            continue
        visited.add(id(current))
        ai_message = getattr(current, "ai_message", None)
        if ai_message is not None:
            details = _agent_response_details({"messages": [ai_message]}, requested_model)
            if details[1]:
                return details
        for candidate in (
            getattr(current, "source", None),
            current.__cause__,
            current.__context__,
        ):
            if isinstance(candidate, BaseException):
                pending.append(candidate)
    return requested_model, ""


def _incomplete_reason(metadata: Mapping[str, Any]) -> str:
    """Extract OpenAI-style incomplete details when no finish reason is present."""
    details = metadata.get("incomplete_details")
    if isinstance(details, Mapping):
        return str(details.get("reason") or "")
    return ""


def _structured_manim_code(
    state: Any,
    response_schema: type[ManimCodeOutput],
) -> str:
    """Validate and extract code from a native structured agent response."""
    if not isinstance(state, Mapping):
        raise ValueError("O agente não retornou um estado estruturado.")
    structured = state.get("structured_response")
    if structured is None:
        raise ValueError("O agente não retornou a saída estruturada solicitada.")
    if not isinstance(structured, response_schema):
        structured = response_schema.model_validate(structured)
    return structured.code


def build_user_content(
    prompt: str,
    images: Sequence[LLMImage],
) -> str | list[str | dict[str, Any]]:
    """Build native LangChain user content, adding vision only when requested."""
    if not images:
        return prompt
    blocks: list[str | dict[str, Any]] = [{"type": "text", "text": prompt}]
    for image in images:
        blocks.extend(
            (
                {"type": "text", "text": image.label},
                image.content_block(),
            )
        )
    return blocks
