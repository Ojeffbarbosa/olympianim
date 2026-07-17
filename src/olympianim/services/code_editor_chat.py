"""Persistent LangGraph chat for proposing safe Manim code edits."""

from __future__ import annotations

import hashlib
import sqlite3
from collections.abc import Callable, Mapping
from dataclasses import dataclass
from typing import Annotated, Any, Literal, NotRequired
from uuid import uuid4

from langchain.agents import create_agent
from langchain.agents.middleware import (
    AgentState,
    SummarizationMiddleware,
    ToolCallLimitMiddleware,
    ToolRetryMiddleware,
)
from langchain.agents.structured_output import ToolStrategy
from langchain_core.callbacks import UsageMetadataCallbackHandler
from langchain_core.messages import AIMessage, BaseMessage, HumanMessage, RemoveMessage
from langchain_core.runnables import RunnableConfig
from langgraph.channels import UntrackedValue
from langgraph.checkpoint.serde.jsonplus import JsonPlusSerializer
from langgraph.checkpoint.sqlite import SqliteSaver
from langgraph.graph.message import REMOVE_ALL_MESSAGES
from pydantic import BaseModel, Field, ValidationError

from olympianim.config import MANIM_REFERENCE_TOOL_CALL_LIMIT
from olympianim.database.repository import ProjectRepository
from olympianim.graph.approved_workflow import (
    extract_python_code,
    voiceover_prompt_requirements,
)
from olympianim.manim.presentation import (
    check_generated_code_safety,
    prepare_voiceover_code,
)
from olympianim.prompts.service import PromptService
from olympianim.prompts.validator import render_prompt_template
from olympianim.providers.llm import get_provider
from olympianim.schemas.render import VoiceConfig
from olympianim.services.artifact_service import ArtifactService
from olympianim.services.image_asset_service import ImageAssetService
from olympianim.services.llm_service import LLMImage, build_user_content
from olympianim.services.project_logging import ProjectLogger, ProjectToolCallback
from olympianim.services.usage_service import UsageContext, UsageService
from olympianim.tools import search_manim_reference
from olympianim.utils.logging import redact

VideoMode = Literal["presentation", "solution"]
InteractionMode = Literal["conversation", "edit"]
_EDITOR_PROMPT_NAME = "Editor Manim com IA - padrão"
_CONVERSATION_PROMPT_NAME = "Conversa sobre código Manim - padrão"
_MAX_SAFETY_REPAIRS = 2
_INTERACTION_KEY = "olympianim_interaction"
_VALIDATED_EDIT_SUMMARY_KEY = "olympianim_validated_edit_summary"
_STRUCTURED_EDIT_RETRY_MESSAGE = (
    "A proposta estruturada está incompleta. Chame novamente a ferramenta de resposta "
    "incluindo changed, summary e o arquivo Python completo no campo code."
)


class _StructuredEdit(BaseModel):
    """Validated response produced by the native LangChain agent."""

    changed: bool = Field(
        default=True,
        description="False when the user did not request a code change.",
    )
    summary: str = Field(min_length=1, max_length=500)
    code: str = Field(min_length=1)


class _EditorAgentState(AgentState[_StructuredEdit]):
    """Persistent messages with a response scoped to one graph invocation."""

    structured_response: NotRequired[Annotated[_StructuredEdit, UntrackedValue]]


@dataclass(frozen=True)
class CodeEditProposal:
    """One previewable edit that has not been applied or persisted as code."""

    summary: str
    code: str
    base_code_hash: str
    valid: bool
    changed: bool = True
    errors: tuple[str, ...] = ()


@dataclass(frozen=True)
class ChatMessage:
    """One user-visible message recovered from the checkpoint thread."""

    role: Literal["user", "assistant"]
    content: str


@dataclass(frozen=True)
class ChatReply:
    """Markdown answer from the non-mutating conversation mode."""

    content: str


class CodeEditorChatService:
    """Own AI edit proposals and project/mode-scoped conversational memory."""

    def __init__(
        self,
        repository: ProjectRepository | None = None,
        *,
        agent_factory: Callable[..., Any] = create_agent,
    ) -> None:
        self.repository = repository or ProjectRepository()
        self.prompt_service = PromptService(repository=self.repository)
        self.usage_service = UsageService(self.repository)
        self.artifacts = ArtifactService(repository=self.repository)
        self.agent_factory = agent_factory

    def propose_edit(
        self,
        project_id: str,
        mode: VideoMode,
        request: str,
        current_code: str,
        *,
        api_key: str,
        provider: str | None = None,
        model_name: str | None = None,
        images: tuple[LLMImage, ...] = (),
    ) -> CodeEditProposal:
        """Ask the native agent for an edit without changing project artifacts."""
        project = self._project(project_id)
        selected_provider = provider or project.llm_provider
        selected_model = model_name or project.llm_model
        instruction = request.strip()
        if not instruction and images:
            instruction = (
                "Analise as imagens anexadas e proponha as alterações correspondentes "
                "no código Manim atual."
            )
        if not instruction:
            raise ValueError("Descreva a alteração desejada.")
        if not current_code.strip():
            raise ValueError("Não há código Manim para editar.")
        if not api_key:
            raise ValueError("A chave de API do provedor de IA não está disponível.")

        prompt = next(
            item
            for item in self.prompt_service.list_prompts("code_editor_agent")
            if item.prompt.name == _EDITOR_PROMPT_NAME
        )
        voice = VoiceConfig(
            enabled=project.voiceover_enabled,
            provider=project.voice_provider,
            model=project.voice_model,
            voice=project.voice,
            language=project.voice_language,
            speed=project.voice_speed,
        )
        system_prompt = render_prompt_template(
            prompt.latest_version.template_text,
            {
                "manim_code": current_code,
                "video_mode": "apresentação" if mode == "presentation" else "resolução",
                "voiceover_requirements": voiceover_prompt_requirements(voice),
            },
        )
        self.repository.record_project_prompt(
            project_id,
            agent_type=prompt.prompt.agent_type,
            prompt_id=prompt.prompt.id,
            prompt_version=prompt.latest_version.version,
            rendered_prompt_snapshot=system_prompt,
        )

        logger = ProjectLogger(self.repository, project_id, secrets=(api_key,))
        usage_context = UsageContext(
            project_id=project_id,
            execution_id=f"editor-chat-{uuid4().hex}",
            call_key=f"editor-chat:{mode}:{uuid4().hex}",
            agent_type="code_editor",
            stage=mode,
        )
        usage_callback = self.usage_service.callback()
        provider_adapter = get_provider(selected_provider, api_key)
        model = provider_adapter.create_chat_model(
            model=selected_model,
            temperature=0.0,
            max_tokens=provider_adapter.code_generation_max_tokens(selected_model),
        )
        callbacks = [
            usage_callback,
            ProjectToolCallback(logger, role="code_editor", mode=mode),
        ]
        logger.info(f"agent.code_editor.{mode}", "Proposta de edição iniciada.")

        def build_agent(checkpointer: SqliteSaver) -> Any:
            return self.agent_factory(
                model=model,
                tools=[search_manim_reference],
                system_prompt=system_prompt,
                middleware=(
                    SummarizationMiddleware(
                        model=model,
                        trigger=("tokens", 24_000),
                        keep=("messages", 6),
                    ),
                    ToolCallLimitMiddleware(
                        tool_name=search_manim_reference.name,
                        run_limit=MANIM_REFERENCE_TOOL_CALL_LIMIT,
                        exit_behavior="continue",
                    ),
                    ToolRetryMiddleware(
                        max_retries=2,
                        tools=[search_manim_reference.name],
                        on_failure="continue",
                    ),
                ),
                response_format=ToolStrategy(
                    _StructuredEdit,
                    handle_errors=_STRUCTURED_EDIT_RETRY_MESSAGE,
                ),
                state_schema=_EditorAgentState,
                checkpointer=checkpointer,
                name="olympianim_code_editor",
            )

        def invoke_agent(message: HumanMessage, *, normalize_history: bool = False) -> object:
            with self._checkpointer() as (checkpointer, _connection):
                agent = build_agent(checkpointer)
                if normalize_history:
                    self._normalize_checkpoint_history(
                        agent,
                        checkpointer,
                        project_id,
                        mode,
                    )
                return agent.invoke(
                    {"messages": [message]},
                    config=self._config(project_id, mode, callbacks=callbacks),
                )

        try:
            result = invoke_agent(
                HumanMessage(
                    content=build_user_content(instruction, images),
                    additional_kwargs={_INTERACTION_KEY: "edit"},
                ),
                normalize_history=True,
            )
            structured = self._require_structured_edit(result)
            proposed_code = structured.code
            code, errors = self._check_proposal_safety(
                project_id,
                mode,
                proposed_code,
                require_voiceover=voice.enabled,
            )
            repair_count = 0
            while errors and repair_count < _MAX_SAFETY_REPAIRS:
                repair_count += 1
                logger.info(
                    f"agent.code_editor.{mode}",
                    f"Correção automática da proposta {repair_count}/{_MAX_SAFETY_REPAIRS}.",
                )
                result = invoke_agent(
                    HumanMessage(
                        content=self._repair_instruction(errors),
                        additional_kwargs={
                            "olympianim_internal": True,
                        },
                    )
                )
                structured = self._require_structured_edit(result)
                proposed_code = structured.code
                code, errors = self._check_proposal_safety(
                    project_id,
                    mode,
                    proposed_code,
                    require_voiceover=voice.enabled,
                )
            with self._checkpointer() as (checkpointer, _connection):
                agent = build_agent(checkpointer)
                agent.update_state(
                    self._thread_config(project_id, mode),
                    {
                        "messages": [
                            AIMessage(
                                content=structured.summary,
                                additional_kwargs={_VALIDATED_EDIT_SUMMARY_KEY: True},
                            )
                        ]
                    },
                )
        except Exception as exc:
            self._record_usage(
                usage_context,
                usage_callback,
                provider=selected_provider,
                model=selected_model,
                completed=False,
            )
            message = redact(str(exc), [api_key])
            logger.error(f"agent.code_editor.{mode}", message)
            raise RuntimeError(f"Falha ao editar o código com IA: {message}") from exc

        self._record_usage(
            usage_context,
            usage_callback,
            provider=selected_provider,
            model=selected_model,
            completed=True,
        )
        logger.info(
            f"agent.code_editor.{mode}",
            (
                "Proposta concluída."
                if not errors
                else "Proposta bloqueada pela proteção de execução."
            ),
        )
        normalized_current = extract_python_code(current_code)
        changed = self.code_hash(code) != self.code_hash(normalized_current)
        return CodeEditProposal(
            summary=structured.summary.strip(),
            code=code,
            base_code_hash=self.code_hash(current_code),
            changed=changed,
            valid=not errors,
            errors=errors,
        )

    def discuss_code(
        self,
        project_id: str,
        mode: VideoMode,
        request: str,
        current_code: str,
        *,
        api_key: str,
        provider: str | None = None,
        model_name: str | None = None,
        images: tuple[LLMImage, ...] = (),
    ) -> ChatReply:
        """Discuss the current code without producing an applicable edit."""
        project = self._project(project_id)
        selected_provider = provider or project.llm_provider
        selected_model = model_name or project.llm_model
        instruction = request.strip()
        if not instruction and images:
            instruction = "Analise as imagens anexadas em relação ao código Manim atual."
        if not instruction:
            raise ValueError("Digite uma pergunta sobre o código.")
        if not current_code.strip():
            raise ValueError("Não há código Manim para analisar.")
        if not api_key:
            raise ValueError("A chave de API do provedor de IA não está disponível.")

        prompt = next(
            item
            for item in self.prompt_service.list_prompts("code_editor_agent")
            if item.prompt.name == _CONVERSATION_PROMPT_NAME
        )
        voice = VoiceConfig(
            enabled=project.voiceover_enabled,
            provider=project.voice_provider,
            model=project.voice_model,
            voice=project.voice,
            language=project.voice_language,
            speed=project.voice_speed,
        )
        system_prompt = render_prompt_template(
            prompt.latest_version.template_text,
            {
                "manim_code": current_code,
                "video_mode": "apresentação" if mode == "presentation" else "resolução",
                "voiceover_requirements": voiceover_prompt_requirements(voice),
            },
        )
        self.repository.record_project_prompt(
            project_id,
            agent_type=prompt.prompt.agent_type,
            prompt_id=prompt.prompt.id,
            prompt_version=prompt.latest_version.version,
            rendered_prompt_snapshot=system_prompt,
        )

        logger = ProjectLogger(self.repository, project_id, secrets=(api_key,))
        usage_context = UsageContext(
            project_id=project_id,
            execution_id=f"editor-conversation-{uuid4().hex}",
            call_key=f"editor-conversation:{mode}:{uuid4().hex}",
            agent_type="code_consultant",
            stage=mode,
        )
        usage_callback = self.usage_service.callback()
        model = get_provider(selected_provider, api_key).create_chat_model(
            model=selected_model,
            temperature=0.1,
        )
        callbacks = [
            usage_callback,
            ProjectToolCallback(logger, role="code_consultant", mode=mode),
        ]
        logger.info(f"agent.code_consultant.{mode}", "Consulta sobre código iniciada.")
        try:
            with self._checkpointer() as (checkpointer, _connection):
                agent = self.agent_factory(
                    model=model,
                    tools=[search_manim_reference],
                    system_prompt=system_prompt,
                    middleware=(
                        SummarizationMiddleware(
                            model=model,
                            trigger=("tokens", 24_000),
                            keep=("messages", 6),
                        ),
                        ToolCallLimitMiddleware(
                            tool_name=search_manim_reference.name,
                            run_limit=MANIM_REFERENCE_TOOL_CALL_LIMIT,
                            exit_behavior="continue",
                        ),
                        ToolRetryMiddleware(
                            max_retries=2,
                            tools=[search_manim_reference.name],
                            on_failure="continue",
                        ),
                    ),
                    response_format=None,
                    checkpointer=checkpointer,
                    name="olympianim_code_assistant",
                )
                self._normalize_checkpoint_history(
                    agent,
                    checkpointer,
                    project_id,
                    mode,
                )
                result = agent.invoke(
                    {
                        "messages": [
                            HumanMessage(
                                content=build_user_content(instruction, images),
                                additional_kwargs={_INTERACTION_KEY: "conversation"},
                            )
                        ]
                    },
                    config=self._config(project_id, mode, callbacks=callbacks),
                )
                content = self._last_assistant_text(result)
                if not content:
                    raise RuntimeError("O modelo não retornou uma resposta textual.")
        except Exception as exc:
            self._record_usage(
                usage_context,
                usage_callback,
                provider=selected_provider,
                model=selected_model,
                completed=False,
            )
            message = redact(str(exc), [api_key])
            logger.error(f"agent.code_consultant.{mode}", message)
            raise RuntimeError(f"Falha ao consultar o código com IA: {message}") from exc

        self._record_usage(
            usage_context,
            usage_callback,
            provider=selected_provider,
            model=selected_model,
            completed=True,
        )
        logger.info(f"agent.code_consultant.{mode}", "Consulta sobre código concluída.")
        return ChatReply(content=content)

    def _check_proposal_safety(
        self,
        project_id: str,
        mode: VideoMode,
        proposed_code: str,
        *,
        require_voiceover: bool,
    ) -> tuple[str, tuple[str, ...]]:
        code = prepare_voiceover_code(
            extract_python_code(proposed_code),
            require_voiceover=require_voiceover,
        )
        errors = check_generated_code_safety(
            code,
            require_voiceover=require_voiceover,
            allowed_image_paths=self._allowed_image_paths(project_id),
        )
        return code, tuple(errors)

    @staticmethod
    def _require_structured_edit(result: object) -> _StructuredEdit:
        """Accept only a complete response emitted by the current agent invocation."""
        if not isinstance(result, Mapping) or "structured_response" not in result:
            raise ValueError(
                "O modelo não retornou uma proposta de edição estruturada completa. "
                "O código atual foi preservado."
            )
        structured = result["structured_response"]
        if isinstance(structured, _StructuredEdit):
            return structured
        try:
            return _StructuredEdit.model_validate(structured)
        except ValidationError as exc:
            raise ValueError(
                "O modelo retornou uma proposta de edição incompleta ou inválida. "
                "O código atual foi preservado."
            ) from exc

    @staticmethod
    def _repair_instruction(errors: tuple[str, ...]) -> str:
        details = "\n".join(f"- {error}" for error in errors)
        return f"""Sua proposta anterior foi bloqueada pela proteção de execução:
{details}

Corrija esses erros e retorne novamente o resumo e o código Python completo.
Mantenha a alteração solicitada pelo professor e não modifique nenhuma outra
parte do código. Verifique especialmente aspas, barras invertidas, strings raw,
indentação e parênteses antes de responder."""

    def conversation(self, project_id: str, mode: VideoMode) -> tuple[ChatMessage, ...]:
        """Return user-visible turns from the native checkpoint state."""
        with self._checkpointer() as (checkpointer, _connection):
            checkpoint = checkpointer.get_tuple(self._thread_config(project_id, mode))
        if checkpoint is None:
            return ()
        messages = checkpoint.checkpoint.get("channel_values", {}).get("messages", ())
        visible: list[ChatMessage] = []
        interaction = ""
        for message in messages:
            content = self._message_text(getattr(message, "content", ""))
            if not content:
                continue
            if isinstance(message, HumanMessage):
                if self._is_internal_message(message, content):
                    continue
                interaction = str(message.additional_kwargs.get(_INTERACTION_KEY, ""))
                visible.append(ChatMessage("user", content.strip()))
            elif isinstance(message, AIMessage) and not getattr(message, "tool_calls", None):
                if interaction == "edit" and not message.additional_kwargs.get(
                    _VALIDATED_EDIT_SUMMARY_KEY
                ):
                    continue
                visible.append(ChatMessage("assistant", content.strip()))
        return tuple(visible)

    def clear_conversation(self, project_id: str, mode: VideoMode) -> None:
        """Delete only the selected editor chat thread."""
        with self._checkpointer() as (checkpointer, _connection):
            checkpointer.delete_thread(self.thread_id(project_id, mode))

    def _normalize_checkpoint_history(
        self,
        agent: Any,
        checkpointer: SqliteSaver,
        project_id: str,
        mode: VideoMode,
    ) -> None:
        """Replace completed provider traces with portable conversation messages."""
        config = self._thread_config(project_id, mode)
        checkpoint = checkpointer.get_tuple(config)
        if checkpoint is None:
            return
        messages = checkpoint.checkpoint.get("channel_values", {}).get("messages", ())
        if not messages:
            return
        agent.update_state(
            config,
            {
                "messages": [
                    RemoveMessage(id=REMOVE_ALL_MESSAGES),
                    *self._portable_history(messages),
                ]
            },
        )

    @classmethod
    def _portable_history(cls, messages: object) -> list[BaseMessage]:
        """Keep user context and final answers, excluding provider-private traces."""
        if not isinstance(messages, (list, tuple)):
            return []
        portable: list[BaseMessage] = []
        interaction = ""
        for message in messages:
            content = getattr(message, "content", "")
            text = cls._message_text(content)
            if isinstance(message, HumanMessage):
                if not text or cls._is_internal_message(message, text):
                    continue
                interaction = str(message.additional_kwargs.get(_INTERACTION_KEY, ""))
                human_kwargs: dict[str, Any] = (
                    {_INTERACTION_KEY: interaction} if interaction else {}
                )
                portable.append(
                    HumanMessage(
                        content=cls._portable_user_content(content),
                        additional_kwargs=human_kwargs,
                    )
                )
                continue
            if not isinstance(message, AIMessage) or getattr(message, "tool_calls", None):
                continue
            if not text:
                continue
            validated_edit = bool(message.additional_kwargs.get(_VALIDATED_EDIT_SUMMARY_KEY))
            if interaction == "edit" and not validated_edit:
                continue
            ai_kwargs: dict[str, Any] = (
                {_VALIDATED_EDIT_SUMMARY_KEY: True} if validated_edit else {}
            )
            portable.append(AIMessage(content=text, additional_kwargs=ai_kwargs))
        return portable

    @staticmethod
    def _portable_user_content(content: object) -> str | list[str | dict[Any, Any]]:
        """Copy only provider-neutral text and image blocks from a user turn."""
        if isinstance(content, str):
            return content.strip()
        if not isinstance(content, list):
            return ""
        blocks: list[str | dict[Any, Any]] = []
        for block in content:
            if not isinstance(block, dict):
                continue
            block_type = block.get("type")
            if block_type == "text" and isinstance(block.get("text"), str):
                blocks.append({"type": "text", "text": block["text"]})
            elif (
                block_type == "image"
                and isinstance(block.get("base64"), str)
                and isinstance(block.get("mime_type"), str)
            ):
                blocks.append(
                    {
                        "type": "image",
                        "base64": block["base64"],
                        "mime_type": block["mime_type"],
                    }
                )
        return blocks

    @staticmethod
    def code_hash(code: str) -> str:
        return hashlib.sha256(code.encode("utf-8")).hexdigest()

    @staticmethod
    def thread_id(project_id: str, mode: VideoMode) -> str:
        return f"editor:{project_id}:{mode}"

    def _allowed_image_paths(self, project_id: str) -> set[str]:
        return {
            asset.manim_path
            for asset in ImageAssetService(
                repository=self.repository,
                projects_dir=self.artifacts.projects_dir,
            ).list_assets(project_id)
        }

    def _project(self, project_id: str) -> Any:
        project = self.repository.get_project(project_id)
        if project is None:
            raise ValueError("Projeto não encontrado.")
        return project

    def _checkpointer(self) -> _CheckpointerContext:
        return _CheckpointerContext(self.repository.database_path)

    def _record_usage(
        self,
        context: UsageContext,
        callback: UsageMetadataCallbackHandler,
        *,
        provider: str,
        model: str,
        completed: bool,
    ) -> None:
        self.usage_service.record_attempt(
            context,
            callback,
            provider=provider,
            model=model,
            completed=completed,
            attempt_type="primary",
            attempt_number=1,
        )

    @staticmethod
    def _message_text(content: object) -> str:
        """Extract only displayable text from text or multimodal messages."""
        if isinstance(content, str):
            return content.strip()
        if not isinstance(content, list):
            return ""
        parts = [
            str(block.get("text", "")).strip()
            for block in content
            if isinstance(block, dict) and block.get("type") == "text"
        ]
        return "\n".join(part for part in parts if part)

    @classmethod
    def _last_assistant_text(cls, result: object) -> str:
        if not isinstance(result, dict):
            return ""
        messages = result.get("messages", ())
        if not isinstance(messages, (list, tuple)):
            return ""
        for message in reversed(messages):
            if isinstance(message, AIMessage) and not getattr(message, "tool_calls", None):
                content = cls._message_text(getattr(message, "content", ""))
                if content:
                    return content
        return ""

    @staticmethod
    def _is_internal_message(message: HumanMessage, content: str) -> bool:
        """Hide safety retries, including checkpoints made before tagging."""
        internal_prefixes = (
            "Sua proposta anterior foi bloqueada pela proteção de execução:",
            "Sua proposta anterior foi rejeitada pela validação:",
            "Here is a summary of the conversation to date:",
        )
        return bool(message.additional_kwargs.get("olympianim_internal")) or content.startswith(
            internal_prefixes
        )

    @classmethod
    def _thread_config(cls, project_id: str, mode: VideoMode) -> RunnableConfig:
        return {"configurable": {"thread_id": cls.thread_id(project_id, mode)}}

    @classmethod
    def _config(
        cls,
        project_id: str,
        mode: VideoMode,
        *,
        callbacks: list[Any],
    ) -> RunnableConfig:
        return {**cls._thread_config(project_id, mode), "callbacks": callbacks}


class _CheckpointerContext:
    """Keep the SQLite connection alive for one saver operation."""

    def __init__(self, database_path: Any) -> None:
        self.database_path = database_path
        self.connection: sqlite3.Connection | None = None

    def __enter__(self) -> tuple[SqliteSaver, sqlite3.Connection]:
        self.connection = sqlite3.connect(
            self.database_path,
            timeout=30,
            check_same_thread=False,
        )
        self.connection.execute("PRAGMA busy_timeout = 30000")
        self.connection.execute("PRAGMA journal_mode = WAL")
        saver = SqliteSaver(
            self.connection,
            serde=JsonPlusSerializer(
                allowed_msgpack_modules=(
                    ("olympianim.services.code_editor_chat", "_StructuredEdit"),
                )
            ),
        )
        saver.setup()
        return saver, self.connection

    def __exit__(self, *_args: object) -> None:
        if self.connection is not None:
            self.connection.close()
