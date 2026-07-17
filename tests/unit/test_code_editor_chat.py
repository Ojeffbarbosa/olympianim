"""Tests for the persistent AI chat in the Manim editor."""

from __future__ import annotations

import sqlite3
from pathlib import Path

import pytest
from langchain.agents import create_agent
from langchain.agents.structured_output import ToolStrategy
from langchain_core.language_models.fake_chat_models import FakeMessagesListChatModel
from langchain_core.messages import AIMessage, HumanMessage, ToolMessage
from langgraph.checkpoint.sqlite import SqliteSaver
from langgraph.graph import END, START, MessagesState, StateGraph

from olympianim.database.models import ProjectCreate
from olympianim.database.repository import ProjectRepository
from olympianim.services.code_editor_chat import (
    CodeEditorChatService,
    _EditorAgentState,
    _StructuredEdit,
)
from olympianim.services.llm_service import LLMImage

_CODE = """from manim import *

class ExampleScene(Scene):
    def construct(self):
        self.add(Text("Original"))
"""

_EDITED_CODE = _CODE.replace("Original", "Editado")


class _FakeAgent:
    def invoke(self, state: dict[str, object], config: dict[str, object]) -> dict[str, object]:
        _ = (state, config)
        return {
            "structured_response": _StructuredEdit(
                summary="Texto principal atualizado.",
                code=_EDITED_CODE,
            )
        }

    def update_state(self, config: dict[str, object], values: dict[str, object]) -> None:
        _ = (config, values)


class _FakeProvider:
    def code_generation_max_tokens(self, _model: str) -> int:
        return 128_000

    def create_chat_model(self, **kwargs: object) -> object:
        _ = kwargs
        return _FakeModel()


class _FakeModel:
    _llm_type = "fake-chat"


def _service(tmp_path: Path) -> tuple[CodeEditorChatService, ProjectRepository, str]:
    repository = ProjectRepository(tmp_path / "olympianim.sqlite3")
    project = repository.create_project(
        ProjectCreate(
            title="Editor",
            problem_statement="Problema",
            llm_provider="OpenAI",
            llm_model="test-model",
        )
    )
    return (
        CodeEditorChatService(
            repository,
            agent_factory=lambda **_kwargs: _FakeAgent(),
        ),
        repository,
        project.id,
    )


def test_proposal_is_preview_only_and_records_usage(tmp_path, monkeypatch) -> None:
    service, repository, project_id = _service(tmp_path)
    monkeypatch.setattr(
        "olympianim.services.code_editor_chat.get_provider",
        lambda *_args: _FakeProvider(),
    )

    proposal = service.propose_edit(
        project_id,
        "presentation",
        "Troque o texto principal.",
        _CODE,
        api_key="secret",
    )

    project = repository.get_project(project_id)
    assert project is not None
    assert proposal.valid is True
    assert proposal.code == _EDITED_CODE.strip()
    assert proposal.base_code_hash == service.code_hash(_CODE)
    assert project.presentation_code_path == ""
    assert repository.list_generated_files(project_id) == []
    usage = repository.list_ai_usage(project_id)
    assert usage[0].agent_type == "code_editor"


def test_edit_agent_uses_untracked_structured_response_state(tmp_path, monkeypatch) -> None:
    service, _, project_id = _service(tmp_path)
    captured: dict[str, object] = {}

    def agent_factory(**kwargs: object) -> _FakeAgent:
        captured.update(kwargs)
        return _FakeAgent()

    monkeypatch.setattr(
        "olympianim.services.code_editor_chat.get_provider",
        lambda *_args: _FakeProvider(),
    )
    service.agent_factory = agent_factory

    service.propose_edit(
        project_id,
        "presentation",
        "Troque o texto principal.",
        _CODE,
        api_key="secret",
    )

    assert captured["state_schema"] is _EditorAgentState
    response_format = captured["response_format"]
    assert response_format.schema is _StructuredEdit
    assert "arquivo Python completo" in response_format.handle_errors
    tool_limiter = captured["middleware"][1]
    assert tool_limiter.tool_name == "search_manim_reference"
    assert tool_limiter.run_limit == 10


def test_editor_uses_provider_output_ceiling_for_complete_code(tmp_path, monkeypatch) -> None:
    service, _, project_id = _service(tmp_path)
    captured: dict[str, object] = {}

    class CapturingProvider(_FakeProvider):
        def create_chat_model(self, **kwargs: object) -> object:
            captured.update(kwargs)
            return _FakeModel()

    monkeypatch.setattr(
        "olympianim.services.code_editor_chat.get_provider",
        lambda *_args: CapturingProvider(),
    )

    service.propose_edit(
        project_id,
        "presentation",
        "Troque o texto principal.",
        _CODE,
        api_key="secret",
        provider="Anthropic",
        model_name="claude-sonnet-5",
    )

    assert captured["max_tokens"] == 128_000


def test_untracked_response_survives_the_run_but_not_the_next_request(tmp_path) -> None:
    service, _, project_id = _service(tmp_path)
    response_seen_during_run: list[bool] = []

    def respond(state: _EditorAgentState) -> dict[str, object]:
        message = state["messages"][-1]
        update: dict[str, object] = {"messages": [AIMessage(content="resposta")]}
        if message.content == "primeira":
            update["structured_response"] = _StructuredEdit(
                summary="Primeira proposta.",
                code=_EDITED_CODE,
            )
        return update

    def inspect_response(state: _EditorAgentState) -> dict[str, object]:
        response_seen_during_run.append("structured_response" in state)
        return {"messages": [AIMessage(content="inspecionada")]}

    with service._checkpointer() as (checkpointer, _connection):
        builder = StateGraph(_EditorAgentState)
        builder.add_node("respond", respond)
        builder.add_node("inspect_response", inspect_response)
        builder.add_edge(START, "respond")
        builder.add_edge("respond", "inspect_response")
        builder.add_edge("inspect_response", END)
        graph = builder.compile(checkpointer=checkpointer)
        config = service._thread_config(project_id, "presentation")

        first = graph.invoke({"messages": [HumanMessage(content="primeira")]}, config)
        second = graph.invoke({"messages": [HumanMessage(content="segunda")]}, config)

    assert isinstance(first.get("structured_response"), _StructuredEdit)
    assert "structured_response" not in second
    assert response_seen_during_run == [True, False]


def test_native_agent_exits_after_valid_untracked_structured_response(tmp_path) -> None:
    class ToolCallingModel(FakeMessagesListChatModel):
        def bind_tools(self, *_args: object, **_kwargs: object) -> ToolCallingModel:
            return self

    model = ToolCallingModel(
        responses=[
            AIMessage(
                content="",
                tool_calls=[
                    {
                        "name": "_StructuredEdit",
                        "args": {
                            "changed": True,
                            "summary": "Proposta válida.",
                            "code": _EDITED_CODE,
                        },
                        "id": "structured-edit-1",
                        "type": "tool_call",
                    }
                ],
            ),
            AIMessage(
                content="",
                tool_calls=[
                    {
                        "name": "_StructuredEdit",
                        "args": {
                            "changed": False,
                            "summary": "Segunda proposta.",
                            "code": _CODE,
                        },
                        "id": "structured-edit-2",
                        "type": "tool_call",
                    }
                ],
            ),
        ]
    )
    database_path = tmp_path / "agent.sqlite3"

    def invoke(message: dict[str, list[HumanMessage]]):
        connection = sqlite3.connect(database_path, check_same_thread=False)
        saver = SqliteSaver(connection)
        saver.setup()
        return (
            create_agent(
                model=model,
                tools=[],
                response_format=ToolStrategy(_StructuredEdit),
                state_schema=_EditorAgentState,
                checkpointer=saver,
            ).invoke(message, config=config),
            connection,
        )

    config = {"configurable": {"thread_id": "editor-test"}, "recursion_limit": 5}

    first, first_connection = invoke({"messages": [HumanMessage(content="Primeira edição.")]})
    first_connection.close()
    second, second_connection = invoke({"messages": [HumanMessage(content="Segunda edição.")]})
    second_connection.close()

    assert isinstance(first.get("structured_response"), _StructuredEdit)
    assert first["structured_response"].summary == "Proposta válida."
    assert isinstance(second.get("structured_response"), _StructuredEdit)
    assert second["structured_response"].summary == "Segunda proposta."


def test_portable_history_removes_provider_private_reasoning_and_tool_trace() -> None:
    user_content = [
        {"type": "text", "text": "Use esta imagem."},
        {"type": "text", "text": "Imagem 1: exemplo.png"},
        {"type": "image", "base64": "aW1hZ2U=", "mime_type": "image/png"},
    ]
    history = [
        HumanMessage(
            content=user_content,
            additional_kwargs={"olympianim_interaction": "edit"},
        ),
        AIMessage(
            content=[
                {
                    "type": "reasoning",
                    "content": [],
                    "encrypted_content": "provider-private",
                }
            ],
            tool_calls=[
                {
                    "name": "search_manim_reference",
                    "args": {"query": "Transform"},
                    "id": "call-1",
                    "type": "tool_call",
                }
            ],
            response_metadata={"provider": "OpenAI"},
        ),
        ToolMessage(content="Referência encontrada.", tool_call_id="call-1"),
        AIMessage(
            content="Transformação corrigida.",
            additional_kwargs={"olympianim_validated_edit_summary": True},
            response_metadata={"provider": "OpenAI"},
            usage_metadata={"input_tokens": 10, "output_tokens": 5, "total_tokens": 15},
        ),
    ]

    portable = CodeEditorChatService._portable_history(history)

    assert len(portable) == 2
    assert isinstance(portable[0], HumanMessage)
    assert portable[0].content == user_content
    assert portable[0].additional_kwargs == {"olympianim_interaction": "edit"}
    assert isinstance(portable[1], AIMessage)
    assert portable[1].content == "Transformação corrigida."
    assert portable[1].tool_calls == []
    assert portable[1].response_metadata == {}
    assert portable[1].usage_metadata is None
    assert portable[1].additional_kwargs == {"olympianim_validated_edit_summary": True}


def test_checkpoint_is_made_provider_neutral_before_next_request(tmp_path) -> None:
    service, repository, project_id = _service(tmp_path)
    connection = sqlite3.connect(repository.database_path, check_same_thread=False)
    saver = SqliteSaver(connection)
    saver.setup()

    def previous_provider(_state: MessagesState) -> dict[str, list[object]]:
        return {
            "messages": [
                AIMessage(
                    content=[
                        {
                            "type": "reasoning",
                            "content": [],
                            "encrypted_content": "openai-only",
                        }
                    ],
                    tool_calls=[
                        {
                            "name": "search_manim_reference",
                            "args": {"query": "Transform"},
                            "id": "call-1",
                            "type": "tool_call",
                        }
                    ],
                ),
                ToolMessage(content="Resultado interno.", tool_call_id="call-1"),
                AIMessage(
                    content="Edição anterior concluída.",
                    additional_kwargs={"olympianim_validated_edit_summary": True},
                ),
            ]
        }

    builder = StateGraph(MessagesState)
    builder.add_node("previous_provider", previous_provider)
    builder.add_edge(START, "previous_provider")
    builder.add_edge("previous_provider", END)
    graph = builder.compile(checkpointer=saver)
    config = service._thread_config(project_id, "solution")
    graph.invoke(
        {
            "messages": [
                HumanMessage(
                    content="Corrija a transformação.",
                    additional_kwargs={"olympianim_interaction": "edit"},
                )
            ]
        },
        config,
    )

    service._normalize_checkpoint_history(
        graph,
        saver,
        project_id,
        "solution",
    )
    checkpoint = saver.get_tuple(config)
    connection.close()

    assert checkpoint is not None
    messages = checkpoint.checkpoint["channel_values"]["messages"]
    assert len(messages) == 2
    assert isinstance(messages[0], HumanMessage)
    assert messages[0].content == "Corrija a transformação."
    assert isinstance(messages[1], AIMessage)
    assert messages[1].content == "Edição anterior concluída."
    assert messages[1].tool_calls == []


def test_missing_current_structured_response_fails_instead_of_reusing_code(
    tmp_path,
    monkeypatch,
) -> None:
    service, repository, project_id = _service(tmp_path)
    monkeypatch.setattr(
        "olympianim.services.code_editor_chat.get_provider",
        lambda *_args: _FakeProvider(),
    )

    class MissingResponseAgent(_FakeAgent):
        def invoke(
            self,
            state: dict[str, object],
            config: dict[str, object],
        ) -> dict[str, object]:
            _ = (state, config)
            return {"messages": [AIMessage(content="Resumo sem código.")]}

    service.agent_factory = lambda **_kwargs: MissingResponseAgent()

    with pytest.raises(RuntimeError, match="proposta de edição estruturada completa"):
        service.propose_edit(
            project_id,
            "presentation",
            "Faça outra alteração.",
            _EDITED_CODE,
            api_key="secret",
        )

    assert repository.list_generated_files(project_id) == []
    usage = repository.list_ai_usage(project_id)
    assert usage[0].status == "failed"


def test_conversation_returns_markdown_without_creating_artifacts(
    tmp_path,
    monkeypatch,
) -> None:
    service, repository, project_id = _service(tmp_path)
    captured: dict[str, object] = {}

    class ConversationAgent:
        def invoke(
            self,
            state: dict[str, object],
            config: dict[str, object],
        ) -> dict[str, object]:
            captured["state"] = state
            captured["config"] = config
            return {
                "messages": [AIMessage(content="Use este trecho:\n```python\nText('Olá')\n```")]
            }

    def agent_factory(**kwargs: object) -> ConversationAgent:
        captured["agent_kwargs"] = kwargs
        return ConversationAgent()

    monkeypatch.setattr(
        "olympianim.services.code_editor_chat.get_provider",
        lambda *_args: _FakeProvider(),
    )
    service.agent_factory = agent_factory

    reply = service.discuss_code(
        project_id,
        "presentation",
        "Como posso melhorar o texto?",
        _CODE,
        api_key="secret",
    )

    assert "```python" in reply.content
    assert repository.list_generated_files(project_id) == []
    assert captured["agent_kwargs"]["response_format"] is None
    tool_limiter = captured["agent_kwargs"]["middleware"][1]
    assert tool_limiter.run_limit == 10
    usage = repository.list_ai_usage(project_id)
    assert usage[0].agent_type == "code_consultant"
    assert usage[0].stage == "presentation"


def test_invalid_proposal_cannot_be_applied(tmp_path, monkeypatch) -> None:
    service, _, project_id = _service(tmp_path)
    monkeypatch.setattr(
        "olympianim.services.code_editor_chat.get_provider",
        lambda *_args: _FakeProvider(),
    )

    class InvalidAgent(_FakeAgent):
        def invoke(self, state: dict[str, object], config: dict[str, object]) -> dict[str, object]:
            _ = (state, config)
            return {
                "structured_response": _StructuredEdit(
                    summary="Código inválido.",
                    code="print('sem cena')",
                )
            }

    service.agent_factory = lambda **_kwargs: InvalidAgent()
    proposal = service.propose_edit(
        project_id,
        "presentation",
        "Altere tudo.",
        _CODE,
        api_key="secret",
    )

    assert proposal.valid is False
    assert proposal.errors


def test_invalid_syntax_is_repaired_before_proposal_is_returned(
    tmp_path,
    monkeypatch,
) -> None:
    service, _, project_id = _service(tmp_path)
    monkeypatch.setattr(
        "olympianim.services.code_editor_chat.get_provider",
        lambda *_args: _FakeProvider(),
    )

    class RepairingAgent(_FakeAgent):
        calls = 0

        def invoke(
            self,
            state: dict[str, object],
            config: dict[str, object],
        ) -> dict[str, object]:
            _ = (state, config)
            self.calls += 1
            code = 'from manim import *\nclass Broken(Scene):\n    value = "aberto'
            if self.calls > 1:
                code = _EDITED_CODE
            return {
                "structured_response": _StructuredEdit(
                    summary="Código corrigido.",
                    code=code,
                )
            }

    agent = RepairingAgent()
    build_count = 0

    def agent_factory(**_kwargs: object) -> RepairingAgent:
        nonlocal build_count
        build_count += 1
        return agent

    service.agent_factory = agent_factory

    proposal = service.propose_edit(
        project_id,
        "presentation",
        "Edite o texto.",
        _CODE,
        api_key="secret",
    )

    assert agent.calls == 2
    assert build_count == 3
    assert proposal.valid is True
    assert proposal.code == _EDITED_CODE.strip()


def test_proposal_uses_selected_model_and_multimodal_images(
    tmp_path,
    monkeypatch,
) -> None:
    service, repository, project_id = _service(tmp_path)
    captured: dict[str, object] = {}

    class CapturingAgent(_FakeAgent):
        def invoke(
            self,
            state: dict[str, object],
            config: dict[str, object],
        ) -> dict[str, object]:
            captured["state"] = state
            return super().invoke(state, config)

    def provider_factory(provider: str, api_key: str) -> _FakeProvider:
        captured["provider"] = provider
        captured["api_key"] = api_key
        return _FakeProvider()

    monkeypatch.setattr(
        "olympianim.services.code_editor_chat.get_provider",
        provider_factory,
    )
    service.agent_factory = lambda **_kwargs: CapturingAgent()

    service.propose_edit(
        project_id,
        "presentation",
        "Use a imagem como referência.",
        _CODE,
        api_key="google-key",
        provider="Google",
        model_name="gemini-test",
        images=(LLMImage(data=b"image", mime_type="image/png", label="Imagem 1: quadro.png"),),
    )

    assert captured["provider"] == "Google"
    assert captured["api_key"] == "google-key"
    message = captured["state"]["messages"][0]
    assert message.content[0] == {
        "type": "text",
        "text": "Use a imagem como referência.",
    }
    assert message.content[1] == {"type": "text", "text": "Imagem 1: quadro.png"}
    assert message.content[2]["type"] == "image"
    usage = repository.list_ai_usage(project_id)
    assert usage[0].provider == "Google"
    assert usage[0].model == "gemini-test"


def test_image_only_request_receives_a_clear_instruction(tmp_path, monkeypatch) -> None:
    service, _, project_id = _service(tmp_path)
    captured: dict[str, object] = {}

    class CapturingAgent(_FakeAgent):
        def invoke(
            self,
            state: dict[str, object],
            config: dict[str, object],
        ) -> dict[str, object]:
            captured["state"] = state
            return super().invoke(state, config)

    monkeypatch.setattr(
        "olympianim.services.code_editor_chat.get_provider",
        lambda *_args: _FakeProvider(),
    )
    service.agent_factory = lambda **_kwargs: CapturingAgent()

    proposal = service.propose_edit(
        project_id,
        "presentation",
        "",
        _CODE,
        api_key="secret",
        images=(LLMImage(data=b"image", mime_type="image/png"),),
    )

    message = captured["state"]["messages"][0]
    assert "Analise as imagens anexadas" in message.content[0]["text"]
    assert proposal.valid is True


def test_code_diff_is_authoritative_when_model_marks_response_unchanged(
    tmp_path,
    monkeypatch,
) -> None:
    service, _, project_id = _service(tmp_path)
    monkeypatch.setattr(
        "olympianim.services.code_editor_chat.get_provider",
        lambda *_args: _FakeProvider(),
    )

    class NoChangeAgent(_FakeAgent):
        def invoke(
            self,
            state: dict[str, object],
            config: dict[str, object],
        ) -> dict[str, object]:
            _ = (state, config)
            return {
                "structured_response": _StructuredEdit(
                    changed=False,
                    summary="Sem alteração solicitada.",
                    code=_EDITED_CODE,
                )
            }

    service.agent_factory = lambda **_kwargs: NoChangeAgent()

    proposal = service.propose_edit(
        project_id,
        "presentation",
        "oi",
        _CODE,
        api_key="secret",
    )

    assert proposal.changed is True
    assert proposal.code == _EDITED_CODE.strip()
    assert proposal.valid is True


def test_identical_code_is_reported_as_unchanged(tmp_path, monkeypatch) -> None:
    service, _, project_id = _service(tmp_path)
    monkeypatch.setattr(
        "olympianim.services.code_editor_chat.get_provider",
        lambda *_args: _FakeProvider(),
    )

    class SameCodeAgent(_FakeAgent):
        def invoke(
            self,
            state: dict[str, object],
            config: dict[str, object],
        ) -> dict[str, object]:
            _ = (state, config)
            return {
                "structured_response": _StructuredEdit(
                    changed=True,
                    summary="Nenhuma diferença necessária.",
                    code=_CODE,
                )
            }

    service.agent_factory = lambda **_kwargs: SameCodeAgent()
    proposal = service.propose_edit(
        project_id,
        "presentation",
        "Revise o texto.",
        _CODE,
        api_key="secret",
    )

    assert proposal.changed is False
    assert proposal.code == _CODE.strip()
    assert proposal.valid is True


def test_checkpoint_memory_is_isolated_and_can_be_cleared(tmp_path) -> None:
    service, repository, project_id = _service(tmp_path)
    connection = sqlite3.connect(repository.database_path, check_same_thread=False)
    saver = SqliteSaver(connection)
    saver.setup()

    def reply(state: MessagesState) -> dict[str, list[AIMessage]]:
        return {"messages": [AIMessage(content=f"Resposta a {state['messages'][-1].content}")]}

    builder = StateGraph(MessagesState)
    builder.add_node("reply", reply)
    builder.add_edge(START, "reply")
    builder.add_edge("reply", END)
    graph = builder.compile(checkpointer=saver)
    graph.invoke(
        {"messages": [HumanMessage(content="apresentação")]},
        {"configurable": {"thread_id": service.thread_id(project_id, "presentation")}},
    )
    graph.invoke(
        {"messages": [HumanMessage(content="resolução")]},
        {"configurable": {"thread_id": service.thread_id(project_id, "solution")}},
    )
    connection.close()

    presentation = service.conversation(project_id, "presentation")
    solution = service.conversation(project_id, "solution")
    assert [message.content for message in presentation] == [
        "apresentação",
        "Resposta a apresentação",
    ]
    assert [message.content for message in solution] == [
        "resolução",
        "Resposta a resolução",
    ]

    service.clear_conversation(project_id, "presentation")
    assert service.conversation(project_id, "presentation") == ()
    assert service.conversation(project_id, "solution") == solution


def test_conversation_shows_only_validated_summary_for_edit_turn(tmp_path) -> None:
    service, repository, project_id = _service(tmp_path)
    connection = sqlite3.connect(repository.database_path, check_same_thread=False)
    saver = SqliteSaver(connection)
    saver.setup()

    def reply(_state: MessagesState) -> dict[str, list[AIMessage]]:
        return {
            "messages": [
                AIMessage(content="Resposta incompleta sem código."),
                AIMessage(
                    content="Proposta validada.",
                    additional_kwargs={"olympianim_validated_edit_summary": True},
                ),
            ]
        }

    builder = StateGraph(MessagesState)
    builder.add_node("reply", reply)
    builder.add_edge(START, "reply")
    builder.add_edge("reply", END)
    graph = builder.compile(checkpointer=saver)
    graph.invoke(
        {
            "messages": [
                HumanMessage(
                    content="Altere a cena.",
                    additional_kwargs={"olympianim_interaction": "edit"},
                )
            ]
        },
        {"configurable": {"thread_id": service.thread_id(project_id, "presentation")}},
    )
    connection.close()

    conversation = service.conversation(project_id, "presentation")

    assert [(message.role, message.content) for message in conversation] == [
        ("user", "Altere a cena."),
        ("assistant", "Proposta validada."),
    ]


def test_thread_and_hash_are_stable_per_mode() -> None:
    assert CodeEditorChatService.thread_id("p1", "presentation") == "editor:p1:presentation"
    assert CodeEditorChatService.thread_id("p1", "solution") == "editor:p1:solution"
    assert CodeEditorChatService.code_hash(_CODE) == CodeEditorChatService.code_hash(_CODE)


def test_validation_retry_messages_are_internal() -> None:
    legacy = HumanMessage(content="Sua proposta anterior foi rejeitada pela validação:\n- erro")
    tagged = HumanMessage(
        content="Corrija o código.",
        additional_kwargs={"olympianim_internal": True},
    )
    regular = HumanMessage(content="Aumente o texto.")

    assert CodeEditorChatService._is_internal_message(legacy, legacy.content)
    assert CodeEditorChatService._is_internal_message(tagged, tagged.content)
    assert not CodeEditorChatService._is_internal_message(regular, regular.content)


def test_summarization_middleware_messages_are_internal() -> None:
    summary = HumanMessage(
        content=(
            "Here is a summary of the conversation to date:\n\n"
            "Previous conversation was too long to summarize."
        )
    )

    assert CodeEditorChatService._is_internal_message(summary, summary.content)
