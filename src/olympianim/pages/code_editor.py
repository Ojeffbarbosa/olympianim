"""Post-generation Manim editor with immutable version history."""

from __future__ import annotations

import difflib
from pathlib import Path
from typing import Literal

import streamlit as st
from code_editor import code_editor

from olympianim.database.models import ProjectRecord
from olympianim.schemas.render import VoiceConfig
from olympianim.services.code_assistant_preferences import (
    CodeAssistantPreferencesService,
)
from olympianim.services.code_editor_chat import (
    CodeEditorChatService,
    CodeEditProposal,
    InteractionMode,
)
from olympianim.services.code_editor_service import CodeEditorService
from olympianim.services.llm_service import LLMImage
from olympianim.services.project_service import ProjectService
from olympianim.ui import options, state
from olympianim.ui.credentials import (
    get_credential_store,
    key_help_text,
    key_source_badge,
    resolve_voice_key,
    voice_key_hint_text,
)

VideoMode = Literal["presentation", "solution"]


def _render_mode(
    editor_service: CodeEditorService,
    chat_service: CodeEditorChatService,
    project: ProjectRecord,
    mode: VideoMode,
    voice_api_key: str,
    voice_config: VoiceConfig,
) -> None:
    label = "apresentação" if mode == "presentation" else "resolução"
    code = editor_service.current_code(project.id, mode)
    if not code:
        st.info(f"Ainda não há código de {label} neste projeto.")
        return

    action_bar = st.container(
        border=True,
        horizontal=True,
        horizontal_alignment="right",
        vertical_alignment="center",
    )
    versions = editor_service.list_versions(project.id, mode)
    selected_version = None
    with action_bar:
        if versions:
            st.caption("Versão")
            selected_version = st.selectbox(
                "Versão do código",
                versions,
                format_func=lambda item: f"v{item.version}",
                key=(f"history_{project.id}_{mode}_latest_{versions[0].version}"),
                label_visibility="collapsed",
                width=110,
            )

    if selected_version is not None:
        if selected_version.video_path and Path(selected_version.video_path).is_file():
            with st.expander(
                f"Prévia da v{selected_version.version}",
                expanded=False,
            ):
                st.video(selected_version.video_path)
        else:
            st.info("Esta versão não tem vídeo disponível.")

    selected_code = (
        Path(selected_version.code_path).read_text(encoding="utf-8")
        if selected_version is not None
        else code
    )

    draft_key = f"editor_draft_{project.id}_{mode}"
    source_key = f"editor_source_{project.id}_{mode}"
    revision_key = f"editor_revision_{project.id}_{mode}"
    event_key = f"editor_event_{project.id}_{mode}"
    source_hash = chat_service.code_hash(selected_code)
    persisted_draft = editor_service.load_draft(project.id, mode)
    draft_matches_source = bool(
        persisted_draft is not None and persisted_draft.source_code_sha256 == source_hash
    )
    initial_code = (
        persisted_draft.code_content
        if persisted_draft is not None and draft_matches_source
        else selected_code
    )
    initial_hash = chat_service.code_hash(initial_code)
    if st.session_state.get(source_key) != initial_hash:
        st.session_state[draft_key] = initial_code
        st.session_state[source_key] = initial_hash
        st.session_state[revision_key] = int(st.session_state.get(revision_key, 0)) + 1

    if persisted_draft is not None:
        if draft_matches_source:
            st.caption("Rascunho aplicado e salvo para este vídeo.")
        else:
            st.warning(
                "Existe um rascunho baseado em outra versão. Volte à versão de origem "
                "ou descarte esse rascunho."
            )
        if st.button(
            "Descartar rascunho salvo",
            key=f"discard_saved_draft_{project.id}_{mode}",
            icon=":material/delete:",
        ):
            editor_service.discard_draft(project.id, mode)
            st.session_state[draft_key] = selected_code
            st.session_state[source_key] = source_hash
            st.session_state[revision_key] = int(st.session_state.get(revision_key, 0)) + 1
            st.rerun()

    editor_column, chat_column = st.columns([3, 2], gap="large")
    with editor_column:
        can_render = not (voice_config.enabled and not voice_api_key)
        editor_buttons = [
            {
                "name": "Salvar rascunho",
                "feather": "Save",
                "primary": False,
                "hasText": True,
                "alwaysOn": True,
                "commands": [["response", "save"]],
                "bindKey": {"win": "Ctrl-S", "mac": "Command-S"},
                "style": {"position": "static"},
            }
        ]
        if can_render:
            editor_buttons.append(
                {
                    "name": "Renderizar",
                    "feather": "Film",
                    "primary": True,
                    "hasText": True,
                    "alwaysOn": True,
                    "commands": [["response", "render"]],
                    "style": {"position": "static"},
                }
            )
        editor_menu = {
            "style": {
                "display": "flex",
                "justifyContent": "flex-end",
                "padding": "0.4rem",
            },
            "groups": [
                {
                    "name": "Ações",
                    "toggleOnlyOne": False,
                    "style": {
                        "display": "flex",
                        "justifyContent": "flex-end",
                        "gap": "0.5rem",
                        "width": "100%",
                    },
                    "buttons": editor_buttons,
                }
            ],
        }
        editor_response = code_editor(
            str(st.session_state.get(draft_key, selected_code)),
            lang="python",
            height="700px",
            response_mode="default",
            menu=editor_menu,
            props={
                "showGutter": True,
                "showPrintMargin": False,
            },
            options={
                "showLineNumbers": True,
                "displayIndentGuides": True,
                "wrap": False,
            },
            key=(f"manual_editor_{mode}_{project.id}_{st.session_state.get(revision_key, 0)}"),
        )
        response_type = str(editor_response.get("type") or "")
        response_id = str(editor_response.get("id") or "")
        response_text = editor_response.get("text")
        is_new_event = bool(response_id and st.session_state.get(event_key) != response_id)
        if is_new_event:
            st.session_state[event_key] = response_id
        action = response_type if is_new_event and response_type in {"save", "render"} else ""
        edited = (
            response_text
            if action and isinstance(response_text, str)
            else str(st.session_state.get(draft_key, selected_code))
        )
        if action:
            st.session_state[draft_key] = edited
            st.session_state[source_key] = chat_service.code_hash(edited)
        st.caption(
            "O código não é salvo enquanto você digita. Salve antes de usar a IA, "
            "trocar de versão ou sair da página."
        )
        if not can_render:
            st.info(
                "Informe a chave do provedor de voz para liberar a renderização. "
                "O rascunho ainda pode ser salvo."
            )
        if action == "save":
            try:
                saved_draft = editor_service.save_or_discard_draft(
                    project.id,
                    mode,
                    edited,
                    source_code=selected_code,
                    source_code_sha256=source_hash,
                )
            except ValueError as exc:
                st.error(str(exc))
            else:
                if saved_draft is None:
                    st.success("O código voltou à versão de origem; o rascunho foi removido.")
                else:
                    st.success("Rascunho salvo. Nenhuma versão foi criada.")
        elif action == "render" and can_render:
            with st.status(
                f"Renderizando nova versão da {label}...",
                expanded=True,
            ) as render_status:
                try:
                    editor_service.save_or_discard_draft(
                        project.id,
                        mode,
                        edited,
                        source_code=selected_code,
                        source_code_sha256=source_hash,
                    )
                    result = editor_service.render_new_version(
                        project.id,
                        mode,
                        edited,
                        voice_api_key=voice_api_key,
                        voice_config=voice_config,
                    )
                except Exception as exc:
                    render_status.update(label="Falha na renderização", state="error")
                    _render_failure(str(exc))
                else:
                    if result.success:
                        render_status.update(
                            label="Nova versão renderizada",
                            state="complete",
                            expanded=False,
                        )
                        st.success("Nova versão renderizada e preservada no histórico.")
                        st.rerun()
                    else:
                        render_status.update(
                            label="Falha na renderização",
                            state="error",
                        )
                        _render_failure(result.stderr or result.error_traceback)

    with chat_column:
        _render_ai_chat(
            editor_service,
            chat_service,
            project,
            mode,
            edited,
            source_code_sha256=source_hash,
            draft_key=draft_key,
            revision_key=revision_key,
            source_key=source_key,
        )

    with st.expander("Histórico de versões", expanded=False):
        if selected_version is None:
            st.caption("Nenhuma versão registrada.")
            return
        st.caption(f"v{selected_version.version} · {selected_version.created_at}")
        st.code(
            Path(selected_version.code_path).read_text(encoding="utf-8"),
            language="python",
        )
        if st.button(
            "Restaurar como nova versão",
            key=f"restore_{mode}_{selected_version.version}",
            icon=":material/restore:",
            width="stretch",
            disabled=voice_config.enabled and not voice_api_key,
        ):
            with st.spinner("Restaurando e renderizando..."):
                try:
                    result = editor_service.restore_version(
                        project.id,
                        mode,
                        selected_version.version,
                        voice_api_key=voice_api_key,
                        voice_config=voice_config,
                    )
                except Exception as exc:
                    _render_failure(str(exc))
                else:
                    if result.success:
                        st.success("Versão restaurada sem apagar o histórico.")
                        st.rerun()
                    else:
                        _render_failure(result.stderr or result.error_traceback)


def _render_failure(message: str) -> None:
    """Show an actionable summary while retaining complete sanitized diagnostics."""
    detail = message.strip() or "Falha sem diagnóstico disponível."
    summary = detail.splitlines()[0][:240]
    st.error(summary)
    with st.expander("Detalhes técnicos do erro"):
        st.code(detail, language=None)


def _render_voice_override(project: ProjectRecord) -> tuple[VoiceConfig, str]:
    """Render a session-only voice configuration used by manual renders."""
    if not project.voiceover_enabled:
        return VoiceConfig(enabled=False), ""

    prefix = f"editor_voice_{project.id}"
    provider_key = f"{prefix}_provider"
    model_key = f"{prefix}_model"
    voice_key = f"{prefix}_voice"
    language_key = f"{prefix}_language"
    speed_key = f"{prefix}_speed"
    providers = options.active_voice_providers()
    if not providers:
        st.error("Ative ao menos um modelo de voz em Configurações.")
        return VoiceConfig(enabled=True), ""
    if st.session_state.get(provider_key) not in providers:
        st.session_state[provider_key] = (
            project.voice_provider if project.voice_provider in providers else providers[0]
        )
    with st.expander("Configuração de voz desta renderização", expanded=False):
        provider = st.selectbox("Provedor de voz", providers, key=provider_key)
        models = options.voice_models_for(provider)
        if st.session_state.get(model_key) not in models:
            preferred = project.voice_model if provider == project.voice_provider else ""
            st.session_state[model_key] = preferred if preferred in models else models[0]
        model = st.selectbox(
            "Modelo de voz",
            models,
            format_func=lambda value: options.model_label(provider, value, modality="speech"),
            key=model_key,
        )
        voices = options.voices_for(provider)
        if st.session_state.get(voice_key) not in voices:
            preferred_voice = project.voice if provider == project.voice_provider else ""
            st.session_state[voice_key] = (
                preferred_voice if preferred_voice in voices else voices[0]
            )
        voice = st.selectbox("Voz", voices, key=voice_key)
        if language_key not in st.session_state:
            st.session_state[language_key] = project.voice_language
        language = st.selectbox("Idioma", options.VOICE_LANGUAGES, key=language_key)
        if speed_key not in st.session_state:
            st.session_state[speed_key] = project.voice_speed
        speed = st.slider(
            "Velocidade",
            min_value=0.5,
            max_value=2.0,
            step=0.1,
            key=speed_key,
        )
        store = get_credential_store()
        resolved = store.resolve_voice(provider)
        typed_key = ""
        if not resolved.found:
            typed_key = st.text_input(
                "Chave da API de voz",
                type="password",
                placeholder=voice_key_hint_text(provider),
                help=key_help_text(),
                key=f"{prefix}_api_key_{provider}",
            )
            resolved = resolve_voice_key(provider, typed_key)
        st.caption(f"Origem da chave: **{key_source_badge(resolved)}**")
    return (
        VoiceConfig(
            enabled=True,
            provider=provider,
            model=model,
            voice=voice,
            language=language,
            speed=speed,
        ),
        resolved.value,
    )


def _render_ai_chat(
    editor_service: CodeEditorService,
    service: CodeEditorChatService,
    project: ProjectRecord,
    mode: VideoMode,
    current_code: str,
    *,
    source_code_sha256: str,
    draft_key: str,
    revision_key: str,
    source_key: str,
) -> None:
    """Render persistent chat and a non-destructive code proposal."""
    header_column, clear_column = st.columns([4, 1], vertical_alignment="center")
    header_column.subheader("Editar com IA")
    state_scope = f"{project.id}_{mode}"
    provider_key = f"editor_ai_provider_{state_scope}"
    model_key = f"editor_ai_model_{state_scope}"
    providers = options.active_llm_providers()
    if not providers:
        st.error("Ative ao menos um modelo de IA em Configurações.")
        return
    preferences_service = CodeAssistantPreferencesService(service.repository)
    assistant_default = preferences_service.resolve()
    default_source_key = f"editor_ai_default_source_{state_scope}"
    default_source = f"{assistant_default.provider}:{assistant_default.model}"
    if st.session_state.get(default_source_key) != default_source:
        st.session_state[provider_key] = assistant_default.provider
        st.session_state[model_key] = assistant_default.model
        st.session_state[default_source_key] = default_source
    elif st.session_state.get(provider_key) not in providers:
        st.session_state[provider_key] = assistant_default.provider
    provider = str(st.session_state[provider_key])
    proposal_key = f"editor_proposal_{project.id}_{mode}"
    messages = service.conversation(project.id, mode)
    with clear_column.popover(
        "",
        icon=":material/delete:",
        help="Limpar conversa",
        key=f"clear_editor_chat_top_{project.id}_{mode}",
        disabled=not messages,
    ):
        st.write("Apagar toda a conversa deste vídeo?")
        st.caption("Essa ação não altera o código nem as versões renderizadas.")
        if st.button(
            "Confirmar exclusão",
            type="primary",
            icon=":material/delete:",
            key=f"confirm_clear_editor_chat_{project.id}_{mode}",
            width="stretch",
        ):
            service.clear_conversation(project.id, mode)
            st.session_state.pop(proposal_key, None)
            st.rerun()

    interaction = st.segmented_control(
        "Modo do assistente",
        options=("conversation", "edit"),
        default="edit",
        format_func=lambda value: "Conversa" if value == "conversation" else "Editor",
        key=f"editor_ai_interaction_{project.id}_{mode}",
        selection_mode="single",
        required=True,
        width="stretch",
    )
    interaction_mode: InteractionMode = "conversation" if interaction == "conversation" else "edit"

    with st.popover(
        f"{provider} · selecionar modelo",
        icon=":material/tune:",
        width="stretch",
    ):
        provider = st.selectbox("Provedor", providers, key=provider_key)
        models = options.models_for(provider)
        provider_default = preferences_service.resolve_for_provider(provider)
        if st.session_state.get(model_key) not in models:
            st.session_state[model_key] = provider_default.model
        model = st.selectbox(
            "Modelo",
            models,
            format_func=lambda value: options.model_label(provider, value),
            key=model_key,
        )

    history = st.container(
        height=460,
        border=True,
        key=f"editor_chat_history_{project.id}_{mode}",
        autoscroll=True,
    )
    with history:
        if not messages:
            if interaction_mode == "conversation":
                st.caption("Pergunte sobre o código ou peça sugestões de melhoria.")
            else:
                st.caption("Descreva uma alteração para receber uma proposta de código.")
        for message in messages:
            with st.chat_message(message.role):
                if message.role == "assistant":
                    st.markdown(message.content, unsafe_allow_html=False)
                else:
                    st.write(message.content)

    request = st.chat_input(
        (
            "Pergunte sobre o código ou peça sugestões"
            if interaction_mode == "conversation"
            else "Descreva a alteração ou anexe imagens de referência"
        ),
        key=f"editor_chat_input_{project.id}_{mode}_{interaction_mode}",
        accept_file="multiple",
        file_type=("png", "jpg", "jpeg", "webp"),
        submit_mode="disable",
    )
    if request:
        if interaction_mode == "edit":
            st.session_state.pop(proposal_key, None)
        request_text = request if isinstance(request, str) else request.text
        uploads = () if isinstance(request, str) else tuple(request.files)
        api_key = get_credential_store().resolve_llm(provider).value
        images = tuple(
            LLMImage(
                data=upload.getvalue(),
                mime_type=upload.type,
                label=f"Imagem {index + 1}: {upload.name}",
            )
            for index, upload in enumerate(uploads or ())
        )
        with history:
            with st.chat_message("user"):
                if request_text:
                    st.write(request_text)
                if uploads:
                    st.image(list(uploads), width=120)
            assistant_message = st.chat_message("assistant")
        request_completed = False
        with (
            assistant_message,
            st.status(
                "Processando sua solicitação...",
                expanded=True,
            ) as edit_status,
        ):
            st.write(f"Modelo: {options.model_label(provider, model)}")
            if images:
                st.write(f"Lendo {len(images)} imagem(ns) anexada(s).")
            try:
                if interaction_mode == "conversation":
                    service.discuss_code(
                        project.id,
                        mode,
                        request_text,
                        current_code,
                        api_key=api_key,
                        provider=provider,
                        model_name=model,
                        images=images,
                    )
                else:
                    new_proposal = service.propose_edit(
                        project.id,
                        mode,
                        request_text,
                        current_code,
                        api_key=api_key,
                        provider=provider,
                        model_name=model,
                        images=images,
                    )
            except (RuntimeError, ValueError) as exc:
                if interaction_mode == "edit":
                    st.session_state.pop(proposal_key, None)
                edit_status.update(
                    label="Não foi possível concluir a solicitação",
                    state="error",
                )
                edit_status.error(str(exc))
            else:
                if interaction_mode == "edit":
                    st.session_state[proposal_key] = new_proposal
                edit_status.update(
                    label=(
                        "Resposta concluída"
                        if interaction_mode == "conversation"
                        else (
                            "Proposta preparada"
                            if new_proposal.changed
                            else "Nenhuma alteração produzida"
                        )
                    ),
                    state="complete",
                    expanded=False,
                )
                request_completed = True
        if request_completed:
            st.rerun()

    proposal = st.session_state.get(proposal_key)
    if interaction_mode != "edit" or not isinstance(proposal, CodeEditProposal):
        return

    st.markdown("**Proposta atual**")
    st.write(proposal.summary)
    if not proposal.changed:
        st.info(
            "A proposta atual não contém diferenças em relação ao código em edição. "
            "Nenhuma alteração está disponível para aplicar."
        )
        if st.button(
            "Descartar resultado",
            key=f"discard_unchanged_editor_proposal_{project.id}_{mode}",
            icon=":material/delete:",
            width="stretch",
        ):
            del st.session_state[proposal_key]
            st.rerun()
        return
    if proposal.errors:
        st.error(
            "A proposta foi bloqueada pela proteção de execução: " + " ".join(proposal.errors)
        )

    diff = "\n".join(
        difflib.unified_diff(
            current_code.splitlines(),
            proposal.code.splitlines(),
            fromfile="código atual",
            tofile="código proposto",
            lineterm="",
        )
    )
    with st.expander("Comparar alterações", expanded=True):
        st.code(diff or "Nenhuma diferença encontrada.", language="diff")
    with st.expander("Ver código completo"):
        st.code(proposal.code, language="python")

    stale = service.code_hash(current_code) != proposal.base_code_hash
    if stale:
        st.warning("O código mudou depois desta proposta. Solicite uma nova edição.")
    action_column, discard_column = st.columns(2, gap="medium")
    if action_column.button(
        "Aplicar e salvar rascunho",
        type="primary",
        key=f"apply_editor_proposal_{project.id}_{mode}",
        icon=":material/check:",
        width="stretch",
        disabled=not proposal.valid or stale,
    ):
        editor_service.save_draft(
            project.id,
            mode,
            proposal.code,
            source_code_sha256=source_code_sha256,
        )
        st.session_state[draft_key] = proposal.code
        st.session_state[source_key] = service.code_hash(proposal.code)
        st.session_state[revision_key] = int(st.session_state.get(revision_key, 0)) + 1
        del st.session_state[proposal_key]
        st.rerun()
    if discard_column.button(
        "Descartar proposta",
        key=f"discard_editor_proposal_{project.id}_{mode}",
        icon=":material/delete:",
        width="stretch",
    ):
        del st.session_state[proposal_key]
        st.rerun()


st.title("Editor de código Manim")
project_id = str(state.get(state.KEY_CURRENT_PROJECT_ID, ""))
if not project_id:
    st.info("Abra um projeto para editar seus vídeos.")
    st.stop()

project = ProjectService().open_project(project_id)
if project is None:
    st.error("Projeto não encontrado.")
    st.stop()

voice_config, voice_api_key = _render_voice_override(project)
editor_service = CodeEditorService()
chat_service = CodeEditorChatService()
selected_mode = st.segmented_control(
    "Vídeo em edição",
    options=("presentation", "solution"),
    default="presentation",
    format_func=lambda value: "Apresentação" if value == "presentation" else "Resolução",
    key=f"editor_mode_{project.id}",
    selection_mode="single",
    required=True,
    width="stretch",
)
mode: VideoMode = "solution" if selected_mode == "solution" else "presentation"
_render_mode(
    editor_service,
    chat_service,
    project,
    mode,
    voice_api_key,
    voice_config,
)
