"""Bridge between the Streamlit session state and the credential service.

This module owns the in-memory ``CredentialStore`` for the session and
implements the credential flow:

1. load ``.env`` once;
2. when a provider is selected, look up its key (``.env`` first);
3. if missing, show the protected input + the session-only warning;
4. store the typed key only in memory;
5. offer a "test connection" button that probes the provider without
   persisting the key.

No business logic that belongs in ``services/`` is duplicated here; we
only orchestrate UI ↔ service interactions so page scripts stay thin.
"""

from __future__ import annotations

from collections.abc import Mapping
from typing import TYPE_CHECKING, Any

import streamlit as st

from olympianim.providers.llm import get_provider
from olympianim.services.credential_service import (
    PROVIDER_ENV_VARS,
    CredentialStore,
    ResolvedKey,
    env_var_for_llm,
    env_var_for_voice,
    session_warning_text,
)
from olympianim.ui import state

if TYPE_CHECKING:
    from olympianim.services.langgraph_workflow import WorkflowCredentials


def get_credential_store() -> CredentialStore:
    """Return the singleton ``CredentialStore`` for this session."""
    store: CredentialStore | None = state.get(state.KEY_CREDENTIAL_STORE, None)
    if store is None:
        CredentialStore.load_env()
        store = CredentialStore()
        state.set(state.KEY_CREDENTIAL_STORE, store)
    return store


def workflow_credentials_for_project(
    project: object,
    llm_provider: str | None = None,
) -> WorkflowCredentials:
    """Resolve the active LLM and voice keys for a persisted project."""
    from olympianim.services.langgraph_workflow import WorkflowCredentials

    project_provider = str(getattr(project, "llm_provider", ""))
    selected_provider = llm_provider or project_provider
    llm_key = ""
    if selected_provider == project_provider:
        llm_key = str(
            state.get(state.KEY_API_KEY_RESOLVED, "") or state.get(state.KEY_LLM_API_KEY, "")
        )
    if not llm_key:
        llm_key = get_credential_store().resolve_llm(selected_provider).value
    voice_key = ""
    if getattr(project, "voiceover_enabled", False):
        if (
            getattr(project, "reuse_llm_api_key", False)
            and getattr(project, "voice_provider", "") == selected_provider
        ):
            voice_key = llm_key
        else:
            voice_key = str(state.get(state.KEY_VOICE_API_KEY_RESOLVED, ""))
            if not voice_key:
                voice_key = (
                    get_credential_store()
                    .resolve_voice(str(getattr(project, "voice_provider", "")))
                    .value
                )
    return WorkflowCredentials(llm_api_key=llm_key, voice_api_key=voice_key)


def resolve_llm_key(provider: str, typed_key: str) -> ResolvedKey:
    """Resolve the API key for ``provider`` using the configured precedence.

    ``typed_key`` is whatever the teacher entered in the protected
    field this run. When non-empty it overrides (or fills) the session
    store. When empty, the store + ``.env`` are consulted.
    """
    store = get_credential_store()
    if typed_key:
        store.set_session_key(provider, typed_key)
    resolved = store.resolve_llm(provider)
    state.set(state.KEY_API_KEY_RESOLVED, resolved.value)
    state.set(state.KEY_API_KEY_SOURCE, resolved.source)
    return resolved


def resolve_voice_key(provider: str, typed_key: str) -> ResolvedKey:
    """Resolve a TTS key in its own session namespace."""
    store = get_credential_store()
    if typed_key:
        store.set_voice_session_key(provider, typed_key)
        resolved = ResolvedKey(provider=provider, value=typed_key, source="session")
    else:
        resolved = store.resolve_voice(provider)
    state.set(state.KEY_VOICE_API_KEY_RESOLVED, resolved.value)
    state.set(state.KEY_VOICE_API_KEY_SOURCE, resolved.source)
    return resolved


def key_hint_text(provider: str) -> str:
    """Return the placeholder text for the API-key input field."""
    return f"Digite sua chave (variável {env_var_for_llm(provider)})"


def voice_key_hint_text(provider: str) -> str:
    """Return the placeholder for a provider-specific TTS key."""
    return f"Digite sua chave ({env_var_for_voice(provider)})"


def key_help_text() -> str:
    """Return the help/tooltip text shown beside the API-key input."""
    return (
        f"{session_warning_text()} Se a chave existir no arquivo .env, "
        "ela será usada automaticamente."
    )


def session_warning_text_from_service() -> str:
    """Expose the canonical session-only warning text."""
    return session_warning_text()


def key_source_badge(resolved: ResolvedKey) -> str:
    """Return a short Portuguese label describing where the key came from."""
    if not resolved.found:
        return "chave não encontrada"
    if resolved.source == "env":
        return "chave carregada do .env"
    if resolved.source == "session":
        return "chave digitada nesta sessão"
    return "origem desconhecida"


def render_connection_test(provider: str, resolved: ResolvedKey) -> Mapping[str, Any]:
    """Render the "test connection" button and run the probe.

    Returns a mapping with the probe result so the caller can decide
    whether to proceed. The key is never logged: the active
    ``SecretFilter`` redacts it.
    """
    if not resolved.found:
        st.info(
            f"Não há chave para {provider}. Preencha o campo acima ou "
            f"defina {PROVIDER_ENV_VARS.get(provider, '')} no .env."
        )
        return {"ok": False, "message": "Sem chave para testar.", "tested": False}

    if st.button(
        "Testar conexão",
        key=f"test_connection_{provider}",
        icon=":material/cable:",
    ):
        with st.spinner(f"Testando conexão com {provider}..."):
            provider_instance = get_provider(provider, resolved.value)
            result = provider_instance.test_connection()
        state.set(state.KEY_CONNECTION_RESULT, result)
        if result:
            st.success(result.message)
        else:
            st.error(result.message)
        return {"ok": result.ok, "message": result.message, "tested": True}

    previous = state.get(state.KEY_CONNECTION_RESULT, None)
    if previous is not None:
        if previous.ok:
            st.success(previous.message)
        else:
            st.error(previous.message)
    return {"ok": previous.ok if previous else False, "message": "", "tested": False}
