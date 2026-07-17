"""Tests for resolving workflow credentials outside the creation form."""

from __future__ import annotations

from types import SimpleNamespace

from olympianim.services.credential_service import CredentialStore
from olympianim.ui import credentials, state


def test_workflow_credentials_fall_back_to_environment(
    monkeypatch,
) -> None:
    project = SimpleNamespace(
        llm_provider="OpenAI",
        voiceover_enabled=False,
    )
    store = CredentialStore()
    monkeypatch.setenv("OPENAI_API_KEY", "key-from-environment")
    monkeypatch.setattr(state, "get", lambda _key, default="": default)
    monkeypatch.setattr(credentials, "get_credential_store", lambda: store)

    resolved = credentials.workflow_credentials_for_project(project)

    assert resolved.llm_api_key == "key-from-environment"
    assert resolved.voice_api_key == ""


def test_workflow_credentials_prefer_active_session_key(
    monkeypatch,
) -> None:
    project = SimpleNamespace(
        llm_provider="OpenAI",
        voiceover_enabled=False,
    )
    values = {state.KEY_API_KEY_RESOLVED: "key-from-session"}
    monkeypatch.setattr(state, "get", lambda key, default="": values.get(key, default))

    resolved = credentials.workflow_credentials_for_project(project)

    assert resolved.llm_api_key == "key-from-session"


def test_workflow_credentials_resolve_the_selected_provider(monkeypatch) -> None:
    project = SimpleNamespace(
        llm_provider="OpenAI",
        voiceover_enabled=False,
    )
    store = CredentialStore()
    monkeypatch.setenv("GOOGLE_API_KEY", "google-key")
    values = {state.KEY_API_KEY_RESOLVED: "openai-session-key"}
    monkeypatch.setattr(state, "get", lambda key, default="": values.get(key, default))
    monkeypatch.setattr(credentials, "get_credential_store", lambda: store)

    resolved = credentials.workflow_credentials_for_project(project, "Google")

    assert resolved.llm_api_key == "google-key"
