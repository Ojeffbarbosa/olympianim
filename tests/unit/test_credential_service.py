"""Unit tests for the credential service."""

from __future__ import annotations

import os
from pathlib import Path

import pytest

from olympianim.services.credential_service import (
    SOURCE_ENV,
    SOURCE_SESSION,
    CredentialStore,
    env_var_for_llm,
    env_var_for_voice,
    session_warning_text,
)


@pytest.fixture()
def isolated_env(monkeypatch: pytest.MonkeyPatch) -> None:
    """Strip the known API-key env vars so tests start from a clean slate."""
    for var in ("OPENAI_API_KEY", "GOOGLE_API_KEY", "ANTHROPIC_API_KEY"):
        monkeypatch.delenv(var, raising=False)


@pytest.fixture()
def store(isolated_env: None) -> CredentialStore:
    """A fresh in-memory credential store."""
    return CredentialStore()


# --- env var mapping -------------------------------------------------------
def test_env_var_mapping_for_llm_providers() -> None:
    assert env_var_for_llm("OpenAI") == "OPENAI_API_KEY"
    assert env_var_for_llm("Google") == "GOOGLE_API_KEY"
    assert env_var_for_llm("Anthropic") == "ANTHROPIC_API_KEY"
    assert env_var_for_llm("Unknown") == ""


def test_env_var_mapping_for_voice_providers() -> None:
    assert env_var_for_voice("OpenAI") == "OPENAI_API_KEY"
    assert env_var_for_voice("Google") == "GOOGLE_API_KEY"
    assert env_var_for_voice("Unknown") == ""


def test_session_warning_text_is_portuguese() -> None:
    text = session_warning_text()
    assert "sessão" in text
    assert "banco de dados" in text


# --- .env loading ----------------------------------------------------------
def test_load_env_reads_dotenv(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    env_file = tmp_path / ".env"
    env_file.write_text("OPENAI_API_KEY=sk-test-from-env\n", encoding="utf-8")
    monkeypatch.delenv("OPENAI_API_KEY", raising=False)
    CredentialStore.load_env(env_path=env_file)
    assert os.environ.get("OPENAI_API_KEY") == "sk-test-from-env"
    monkeypatch.delenv("OPENAI_API_KEY", raising=False)


def test_load_env_missing_file_is_noop(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.delenv("OPENAI_API_KEY", raising=False)
    CredentialStore.load_env(env_path=tmp_path / "nope.env")  # should not raise
    assert os.environ.get("OPENAI_API_KEY") is None


# --- resolution ------------------------------------------------------------
def test_resolve_prefers_env_over_session(
    store: CredentialStore,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("OPENAI_API_KEY", "sk-env-value")
    store.set_session_key("OpenAI", "sk-session-value")
    resolved = store.resolve_llm("OpenAI")
    assert resolved.found
    assert resolved.source == SOURCE_ENV
    assert resolved.value == "sk-env-value"


def test_resolve_falls_back_to_session(
    store: CredentialStore,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.delenv("OPENAI_API_KEY", raising=False)
    store.set_session_key("OpenAI", "sk-session-value")
    resolved = store.resolve_llm("OpenAI")
    assert resolved.found
    assert resolved.source == SOURCE_SESSION
    assert resolved.value == "sk-session-value"


def test_resolve_returns_empty_when_missing(
    store: CredentialStore,
    isolated_env: None,
) -> None:
    resolved = store.resolve_llm("OpenAI")
    assert not resolved.found
    assert resolved.value == ""
    assert resolved.source == ""


def test_voice_and_llm_session_keys_are_independent(
    store: CredentialStore,
    isolated_env: None,
) -> None:
    store.set_session_key("Google", "llm-key")
    store.set_voice_session_key("Google", "voice-key")

    assert store.resolve_llm("Google").value == "llm-key"
    assert store.resolve_voice("Google").value == "voice-key"


def test_tts_session_namespace_does_not_replace_llm_key(
    store: CredentialStore,
    isolated_env: None,
) -> None:
    store.set_session_key("Google", "llm-key")
    store.set_voice_session_key("Google", "voice-key")

    assert store.get_session_key("Google") == "llm-key"
    assert store.get_voice_session_key("Google") == "voice-key"


def test_resolve_voice_uses_voice_mapping(
    store: CredentialStore,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("GOOGLE_API_KEY", "google-key")
    resolved = store.resolve_voice("Google")
    assert resolved.found
    assert resolved.source == SOURCE_ENV
    assert resolved.value == "google-key"


def test_resolve_dispatch_flag(
    store: CredentialStore,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("GOOGLE_API_KEY", "google-key")
    assert store.resolve("Google", voice=True).source == SOURCE_ENV
    monkeypatch.setenv("ANTHROPIC_API_KEY", "ant-key")
    assert store.resolve("Anthropic", voice=False).source == SOURCE_ENV


def test_resolve_unknown_provider_is_empty(store: CredentialStore, isolated_env: None) -> None:
    assert not store.resolve_llm("Unknown").found


# --- in-memory semantics ---------------------------------------------------
def test_session_key_is_kept_only_in_memory(store: CredentialStore, isolated_env: None) -> None:
    store.set_session_key("OpenAI", "sk-mem-only")
    assert store.get_session_key("OpenAI") == "sk-mem-only"
    # Nothing was written to the actual environment.
    assert os.environ.get("OPENAI_API_KEY") is None


def test_clear_forgets_session_keys(store: CredentialStore, isolated_env: None) -> None:
    store.set_session_key("OpenAI", "sk-one")
    store.set_session_key("Google", "g-two")
    store.clear()
    assert store.get_session_key("OpenAI") == ""
    assert store.get_session_key("Google") == ""


def test_set_session_key_registers_secret_in_filter(
    store: CredentialStore, isolated_env: None
) -> None:
    store.set_session_key("OpenAI", "sk-secret-abc")
    from olympianim.utils.logging import configure_logging

    filt = configure_logging()
    assert filt.registered_count >= 1
    assert "sk-secret-abc" not in filt._redact("the key is sk-secret-abc here")
