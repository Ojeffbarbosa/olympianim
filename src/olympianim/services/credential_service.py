"""Credential service: load ``.env`` and resolve API keys per provider.

Security rules:

1. Look up the key first in the environment loaded from ``.env``.
2. If missing, ask the teacher in the interface and keep the typed key
   only in memory for the session.
3. Never persist keys to SQLite.
4. Never write keys into project files.
5. Never log keys (the active ``SecretFilter`` redacts them).

This module is intentionally pure-Python and free of Streamlit imports
so it can be unit-tested without a running app. The in-memory session
store is a plain dict the UI passes in; nothing here touches disk
beyond the read-only ``.env`` lookup.
"""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass, field
from pathlib import Path
from typing import Final

from dotenv import load_dotenv

from olympianim.utils.logging import SecretFilter, configure_logging

# ---------------------------------------------------------------------------
# Provider -> environment variable name
# ---------------------------------------------------------------------------
PROVIDER_ENV_VARS: Final[Mapping[str, str]] = {
    "OpenAI": "OPENAI_API_KEY",
    "Google": "GOOGLE_API_KEY",
    "Anthropic": "ANTHROPIC_API_KEY",
}

# Voice providers may reuse the language-model key when the provider matches.
VOICE_PROVIDER_ENV_VARS: Final[Mapping[str, str]] = {
    "OpenAI": "OPENAI_API_KEY",
    "Google": "GOOGLE_API_KEY",
}

# Where a key came from. Only this non-sensitive tag is ever persisted
# to SQLite.
SOURCE_ENV: Final[str] = "env"
SOURCE_SESSION: Final[str] = "session"


@dataclass(frozen=True)
class ResolvedKey:
    """The result of resolving a key for a provider.

    ``value`` is empty when no key is available; in that case the UI
    must prompt the teacher. The ``source`` tag is safe to persist.
    """

    provider: str
    value: str
    source: str  # SOURCE_ENV | SOURCE_SESSION | ""

    @property
    def found(self) -> bool:
        """Whether a non-empty key was resolved."""
        return bool(self.value)


@dataclass
class CredentialStore:
    """In-memory session store for keys typed in the interface.

    Lives only for the duration of the Streamlit session and is never
    written to disk. The caller (UI) is responsible for tying it to
    ``st.session_state``.
    """

    session_keys: dict[str, str] = field(default_factory=dict)
    voice_session_keys: dict[str, str] = field(default_factory=dict)
    _secret_filter: SecretFilter = field(init=False)

    def __post_init__(self) -> None:
        # Register the secret filter so any accidental log of a key is redacted.
        self._secret_filter = configure_logging()

    # -- session (typed in UI) -------------------------------------------------
    def set_session_key(self, provider: str, value: str) -> None:
        """Store a key the teacher typed in the interface (memory only)."""
        self.session_keys[provider] = value
        self._secret_filter.register(value)

    def get_session_key(self, provider: str) -> str:
        """Return the session-only key for ``provider`` (empty if absent)."""
        return self.session_keys.get(provider, "")

    def set_voice_session_key(self, provider: str, value: str) -> None:
        """Store a TTS-specific key without replacing the LLM key."""
        self.voice_session_keys[provider] = value
        self._secret_filter.register(value)

    def get_voice_session_key(self, provider: str) -> str:
        """Return a TTS-specific session key."""
        return self.voice_session_keys.get(provider, "")

    def clear(self) -> None:
        """Forget all session keys (call at logout / session end)."""
        for value in self.session_keys.values():
            self._unregister_secret(value)
        self.session_keys.clear()
        self.voice_session_keys.clear()

    # -- env (.env) -----------------------------------------------------------
    @staticmethod
    def load_env(env_path: Path | None = None) -> None:
        """Load variables from ``.env`` into ``os.environ``.

        Called once at startup. Missing file is a no-op.
        """
        if env_path is None:
            from olympianim.config import ENV_FILE

            env_path = ENV_FILE
        load_dotenv(dotenv_path=env_path, override=False)

    # -- resolution -----------------------------------------------------------
    def resolve_llm(self, provider: str) -> ResolvedKey:
        """Resolve the API key for an LLM provider."""
        return self._resolve(provider, PROVIDER_ENV_VARS)

    def resolve_voice(self, provider: str) -> ResolvedKey:
        """Resolve the API key for a voice provider."""
        env_var = VOICE_PROVIDER_ENV_VARS.get(provider, "")
        if env_var:
            import os

            if env_value := os.environ.get(env_var, ""):
                self._secret_filter.register(env_value)
                return ResolvedKey(provider=provider, value=env_value, source=SOURCE_ENV)
        value = self.get_voice_session_key(provider)
        return ResolvedKey(
            provider=provider,
            value=value,
            source=SOURCE_SESSION if value else "",
        )

    def resolve(self, provider: str, *, voice: bool = False) -> ResolvedKey:
        """Resolve a key, dispatching to the voice map when ``voice`` is True."""
        return self.resolve_voice(provider) if voice else self.resolve_llm(provider)

    # -- internals ------------------------------------------------------------
    def _resolve(
        self,
        provider: str,
        env_vars: Mapping[str, str],
    ) -> ResolvedKey:
        import os

        env_var = env_vars.get(provider, "")
        if env_var:
            env_value = os.environ.get(env_var, "")
            if env_value:
                self._secret_filter.register(env_value)
                return ResolvedKey(provider=provider, value=env_value, source=SOURCE_ENV)

        session_value = self.get_session_key(provider)
        if session_value:
            return ResolvedKey(provider=provider, value=session_value, source=SOURCE_SESSION)

        return ResolvedKey(provider=provider, value="", source="")

    def _unregister_secret(self, value: str) -> None:
        """Best-effort removal of a secret from the redaction filter.

        ``SecretFilter`` does not expose removal by value; clearing is
        the supported path. We skip per-value removal to keep the
        filter conservative (safer to over-redact than to leak).
        """


def env_var_for_llm(provider: str) -> str:
    """Return the ``.env`` variable name for an LLM provider."""
    return PROVIDER_ENV_VARS.get(provider, "")


def env_var_for_voice(provider: str) -> str:
    """Return the ``.env`` variable name for a voice provider."""
    return VOICE_PROVIDER_ENV_VARS.get(provider, "")


def session_warning_text() -> str:
    """Return the user-facing warning shown beside the key input."""
    return (
        "A chave informada será utilizada apenas nesta sessão e não será salva no banco de dados."
    )
