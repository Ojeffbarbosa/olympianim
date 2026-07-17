"""Secure logging helpers.

API keys must never be written to logs. This module provides:

* ``redact`` - replace any known secret value with a placeholder.
* ``SecretFilter`` - a ``logging.Filter`` that scrubs secret substrings
  from every record before it is emitted.
* ``configure_logging`` - install the filter on the root logger.

The filter is intentionally simple: it walks the log message and the
stringified record arguments, replacing each registered secret with
``REDACTED_PLACEHOLDER``. Secrets are registered at runtime via
``SecretFilter.register`` and kept only in memory.
"""

from __future__ import annotations

import logging
from collections.abc import Iterable
from typing import Final

REDACTED_PLACEHOLDER: Final[str] = "***REDACTED***"

# Substring markers commonly surrounding a key in error text. Used only
# as a defensive best-effort; the authoritative protection is the
# explicit registry of secret values.
_KEY_MARKERS: Final[tuple[str, ...]] = (
    "sk-",
    "AIza",
    "sk-ant-",
)


class SecretFilter(logging.Filter):
    """Logging filter that redacts registered secret values.

    Secrets are registered in memory only and are never persisted.
    """

    def __init__(self) -> None:
        super().__init__()
        # Sort by descending length so longer secrets are replaced first,
        # preventing partial overlaps from leaking a shorter prefix.
        self._secrets: list[str] = []

    def register(self, secret: str) -> None:
        """Register a secret value to be redacted from log records."""
        if not secret:
            return
        if secret not in self._secrets:
            self._secrets.append(secret)
            self._secrets.sort(key=len, reverse=True)

    def register_many(self, secrets: Iterable[str]) -> None:
        """Register several secret values at once."""
        for secret in secrets:
            self.register(secret)

    def clear(self) -> None:
        """Forget all registered secrets (e.g. at session end)."""
        self._secrets.clear()

    @property
    def registered_count(self) -> int:
        """Return how many secrets are currently registered."""
        return len(self._secrets)

    def _redact(self, text: str) -> str:
        redacted = text
        for secret in self._secrets:
            if secret and secret in redacted:
                redacted = redacted.replace(secret, REDACTED_PLACEHOLDER)
        return redacted

    def filter(self, record: logging.LogRecord) -> bool:
        """Redact secrets in the record message and its arguments."""
        if isinstance(record.msg, str):
            record.msg = self._redact(record.msg)
        if record.args:
            record.args = tuple(
                self._redact(arg) if isinstance(arg, str) else arg
                for arg in (record.args if isinstance(record.args, tuple) else (record.args,))
            )
        return True


_ACTIVE_FILTER: SecretFilter | None = None


def configure_logging(level: int = logging.INFO) -> SecretFilter:
    """Install (or return) the shared ``SecretFilter`` on the root logger."""
    global _ACTIVE_FILTER
    if _ACTIVE_FILTER is None:
        _ACTIVE_FILTER = SecretFilter()
        root = logging.getLogger()
        if not any(isinstance(f, SecretFilter) for f in root.filters):
            root.addFilter(_ACTIVE_FILTER)
        if root.level == logging.NOTSET or root.level > level:
            root.setLevel(level)
    return _ACTIVE_FILTER


def redact(text: str, secrets: Iterable[str]) -> str:
    """Return ``text`` with each secret replaced by the placeholder."""
    redacted = text
    for secret in sorted(secrets, key=len, reverse=True):
        if secret and secret in redacted:
            redacted = redacted.replace(secret, REDACTED_PLACEHOLDER)
    return redacted


def looks_like_key_prefix(text: str) -> bool:
    """Heuristic: does ``text`` start with a known provider key prefix?"""
    return any(text.startswith(marker) for marker in _KEY_MARKERS)
