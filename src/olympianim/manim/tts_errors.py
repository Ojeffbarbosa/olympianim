"""Sanitized provider-error diagnostics shared by speech adapters."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import httpx


@dataclass(frozen=True)
class TTSErrorDetails:
    exception_type: str
    code: str = ""
    status: str = ""
    message: str = ""
    transient: bool = False

    def summary(self) -> str:
        identity = self.code or self.status or self.exception_type
        return f"{identity}: {self.message}" if self.message else identity


def describe_tts_error(exc: Exception, *, transient: bool = False) -> TTSErrorDetails:
    """Extract stable, secret-free fields without storing request content or headers."""
    response: Any = getattr(exc, "response", None)
    code = getattr(exc, "code", "") or getattr(response, "status_code", "")
    status = getattr(exc, "status", "") or getattr(response, "reason_phrase", "")
    raw_message = getattr(exc, "message", "") or str(exc)
    message = " ".join(str(raw_message).split())[:1000]
    if message in {"Error code:", "Error code"}:
        message = "O provedor não enviou uma descrição do erro."
    return TTSErrorDetails(
        exception_type=type(exc).__name__,
        code=str(code or ""),
        status=str(status or ""),
        message=message,
        transient=transient,
    )


def is_transient_tts_error(exc: Exception) -> bool:
    """Recognize transport, throttling and server failures across provider SDKs."""
    response: Any = getattr(exc, "response", None)
    code = (
        getattr(exc, "code", None)
        or getattr(exc, "status_code", None)
        or getattr(response, "status_code", None)
    )
    return (
        isinstance(exc, httpx.TransportError)
        or type(exc).__name__ in {"APIConnectionError", "APITimeoutError", "RateLimitError"}
        or code in {408, 429}
        or (isinstance(code, int) and code >= 500)
    )
