"""Sanitized JSONL transport for usage emitted inside Manim subprocesses."""

from __future__ import annotations

import os
import threading
from pathlib import Path

from olympianim.schemas.render import AIUsageEvent

USAGE_PATH_ENV = "OLYMPIANIM_USAGE_EVENTS_PATH"
_LOCK = threading.Lock()
_CALL_INDEX = 0


def next_call_index() -> int:
    """Return the next speech block index for the current render process."""
    global _CALL_INDEX
    with _LOCK:
        _CALL_INDEX += 1
        return _CALL_INDEX


def emit_usage_event(event: AIUsageEvent) -> None:
    """Append one secret-free event when metering is configured."""
    raw_path = os.environ.get(USAGE_PATH_ENV, "").strip()
    if not raw_path:
        return
    path = Path(raw_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with _LOCK, path.open("a", encoding="utf-8") as stream:
        stream.write(event.model_dump_json() + "\n")


def read_usage_events(path: Path) -> list[AIUsageEvent]:
    """Read valid events while ignoring a final partial line after a crash."""
    if not path.is_file():
        return []
    events: list[AIUsageEvent] = []
    for line in path.read_text(encoding="utf-8").splitlines():
        try:
            events.append(AIUsageEvent.model_validate_json(line))
        except ValueError:
            continue
    return events
