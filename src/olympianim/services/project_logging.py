"""Project-scoped logging and LangChain callback integration."""

from __future__ import annotations

import json
import threading
from typing import Any
from uuid import UUID

from langchain_core.callbacks import BaseCallbackHandler

from olympianim.database.repository import ProjectRepository
from olympianim.utils.logging import redact

_MAX_INFO_LOG_VALUE_LENGTH = 500


class ProjectLogger:
    """Persist non-sensitive events and complete error diagnostics."""

    def __init__(
        self,
        repository: ProjectRepository,
        project_id: str,
        *,
        secrets: tuple[str, ...] = (),
    ) -> None:
        self.repository = repository
        self.project_id = project_id
        self.secrets = secrets

    def info(self, step: str, message: str) -> None:
        self._write("info", step, message)

    def error(self, step: str, message: str) -> None:
        self._write("error", step, message)

    def _write(self, level: str, step: str, message: str) -> None:
        sanitized = redact(message, self.secrets).strip()
        self.repository.add_log(
            self.project_id,
            level=level,
            step=step,
            message=sanitized if level == "error" else _compact(sanitized),
        )


class ProjectToolCallback(BaseCallbackHandler):
    """Record native LangChain tool calls against their owning project."""

    def __init__(self, logger: ProjectLogger, *, role: str, mode: str) -> None:
        self.logger = logger
        self.role = role
        self.mode = mode
        self._tools_by_run: dict[UUID, str] = {}
        self._lock = threading.Lock()

    def on_tool_start(
        self,
        serialized: dict[str, Any],
        input_str: str,
        *,
        run_id: UUID,
        parent_run_id: UUID | None = None,
        tags: list[str] | None = None,
        metadata: dict[str, Any] | None = None,
        inputs: dict[str, Any] | None = None,
        **kwargs: Any,
    ) -> None:
        _ = (parent_run_id, tags, metadata, kwargs)
        name = str(serialized.get("name") or "tool")
        with self._lock:
            self._tools_by_run[run_id] = name
        tool_input = inputs if inputs is not None else input_str
        self.logger.info(
            self._step(name),
            f"Ferramenta chamada. Entrada: {_safe_json(tool_input)}",
        )

    def on_tool_end(
        self,
        output: Any,
        *,
        run_id: UUID,
        parent_run_id: UUID | None = None,
        **kwargs: Any,
    ) -> None:
        _ = (output, parent_run_id, kwargs)
        name = self._pop_tool_name(run_id)
        self.logger.info(self._step(name), "Ferramenta concluída com sucesso.")

    def on_tool_error(
        self,
        error: BaseException,
        *,
        run_id: UUID,
        parent_run_id: UUID | None = None,
        **kwargs: Any,
    ) -> None:
        _ = (parent_run_id, kwargs)
        name = self._pop_tool_name(run_id)
        self.logger.error(self._step(name), f"Ferramenta falhou: {error}")

    def _step(self, tool_name: str) -> str:
        return f"tool.{self.role}.{self.mode}.{tool_name}"

    def _pop_tool_name(self, run_id: UUID) -> str:
        with self._lock:
            return self._tools_by_run.pop(run_id, "tool")


def _safe_json(value: Any) -> str:
    try:
        rendered = json.dumps(value, ensure_ascii=False, sort_keys=True, default=str)
    except (TypeError, ValueError):
        rendered = str(value)
    return _compact(rendered)


def _compact(value: str) -> str:
    text = " ".join(str(value).split())
    if len(text) <= _MAX_INFO_LOG_VALUE_LENGTH:
        return text
    return f"{text[: _MAX_INFO_LOG_VALUE_LENGTH - 3]}..."
