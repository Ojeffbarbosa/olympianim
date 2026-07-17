"""Sanitized project evidence export for audit and reproducibility."""

from __future__ import annotations

import hashlib
import io
import json
import platform
import re
import subprocess
import zipfile
from dataclasses import asdict
from datetime import UTC, datetime
from importlib.metadata import PackageNotFoundError, version
from pathlib import Path, PurePosixPath
from typing import Any

from olympianim.config import APP_VERSION, PROJECT_ROOT, PROJECTS_DIR
from olympianim.database.repository import ProjectRepository
from olympianim.database.sqlite import LATEST_SCHEMA_VERSION

_EXCLUDED_FILE_TYPES = frozenset(
    {
        "presentation_render_log",
        "solution_render_log",
    }
)
_RESTRICTED_FILE_TYPES = frozenset(
    {
        "problem_image",
        "solution_image",
        "animation_asset",
        "animation_asset_transparent",
        "presentation_video",
        "solution_video",
        "final_video",
    }
)
_PACKAGE_NAMES = (
    "manim",
    "manim-voiceover",
    "langgraph",
    "langgraph-checkpoint-sqlite",
    "langchain",
    "streamlit",
)
_SECRET_PATTERNS = (
    re.compile(r"\bsk-(?:proj-|ant-)?[A-Za-z0-9_-]{12,}\b"),
    re.compile(r"\bAIza[A-Za-z0-9_-]{20,}\b"),
    re.compile(r"(?im)^(OPENAI_API_KEY|GOOGLE_API_KEY|ANTHROPIC_API_KEY)\s*=\s*.+$"),
)


class ProjectExportService:
    """Build an in-memory ZIP without database files, keys or host paths."""

    def __init__(
        self,
        repository: ProjectRepository | None = None,
        *,
        projects_dir: Path = PROJECTS_DIR,
    ) -> None:
        self.repository = repository or ProjectRepository()
        self.projects_dir = projects_dir

    def build_zip(
        self,
        project_id: str,
        *,
        include_originals: bool = False,
        rights_confirmed: bool = False,
    ) -> bytes:
        """Return a sanitized ZIP; media requires an explicit rights confirmation."""
        if include_originals and not rights_confirmed:
            raise ValueError(
                "Confirme os direitos de uso antes de incluir imagens e vídeos originais."
            )
        project = self.repository.get_project(project_id)
        if project is None:
            raise ValueError("Projeto não encontrado.")
        project_root = (self.projects_dir / project_id).resolve()
        entries: dict[str, bytes] = {}

        portable_project = asdict(project)
        for key in tuple(portable_project):
            if key.endswith("_path"):
                portable_project[key] = self._portable_registered_path(
                    str(portable_project[key]), project_root
                )
        self._add_json(entries, "metadata/project.json", portable_project, project_root)

        jobs = []
        for job in self.repository.list_jobs(project_id):
            item = asdict(job)
            item["payload"] = _json_or_empty(job.payload)
            item["result"] = _json_or_empty(job.result)
            jobs.append(item)
        self._add_json(entries, "audit/jobs.json", jobs, project_root)

        events = self.repository.list_workflow_events(project_id)
        event_lines = [
            self._sanitize(
                json.dumps(
                    {**asdict(event), "payload": _json_or_empty(event.payload)},
                    ensure_ascii=False,
                    sort_keys=True,
                ),
                project_root,
            )
            for event in events
        ]
        entries["audit/events.jsonl"] = (
            "\n".join(event_lines) + ("\n" if event_lines else "")
        ).encode("utf-8")

        usage = [asdict(item) for item in self.repository.list_ai_usage(project_id)]
        self._add_json(entries, "audit/ai_usage.json", usage, project_root)

        for index, prompt in enumerate(self.repository.list_project_prompts(project_id), start=1):
            base = f"prompts/{index:03d}_{_safe_name(prompt.agent_type)}"
            metadata = {
                "agent_type": prompt.agent_type,
                "prompt_id": prompt.prompt_id,
                "prompt_version": prompt.prompt_version,
                "prompt_sha256": prompt.prompt_sha256,
                "operation_id": prompt.operation_id,
                "created_at": prompt.created_at,
            }
            self._add_json(entries, f"{base}.json", metadata, project_root)
            system_text = prompt.rendered_system_snapshot
            user_text = prompt.rendered_user_snapshot
            if not system_text and not user_text:
                user_text = prompt.rendered_prompt_snapshot
            entries[f"{base}.system.md"] = self._sanitize(system_text, project_root).encode(
                "utf-8"
            )
            entries[f"{base}.user.md"] = self._sanitize(user_text, project_root).encode("utf-8")

        for call in self.repository.list_llm_call_cache(project_id):
            name = (
                f"ai_outputs/{_safe_name(call.role)}_{_safe_name(call.mode)}_"
                f"{call.cache_key[:12]}.txt"
            )
            entries[name] = self._sanitize(call.response_text, project_root).encode("utf-8")

        exported_artifacts: list[dict[str, Any]] = []
        for artifact in self.repository.list_generated_files(project_id):
            if artifact.file_type in _EXCLUDED_FILE_TYPES:
                continue
            if artifact.file_type in _RESTRICTED_FILE_TYPES and not include_originals:
                continue
            source = Path(artifact.path)
            try:
                resolved = source.resolve(strict=True)
            except (FileNotFoundError, OSError):
                continue
            if not resolved.is_relative_to(project_root) or not resolved.is_file():
                continue
            relative = resolved.relative_to(project_root).as_posix()
            archive_name = _safe_archive_name(f"artifacts/{relative}")
            content = resolved.read_bytes()
            if resolved.suffix.casefold() in {
                ".py",
                ".diff",
                ".md",
                ".txt",
                ".srt",
                ".json",
                ".jsonl",
            }:
                content = self._sanitize(
                    content.decode("utf-8", errors="replace"), project_root
                ).encode()
            entries[archive_name] = content
            exported_artifacts.append(
                {
                    "artifact_key": artifact.artifact_key,
                    "file_type": artifact.file_type,
                    "version": artifact.version,
                    "archive_path": archive_name,
                    "sha256": hashlib.sha256(content).hexdigest(),
                    "size_bytes": len(content),
                }
            )

        manifest = {
            "format": "olympianim-project-evidence",
            "format_version": 1,
            "created_at": datetime.now(UTC).isoformat(timespec="seconds"),
            "project_id": project_id,
            "schema_version": LATEST_SCHEMA_VERSION,
            "workflow_revision": project.workflow_revision,
            "original_media_included": include_originals,
            "artifacts": sorted(
                exported_artifacts,
                key=lambda item: (item["file_type"], item["version"], item["archive_path"]),
            ),
            "environment": self._environment_manifest(project),
        }
        self._add_json(entries, "manifest.json", manifest, project_root)
        entries["README.txt"] = (
            "Exportação auditável do Olympianim.\n"
            "O banco SQLite, variáveis de ambiente, chaves e logs brutos não fazem parte "
            "deste arquivo. Verifique manifest.json para hashes e versões.\n"
        ).encode()
        return _build_zip(entries)

    def _environment_manifest(self, project: Any) -> dict[str, Any]:
        packages: dict[str, str] = {}
        for package in _PACKAGE_NAMES:
            try:
                packages[package] = version(package)
            except PackageNotFoundError:
                packages[package] = "not-installed"
        cached_calls = self.repository.list_llm_call_cache(project.id)
        return {
            "app_version": APP_VERSION,
            "git_commit": _command_version(["git", "-C", str(PROJECT_ROOT), "rev-parse", "HEAD"]),
            "python": platform.python_version(),
            "python_implementation": platform.python_implementation(),
            "operating_system": platform.platform(),
            "packages": packages,
            "manim_documentation": "stable/0.20.1",
            "ffmpeg": _command_version(["ffmpeg", "-version"]),
            "latex": _command_version(["latex", "--version"]),
            "render_quality": self.repository.get_setting("render_quality", "low_quality"),
            "requested_text_model": {
                "provider": project.llm_provider,
                "model": project.llm_model,
            },
            "resolved_text_models": sorted(
                {
                    f"{item.provider}:{item.resolved_model or item.requested_model}"
                    for item in cached_calls
                }
            ),
            "voice": {
                "enabled": project.voiceover_enabled,
                "provider": project.voice_provider,
                "model": project.voice_model,
                "voice": project.voice,
                "language": project.voice_language,
                "speed": project.voice_speed,
            },
            "color_palette_snapshot": _json_or_text(project.color_palette_snapshot),
        }

    @staticmethod
    def _portable_registered_path(value: str, project_root: Path) -> str:
        if not value:
            return ""
        try:
            resolved = Path(value).resolve()
        except OSError:
            return ""
        if not resolved.is_relative_to(project_root):
            return ""
        return resolved.relative_to(project_root).as_posix()

    @staticmethod
    def _sanitize(value: str, project_root: Path) -> str:
        sanitized = str(value).replace(str(project_root), "<PROJECT_ROOT>")
        sanitized = sanitized.replace(str(PROJECT_ROOT), "<APP_ROOT>")
        for pattern in _SECRET_PATTERNS:
            sanitized = pattern.sub("[REDACTED]", sanitized)
        return sanitized

    def _add_json(
        self,
        entries: dict[str, bytes],
        name: str,
        value: Any,
        project_root: Path,
    ) -> None:
        rendered = json.dumps(value, ensure_ascii=False, indent=2, sort_keys=True, default=str)
        entries[_safe_archive_name(name)] = self._sanitize(rendered, project_root).encode("utf-8")


def _safe_archive_name(value: str) -> str:
    path = PurePosixPath(value)
    if path.is_absolute() or not path.parts or any(part in {"", ".", ".."} for part in path.parts):
        raise ValueError("Caminho inseguro recusado na exportação ZIP.")
    return path.as_posix()


def _safe_name(value: str) -> str:
    return re.sub(r"[^A-Za-z0-9_.-]+", "_", value).strip("._") or "item"


def _json_or_empty(value: str) -> Any:
    try:
        parsed = json.loads(value or "{}")
    except (TypeError, ValueError):
        return {}
    return parsed


def _json_or_text(value: str) -> Any:
    try:
        return json.loads(value)
    except (TypeError, ValueError):
        return value


def _command_version(command: list[str]) -> str:
    try:
        completed = subprocess.run(
            command,
            capture_output=True,
            text=True,
            check=False,
            timeout=5,
        )
    except (OSError, subprocess.TimeoutExpired):
        return "unavailable"
    output = (completed.stdout or completed.stderr).strip().splitlines()
    return output[0] if completed.returncode == 0 and output else "unavailable"


def _build_zip(entries: dict[str, bytes]) -> bytes:
    buffer = io.BytesIO()
    with zipfile.ZipFile(buffer, mode="w", compression=zipfile.ZIP_DEFLATED) as archive:
        for name in sorted(entries):
            safe_name = _safe_archive_name(name)
            info = zipfile.ZipInfo(safe_name, date_time=(1980, 1, 1, 0, 0, 0))
            info.compress_type = zipfile.ZIP_DEFLATED
            info.external_attr = 0o600 << 16
            archive.writestr(info, entries[name])
    return buffer.getvalue()
