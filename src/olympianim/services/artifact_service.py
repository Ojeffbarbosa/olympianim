"""Atomic, content-addressed persistence of generated project artifacts."""

from __future__ import annotations

import difflib
import hashlib
import os
import tempfile
from pathlib import Path

from olympianim.config import PROJECTS_DIR
from olympianim.database.repository import ProjectRepository


class ArtifactService:
    """Write project files atomically and register reproducible metadata."""

    def __init__(
        self,
        repository: ProjectRepository | None = None,
        projects_dir: Path = PROJECTS_DIR,
    ) -> None:
        self.repository = repository or ProjectRepository()
        self.projects_dir = projects_dir

    def save_text(
        self,
        project_id: str,
        *,
        relative_path: str,
        content: str,
        file_type: str,
        description: str,
        version: int = 1,
        artifact_key: str = "",
    ) -> Path:
        """Atomically save one UTF-8 artifact and upsert its identity."""
        return self.save_binary(
            project_id,
            relative_path=relative_path,
            content=content.encode("utf-8"),
            file_type=file_type,
            description=description,
            version=version,
            artifact_key=artifact_key,
        )

    def save_binary(
        self,
        project_id: str,
        *,
        relative_path: str,
        content: bytes,
        file_type: str,
        description: str,
        version: int = 1,
        artifact_key: str = "",
    ) -> Path:
        """Atomically save binary content and register its SHA-256 digest."""
        path = self._project_path(project_id, relative_path)
        self._atomic_write(path, content)
        self._register(
            project_id,
            file_type=file_type,
            path=path,
            description=description,
            version=version,
            artifact_key=artifact_key,
        )
        return path

    def save_manim_code(self, project_id: str, *, mode: str, code: str, version: int) -> Path:
        """Version Manim source and store a unified diff for every repair."""
        previous_path = self._project_path(project_id, f"{mode}/versions/{mode}_v{version - 1}.py")
        previous = previous_path.read_text(encoding="utf-8") if previous_path.is_file() else ""
        path = self.save_text(
            project_id,
            relative_path=f"{mode}/versions/{mode}_v{version}.py",
            content=code,
            file_type=f"{mode}_code",
            description=f"Código Manim gerado para o vídeo de {mode}.",
            version=version,
            artifact_key=f"{mode}_code:v{version}",
        )
        if version > 1 and previous:
            diff = "".join(
                difflib.unified_diff(
                    previous.splitlines(keepends=True),
                    code.splitlines(keepends=True),
                    fromfile=f"{mode}_v{version - 1}.py",
                    tofile=f"{mode}_v{version}.py",
                )
            )
            self.save_text(
                project_id,
                relative_path=(f"{mode}/versions/{mode}_v{version - 1}_to_v{version}.diff"),
                content=diff,
                file_type=f"{mode}_code_diff",
                description="Diferença técnica produzida pela correção automática.",
                version=version,
                artifact_key=f"{mode}_code_diff:v{version}",
            )
        self.repository.update_project_artifacts(
            project_id,
            **{f"{mode}_code_path": str(path)},
        )
        return path

    def register_video(
        self,
        project_id: str,
        *,
        mode: str,
        video_path: Path,
        version: int = 1,
    ) -> None:
        """Register a completed render for either presentation or solution."""
        self.register_existing(
            project_id,
            file_type=f"{mode}_video",
            path=video_path,
            description=f"Vídeo Manim de {mode} renderizado.",
            version=version,
            artifact_key=f"{mode}_video:v{version}",
        )
        self.repository.update_project_artifacts(
            project_id,
            **{f"{mode}_video_path": str(video_path)},
        )

    def register_subtitle(
        self,
        project_id: str,
        *,
        mode: str,
        subtitle_path: Path,
        transcript_path: Path,
        version: int = 1,
    ) -> None:
        """Register native Manim SRT and its plain-text transcript."""
        self.register_existing(
            project_id,
            file_type=f"{mode}_subtitle",
            path=subtitle_path,
            description=f"Legendas SRT do vídeo de {mode}.",
            version=version,
            artifact_key=f"{mode}_subtitle:v{version}",
        )
        self.register_existing(
            project_id,
            file_type=f"{mode}_transcript",
            path=transcript_path,
            description=f"Transcrição textual do vídeo de {mode}.",
            version=version,
            artifact_key=f"{mode}_transcript:v{version}",
        )

    def register_existing(
        self,
        project_id: str,
        *,
        file_type: str,
        path: Path,
        description: str,
        version: int = 1,
        artifact_key: str = "",
    ) -> None:
        """Register an existing project-contained file with its digest."""
        resolved = path.resolve()
        project_root = self.project_directory(project_id).resolve()
        if not resolved.is_relative_to(project_root):
            raise ValueError("O artefato precisa permanecer dentro da pasta do projeto.")
        if not resolved.is_file():
            raise FileNotFoundError(resolved)
        self._register(
            project_id,
            file_type=file_type,
            path=resolved,
            description=description,
            version=version,
            artifact_key=artifact_key,
        )

    def register_final_video(self, project_id: str, video_path: Path) -> None:
        """Register the assembled delivery video without replacing source renders."""
        self.register_existing(
            project_id,
            file_type="final_video",
            path=video_path,
            description="Vídeo final com apresentação e resolução.",
            artifact_key="final_video:v1",
        )
        self.repository.update_project_artifacts(project_id, final_video_path=str(video_path))

    def clear_final_video(self, project_id: str) -> None:
        """Invalidate an assembled video after one source video changes."""
        project = self.repository.get_project(project_id)
        if project is None or not project.final_video_path:
            return
        Path(project.final_video_path).unlink(missing_ok=True)
        self.repository.update_project_artifacts(project_id, final_video_path="")

    def project_directory(self, project_id: str) -> Path:
        """Return a safe project workspace and create its canonical folders."""
        root = self.projects_dir.resolve()
        project_dir = (root / project_id).resolve()
        if not project_dir.is_relative_to(root) or project_dir == root:
            raise ValueError("Identificador de projeto inválido para o workspace.")
        for relative in (
            "analysis",
            "presentation",
            "presentation/versions",
            "solution",
            "solution/versions",
            "final",
            "logs",
        ):
            (project_dir / relative).mkdir(parents=True, exist_ok=True)
        return project_dir

    def _project_path(self, project_id: str, relative_path: str) -> Path:
        project_dir = self.project_directory(project_id).resolve()
        path = (project_dir / relative_path).resolve()
        if not path.is_relative_to(project_dir) or path == project_dir:
            raise ValueError("Caminho de artefato inválido ou fora do projeto.")
        path.parent.mkdir(parents=True, exist_ok=True)
        return path

    @staticmethod
    def _atomic_write(path: Path, content: bytes) -> None:
        temporary_path: Path | None = None
        try:
            with tempfile.NamedTemporaryFile(
                mode="wb",
                dir=path.parent,
                prefix=f".{path.name}.",
                suffix=".tmp",
                delete=False,
            ) as temporary:
                temporary.write(content)
                temporary.flush()
                os.fsync(temporary.fileno())
                temporary_path = Path(temporary.name)
            os.replace(temporary_path, path)
        finally:
            if temporary_path is not None:
                temporary_path.unlink(missing_ok=True)

    def _register(
        self,
        project_id: str,
        *,
        file_type: str,
        path: Path,
        description: str,
        version: int = 1,
        artifact_key: str = "",
    ) -> None:
        content = path.read_bytes()
        self.repository.add_generated_file(
            project_id,
            file_type=file_type,
            path=str(path),
            description=description,
            version=version,
            artifact_key=artifact_key or f"{file_type}:v{version}",
            sha256=hashlib.sha256(content).hexdigest(),
            size_bytes=len(content),
        )
