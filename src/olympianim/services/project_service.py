"""Project use cases for local persistence and workspace files."""

from __future__ import annotations

import hashlib
import re
from dataclasses import dataclass
from pathlib import Path

from olympianim.config import PROJECTS_DIR
from olympianim.database.models import GenerationLogRecord, ProjectCreate, ProjectRecord
from olympianim.database.repository import ProjectRepository, new_id
from olympianim.services.image_asset_service import (
    AnimationAsset,
    AnimationAssetInput,
    ImageAssetService,
)


@dataclass(frozen=True)
class ProjectImageInput:
    """One ordered image attached to a project input."""

    filename: str
    content: bytes


@dataclass(frozen=True)
class ProjectInput:
    """Non-sensitive project data captured from the teacher interface."""

    problem_statement: str
    title: str = ""
    problem_images: tuple[ProjectImageInput, ...] = ()
    problem_source: str = ""
    problem_level: str = ""
    math_area: str = "Automática"
    teacher_solution: str = ""
    solution_images: tuple[ProjectImageInput, ...] = ()
    teacher_instructions: str = ""
    llm_provider: str = ""
    llm_model: str = ""
    llm_api_key_source: str = ""
    voice_provider: str = ""
    voice_model: str = ""
    voice: str = ""
    voice_language: str = ""
    voice_speed: float = 1.0
    voice_api_key_source: str = ""
    reuse_llm_api_key: bool = False
    voiceover_enabled: bool = False
    color_palette_id: str = ""
    color_palette_snapshot: str = ""
    output_delivery_mode: str = "separate"
    animation_assets: tuple[AnimationAssetInput, ...] = ()


class ProjectService:
    """Coordinate project persistence and generated workspace folders."""

    def __init__(
        self,
        repository: ProjectRepository | None = None,
        projects_dir: Path = PROJECTS_DIR,
    ) -> None:
        self.repository = repository or ProjectRepository()
        self.projects_dir = projects_dir

    def create_project(self, data: ProjectInput) -> ProjectRecord:
        """Create a local project and persist its non-sensitive metadata."""
        image_asset_service = ImageAssetService(
            repository=self.repository,
            projects_dir=self.projects_dir,
        )
        prepared_assets = image_asset_service.prepare_many(data.animation_assets)
        project_id = new_id()
        project_dir = self.ensure_project_directories(project_id)
        problem_image_paths = self._save_images(data.problem_images, project_dir / "input")
        solution_image_paths = self._save_images(data.solution_images, project_dir / "teacher")
        record = self.repository.create_project(
            ProjectCreate(
                title=data.title.strip()[:120] or self._derive_title(data.problem_statement),
                problem_statement=data.problem_statement,
                problem_source=data.problem_source,
                problem_level=data.problem_level,
                math_area=data.math_area,
                teacher_solution=data.teacher_solution,
                teacher_instructions=data.teacher_instructions,
                llm_provider=data.llm_provider,
                llm_model=data.llm_model,
                llm_api_key_source=data.llm_api_key_source,
                voice_provider=data.voice_provider,
                voice_model=data.voice_model,
                voice=data.voice,
                voice_language=data.voice_language,
                voice_speed=data.voice_speed,
                voice_api_key_source=data.voice_api_key_source,
                reuse_llm_api_key=data.reuse_llm_api_key,
                voiceover_enabled=data.voiceover_enabled,
                color_palette_id=data.color_palette_id,
                color_palette_snapshot=data.color_palette_snapshot,
                output_delivery_mode=data.output_delivery_mode,
                status="created",
            ),
            project_id=project_id,
        )
        self.repository.add_log(
            record.id,
            message="Projeto criado e metadados salvos no SQLite local.",
            step="create_project",
        )
        for index, saved_path in enumerate(problem_image_paths, start=1):
            self.repository.add_generated_file(
                record.id,
                file_type="problem_image",
                path=str(saved_path),
                version=index,
                description=f"Imagem {index} anexada ao enunciado.",
                artifact_key=f"problem_image:v{index}",
                sha256=hashlib.sha256(saved_path.read_bytes()).hexdigest(),
                size_bytes=saved_path.stat().st_size,
            )
        for index, saved_path in enumerate(solution_image_paths, start=1):
            self.repository.add_generated_file(
                record.id,
                file_type="solution_image",
                path=str(saved_path),
                version=index,
                description=f"Imagem {index} anexada à solução do professor.",
                artifact_key=f"solution_image:v{index}",
                sha256=hashlib.sha256(saved_path.read_bytes()).hexdigest(),
                size_bytes=saved_path.stat().st_size,
            )
        saved_assets = image_asset_service.save_many(record.id, prepared_assets)
        for asset in saved_assets:
            self.repository.add_log(
                record.id,
                step="animation_asset.save",
                message=(
                    f"Objeto visual salvo: {asset.filename}; "
                    f"fundo removido: {'sim' if asset.background_removed else 'não'}."
                ),
            )
        return record

    def list_projects(self) -> list[ProjectRecord]:
        """List previously saved projects."""
        return self.repository.list_projects()

    def open_project(self, project_id: str) -> ProjectRecord | None:
        """Load one existing project by id."""
        return self.repository.get_project(project_id)

    def list_logs(self, project_id: str) -> list[GenerationLogRecord]:
        """Return the persisted generation history for one project."""
        return self.repository.list_logs(project_id)

    def list_animation_assets(self, project_id: str) -> tuple[AnimationAsset, ...]:
        """Return object images available to the project's video workflow."""
        return ImageAssetService(
            repository=self.repository,
            projects_dir=self.projects_dir,
        ).list_assets(project_id)

    def ensure_project_directories(self, project_id: str) -> Path:
        """Create the canonical workspace directory tree for a project."""
        project_dir = self.projects_dir / project_id
        for child in (
            "input",
            "analysis",
            "prompts",
            "teacher",
            "presentation/versions",
            "solution/versions",
            "final",
            "logs",
        ):
            (project_dir / child).mkdir(parents=True, exist_ok=True)
        return project_dir

    @staticmethod
    def _derive_title(problem_statement: str) -> str:
        first_line = next(
            (line.strip() for line in problem_statement.splitlines() if line.strip()),
            "Novo projeto",
        )
        return first_line[:80]

    def _save_images(
        self,
        images: tuple[ProjectImageInput, ...],
        destination: Path,
    ) -> tuple[Path, ...]:
        saved: list[Path] = []
        for index, image in enumerate(images, start=1):
            filename = self._safe_filename(image.filename)
            image_path = destination / f"{index:02d}_{filename}"
            image_path.write_bytes(image.content)
            saved.append(image_path)
        return tuple(saved)

    @staticmethod
    def _safe_filename(filename: str) -> str:
        name = Path(filename).name
        safe = re.sub(r"[^A-Za-z0-9._-]+", "_", name).strip("._")
        return safe or "problem_image"
