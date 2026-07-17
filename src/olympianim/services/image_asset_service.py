"""Validation and preprocessing for teacher-provided animation images."""

from __future__ import annotations

import io
import re
from dataclasses import dataclass
from pathlib import Path
from typing import cast

from PIL import Image, ImageDraw, ImageOps, UnidentifiedImageError

from olympianim.config import PROJECTS_DIR
from olympianim.database.models import GeneratedFileRecord
from olympianim.database.repository import ProjectRepository
from olympianim.services.artifact_service import ArtifactService

MAX_ANIMATION_ASSETS = 5
MAX_ASSET_BYTES = 10 * 1024 * 1024
MAX_ASSET_DIMENSION = 4096
MIN_DESCRIPTION_LENGTH = 10
MAX_DESCRIPTION_LENGTH = 500
_SUPPORTED_FORMATS = {"JPEG", "PNG", "WEBP"}
_CORNER_MATCH_TOLERANCE = 35
_FLOOD_FILL_TOLERANCE = 48


@dataclass(frozen=True)
class AnimationAssetInput:
    """One image and its required teacher-authored description."""

    filename: str
    content: bytes
    description: str
    remove_background: bool = False


@dataclass(frozen=True)
class PreparedAnimationAsset:
    """Validated PNG ready to be written inside a project."""

    filename: str
    content: bytes
    description: str
    background_removed: bool


@dataclass(frozen=True)
class AnimationAsset:
    """Persisted animation resource exposed to the workflow."""

    filename: str
    path: str
    manim_path: str
    description: str
    background_removed: bool


class ImageAssetService:
    """Prepare and persist portable image resources for ``ImageMobject``."""

    def __init__(
        self,
        repository: ProjectRepository | None = None,
        artifact_service: ArtifactService | None = None,
        projects_dir: Path = PROJECTS_DIR,
    ) -> None:
        self.repository = repository or ProjectRepository()
        self.artifacts = artifact_service or ArtifactService(
            repository=self.repository,
            projects_dir=projects_dir,
        )

    def prepare_many(
        self,
        assets: tuple[AnimationAssetInput, ...],
    ) -> tuple[PreparedAnimationAsset, ...]:
        """Validate an upload batch before any project data is persisted."""
        if len(assets) > MAX_ANIMATION_ASSETS:
            raise ValueError(f"Envie no máximo {MAX_ANIMATION_ASSETS} imagens de objetos.")
        prepared: list[PreparedAnimationAsset] = []
        for index, asset in enumerate(assets, start=1):
            description = " ".join(asset.description.split())
            if not MIN_DESCRIPTION_LENGTH <= len(description) <= MAX_DESCRIPTION_LENGTH:
                raise ValueError(
                    f"A descrição de {asset.filename!r} deve ter entre "
                    f"{MIN_DESCRIPTION_LENGTH} e {MAX_DESCRIPTION_LENGTH} caracteres."
                )
            image = self._load_image(asset)
            if asset.remove_background:
                image = remove_uniform_background(image)
            output = io.BytesIO()
            image.save(output, format="PNG", optimize=True)
            prepared.append(
                PreparedAnimationAsset(
                    filename=f"{index:02d}_{_safe_stem(asset.filename)}.png",
                    content=output.getvalue(),
                    description=description,
                    background_removed=asset.remove_background,
                )
            )
        return tuple(prepared)

    def save_many(
        self,
        project_id: str,
        assets: tuple[PreparedAnimationAsset, ...],
    ) -> tuple[AnimationAsset, ...]:
        """Write a prepared batch and register it in ``generated_files``."""
        saved: list[AnimationAsset] = []
        for index, asset in enumerate(assets, start=1):
            relative_path = f"input/objects/{asset.filename}"
            path = self.artifacts.save_binary(
                project_id,
                relative_path=relative_path,
                content=asset.content,
                file_type=(
                    "animation_asset_transparent"
                    if asset.background_removed
                    else "animation_asset"
                ),
                description=asset.description,
                version=index,
                artifact_key=f"animation_asset:v{index}",
            )
            saved.append(
                AnimationAsset(
                    filename=asset.filename,
                    path=str(path),
                    manim_path=relative_path,
                    description=asset.description,
                    background_removed=asset.background_removed,
                )
            )
        return tuple(saved)

    def list_assets(self, project_id: str) -> tuple[AnimationAsset, ...]:
        """Return persisted object images in stable upload order."""
        records = [
            record
            for record in self.repository.list_generated_files(project_id)
            if record.file_type in {"animation_asset", "animation_asset_transparent"}
        ]
        return tuple(self._from_record(record, project_id) for record in records)

    @staticmethod
    def _load_image(asset: AnimationAssetInput) -> Image.Image:
        if not asset.content:
            raise ValueError(f"O arquivo {asset.filename!r} está vazio.")
        if len(asset.content) > MAX_ASSET_BYTES:
            raise ValueError(f"O arquivo {asset.filename!r} excede 10 MB.")
        try:
            with Image.open(io.BytesIO(asset.content)) as source:
                source.load()
                if source.format not in _SUPPORTED_FORMATS:
                    raise ValueError(
                        f"Formato não suportado em {asset.filename!r}: {source.format}."
                    )
                image = ImageOps.exif_transpose(source).convert("RGBA")
        except (UnidentifiedImageError, OSError) as exc:
            raise ValueError(f"O arquivo {asset.filename!r} não é uma imagem válida.") from exc
        if max(image.size) > MAX_ASSET_DIMENSION:
            raise ValueError(
                f"A imagem {asset.filename!r} excede {MAX_ASSET_DIMENSION} pixels por dimensão."
            )
        return image

    @staticmethod
    def _from_record(record: GeneratedFileRecord, project_id: str) -> AnimationAsset:
        path = Path(record.path)
        marker = f"{project_id}/"
        normalized = path.as_posix()
        manim_path = normalized.split(marker, 1)[1] if marker in normalized else path.name
        return AnimationAsset(
            filename=path.name,
            path=record.path,
            manim_path=manim_path,
            description=record.description,
            background_removed=record.file_type == "animation_asset_transparent",
        )


def remove_uniform_background(image: Image.Image) -> Image.Image:
    """Make a solid edge-connected background transparent using Pillow."""
    result = image.convert("RGBA")
    width, height = result.size
    corners = (
        _rgb_pixel(result, (0, 0)),
        _rgb_pixel(result, (width - 1, 0)),
        _rgb_pixel(result, (0, height - 1)),
        _rgb_pixel(result, (width - 1, height - 1)),
    )
    clusters = [
        [candidate for candidate in corners if _color_distance(seed, candidate) <= 35]
        for seed in corners
    ]
    matching = max(clusters, key=len)
    if len(matching) < 3:
        raise ValueError(
            "A remoção de fundo exige uma cor uniforme em pelo menos três cantos da imagem."
        )
    background = tuple(
        sum(color[channel] for color in matching) // len(matching) for channel in range(3)
    )
    transparent = (background[0], background[1], background[2], 0)
    for point, color in zip(
        ((0, 0), (width - 1, 0), (0, height - 1), (width - 1, height - 1)),
        corners,
        strict=True,
    ):
        if _color_distance(background, color) <= _CORNER_MATCH_TOLERANCE:
            ImageDraw.floodfill(
                result,
                point,
                transparent,
                thresh=_FLOOD_FILL_TOLERANCE,
            )
    return result


def _color_distance(first: tuple[int, ...], second: tuple[int, ...]) -> int:
    return max(abs(first[index] - second[index]) for index in range(3))


def _rgb_pixel(image: Image.Image, point: tuple[int, int]) -> tuple[int, int, int]:
    pixel = cast(tuple[int, int, int, int], image.getpixel(point))
    return pixel[:3]


def _safe_stem(filename: str) -> str:
    stem = Path(filename).stem
    safe = re.sub(r"[^A-Za-z0-9_-]+", "_", stem).strip("_-").lower()
    return safe[:60] or "objeto"
