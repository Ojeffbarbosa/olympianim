"""Tests for teacher-provided images used by Manim scenes."""

from __future__ import annotations

import io
from pathlib import Path

import pytest
from PIL import Image, ImageDraw

from olympianim.database.repository import ProjectRepository
from olympianim.graph.approved_workflow import build_animation_asset_context
from olympianim.services.image_asset_service import (
    MAX_ANIMATION_ASSETS,
    AnimationAsset,
    AnimationAssetInput,
    ImageAssetService,
    remove_uniform_background,
)
from olympianim.services.project_service import ProjectInput, ProjectService


def _png_bytes(background: str = "white") -> bytes:
    image = Image.new("RGB", (80, 60), background)
    ImageDraw.Draw(image).rectangle((20, 15, 60, 45), fill="red")
    output = io.BytesIO()
    image.save(output, format="PNG")
    return output.getvalue()


@pytest.mark.parametrize("background", ("white", "#00ff00"))
def test_uniform_background_removal_makes_edge_transparent(background: str) -> None:
    image = Image.open(io.BytesIO(_png_bytes(background))).convert("RGBA")

    result = remove_uniform_background(image)

    assert result.getpixel((0, 0))[3] == 0
    assert result.getpixel((40, 30))[3] == 255


def test_background_removal_preserves_disconnected_matching_color() -> None:
    image = Image.new("RGBA", (60, 60), "white")
    draw = ImageDraw.Draw(image)
    draw.rectangle((15, 15, 45, 45), fill="red")
    draw.rectangle((25, 25, 35, 35), fill="white")

    result = remove_uniform_background(image)

    assert result.getpixel((0, 0))[3] == 0
    assert result.getpixel((30, 30))[3] == 255


def test_prepare_requires_description_and_rejects_nonuniform_corners() -> None:
    service = ImageAssetService()
    with pytest.raises(ValueError, match="descrição"):
        service.prepare_many((AnimationAssetInput("moto.png", _png_bytes(), "curta"),))

    image = Image.new("RGB", (20, 20), "white")
    pixels = image.load()
    pixels[0, 0], pixels[19, 0], pixels[0, 19], pixels[19, 19] = (
        (255, 0, 0),
        (0, 255, 0),
        (0, 0, 255),
        (255, 255, 255),
    )
    output = io.BytesIO()
    image.save(output, format="PNG")
    with pytest.raises(ValueError, match="cor uniforme"):
        service.prepare_many(
            (
                AnimationAssetInput(
                    "objeto.png",
                    output.getvalue(),
                    "Objeto colorido visto de perfil.",
                    remove_background=True,
                ),
            )
        )


def test_prepare_rejects_excess_corrupt_and_oversized_uploads() -> None:
    service = ImageAssetService()
    valid = AnimationAssetInput(
        "objeto.png",
        _png_bytes(),
        "Objeto vermelho sobre fundo branco.",
    )
    with pytest.raises(ValueError, match="no máximo"):
        service.prepare_many(tuple(valid for _ in range(MAX_ANIMATION_ASSETS + 1)))
    with pytest.raises(ValueError, match="não é uma imagem válida"):
        service.prepare_many(
            (AnimationAssetInput("falsa.png", b"not-an-image", valid.description),)
        )

    oversized = Image.new("RGB", (4097, 1), "white")
    output = io.BytesIO()
    oversized.save(output, format="PNG")
    with pytest.raises(ValueError, match="4096"):
        service.prepare_many(
            (AnimationAssetInput("grande.png", output.getvalue(), valid.description),)
        )


def test_project_persists_portable_asset_metadata(tmp_path: Path) -> None:
    repository = ProjectRepository(tmp_path / "olympianim.db")
    service = ProjectService(repository=repository, projects_dir=tmp_path / "projects")

    project = service.create_project(
        ProjectInput(
            problem_statement="Uma moto percorre uma estrada.",
            animation_assets=(
                AnimationAssetInput(
                    "../../Moto vermelha.JPG",
                    _png_bytes(),
                    "Motocicleta vermelha vista de perfil para a direita.",
                ),
            ),
        )
    )

    assets = service.list_animation_assets(project.id)
    assert len(assets) == 1
    assert assets[0].filename == "01_moto_vermelha.png"
    assert assets[0].manim_path == "input/objects/01_moto_vermelha.png"
    assert Path(assets[0].path).is_file()
    assert ".." not in assets[0].path


def test_prompt_context_exists_only_for_image_aware_roles() -> None:
    asset = AnimationAsset(
        filename="01_moto.png",
        path="/project/input/objects/01_moto.png",
        manim_path="input/objects/01_moto.png",
        description="Motocicleta vermelha vista de perfil.",
        background_removed=True,
    )

    assert build_animation_asset_context((), "builder") == ""
    assert build_animation_asset_context((asset,), "solver") == ""
    planner = build_animation_asset_context((asset,), "planner")
    builder = build_animation_asset_context((asset,), "builder")
    assert "Motocicleta vermelha" in planner
    assert "caminho Manim" not in planner
    assert "input/objects/01_moto.png" in builder
