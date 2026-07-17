"""Real low-quality Manim render integration tests."""

from __future__ import annotations

from pathlib import Path

import pytest
from PIL import Image

from olympianim.manim.presentation import PresentationRenderer
from olympianim.schemas.render import ManimCodeResult


def _render_source(tmp_path: Path, source: str, scene_name: str) -> Path:
    code_path = tmp_path / f"{scene_name}.py"
    code_path.write_text(source, encoding="utf-8")
    result = PresentationRenderer(timeout_seconds=60).render(
        ManimCodeResult(
            mode="presentation",
            scene_name=scene_name,
            code=source,
            code_path=str(code_path),
        ),
        project_directory=tmp_path,
        quality="low_quality",
    )
    assert result.success, result.stderr
    return Path(result.video_path)


@pytest.mark.integration
def test_renderer_produces_a_real_video(tmp_path: Path) -> None:
    source = """from manim import Scene, Text, Write

class PresentationScene(Scene):
    def construct(self):
        title = Text("Olympianim")
        self.play(Write(title), run_time=0.1)
        self.wait(0.1)
"""

    assert _render_source(tmp_path, source, "PresentationScene").is_file()


@pytest.mark.integration
def test_renderer_loads_a_project_relative_image(tmp_path: Path) -> None:
    image_path = tmp_path / "input" / "objects" / "01_object.png"
    image_path.parent.mkdir(parents=True)
    Image.new("RGBA", (80, 40), "white").save(image_path)
    source = """from manim import FadeIn, ImageMobject, Scene

class AssetScene(Scene):
    def construct(self):
        image = ImageMobject("input/objects/01_object.png")
        self.play(FadeIn(image), run_time=0.1)
        self.wait(0.1)
"""

    assert _render_source(tmp_path, source, "AssetScene").is_file()


@pytest.mark.integration
def test_renderer_composes_multiline_inline_math_and_raster_image(tmp_path: Path) -> None:
    image_path = tmp_path / "input" / "objects" / "01_object.png"
    image_path.parent.mkdir(parents=True)
    Image.new("RGBA", (80, 40), "white").save(image_path)
    source = """from manim import *

class MixedStatementScene(Scene):
    def construct(self):
        statement_lines = (
            r"Uma sequência tem $a_1=2$ e $a_{n+1}=a_n+3$.",
            r"Determine $S=\\sum_{k=1}^{10}a_k$ e verifique os $20\\%$ finais.",
        )
        statement = VGroup(
            *(Tex(line, font_size=28) for line in statement_lines)
        ).arrange(DOWN, aligned_edge=LEFT, buff=0.22)
        image = ImageMobject("input/objects/01_object.png").scale_to_fit_height(0.6)
        screen = Group(statement, image).arrange(DOWN, buff=0.3)
        self.play(FadeIn(screen), run_time=0.1)
        self.wait(0.1)
"""

    assert _render_source(tmp_path, source, "MixedStatementScene").is_file()


@pytest.mark.integration
def test_renderer_preserves_native_manim_srt(tmp_path: Path) -> None:
    source = """from manim import Scene, Text

class CaptionScene(Scene):
    def construct(self):
        self.add_subcaption("Leia o problema.", duration=0.2)
        self.add(Text("Problema"))
        self.wait(0.2)
"""
    code_path = tmp_path / "CaptionScene.py"
    code_path.write_text(source, encoding="utf-8")

    result = PresentationRenderer(timeout_seconds=60).render(
        ManimCodeResult(
            mode="presentation",
            scene_name="CaptionScene",
            code=source,
            code_path=str(code_path),
        ),
        project_directory=tmp_path,
        quality="low_quality",
    )

    assert result.success, result.stderr
    subtitle = Path(result.subtitle_path)
    assert subtitle.is_file()
    assert "Leia o problema." in subtitle.read_text(encoding="utf-8")
