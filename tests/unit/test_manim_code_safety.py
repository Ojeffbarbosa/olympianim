"""Regression tests for generated-code execution safety."""

from __future__ import annotations

import pytest

from olympianim.manim.presentation import (
    check_generated_code_safety,
    prepare_source_watermark_code,
    strip_source_watermark_code,
)

_SOLUTION_CODE = """from manim import *

class SolutionScene(Scene):
    def construct(self):
        self.add(Text("A solução é 150 reais."))
"""


def test_pedagogical_content_is_not_inspected() -> None:
    assert check_generated_code_safety(_SOLUTION_CODE, require_voiceover=False) == []


@pytest.mark.parametrize(
    "import_line",
    ("import numpy as np", "from numpy import array"),
)
def test_generated_code_can_import_numpy(import_line: str) -> None:
    code = f"""from manim import Scene, Dot
{import_line}

class Demo(Scene):
    def construct(self):
        self.add(Dot(array([0, 0, 0])))
"""

    assert check_generated_code_safety(code, require_voiceover=False) == []


def test_source_watermark_is_deterministic_and_idempotent() -> None:
    code = """from __future__ import annotations
from manim import *

class DemoScene(Scene):
    def construct(self):
        self.clear()
"""

    prepared = prepare_source_watermark_code(code, "OBMEP 2024")

    assert prepared.startswith("from __future__ import annotations\n")
    assert "Text('Fonte: OBMEP 2024'" in prepared
    assert ".set_opacity(0.3)" in prepared
    assert ".set_z_index(-100)" in prepared
    assert "class DemoScene(OlympianimSourceWatermarkMixin, Scene):" in prepared
    assert prepare_source_watermark_code(prepared, "OBMEP 2024") == prepared
    assert check_generated_code_safety(prepared, require_voiceover=False) == []


def test_source_watermark_is_not_added_without_source() -> None:
    assert prepare_source_watermark_code(_SOLUTION_CODE, "  ") == _SOLUTION_CODE


def test_source_watermark_normalizes_legacy_duplicates_without_marker() -> None:
    code = """class OlympianimSourceWatermarkMixin:
    pass

class OlympianimSourceWatermarkMixin:
    pass

from manim import *

class DemoScene(OlympianimSourceWatermarkMixin, OlympianimSourceWatermarkMixin, Scene):
    def construct(self):
        self.add(Text("Demo"))
"""

    prepared = prepare_source_watermark_code(code, "OBMEP")
    canonical = strip_source_watermark_code(prepared)

    assert prepared.count("class OlympianimSourceWatermarkMixin:") == 1
    assert prepared.count("OlympianimSourceWatermarkMixin, Scene") == 1
    assert "OlympianimSourceWatermarkMixin" not in canonical
    assert "class DemoScene(Scene):" in canonical
    assert prepare_source_watermark_code(prepared, "OBMEP") == prepared


def test_image_mobject_requires_registered_literal_path() -> None:
    code = """from manim import *
class Demo(Scene):
    def construct(self):
        self.add(ImageMobject('input/objects/01_moto.png'))
"""
    assert (
        check_generated_code_safety(
            code,
            require_voiceover=False,
            allowed_image_paths={"input/objects/01_moto.png"},
        )
        == []
    )
    errors = check_generated_code_safety(code, require_voiceover=False)
    assert "Imagem não registrada" in errors[0]


def test_manim_api_and_grouping_choices_are_not_inspected() -> None:
    code = """from manim import *
class Demo(Scene):
    def construct(self):
        image = ImageMobject('input/objects/01_moto.png')
        self.add(VGroup(image, Text('ponto de partida')))
"""

    assert (
        check_generated_code_safety(
            code,
            require_voiceover=False,
            allowed_image_paths={"input/objects/01_moto.png"},
        )
        == []
    )


@pytest.mark.parametrize(
    "expression",
    ("'/tmp/moto.png'", "'../moto.png'", "asset_path"),
)
def test_image_mobject_rejects_unsafe_paths(expression: str) -> None:
    code = f"""from manim import *
class Demo(Scene):
    def construct(self):
        self.add(ImageMobject({expression}))
"""
    assert check_generated_code_safety(
        code,
        require_voiceover=False,
        allowed_image_paths={"input/objects/01_moto.png"},
    )


def test_generated_code_cannot_import_application_internals() -> None:
    code = """from manim import Scene
from olympianim.manim.presentation import subprocess

class Demo(Scene):
    def construct(self):
        subprocess.run(["echo", "unsafe"])
"""

    errors = check_generated_code_safety(
        code,
        require_voiceover=False,
    )

    assert "Importação não permitida: olympianim.manim.presentation." in errors


def test_generated_code_cannot_use_dunder_introspection() -> None:
    code = """from manim import Scene

class Demo(Scene):
    def construct(self):
        self.add(Scene.__class__)
"""

    errors = check_generated_code_safety(
        code,
        require_voiceover=False,
    )

    assert "Acesso interno não permitido: __class__." in errors
