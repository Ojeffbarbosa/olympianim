"""Structured contracts returned by language-model agents."""

from __future__ import annotations

from pydantic import Field

from olympianim.schemas.base import OlympianimModel


class ManimCodeOutput(OlympianimModel):
    """Complete Python source produced by a Manim code agent."""

    code: str = Field(
        min_length=1,
        description=(
            "Complete executable Manim Python file, without Markdown fences or explanatory text."
        ),
    )
