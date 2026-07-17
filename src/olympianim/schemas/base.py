"""Shared Pydantic configuration and validation helpers."""

from __future__ import annotations

from pydantic import BaseModel, ConfigDict


class OlympianimModel(BaseModel):
    """Base model for structured agent contracts."""

    model_config = ConfigDict(
        extra="forbid",
        str_strip_whitespace=True,
        validate_assignment=True,
    )
