"""Prompt template variable extraction and validation."""

from __future__ import annotations

import string
from dataclasses import dataclass
from typing import Any

from olympianim.prompts.variables import variables_for_agent

_REQUIRED_VARIABLES = {"gemini_tts": {"transcript"}}


@dataclass(frozen=True)
class PromptValidationResult:
    """Result of checking a template against allowed variables."""

    valid: bool
    used_variables: tuple[str, ...]
    unknown_variables: tuple[str, ...]


def extract_template_variables(template_text: str) -> tuple[str, ...]:
    """Extract variables from ``str.format``-style braces."""
    formatter = string.Formatter()
    variables: list[str] = []
    for _, field_name, _, _ in formatter.parse(template_text):
        if not field_name:
            continue
        root_name = field_name.split(".", 1)[0].split("[", 1)[0]
        if root_name and root_name not in variables:
            variables.append(root_name)
    return tuple(variables)


def validate_prompt_template(agent_type: str, template_text: str) -> PromptValidationResult:
    """Validate a prompt template for an agent."""
    allowed = set(variables_for_agent(agent_type))
    used = extract_template_variables(template_text)
    unknown = tuple(variable for variable in used if variable not in allowed)
    missing = _REQUIRED_VARIABLES.get(agent_type, set()) - set(used)
    return PromptValidationResult(
        valid=not unknown and not missing,
        used_variables=used,
        unknown_variables=unknown + tuple(f"ausente:{variable}" for variable in sorted(missing)),
    )


def render_prompt_template(template_text: str, values: dict[str, Any]) -> str:
    """Render a prompt template with missing variables filled as empty strings."""
    variables = extract_template_variables(template_text)
    safe_values = {variable: str(values.get(variable, "")) for variable in variables}
    return template_text.format(**safe_values)
