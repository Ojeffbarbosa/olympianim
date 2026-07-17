"""Prompt template management, variable validation and rendering."""

from olympianim.prompts.defaults import DEFAULT_PROMPTS, DefaultPrompt
from olympianim.prompts.service import PromptService, PromptWithVersion
from olympianim.prompts.validator import (
    PromptValidationResult,
    extract_template_variables,
    render_prompt_template,
    validate_prompt_template,
)
from olympianim.prompts.variables import AGENT_SPECS, AgentSpec, variables_for_agent

__all__ = [
    "AGENT_SPECS",
    "DEFAULT_PROMPTS",
    "AgentSpec",
    "DefaultPrompt",
    "PromptService",
    "PromptValidationResult",
    "PromptWithVersion",
    "extract_template_variables",
    "render_prompt_template",
    "validate_prompt_template",
    "variables_for_agent",
]
