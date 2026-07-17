"""Known prompt variables and active agent metadata."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Final


@dataclass(frozen=True)
class AgentSpec:
    """Prompt-owning agent shown in the prompt editor."""

    agent_type: str
    display_name: str
    description: str
    variables: tuple[str, ...]


AGENT_SPECS: Final[tuple[AgentSpec, ...]] = (
    AgentSpec(
        agent_type="workflow_planner",
        display_name="Planejador",
        description="Define a pedagogia da apresentação e da resolução em planos separados.",
        variables=("problem_statement", "teacher_instructions", "solution_basis"),
    ),
    AgentSpec(
        agent_type="workflow_builder",
        display_name="Builder Manim",
        description="Converte cada plano aprovado em código Manim sem alterar sua pedagogia.",
        variables=("problem_statement", "approved_plan", "voiceover_requirements"),
    ),
    AgentSpec(
        agent_type="solution_solver",
        display_name="Solucionador",
        description=(
            "Consolida a solução fornecida em imagem ou resolve o problema quando necessário."
        ),
        variables=("problem_statement", "teacher_instructions"),
    ),
    AgentSpec(
        agent_type="workflow_debugger",
        display_name="Corretor de renderização",
        description="Corrige falhas técnicas preservando o conteúdo aprovado.",
        variables=("manim_code", "render_error", "voiceover_requirements"),
    ),
    AgentSpec(
        agent_type="code_editor_agent",
        display_name="Editor Manim com IA",
        description="Conversa sobre o código atual e propõe alterações quando solicitado.",
        variables=("manim_code", "video_mode", "voiceover_requirements"),
    ),
    AgentSpec(
        agent_type="gemini_tts",
        display_name="Direção de narração Gemini",
        description="Orienta a interpretação da voz sem alterar a transcrição.",
        variables=("transcript", "language", "video_mode"),
    ),
)


def agent_spec_for(agent_type: str) -> AgentSpec:
    """Return metadata for an active prompt-owning agent."""
    for spec in AGENT_SPECS:
        if spec.agent_type == agent_type:
            return spec
    raise ValueError(f"Unknown prompt agent type: {agent_type!r}")


def variables_for_agent(agent_type: str) -> tuple[str, ...]:
    """Return variables allowed for one agent prompt."""
    return agent_spec_for(agent_type).variables
