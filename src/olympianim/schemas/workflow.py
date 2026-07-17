"""Typed contracts for resumable, teacher-approved workflow transitions."""

from __future__ import annotations

from dataclasses import dataclass
from enum import StrEnum
from typing import Any


class WorkflowPhase(StrEnum):
    """Stable phases persisted by the current workflow revision."""

    CREATED = "created"
    PREPARE_SOLUTION_BASIS = "prepare_solution_basis"
    REVIEW_SOLUTION_BASIS = "review_solution_basis"
    PLAN_PRESENTATION = "plan_presentation"
    REVIEW_PLAN_PRESENTATION = "review_plan_presentation"
    BUILD_PRESENTATION = "build_presentation"
    REVIEW_CODE_PRESENTATION = "review_code_presentation"
    RENDER_PRESENTATION = "render_presentation"
    PRESENTATION_COMPLETE = "presentation_complete"
    PLAN_SOLUTION = "plan_solution"
    REVIEW_PLAN_SOLUTION = "review_plan_solution"
    BUILD_SOLUTION = "build_solution"
    REVIEW_CODE_SOLUTION = "review_code_solution"
    RENDER_SOLUTION = "render_solution"
    COMPLETED = "completed"
    FAILED = "failed"
    STOPPED = "stopped"


class WorkflowJobAction(StrEnum):
    """Background runner entry points."""

    START = "start"
    RESUME = "resume"
    CONTINUE = "continue"


class ReviewAction(StrEnum):
    """Actions accepted at explicit human review boundaries."""

    APPROVE = "approve"
    EDIT = "edit"
    REGENERATE = "regenerate"
    REJECT = "reject"
    GENERATE_SOLUTION = "generate_solution"


REVIEW_PHASES = frozenset(
    {
        WorkflowPhase.REVIEW_SOLUTION_BASIS.value,
        WorkflowPhase.REVIEW_PLAN_PRESENTATION.value,
        WorkflowPhase.REVIEW_CODE_PRESENTATION.value,
        WorkflowPhase.PRESENTATION_COMPLETE.value,
        WorkflowPhase.REVIEW_PLAN_SOLUTION.value,
        WorkflowPhase.REVIEW_CODE_SOLUTION.value,
    }
)


@dataclass(frozen=True)
class ResumeRequest:
    """Serializable identity and boundary contract for one explicit action."""

    operation_id: str
    expected_phase: str
    decision: dict[str, Any]
    workflow_revision: int = 1

    def validate(self) -> None:
        """Reject malformed operations before they can consume a graph interrupt."""
        if not self.operation_id.strip():
            raise ValueError("A operação do workflow não possui identificador estável.")
        if self.expected_phase not in REVIEW_PHASES:
            raise ValueError("A etapa esperada não é uma fronteira de revisão válida.")
        raw_action = str(self.decision.get("action", "")).strip()
        try:
            action = ReviewAction(raw_action)
        except ValueError as exc:
            raise ValueError(f"Ação de revisão desconhecida: {raw_action!r}.") from exc
        if self.expected_phase == WorkflowPhase.PRESENTATION_COMPLETE:
            if action is not ReviewAction.GENERATE_SOLUTION:
                raise ValueError("A apresentação concluída aceita somente gerar a resolução.")
        elif action is ReviewAction.GENERATE_SOLUTION:
            raise ValueError("Gerar a resolução exige a apresentação concluída.")


def is_review_phase(value: str) -> bool:
    """Return whether ``value`` identifies a human-review boundary."""
    return value in REVIEW_PHASES
