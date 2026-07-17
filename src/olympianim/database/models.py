"""Typed records used by the SQLite persistence layer.

The database stores only non-sensitive metadata. API keys are
deliberately absent from every model in this module.
"""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class ProjectCreate:
    """Input required to create a project row."""

    title: str
    problem_statement: str
    problem_source: str = ""
    problem_level: str = ""
    math_area: str = "Automática"
    teacher_solution: str = ""
    teacher_instructions: str = ""
    llm_provider: str = ""
    llm_model: str = ""
    llm_api_key_source: str = ""
    voice_provider: str = ""
    voice_model: str = ""
    voice: str = ""
    voice_language: str = ""
    voice_speed: float = 1.0
    voice_api_key_source: str = ""
    reuse_llm_api_key: bool = False
    voiceover_enabled: bool = False
    color_palette_id: str = ""
    color_palette_snapshot: str = ""
    presentation_video_path: str = ""
    solution_video_path: str = ""
    presentation_code_path: str = ""
    solution_code_path: str = ""
    output_delivery_mode: str = "separate"
    final_video_path: str = ""
    status: str = "created"
    workflow_revision: int = 1


@dataclass(frozen=True)
class ProjectRecord(ProjectCreate):
    """A persisted project row."""

    id: str = ""
    created_at: str = ""
    updated_at: str = ""


@dataclass(frozen=True)
class GeneratedFileRecord:
    """A generated artifact associated with a project."""

    id: str
    project_id: str
    file_type: str
    path: str
    version: int
    description: str
    created_at: str
    artifact_key: str = ""
    sha256: str = ""
    size_bytes: int = 0


@dataclass(frozen=True)
class PromptRecord:
    """Prompt template identity metadata."""

    id: str
    name: str
    agent_type: str
    description: str
    is_default: bool
    created_at: str
    updated_at: str


@dataclass(frozen=True)
class PromptVersionRecord:
    """A versioned prompt template body."""

    id: str
    prompt_id: str
    version: int
    template_text: str
    created_at: str


@dataclass(frozen=True)
class ProjectPromptRecord:
    """Prompt version used by a project, optionally with a rendered snapshot."""

    id: str
    project_id: str
    agent_type: str
    prompt_id: str
    prompt_version: int
    rendered_prompt_snapshot: str
    created_at: str
    rendered_system_snapshot: str = ""
    rendered_user_snapshot: str = ""
    prompt_sha256: str = ""
    operation_id: str = ""


@dataclass(frozen=True)
class GenerationLogRecord:
    """Short generation log line persisted for project history."""

    id: str
    project_id: str
    level: str
    step: str
    message: str
    created_at: str


@dataclass(frozen=True)
class CodeEditorDraftRecord:
    """One durable editor draft scoped to a project and video mode."""

    project_id: str
    mode: str
    code_content: str
    source_code_sha256: str
    updated_at: str


@dataclass(frozen=True)
class GenerationJobRecord:
    """Persistent unit of background workflow execution."""

    id: str
    project_id: str
    action: str
    payload: str
    status: str
    current_step: str
    progress: int
    attempts: int
    result: str
    error_message: str
    cancel_requested: bool
    created_at: str
    updated_at: str
    started_at: str
    finished_at: str
    heartbeat_at: str
    operation_id: str = ""
    expected_phase: str = ""
    workflow_revision: int = 1


@dataclass(frozen=True)
class WorkflowTransitionRecord:
    """One explicit, idempotent workflow operation requested by the teacher."""

    operation_id: str
    project_id: str
    action: str
    expected_phase: str
    decision_sha256: str
    workflow_revision: int
    status: str
    result_snapshot: str
    created_at: str
    updated_at: str


@dataclass(frozen=True)
class LLMCallCacheRecord:
    """A successful model response reusable after a local process restart."""

    cache_key: str
    project_id: str
    operation_id: str
    role: str
    mode: str
    provider: str
    requested_model: str
    resolved_model: str
    finish_reason: str
    prompt_sha256: str
    response_text: str
    created_at: str


@dataclass(frozen=True)
class WorkflowEventRecord:
    """Structured, deduplicated event suitable for audit export."""

    id: str
    event_key: str
    project_id: str
    operation_id: str
    job_id: str
    event_type: str
    phase: str
    payload: str
    created_at: str


@dataclass(frozen=True)
class AIUsageRecord:
    """Provider-neutral consumption for one generative AI attempt."""

    id: str
    project_id: str
    execution_id: str
    call_key: str
    agent_type: str
    stage: str
    provider: str
    model: str
    status: str
    attempt_type: str
    sequence: int
    input_tokens: int
    output_tokens: int
    total_tokens: int
    cache_read_tokens: int
    cache_creation_tokens: int
    reasoning_tokens: int
    modality: str
    input_characters: int
    audio_output_tokens: int
    audio_seconds: float
    estimated_cost_usd: float
    pricing_known: bool
    usage_source: str
    metadata_available: bool
    error_type: str
    error_code: str
    error_status: str
    error_message: str
    error_transient: bool
    created_at: str


@dataclass(frozen=True)
class ModelCatalogRecord:
    """One configurable provider model and its native USD rates."""

    id: str
    provider: str
    modality: str
    model_id: str
    display_name: str
    enabled: bool
    is_default: bool
    is_builtin: bool
    revision: int
    sort_order: int
    input_token_rate: float
    cached_input_token_rate: float
    output_token_rate: float
    input_character_rate: float
    audio_output_token_rate: float
    created_at: str
    updated_at: str


@dataclass(frozen=True)
class ColorPaletteRecord:
    """A semantic color palette available to animation builders."""

    id: str
    name: str
    description: str
    background: str
    primary_text: str
    secondary_text: str
    surface: str
    primary: str
    secondary: str
    highlight: str
    stroke: str
    enabled: bool
    is_builtin: bool
    revision: int
    sort_order: int
    created_at: str
    updated_at: str
