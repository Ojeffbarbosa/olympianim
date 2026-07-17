"""Session-state plumbing for the Streamlit interface.

Streamlit reruns scripts on every interaction, so durable form values
and generation results must live in ``st.session_state``. This module
centralises the key names and offers typed accessors so the page files
do not scatter string literals around.

No API keys are persisted to disk here. Keys typed in
the interface stay only in memory for the duration of the session.
"""

from __future__ import annotations

from collections.abc import Callable
from typing import Any

import streamlit as st

from olympianim.database.models import ProjectRecord

# ---------------------------------------------------------------------------
# Session-state key names
# ---------------------------------------------------------------------------
# Problem input
KEY_PROJECT_TITLE = "project_title"
KEY_PROBLEM_STATEMENT = "problem_statement"
KEY_PROBLEM_IMAGE = "problem_image"
KEY_PROBLEM_SOURCE = "problem_source"
KEY_PROBLEM_LEVEL = "problem_level"
KEY_MATH_AREA = "math_area"
KEY_TEACHER_SOLUTION = "teacher_solution"
KEY_SOLUTION_IMAGES = "solution_images"
KEY_TEACHER_INSTRUCTIONS = "teacher_instructions"

# AI configuration
KEY_LLM_PROVIDER = "llm_provider"
KEY_LLM_MODEL = "llm_model"
KEY_LLM_API_KEY = "llm_api_key"
KEY_LLM_API_KEY_SOURCE = "llm_api_key_source"  # "env" | "session" | ""

# Voice configuration
KEY_VOICE_PROVIDER = "voice_provider"
KEY_VOICE_MODEL = "voice_model"
KEY_VOICE = "voice"
KEY_VOICE_LANGUAGE = "voice_language"
KEY_VOICE_SPEED = "voice_speed"
KEY_VOICEOVER_ENABLED = "voiceover_enabled"
KEY_VOICE_API_KEY = "voice_api_key"
KEY_VOICE_API_KEY_RESOLVED = "voice_api_key_resolved"
KEY_VOICE_API_KEY_SOURCE = "voice_api_key_source"
KEY_REUSE_LLM_API_KEY = "reuse_llm_api_key"

# Visual palette configuration
KEY_COLOR_PALETTE_ID = "color_palette_id"
KEY_GENERATION_PREFERENCES_DRAFT = "generation_preferences_draft"

# Generation progress and results
KEY_CURRENT_PROJECT_ID = "current_project_id"
KEY_PROGRESS_LOG = "progress_log"
KEY_PRESENTATION_VIDEO = "presentation_video_path"
KEY_SOLUTION_VIDEO = "solution_video_path"
KEY_GENERATION_METADATA = "generation_metadata"

# Credential store. Lives only in memory for the session.
KEY_CREDENTIAL_STORE = "credential_store"
KEY_CONNECTION_RESULT = "connection_result"
KEY_API_KEY_RESOLVED = "api_key_resolved"
KEY_API_KEY_SOURCE = "api_key_source"


def get[T](key: str, default: T | None = None) -> T | Any:
    """Return a session-state value, initialising it to ``default``.

    Using this helper guarantees the key exists before the first read,
    avoiding ``KeyError`` on the initial render.
    """
    if key not in st.session_state:
        st.session_state[key] = default
    return st.session_state[key]


def set(key: str, value: Any) -> None:
    """Overwrite a session-state value."""
    st.session_state[key] = value


def update(key: str, default: Any, factory: Callable[[Any], Any]) -> None:
    """Read ``key`` (initialised with ``default``) and replace it via ``factory``."""
    current = get(key, default)
    st.session_state[key] = factory(current)


def append_progress(message: str) -> None:
    """Append a line to the on-screen progress log."""
    log: list[str] = get(KEY_PROGRESS_LOG, [])
    log.append(message)
    st.session_state[KEY_PROGRESS_LOG] = log


def clear_progress() -> None:
    """Reset the progress log before a new generation run."""
    st.session_state[KEY_PROGRESS_LOG] = []


def select_project(project: ProjectRecord) -> None:
    """Select an existing project without mutating the new-project form."""
    set(KEY_CURRENT_PROJECT_ID, project.id)
    set(KEY_PRESENTATION_VIDEO, project.presentation_video_path)
    set(KEY_SOLUTION_VIDEO, project.solution_video_path)
    set(
        KEY_GENERATION_METADATA,
        {
            "project_id": project.id,
            "status": project.status,
            "created_at": project.created_at,
            "updated_at": project.updated_at,
            "presentation_code_path": project.presentation_code_path,
            "solution_code_path": project.solution_code_path,
        },
    )


def init_defaults() -> None:
    """Populate session state with the default form values.

    Call once at the top of the page so selectors have a stable initial
    value across reruns.
    """
    from olympianim.services.generation_preferences import (
        GenerationPreferences,
        GenerationPreferencesService,
    )
    from olympianim.ui.options import (
        DEFAULT_MATH_AREA,
        VOICE_LANGUAGES,
        VOICE_OPTIONS,
        active_llm_providers,
        active_voice_providers,
        models_for,
        voice_models_for,
    )

    llm_providers = active_llm_providers()
    voice_providers = active_voice_providers()
    saved = GenerationPreferencesService().load() or GenerationPreferences()
    draft = get(KEY_GENERATION_PREFERENCES_DRAFT, saved)
    if not isinstance(draft, GenerationPreferences):
        draft = saved
        set(KEY_GENERATION_PREFERENCES_DRAFT, draft)
    llm_provider = _valid_choice(draft.llm_provider, llm_providers)
    voice_provider = _valid_choice(draft.voice_provider, voice_providers)
    llm_models = models_for(llm_provider) if llm_provider else ()
    voice_models = voice_models_for(voice_provider) if voice_provider else ()
    voices = VOICE_OPTIONS.get(voice_provider, ())

    defaults: dict[str, Any] = {
        KEY_PROJECT_TITLE: "",
        KEY_PROBLEM_STATEMENT: "",
        KEY_PROBLEM_SOURCE: "",
        KEY_PROBLEM_LEVEL: "",
        KEY_MATH_AREA: DEFAULT_MATH_AREA,
        KEY_TEACHER_SOLUTION: "",
        KEY_TEACHER_INSTRUCTIONS: "",
        KEY_LLM_PROVIDER: llm_provider,
        KEY_LLM_MODEL: _valid_choice(draft.llm_model, llm_models),
        KEY_LLM_API_KEY: "",
        KEY_LLM_API_KEY_SOURCE: "",
        KEY_VOICE_PROVIDER: voice_provider,
        KEY_VOICE_MODEL: _valid_choice(draft.voice_model, voice_models),
        KEY_VOICE: _valid_choice(draft.voice, voices),
        KEY_VOICE_LANGUAGE: _valid_choice(draft.voice_language, VOICE_LANGUAGES),
        KEY_VOICE_SPEED: draft.voice_speed,
        KEY_VOICEOVER_ENABLED: draft.voiceover_enabled,
        KEY_VOICE_API_KEY: "",
        KEY_VOICE_API_KEY_RESOLVED: "",
        KEY_VOICE_API_KEY_SOURCE: "",
        KEY_REUSE_LLM_API_KEY: draft.reuse_llm_api_key,
        KEY_COLOR_PALETTE_ID: draft.color_palette_id,
        KEY_CURRENT_PROJECT_ID: "",
        KEY_PROGRESS_LOG: [],
        KEY_PRESENTATION_VIDEO: "",
        KEY_SOLUTION_VIDEO: "",
        KEY_GENERATION_METADATA: {},
    }
    for key, value in defaults.items():
        if key not in st.session_state:
            st.session_state[key] = value


def _valid_choice(preferred: str, options: tuple[str, ...]) -> str:
    """Return a saved option when valid, otherwise the current catalog default."""
    if preferred in options:
        return preferred
    return options[0] if options else ""
