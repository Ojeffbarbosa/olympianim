"""Unit tests for the session-state accessors in ``olympianim.ui.state``.

Streamlit's ``st.session_state`` only exists inside a live script run,
so these tests stub ``streamlit`` with a lightweight in-memory dict
before importing the module. This keeps the tests fast and independent
of a running Streamlit server.
"""

from __future__ import annotations

import sys
import types
from collections.abc import Mapping
from typing import Any

import pytest

# ---------------------------------------------------------------------------
# Minimal streamlit stub providing only what ``state.py`` touches.
# ---------------------------------------------------------------------------
_STUB_NS: dict[str, Any] = {}


class _SessionStateStub(dict[str, Any]):
    """Dict subclass that mimics ``st.session_state`` semantics."""


def _make_streamlit_stub() -> types.ModuleType:
    st = types.ModuleType("streamlit")
    st.session_state = _SessionStateStub()
    return st


@pytest.fixture()
def st_stub(monkeypatch: pytest.MonkeyPatch) -> types.ModuleType:
    """Install a fresh streamlit stub and reload the state module.

    ``state.py`` captures ``streamlit`` at import time, so we must drop
    every cached ``olympianim.*`` module (not only ``olympianim.ui.state``)
    to guarantee the state accessors rebind to the new stub's
    ``session_state``.
    """
    stub = _make_streamlit_stub()
    monkeypatch.setitem(sys.modules, "streamlit", stub)
    for mod in list(sys.modules):
        if mod.startswith("olympianim"):
            sys.modules.pop(mod, None)
    return stub


def test_get_initialises_default(st_stub: types.ModuleType) -> None:
    from olympianim.ui import state

    assert state.get("missing_key", "default") == "default"
    assert st_stub.session_state["missing_key"] == "default"


def test_set_overwrites_value(st_stub: types.ModuleType) -> None:
    from olympianim.ui import state

    state.set("colour", "red")
    assert st_stub.session_state["colour"] == "red"
    state.set("colour", "blue")
    assert st_stub.session_state["colour"] == "blue"


def test_update_applies_factory(st_stub: types.ModuleType) -> None:
    from olympianim.ui import state

    state.set("counter", 0)
    state.update("counter", 0, lambda current: current + 5)
    assert st_stub.session_state["counter"] == 5


def test_append_progress_grows_log(st_stub: types.ModuleType) -> None:
    from olympianim.ui import state

    state.append_progress("step one")
    state.append_progress("step two")
    assert st_stub.session_state[state.KEY_PROGRESS_LOG] == ["step one", "step two"]


def test_clear_progress_resets_log(st_stub: types.ModuleType) -> None:
    from olympianim.ui import state

    state.append_progress("hello")
    state.clear_progress()
    assert st_stub.session_state[state.KEY_PROGRESS_LOG] == []


def test_init_defaults_populates_expected_keys(
    st_stub: types.ModuleType,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from olympianim.services.generation_preferences import GenerationPreferencesService
    from olympianim.ui import state

    monkeypatch.setattr(GenerationPreferencesService, "load", lambda _: None)
    state.init_defaults()

    expected: Mapping[str, object] = {
        state.KEY_PROJECT_TITLE: "",
        state.KEY_PROBLEM_STATEMENT: "",
        state.KEY_MATH_AREA: "Automática",
        state.KEY_LLM_PROVIDER: "OpenAI",
        state.KEY_VOICE_PROVIDER: "OpenAI",
        state.KEY_VOICE: "alloy",
        state.KEY_VOICE_LANGUAGE: "Português (Brasil)",
        state.KEY_VOICEOVER_ENABLED: False,
        state.KEY_PROGRESS_LOG: [],
    }
    for key, value in expected.items():
        assert st_stub.session_state[key] == value

    from olympianim.ui import options

    assert st_stub.session_state[state.KEY_LLM_MODEL] in options.models_for("OpenAI")


def test_init_defaults_does_not_overwrite_existing(st_stub: types.ModuleType) -> None:
    from olympianim.ui import state

    st_stub.session_state[state.KEY_LLM_PROVIDER] = "Anthropic"
    state.init_defaults()
    assert st_stub.session_state[state.KEY_LLM_PROVIDER] == "Anthropic"


def test_init_defaults_uses_saved_generation_preferences(
    st_stub: types.ModuleType, monkeypatch: pytest.MonkeyPatch
) -> None:
    from olympianim.services.generation_preferences import (
        GenerationPreferences,
        GenerationPreferencesService,
    )
    from olympianim.ui import state

    saved = GenerationPreferences(
        llm_provider="Google",
        llm_model="gemini-3.5-flash",
        voiceover_enabled=True,
        voice_provider="Google",
        voice_model="gemini-3.1-flash-tts-preview",
        voice="Kore",
        voice_language="Inglês (EUA)",
        voice_speed=1.3,
        reuse_llm_api_key=False,
    )
    monkeypatch.setattr(GenerationPreferencesService, "load", lambda _: saved)

    state.init_defaults()

    assert st_stub.session_state[state.KEY_LLM_PROVIDER] == "Google"
    assert st_stub.session_state[state.KEY_LLM_MODEL] == "gemini-3.5-flash"
    assert st_stub.session_state[state.KEY_VOICEOVER_ENABLED] is True
    assert st_stub.session_state[state.KEY_VOICE] == "Kore"
    assert st_stub.session_state[state.KEY_VOICE_LANGUAGE] == "Inglês (EUA)"
    assert st_stub.session_state[state.KEY_VOICE_SPEED] == 1.3


def test_init_defaults_restores_widget_values_from_session_draft(
    st_stub: types.ModuleType,
) -> None:
    from olympianim.services.generation_preferences import GenerationPreferences
    from olympianim.ui import state

    draft = GenerationPreferences(
        llm_provider="Google",
        llm_model="gemini-3.5-flash",
        voiceover_enabled=True,
        voice_provider="Google",
        voice_model="gemini-3.1-flash-tts-preview",
        voice="Kore",
        color_palette_id="builtin:manim-dark",
    )
    st_stub.session_state[state.KEY_GENERATION_PREFERENCES_DRAFT] = draft

    state.init_defaults()

    assert st_stub.session_state[state.KEY_VOICEOVER_ENABLED] is True
    assert st_stub.session_state[state.KEY_VOICE] == "Kore"
    assert st_stub.session_state[state.KEY_COLOR_PALETTE_ID] == "builtin:manim-dark"


def test_capture_draft_preserves_voice_details_while_disabled(
    st_stub: types.ModuleType,
) -> None:
    from olympianim.services.generation_preferences import GenerationPreferences
    from olympianim.ui import sections, state

    previous = GenerationPreferences(
        voiceover_enabled=True,
        voice_provider="Google",
        voice_model="gemini-3.1-flash-tts-preview",
        voice="Kore",
        voice_language="Português (Brasil)",
        voice_speed=1.2,
        reuse_llm_api_key=True,
        color_palette_id="builtin:manim-dark",
    )
    state.set(state.KEY_GENERATION_PREFERENCES_DRAFT, previous)

    captured = sections.capture_generation_preferences(
        {"llm_provider": "OpenAI", "llm_model": "gpt-5.4-mini"},
        {"voiceover_enabled": False},
        {"color_palette_id": "builtin:manim-light"},
    )

    assert captured.voiceover_enabled is False
    assert captured.voice_provider == "Google"
    assert captured.voice_model == "gemini-3.1-flash-tts-preview"
    assert captured.voice == "Kore"
    assert captured.voice_speed == 1.2
    assert captured.color_palette_id == "builtin:manim-light"


def test_select_project_does_not_replace_new_project_preferences(
    st_stub: types.ModuleType,
) -> None:
    from olympianim.database.models import ProjectRecord
    from olympianim.ui import state

    state.set(state.KEY_PROJECT_TITLE, "Novo projeto em preparação")
    state.set(state.KEY_LLM_PROVIDER, "OpenAI")
    state.set(state.KEY_LLM_MODEL, "gpt-5.4-mini")
    state.set(state.KEY_VOICEOVER_ENABLED, True)
    state.set(state.KEY_COLOR_PALETTE_ID, "builtin:manim-light")
    project = ProjectRecord(
        id="existing-project",
        title="Projeto existente",
        problem_statement="Problema",
        llm_provider="Google",
        llm_model="gemini-3.5-flash",
        voiceover_enabled=False,
        color_palette_id="builtin:manim-dark",
        presentation_video_path="presentation.mp4",
        solution_video_path="solution.mp4",
    )

    state.select_project(project)

    assert state.get(state.KEY_CURRENT_PROJECT_ID) == "existing-project"
    assert state.get(state.KEY_PRESENTATION_VIDEO) == "presentation.mp4"
    assert state.get(state.KEY_PROJECT_TITLE) == "Novo projeto em preparação"
    assert state.get(state.KEY_LLM_PROVIDER) == "OpenAI"
    assert state.get(state.KEY_LLM_MODEL) == "gpt-5.4-mini"
    assert state.get(state.KEY_VOICEOVER_ENABLED) is True
    assert state.get(state.KEY_COLOR_PALETTE_ID) == "builtin:manim-light"


def test_key_constants_are_strings() -> None:
    from olympianim.ui import state

    assert isinstance(state.KEY_PROBLEM_STATEMENT, str)
    assert isinstance(state.KEY_PRESENTATION_VIDEO, str)
    assert state.KEY_PRESENTATION_VIDEO != state.KEY_SOLUTION_VIDEO
