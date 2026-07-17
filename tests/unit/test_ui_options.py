"""Tests for UI options backed by the active model catalog."""

from __future__ import annotations

from olympianim.ui import options as opt


def test_active_catalog_exposes_supported_text_providers() -> None:
    assert set(opt.active_llm_providers()) == {"OpenAI", "Google", "Anthropic"}
    assert opt.models_for("OpenAI")


def test_voice_options_match_active_voice_providers() -> None:
    for provider in opt.active_voice_providers():
        assert opt.voice_models_for(provider)
        assert opt.voices_for(provider)


def test_math_and_voice_options_have_expected_defaults() -> None:
    assert opt.MATH_AREAS[0] == "Automática"
    assert opt.VOICE_LANGUAGES[0] == "Português (Brasil)"


def test_unknown_voice_provider_has_no_voices() -> None:
    assert opt.voices_for("Unknown") == ()
