"""Unit tests for the central configuration module."""

from __future__ import annotations

import sys

import pytest

from olympianim import config


def test_app_identity_is_consistent() -> None:
    assert config.APP_NAME == "Olympianim"
    assert config.APP_SLUG == "olympianim"
    assert config.APP_VERSION == "0.1.0"


def test_package_version_matches_config() -> None:
    from olympianim import __version__

    assert __version__ == config.APP_VERSION


def test_minimum_python_is_312() -> None:
    assert config.MIN_PYTHON_VERSION == (3, 12)
    assert config.MAX_PYTHON_VERSION_EXCLUSIVE == (3, 13)


def test_check_python_version_passes_on_current_interpreter() -> None:
    config.check_python_version()


def test_check_python_version_aborts_when_too_old(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(sys, "version_info", (3, 10, 0, "final", 0))
    with pytest.raises(SystemExit):
        config.check_python_version()


def test_check_python_version_aborts_when_too_new(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(sys, "version_info", (3, 13, 0, "final", 0))
    with pytest.raises(SystemExit):
        config.check_python_version()


def test_render_retry_limit_is_three() -> None:
    assert config.MAX_RENDER_RETRIES == 3


def test_render_timeout_is_bounded() -> None:
    assert config.DEFAULT_RENDER_TIMEOUT_SECONDS == 15 * 60


def test_llm_timeout_defaults_to_ten_minutes(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.delenv("OLYMPIANIM_LLM_TIMEOUT_SECONDS", raising=False)
    assert config.llm_request_timeout_seconds() == 10 * 60


@pytest.mark.parametrize(
    ("configured", "expected"),
    [
        ("900", 900),
        ("20", config.MIN_LLM_REQUEST_TIMEOUT_SECONDS),
        ("7200", config.MAX_LLM_REQUEST_TIMEOUT_SECONDS),
        ("invalid", config.DEFAULT_LLM_REQUEST_TIMEOUT_SECONDS),
    ],
)
def test_llm_timeout_configuration_is_bounded(
    monkeypatch: pytest.MonkeyPatch,
    configured: str,
    expected: int,
) -> None:
    monkeypatch.setenv("OLYMPIANIM_LLM_TIMEOUT_SECONDS", configured)
    assert config.llm_request_timeout_seconds() == expected


def test_project_paths_are_inside_workspace() -> None:
    assert config.PROJECTS_DIR.parent == config.WORKSPACE_DIR
    assert config.DATABASE_PATH.parent == config.WORKSPACE_DIR


def test_logo_variants_are_packaged() -> None:
    assert config.LOGO_LIGHT_PATH.is_file()
    assert config.LOGO_DARK_PATH.is_file()


def test_pep561_marker_is_packaged() -> None:
    assert (config.PACKAGE_ROOT / "py.typed").is_file()


def test_env_example_is_versioned_but_env_is_not() -> None:
    assert config.ENV_EXAMPLE_FILE.name == ".env.example"
    assert config.ENV_FILE.name == ".env"
