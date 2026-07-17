"""Central configuration constants for Olympianim."""

from __future__ import annotations

import os
import sys
from pathlib import Path

# --- Application identity -------------------------------------------------
APP_NAME: str = "Olympianim"
APP_SLUG: str = "olympianim"
APP_VERSION: str = "0.1.0"

# --- Python version policy -------------------------------------------------
# Python 3.12 is the supported runtime for the validated dependency stack.
MIN_PYTHON_VERSION: tuple[int, int] = (3, 12)
MAX_PYTHON_VERSION_EXCLUSIVE: tuple[int, int] = (3, 13)


def check_python_version() -> None:
    """Abort early when the interpreter is outside the supported range."""
    current = sys.version_info[:2]
    if current < MIN_PYTHON_VERSION or current >= MAX_PYTHON_VERSION_EXCLUSIVE:
        raise SystemExit(f"{APP_NAME} requires Python 3.12 (detected {sys.version.split()[0]}).")


# --- Project paths --------------------------------------------------------
PACKAGE_ROOT: Path = Path(__file__).resolve().parent
PROJECT_ROOT: Path = Path(os.environ.get("OLYMPIANIM_HOME", Path.cwd())).resolve()
ASSETS_DIR: Path = PACKAGE_ROOT / "assets"
LOGO_LIGHT_PATH: Path = ASSETS_DIR / "olympianim-logo-light.png"
LOGO_DARK_PATH: Path = ASSETS_DIR / "olympianim-logo-dark.png"
WORKSPACE_DIR: Path = PROJECT_ROOT / "workspace"
PROJECTS_DIR: Path = WORKSPACE_DIR / "projects"
DATABASE_PATH: Path = WORKSPACE_DIR / "olympianim.db"
ENV_FILE: Path = PROJECT_ROOT / ".env"
ENV_EXAMPLE_FILE: Path = PROJECT_ROOT / ".env.example"

# --- Render policy --------------------------------------------------------
MAX_RENDER_RETRIES: int = 3
DEFAULT_RENDER_TIMEOUT_SECONDS: int = 15 * 60
DEFAULT_RENDER_QUALITY: str = "low_quality"

# --- Language-model request policy ---------------------------------------
# Reasoning models may legitimately take several minutes. Keep this aligned
# with the OpenAI Python SDK's default read timeout instead of imposing the
# previous two-minute limit on every provider.
DEFAULT_LLM_REQUEST_TIMEOUT_SECONDS: int = 10 * 60
MIN_LLM_REQUEST_TIMEOUT_SECONDS: int = 60
MAX_LLM_REQUEST_TIMEOUT_SECONDS: int = 60 * 60
DEFAULT_LLM_MAX_RETRIES: int = 2
MANIM_REFERENCE_TOOL_CALL_LIMIT: int = 10


def llm_request_timeout_seconds() -> int:
    """Return the bounded per-request LLM timeout configured for this process."""
    raw_value = os.environ.get("OLYMPIANIM_LLM_TIMEOUT_SECONDS", "").strip()
    if not raw_value:
        return DEFAULT_LLM_REQUEST_TIMEOUT_SECONDS
    try:
        configured = int(raw_value)
    except ValueError:
        return DEFAULT_LLM_REQUEST_TIMEOUT_SECONDS
    return min(
        max(configured, MIN_LLM_REQUEST_TIMEOUT_SECONDS),
        MAX_LLM_REQUEST_TIMEOUT_SECONDS,
    )
