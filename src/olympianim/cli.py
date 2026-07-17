"""Command line entry point for the installed Olympianim application."""

from __future__ import annotations

import subprocess
import sys
from pathlib import Path

from olympianim.config import APP_NAME, APP_VERSION, check_python_version


def main() -> int:
    """Launch the packaged Streamlit interface."""
    check_python_version()
    print(f"{APP_NAME} {APP_VERSION}")
    app_path = Path(__file__).with_name("app.py")
    completed = subprocess.run(
        [sys.executable, "-m", "streamlit", "run", str(app_path), *sys.argv[1:]],
        check=False,
    )
    return completed.returncode


if __name__ == "__main__":
    sys.exit(main())
