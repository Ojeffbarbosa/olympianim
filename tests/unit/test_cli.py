"""Tests for the installed command line entry point."""

from __future__ import annotations

import subprocess
import sys

from olympianim import cli


def test_cli_launches_packaged_streamlit_app(monkeypatch) -> None:
    captured: dict[str, object] = {}

    def fake_run(command: list[str], *, check: bool) -> subprocess.CompletedProcess[str]:
        captured["command"] = command
        captured["check"] = check
        return subprocess.CompletedProcess(command, returncode=0)

    monkeypatch.setattr(subprocess, "run", fake_run)
    monkeypatch.setattr(sys, "argv", ["olympianim", "--server.headless=true"])

    assert cli.main() == 0
    command = captured["command"]
    assert isinstance(command, list)
    assert command[:4] == [sys.executable, "-m", "streamlit", "run"]
    assert command[-1] == "--server.headless=true"
    assert captured["check"] is False
