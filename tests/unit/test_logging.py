"""Unit tests for the secure logging helpers."""

from __future__ import annotations

import logging

import pytest

from olympianim.utils.logging import (
    REDACTED_PLACEHOLDER,
    SecretFilter,
    looks_like_key_prefix,
    redact,
)


def test_redact_replaces_secret() -> None:
    assert redact("token=sk-secret-123 end", ["sk-secret-123"]) == (
        f"token={REDACTED_PLACEHOLDER} end"
    )


def test_redact_handles_multiple_secrets() -> None:
    out = redact("a sk-1 b sk-22 c", ["sk-1", "sk-22"])
    assert "sk-1" not in out
    assert "sk-22" not in out
    assert out.count(REDACTED_PLACEHOLDER) == 2


def test_redact_replaces_longer_secret_first() -> None:
    # If the shorter secret were replaced first, the prefix of the longer
    # one could leak; ordering by length prevents that.
    out = redact("sk-secret sk-secret-extra", ["sk-secret", "sk-secret-extra"])
    assert "sk-secret-extra" not in out
    assert out.count(REDACTED_PLACEHOLDER) == 2


def test_redact_ignores_empty_secret() -> None:
    assert redact("nothing changes", [""]) == "nothing changes"


def test_secret_filter_registers_and_redacts(caplog: pytest.LogCaptureFixture) -> None:
    filt = SecretFilter()
    filt.register("sk-super-secret")
    logger = logging.getLogger("test_secret_filter")
    logger.handlers.clear()
    logger.addFilter(filt)
    logger.setLevel(logging.DEBUG)
    with caplog.at_level(logging.DEBUG, logger="test_secret_filter"):
        logger.info("using key sk-super-secret now")
    assert "sk-super-secret" not in caplog.text
    assert REDACTED_PLACEHOLDER in caplog.text


def test_secret_filter_clear_removes_secrets() -> None:
    filt = SecretFilter()
    filt.register("sk-x")
    assert filt.registered_count == 1
    filt.clear()
    assert filt.registered_count == 0


def test_looks_like_key_prefix_detects_known_prefixes() -> None:
    assert looks_like_key_prefix("sk-abc")
    assert looks_like_key_prefix("AIzaXYZ")
    assert looks_like_key_prefix("sk-ant-xyz")
    assert not looks_like_key_prefix("hello")
    assert not looks_like_key_prefix("")
