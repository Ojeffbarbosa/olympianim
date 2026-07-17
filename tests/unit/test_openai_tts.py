"""Tests for metered OpenAI speech generation."""

from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace

import pytest

from olympianim.manim.openai_tts import OpenAISpeechService
from olympianim.manim.usage_events import read_usage_events


class _StreamingResponse:
    def __enter__(self) -> _StreamingResponse:
        return self

    def __exit__(self, *_args: object) -> None:
        return None

    @staticmethod
    def stream_to_file(path: Path) -> None:
        Path(path).write_bytes(b"fake-mp3")


class _SpeechEndpoint:
    def __init__(self) -> None:
        self.calls: list[dict[str, object]] = []
        self.with_streaming_response = self

    def create(self, **kwargs: object) -> _StreamingResponse:
        self.calls.append(kwargs)
        return _StreamingResponse()


def test_openai_speech_counts_characters_and_skips_cache(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    usage_path = tmp_path / "usage.jsonl"
    monkeypatch.setenv("OLYMPIANIM_USAGE_EVENTS_PATH", str(usage_path))
    monkeypatch.setattr("olympianim.manim.openai_tts.get_duration", lambda _path: 1.25)
    endpoint = _SpeechEndpoint()
    client = SimpleNamespace(audio=SimpleNamespace(speech=endpoint))
    service = OpenAISpeechService(
        api_key="secret",
        voice="alloy",
        model="tts-1",
        cache_dir=tmp_path,
        client=client,
    )

    service._wrap_generate_from_text("Texto para narrar.")
    service._wrap_generate_from_text("Texto para narrar.")

    events = read_usage_events(usage_path)
    assert len(endpoint.calls) == 1
    assert len(events) == 1
    assert events[0].input_characters == len("Texto para narrar.")
    assert events[0].audio_seconds == pytest.approx(1.25)
