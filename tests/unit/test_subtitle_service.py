"""Tests for native Manim subtitle preservation and concatenation."""

from olympianim.services.subtitle_service import SubtitleService

_PRESENTATION = """1
00:00:00,000 --> 00:00:01,250
Leia o problema.

2
00:00:01,500 --> 00:00:02,000
O que você observa?
"""

_SOLUTION = """1
00:00:00,100 --> 00:00:01,000
Vamos resolver.
"""


def test_srt_transcript_preserves_cue_order() -> None:
    assert SubtitleService.transcript(_PRESENTATION) == ("Leia o problema.\nO que você observa?\n")


def test_combined_srt_offsets_solution_by_presentation_duration() -> None:
    combined = SubtitleService.combine(
        _PRESENTATION,
        _SOLUTION,
        offset_seconds=2.5,
    )

    cues = SubtitleService.parse(combined)
    assert len(cues) == 3
    assert cues[2].start_ms == 2600
    assert cues[2].end_ms == 3500
    assert cues[2].text == "Vamos resolver."
