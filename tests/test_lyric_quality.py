# SPDX-License-Identifier: MIT | Copyright (c) 2026 MASH (Khaled Bhuiyan)
"""Tests for the lyric quality filters, the line splitter and the scorer.

The Bangla letters are written as code points on purpose. The Bangla and
Devanagari letters look alike, so a literal in a source file can be silently
changed into its twin.
"""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from mash_mix.tasks.lyrics import (  # noqa: E402
    _has_replacement_char,
    _is_character_loop,
    _merge_short_lines,
    _split_on_pauses,
)
from mash_mix.types import LyricLine, Segment, Word  # noqa: E402

# The danda, the Bangla full stop.
DANDA = "\u0964"
# A syllable that repeats with several letters, so no word repeats with it.
LOOP = "\u09ae\u09be\u09a5\u09cd\u09af\u09be\u09b2\u09cd\u09af\u09be\u09ae\u09cd\u09af\u09be\u09b2\u09cd\u09af\u09be"


def test_is_character_loop_catches_a_repeated_syllable():
    assert _is_character_loop(LOOP * 3)
    assert _is_character_loop("ra ra ra ra ra ra ra ra ra ra")


def test_is_character_loop_catches_a_short_block_at_the_end_of_a_line():
    """A short syllable repeated three times is a loop even in a long line."""
    text = "\u09a4\u09c1\u09ae\u09bf \u09a4\u09c1\u09ae\u09bf \u09a4\u09c1\u09a4\u09cd\u09ac\u09be\u09ac\u09c7 " + "\u09a4\u09ac\u09c7 " * 3
    assert _is_character_loop(text)


def test_is_character_loop_leaves_real_lyrics_alone():
    line = "\u09a6\u09cb\u09b2 \u09a6\u09cb\u09b2 \u09a6\u09c1\u09b2\u09c1\u09a8\u09bf"
    assert not _is_character_loop(line)
    assert not _is_character_loop("")


def test_has_replacement_char():
    assert _has_replacement_char("dol \ufffd dol")
    assert not _has_replacement_char("dol dol")


def test_split_on_pauses_breaks_after_a_full_stop():
    words = [
        Word(start=0.0, end=0.4, text="dol"),
        Word(start=0.4, end=0.9, text="chiruni" + DANDA),
        Word(start=1.0, end=1.4, text="ranga"),
    ]
    segments = [Segment(start=0.0, end=1.4, text="x", words=words)]
    pieces = _split_on_pauses(segments)
    assert len(pieces) == 2
    assert pieces[0].text == "dol chiruni" + DANDA
    assert pieces[1].text == "ranga"


def test_split_on_pauses_keeps_a_short_clause_with_its_phrase():
    words = [
        Word(start=0.0, end=0.4, text="ok"),
        Word(start=0.4, end=0.9, text="ko,"),
        Word(start=0.9, end=1.4, text="ranga"),
    ]
    segments = [Segment(start=0.0, end=1.4, text="x", words=words)]
    pieces = _split_on_pauses(segments)
    assert len(pieces) == 1


def test_split_on_pauses_breaks_a_long_clause():
    words = [
        Word(start=0.0, end=0.4, text="ranga mathar chiruni,"),
        Word(start=0.4, end=0.9, text="ranga"),
    ]
    segments = [Segment(start=0.0, end=0.9, text="x", words=words)]
    pieces = _split_on_pauses(segments)
    assert len(pieces) == 2


def test_merge_short_lines_respects_the_length_cap():
    lines = [
        LyricLine(start=0.0, end=0.5, text="short one"),
        LyricLine(start=0.5, end=1.0, text="x" * 60),
    ]
    merged = _merge_short_lines(lines, max_chars=45)
    assert len(merged) == 2


def test_merge_short_lines_joins_a_short_fragment():
    lines = [
        LyricLine(start=0.0, end=0.5, text="ok"),
        LyricLine(start=0.5, end=3.0, text="ranga mathar chiruni"),
    ]
    merged = _merge_short_lines(lines)
    assert len(merged) == 1
    assert merged[0].text == "ok ranga mathar chiruni"


def test_the_scorer_finds_a_line_that_sits_inside_a_longer_line():
    sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))
    from lyrics_eval import match_lines, phrase_recall

    truth = ["dol dol doluni", "ranga mathar chiruni"]
    hypothesis = ["dol dol doluni ranga mathar chiruni mane tumi koro na"]

    _, line_recall = match_lines(truth, hypothesis)
    _, content_recall = phrase_recall(truth, hypothesis)

    assert line_recall == 0.0
    assert content_recall == 1.0
