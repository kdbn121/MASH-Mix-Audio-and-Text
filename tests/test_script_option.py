# SPDX-License-Identifier: MIT | Copyright (c) 2026 MASH (Khaled Bhuiyan)
"""Tests for the script layer, the Banglish option and the lyric filters."""

from __future__ import annotations

import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from mash_mix.lang.translit import (  # noqa: E402
    apply_script,
    banglish_word_ratio,
    resolve_script_target,
)
from mash_mix.tasks.lyrics import (  # noqa: E402
    _collapse_word_repeats,
    _is_repetition_loop,
    _split_on_pauses,
)
from mash_mix.types import Segment, Word  # noqa: E402

# Written as code points on purpose. The Bangla and Devanagari letters look
# alike, so a literal in a source file can be silently changed into its twin.
BANLA = "\u09a6\u09cb\u09b2"
CHIRUNI = "\u099a\u09bf\u09b0\u09c1\u09a8\u09bf"
MAN_KORO = "\u09ae\u09be\u09a8 \u09a4\u09c1\u09ae\u09bf \u0995\u09cb\u09b0\u09cb \u09a8\u09be"
BANGLISH = "dol dol doluni"
DEVA = "\u0926\u094b\u0932"


def test_banglish_word_ratio_counts_romanised_bangla():
    assert banglish_word_ratio(BANGLISH) > 0.9
    assert banglish_word_ratio(BANLA) == 0.0


def test_apply_script_writes_romanised_bangla_in_bangla():
    out = apply_script(BANGLISH, "bangla")
    assert "dol" not in out.lower()
    assert any("\u0980" <= ch <= "\u09ff" for ch in out)


def test_apply_script_keeps_english_alone():
    text = "the weather is fine today"
    assert apply_script(text, "bangla") == text


def test_apply_script_to_banglish():
    out = apply_script(BANLA, "banglish")
    assert out.isascii()
    assert out.lower().startswith("dol")


def test_apply_script_rejects_an_unknown_script():
    with pytest.raises(ValueError):
        apply_script(BANLA, "klingon")


def test_resolve_script_target_picks_bangla_for_a_romanised_document():
    assert resolve_script_target(BANGLISH, "bangla") == "bangla"


def test_resolve_script_target_leaves_an_english_document_alone():
    assert resolve_script_target("the weather is fine today", "bangla") == ""


def test_resolve_script_target_always_converts_to_banglish():
    assert resolve_script_target(BANLA, "banglish") == "banglish"


def test_collapse_word_repeats():
    assert _collapse_word_repeats("a b c c c c d") == "a b c c d"
    assert _collapse_word_repeats("") == ""


def test_is_repetition_loop():
    assert _is_repetition_loop("ra ra ra ra ra ra ra ra")
    assert not _is_repetition_loop("dol dol doluni ranga mathar chiruni")


def test_split_on_pauses_breaks_a_long_segment():
    words = [
        Word(start=0.0, end=0.4, text="dol"),
        Word(start=0.4, end=0.8, text="dol"),
        Word(start=3.0, end=3.4, text="chiruni"),
    ]
    segments = [Segment(start=0.0, end=3.4, text="dol dol chiruni", words=words)]
    pieces = _split_on_pauses(segments)
    assert len(pieces) == 2
    assert pieces[0].text == "dol dol"
    assert pieces[1].text == "chiruni"


def test_split_on_pauses_keeps_a_segment_without_words():
    segments = [Segment(start=0.0, end=1.0, text="a line")]
    assert _split_on_pauses(segments) == segments
