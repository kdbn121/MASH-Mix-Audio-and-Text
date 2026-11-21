# SPDX-License-Identifier: MIT | Copyright (c) 2026 MASH (Khaled Bhuiyan)
"""The cue builder that turns word timings into karaoke subtitles."""

from __future__ import annotations

from mash_mix.export.subtitles import segments_to_lines, to_srt
from mash_mix.export.word_cues import Cue, build_cues, cues_from_lines
from mash_mix.export.word_cues import to_ass as cues_to_ass
from mash_mix.export.word_cues import to_srt as cues_to_srt
from mash_mix.export.word_cues import to_vtt as cues_to_vtt
from mash_mix.types import LyricLine, Segment, Word


def _words(*pairs):
    """Build words out of (text, start, end) triples."""
    return [
        Word(start=start, end=end, text=text)
        for text, start, end in pairs
    ]


def _line(text, start, end, words=None):
    return LyricLine(start=start, end=end, text=text, words=list(words or []))


def test_a_cue_never_crosses_a_silence():
    words = _words(
        ("one", 0.0, 0.4),
        ("two", 0.5, 0.9),
        ("three", 4.0, 4.4),
    )
    cues = build_cues(words, max_gap=0.8)
    assert len(cues) == 2
    assert cues[0].text == "one two"
    assert cues[1].text == "three"


def test_a_cue_never_holds_more_than_the_word_limit():
    words = _words(*[(f"w{index}", index * 0.2, index * 0.2 + 0.15) for index in range(12)])
    cues = build_cues(words, max_words=4, max_gap=10.0)
    assert all(len(cue.words) <= 4 for cue in cues)
    assert len(cues) == 3


def test_a_cue_never_holds_more_than_the_character_limit():
    words = _words(
        ("aaaaaaaa", 0.0, 0.2),
        ("bbbbbbbb", 0.3, 0.5),
        ("cccccccc", 0.6, 0.8),
    )
    cues = build_cues(words, max_chars=10, max_gap=10.0)
    assert all(len(cue.text) <= 10 for cue in cues)


def test_a_sentence_closes_the_cue_early():
    words = _words(
        ("one", 0.0, 0.3),
        ("two.", 0.4, 0.7),
        ("three", 0.8, 1.1),
    )
    cues = build_cues(words, max_words=8, max_gap=10.0)
    assert cues[0].text == "one two."
    assert cues[1].text == "three"


def test_a_bangla_danda_closes_the_cue():
    words = _words(
        ("ওরে", 0.0, 0.3),
        ("নীল।", 0.4, 0.7),
        ("দরিয়া", 0.8, 1.1),
    )
    cues = build_cues(words, max_words=8, max_gap=10.0)
    assert cues[0].text == "ওরে নীল।"
    assert cues[1].text == "দরিয়া"


def test_the_karaoke_style_marks_the_word_being_sung():
    words = _words(
        ("ওরে", 0.0, 0.5),
        ("নীল", 0.5, 1.0),
        ("দরিয়া", 1.0, 1.6),
    )
    cues = build_cues(words)
    rendered = cues_to_srt(cues, "karaoke")
    assert "<b>ওরে</b> নীল দরিয়া" in rendered
    assert "ওরে <b>নীল</b> দরিয়া" in rendered
    assert "ওরে নীল <b>দরিয়া</b>" in rendered


def test_the_karaoke_style_holds_the_whole_line_on_the_screen():
    words = _words(
        ("ওরে", 0.0, 0.5),
        ("নীল", 0.5, 1.0),
    )
    cues = build_cues(words)
    rendered = cues_to_srt(cues, "karaoke")
    bodies = [block.splitlines()[2] for block in rendered.strip().split("\n\n")]
    assert len(bodies) == 2
    for body in bodies:
        assert "ওরে" in body and "নীল" in body


def test_the_ass_style_uses_the_karaoke_tag():
    words = _words(
        ("ওরে", 0.0, 0.5),
        ("নীল", 0.5, 1.0),
    )
    cues = build_cues(words)
    rendered = cues_to_ass(cues, "karaoke")
    assert r"{\k50}" in rendered


def test_the_plain_style_is_untouched_by_the_word_timings():
    words = _words(("ওরে", 0.0, 0.5), ("নীল", 0.5, 1.0))
    line = _line("ওরে নীল", 0.0, 1.0, words)
    rendered = to_srt([line])
    assert "<b>" not in rendered
    assert rendered.strip().endswith("ওরে নীল")


def test_a_line_with_no_word_timings_is_written_the_plain_way():
    line = _line("শুধু লাইন", 0.0, 2.0)
    rendered = to_srt([line], style="karaoke")
    assert "<b>" not in rendered
    assert "শুধু লাইন" in rendered


def test_the_vtt_style_escapes_the_markup():
    words = _words(("a<b", 0.0, 0.5))
    cues = build_cues(words)
    rendered = cues_to_vtt(cues, "karaoke")
    assert "<b>a&lt;b</b>" in rendered


def test_a_silence_between_two_cues_is_kept():
    words = _words(
        ("one", 0.0, 0.2),
        ("two", 0.3, 0.5),
        ("three", 3.0, 3.2),
    )
    cues = build_cues(words)
    assert len(cues) == 2
    assert cues[0].text == "one two"
    assert cues[1].text == "three"


def test_a_short_cue_is_held_by_pushing_the_next_one_back():
    words = _words(
        ("one", 0.0, 0.2),
        ("two", 0.2, 0.4),
        ("three", 5.0, 5.4),
    )
    cues = build_cues(words, min_seconds=1.0)
    assert cues[0].text == "one two"
    assert cues[0].duration >= 1.0
    assert cues[1].text == "three"


def test_the_cues_of_a_line_keep_the_speaker():
    line = LyricLine(
        start=0.0,
        end=1.0,
        text="ওরে নীল",
        speaker="SPEAKER_00",
        words=_words(("ওরে", 0.0, 0.5), ("নীল", 0.5, 1.0)),
    )
    cues = cues_from_lines([line])
    assert cues[0].speaker == "SPEAKER_00"


def test_the_segments_keep_their_word_timings():
    segment = Segment(
        start=0.0,
        end=1.0,
        text="ওরে নীল",
        words=_words(("ওরে", 0.0, 0.5), ("নীল", 0.5, 1.0)),
    )
    lines = segments_to_lines([segment])
    assert len(lines[0].words) == 2
    assert lines[0].words[0].text == "ওরে"
    assert lines[0].words[1].end == 1.0


def test_a_cue_number_counts_from_one():
    words = _words(("a", 0.0, 0.4), ("b", 0.5, 0.9))
    cues = build_cues(words)
    assert cues[0].index == 1


def test_the_word_by_word_style_gives_every_word_a_cue():
    words = _words(("ওরে", 0.0, 0.5), ("নীল", 0.5, 1.0), ("দরিয়া", 1.0, 1.6))
    cues = build_cues(words)
    rendered = cues_to_srt(cues, "word_by_word")
    bodies = [block.splitlines()[2] for block in rendered.strip().split("\n\n")]
    assert bodies == ["ওরে", "নীল", "দরিয়া"]


def test_the_vtt_file_starts_with_the_header():
    words = _words(("ওরে", 0.0, 0.5))
    cues = build_cues(words)
    assert cues_to_srt(cues, "plain")
    from mash_mix.export.word_cues import to_vtt as cues_to_vtt

    assert cues_to_vtt(cues, "karaoke").startswith("WEBVTT")


def test_an_empty_cue_is_allowed():
    cue = Cue(index=1, start=0.0, end=1.0, text="", words=[])
    assert cue.duration == 1.0
