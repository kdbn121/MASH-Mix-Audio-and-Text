# SPDX-License-Identifier: MIT | Copyright (c) 2026 MASH (Khaled Bhuiyan)
"""The subtitle and plain text exporters."""

from __future__ import annotations

from mash_mix.export.plaintext import (
    lyrics_to_text,
    to_json,
    transcript_to_text,
    write_json,
    write_text,
)
from mash_mix.export.subtitles import (
    format_lrc_timestamp,
    format_srt_timestamp,
    format_vtt_timestamp,
    segments_to_lines,
    to_lrc,
    to_srt,
    to_vtt,
    wrap_text,
    write_subtitles,
)
from mash_mix.types import LyricLine, LyricsResult

LINES = [
    LyricLine(start=0.0, end=2.5, text="প্রথম লাইন"),
    LyricLine(start=2.5, end=5.125, text="দ্বিতীয় লাইন"),
]

EXPECTED_SRT = (
    "1\n"
    "00:00:00,000 --> 00:00:02,500\n"
    "প্রথম লাইন\n"
    "\n"
    "2\n"
    "00:00:02,500 --> 00:00:05,125\n"
    "দ্বিতীয় লাইন\n"
)


def test_to_srt_produces_the_exact_block_format():
    assert to_srt(LINES) == EXPECTED_SRT


def test_srt_timestamps_use_milliseconds():
    assert format_srt_timestamp(0.0) == "00:00:00,000"
    assert format_srt_timestamp(2.5) == "00:00:02,500"
    assert format_srt_timestamp(5.125) == "00:00:05,125"
    assert format_srt_timestamp(3661.5) == "01:01:01,500"


def test_to_vtt_starts_with_webvtt_and_uses_dots():
    result = to_vtt(LINES)
    assert result.startswith("WEBVTT")
    assert "00:00:00.000 --> 00:00:02.500" in result
    assert "," not in result


def test_vtt_can_carry_a_title():
    result = to_vtt(LINES, title="Kobita tuku")
    assert result.startswith("WEBVTT")
    assert "NOTE" in result
    assert "Title: Kobita tuku" in result


def test_vtt_timestamp_format():
    assert format_vtt_timestamp(2.5) == "00:00:02.500"


def test_to_lrc_produces_bracketed_centisecond_timestamps():
    result = to_lrc(LINES)
    assert result == "[00:00.00]প্রথম লাইন\n[00:02.50]দ্বিতীয় লাইন\n"
    assert format_lrc_timestamp(2.5) == "[00:02.50]"
    assert format_lrc_timestamp(65.25) == "[01:05.25]"


def test_wrap_text_respects_max_chars():
    text = "এটি একটি বেশ লম্বা বাংলা লাইন যা সাবটাইটেলের জন্য দুই লাইনে ভাঙা উচিত"
    wrapped = wrap_text(text, max_chars=30)
    assert len(wrapped) > 1
    for line in wrapped:
        assert len(line) <= 30


def test_wrap_text_keeps_every_word():
    text = "এটি একটি বেশ লম্বা বাংলা লাইন যা সাবটাইটেলের জন্য দুই লাইনে ভাঙা উচিত"
    wrapped = wrap_text(text, max_chars=30)
    assert " ".join(wrapped).split() == text.split()


def test_wrap_text_splits_a_word_that_is_too_long():
    wrapped = wrap_text("অ" * 60, max_chars=25)
    assert len(wrapped) >= 2
    for line in wrapped:
        assert len(line) <= 25


def test_wrap_text_on_empty_input():
    assert wrap_text("") == []


def test_write_subtitles_writes_a_file_and_round_trips(tmp_path):
    target = tmp_path / "lyrics.srt"
    written = write_subtitles(LINES, str(target), fmt="srt")
    assert written == str(target)
    assert target.exists()
    content = target.read_text(encoding="utf-8")
    assert content == EXPECTED_SRT
    assert content == to_srt(LINES)


def test_write_subtitles_writes_a_vtt_file(tmp_path):
    target = tmp_path / "lyrics.vtt"
    write_subtitles(LINES, str(target), fmt="vtt")
    assert target.read_text(encoding="utf-8").startswith("WEBVTT")


def test_write_subtitles_writes_a_karaoke_file(tmp_path):
    from mash_mix.types import Word

    lines = [
        LyricLine(
            start=0.0,
            end=1.6,
            text="one two",
            words=[
                Word(start=0.0, end=0.5, text="one"),
                Word(start=0.5, end=1.0, text="two"),
            ],
        )
    ]
    target = tmp_path / "lyrics.karaoke.srt"
    write_subtitles(lines, str(target), fmt="karaoke")
    content = target.read_text(encoding="utf-8")
    assert "<b>one</b>" in content
    assert "one <b>two</b>" in content


def test_write_subtitles_falls_back_to_plain_when_there_are_no_word_times(tmp_path):
    target = tmp_path / "lyrics.karaoke.srt"
    write_subtitles(LINES, str(target), fmt="karaoke")
    content = target.read_text(encoding="utf-8")
    assert "<b>" not in content
    assert "প্রথম লাইন" in content


def test_write_subtitles_rejects_an_unknown_format(tmp_path):
    import pytest

    with pytest.raises(ValueError):
        write_subtitles(LINES, str(tmp_path / "lyrics.xyz"), fmt="xyz")


def test_transcript_to_text(fake_transcription):
    plain = transcript_to_text(fake_transcription)
    assert plain == "প্রথম লাইন\nদ্বিতীয় লাইন\nতৃতীয় লাইন\n"


def test_transcript_to_text_includes_timestamps(fake_transcription):
    stamped = transcript_to_text(fake_transcription, timestamps=True)
    assert "[00:00] প্রথম লাইন" in stamped
    assert "[00:02] দ্বিতীয় লাইন" in stamped
    assert "[00:05] তৃতীয় লাইন" in stamped


def test_transcript_to_text_can_include_speakers(fake_transcription):
    fake_transcription.segments[0].speaker = "ভাই"
    stamped = transcript_to_text(fake_transcription, speakers=True)
    assert "ভাই: প্রথম লাইন" in stamped


def test_segments_to_lines(fake_transcription):
    lines = segments_to_lines(fake_transcription.segments)
    assert len(lines) == 3
    assert lines[0].start == 0.0
    assert lines[0].text == "প্রথম লাইন"
    assert lines[2].end == 8.0


def test_lyrics_to_text():
    result = LyricsResult(lines=LINES, language="bn-BD", title="Kobita tuku", artist="MASH")
    text = lyrics_to_text(result)
    assert "Title: Kobita tuku" in text
    assert "Artist: MASH" in text
    assert "প্রথম লাইন" in text
    assert "[00:00]" not in text
    assert "[00:00] প্রথম লাইন" in lyrics_to_text(result, timestamps=True)


def test_write_text_and_json(tmp_path):
    target = tmp_path / "out" / "transcript.txt"
    write_text(str(target), "hello")
    assert target.read_text(encoding="utf-8") == "hello"

    payload = tmp_path / "out" / "result.json"
    write_json(str(payload), {"tag": "bn-BD"})
    assert "bn-BD" in payload.read_text(encoding="utf-8")


def test_to_json_uses_a_to_dict_when_there_is_one(fake_transcription):
    payload = to_json(fake_transcription)
    assert '"language": "bn-BD"' in payload
