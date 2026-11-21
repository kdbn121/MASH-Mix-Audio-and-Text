# SPDX-License-Identifier: MIT | Copyright (c) 2026 MASH (Khaled Bhuiyan)
"""The command line interface."""

from __future__ import annotations

import json

import pytest

from mash_mix.cli import build_parser, main

COMMANDS = [
    "lyrics",
    "transcribe",
    "speak",
    "sts",
    "analyze",
    "separate",
    "text",
    "languages",
    "backends",
    "models",
    "doctor",
    "serve",
]


def test_the_parser_builds():
    parser = build_parser()
    assert parser.prog == "mash-mix"


def _subcommand_choices(parser):
    """The subcommand names of a parser, whatever the Python version."""
    for action in parser._actions:
        choices = getattr(action, "choices", None)
        if isinstance(choices, dict) and "lyrics" in choices:
            return choices
    return {}


def test_every_subcommand_is_present():
    parser = build_parser()
    help_text = parser.format_help()
    for command in COMMANDS:
        assert command in help_text


def test_the_subcommand_choices():
    parser = build_parser()
    choices = _subcommand_choices(parser)
    assert set(COMMANDS) == set(choices)


def test_the_lyrics_flags():
    parser = build_parser()
    args = parser.parse_args(
        [
            "lyrics",
            "song.mp3",
            "-o",
            "out",
            "--format",
            "srt",
            "--format",
            "txt",
            "--lang",
            "bn-IN",
            "--separate-vocals",
            "--chunk-seconds",
            "20",
        ]
    )
    assert args.audio == "song.mp3"
    assert args.output_dir == "out"
    assert args.formats == ["srt", "txt"]
    assert args.language == "bn-IN"
    assert args.separate_vocals is True
    assert args.chunk_seconds == 20.0
    assert args.separation_model == "htdemucs"
    assert args.max_repeats == 2


def test_the_transcribe_flags():
    parser = build_parser()
    args = parser.parse_args(["transcribe", "a.wav", "-o", "out.txt", "--timestamps"])
    assert args.output == "out.txt"
    assert args.output_format == "txt"
    assert args.timestamps is True
    assert args.task == "transcribe"


def test_the_transcribe_audio_url_flag():
    parser = build_parser()
    args = parser.parse_args(
        ["transcribe", "a.wav", "--backend", "dashscope-asr", "--audio-url", "https://example.com/a.wav"]
    )
    assert args.audio_url == "https://example.com/a.wav"
    assert parser.parse_args(["transcribe", "a.wav"]).audio_url is None


def test_the_karaoke_format_is_offered():
    parser = build_parser()
    lyrics = parser.parse_args(["lyrics", "a.mp3", "--format", "karaoke"])
    assert lyrics.formats == ["karaoke"]
    transcribe = parser.parse_args(["transcribe", "a.wav", "--format", "karaoke"])
    assert transcribe.output_format == "karaoke"


def test_the_speak_flags():
    parser = build_parser()
    args = parser.parse_args(["speak", "hello", "--voice", "bn-female", "--speed", "0.9"])
    assert args.text == "hello"
    assert args.output == "speech.wav"
    assert args.voice == "bn-female"
    assert args.speed == 0.9
    assert args.no_normalize is False


def test_the_text_subcommands():
    parser = build_parser()
    assert parser.parse_args(["text", "numbers", "42"]).text_command == "numbers"
    assert parser.parse_args(["text", "normalize", "abc"]).digits == "keep"
    assert parser.parse_args(["text", "translit", "ami"]).reverse is False
    assert parser.parse_args(["text", "detect", "abc"]).text == "abc"


def test_languages_prints_json_with_three_languages(capsys):
    code = main(["languages"])
    captured = capsys.readouterr()
    assert code == 0
    payload = json.loads(captured.out)
    assert len(payload) == 3
    assert [entry["tag"] for entry in payload] == ["bn", "bn-BD", "bn-IN"]


def test_backends_runs_without_raising(capsys):
    code = main(["backends"])
    captured = capsys.readouterr()
    assert code == 0
    assert captured.out.strip()
    assert "stt" in captured.out


def test_text_numbers_reads_a_number(capsys):
    code = main(["text", "numbers", "42"])
    captured = capsys.readouterr()
    assert code == 0
    assert captured.out.strip() == "বল্লিশ"


def test_text_numbers_reads_a_lakh_and_crore_number(capsys):
    code = main(["text", "numbers", "12345678"])
    captured = capsys.readouterr()
    assert code == 0
    assert "কোটি" in captured.out
    assert "লক্ষ" in captured.out


def test_text_normalize_runs(capsys):
    code = main(["text", "normalize", "123", "--digits", "native"])
    captured = capsys.readouterr()
    assert code == 0
    assert captured.out.strip() == "১২৩"


def test_text_translit_runs(capsys):
    code = main(["text", "translit", "ami"])
    captured = capsys.readouterr()
    assert code == 0
    assert captured.out.strip() == "আমি"


def test_text_detect_prints_json(capsys):
    code = main(["text", "detect", "আপ্নে কেমন আছেন"])
    captured = capsys.readouterr()
    assert code == 0
    payload = json.loads(captured.out)
    assert payload["tag"] == "bn-BD"


def test_main_returns_zero_for_a_text_command():
    assert main(["text", "numbers", "42"]) == 0


def test_main_without_a_subcommand_raises_system_exit():
    with pytest.raises(SystemExit) as excinfo:
        main([])
    assert excinfo.value.code == 2


def test_main_with_an_unknown_subcommand_raises_system_exit():
    with pytest.raises(SystemExit):
        main(["not-a-command"])
