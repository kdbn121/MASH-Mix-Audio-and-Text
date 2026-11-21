# SPDX-License-Identifier: MIT | Copyright (c) 2026 MASH (Khaled Bhuiyan)
"""The corrections that fix the words a recogniser keeps getting wrong."""

from __future__ import annotations

from mash_mix.lang.corrections import (
    DEFAULT_CORRECTIONS,
    apply_corrections,
    correct_words,
    merge_corrections,
)
from mash_mix.types import Word


def test_a_word_that_is_wrong_is_put_right():
    assert apply_corrections("I recieve the news") == "I receive the news"


def test_the_case_of_the_word_is_kept():
    assert apply_corrections("Recieve it") == "Receive it"
    assert apply_corrections("RECIEVE IT") == "RECEIVE IT"


def test_the_punctuation_around_the_word_is_kept():
    assert apply_corrections("wait, recieve it.") == "wait, receive it."


def test_a_word_inside_a_longer_word_is_left_alone():
    assert apply_corrections("cloudy") == "cloudy"
    assert apply_corrections("believed") == "believed"


def test_the_caller_table_wins_over_the_built_in_one():
    fixed = apply_corrections("seperate", {"seperate": "sector"})
    assert fixed == "sector"


def test_a_table_can_be_turned_off():
    assert apply_corrections("I recieve", use_defaults=False) == "I recieve"


def test_the_bangla_text_is_not_touched():
    text = "ওরে নীল দরিয়া বন্দী হইয়া মনোয়া পাখি হায়রে"
    assert apply_corrections(text) == text


def test_the_bangla_words_are_found():
    assert apply_corrections("বাংলা ভাষা recieve") == "বাংলা ভাষা receive"


def test_the_tables_are_joined_with_the_last_one_winning():
    joined = merge_corrections({"a": "b"}, {"a": "c", "d": "e"})
    assert joined["a"] == "c"
    assert joined["d"] == "e"


def test_an_empty_table_changes_nothing():
    assert apply_corrections("nothing to do here") == "nothing to do here"


def test_the_words_keep_their_timings_when_they_are_corrected():
    words = [
        Word(start=0.0, end=0.5, text="recieved"),
        Word(start=0.5, end=1.0, text="today"),
    ]
    fixed = correct_words(words)
    assert fixed[0].text == "received"
    assert fixed[0].start == 0.0
    assert fixed[0].end == 0.5
    assert fixed[1].text == "today"


def test_a_word_that_is_right_is_left_alone():
    words = [Word(start=0.0, end=0.5, text="today")]
    fixed = correct_words(words)
    assert fixed[0] is words[0]


def test_the_built_in_table_has_no_duplicate_keys():
    assert len(DEFAULT_CORRECTIONS) == len(set(DEFAULT_CORRECTIONS))
