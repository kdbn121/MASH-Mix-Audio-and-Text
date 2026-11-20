# SPDX-License-Identifier: MIT | Copyright (c) 2026 MASH (Khaled Bhuiyan)
"""Romanised Bangla to Bangla script and back."""

from __future__ import annotations

from mash_mix.lang.translit import (
    bangla_to_latin,
    banglish_to_bangla,
    convert_sentence,
    has_banglish,
)


def test_banglish_to_bangla_converts_the_common_words():
    assert banglish_to_bangla("ami") == "আমি"
    assert banglish_to_bangla("tomay") == "তোমায়"
    assert banglish_to_bangla("bhalobashi") == "ভালোবাসি"
    assert banglish_to_bangla("dhonnobad") == "ধন্যবাদ"
    assert banglish_to_bangla("bangladesh") == "বাংলাদেশ"
    assert banglish_to_bangla("dhaka") == "ঢাকা"


def test_banglish_to_bangla_converts_a_sentence():
    assert banglish_to_bangla("ami tomay bhalobashi") == "আমি তোমায় ভালোবাসি"


def test_digits_and_punctuation_are_left_alone():
    """Only the words are converted, everything else is carried through."""
    assert banglish_to_bangla("123") == "123"
    assert banglish_to_bangla("ami, tomay!") == "আমি, তোমায়!"


def test_bangla_script_passes_through_unchanged():
    assert banglish_to_bangla("আমি তোমায়") == "আমি তোমায়"
    assert banglish_to_bangla("আমি তোমায় ভালোবাসি") == "আমি তোমায় ভালোবাসি"


def test_bangla_script_is_only_normalised():
    result = banglish_to_bangla("আমি  তোমায়")
    assert result == "আমি তোমায়"


def test_bangla_to_latin_round_trips_a_simple_word():
    assert bangla_to_latin("আমি") == "ami"
    assert bangla_to_latin(banglish_to_bangla("ami")) == "ami"


def test_bangla_to_latin_keeps_the_vowel_structure():
    assert bangla_to_latin("ভালোবাসি") == "bhalobasi"
    assert bangla_to_latin("আমি", style="phonetic") == "ami"


def test_has_banglish():
    assert has_banglish("ami bangladesh dhaka")
    assert has_banglish("bangladesh dhaka chattogram")
    assert not has_banglish("আমি তোমায়")
    assert not has_banglish("")
    assert not has_banglish("12345")


def test_convert_sentence_keeps_the_punctuation():
    result = convert_sentence("ami, dhaka!")
    assert "," in result
    assert "!" in result
    assert "আমি" in result


def test_empty_input():
    assert banglish_to_bangla("") == ""
    assert bangla_to_latin("") == ""
