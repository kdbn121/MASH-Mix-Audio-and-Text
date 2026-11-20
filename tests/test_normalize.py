# SPDX-License-Identifier: MIT | Copyright (c) 2026 MASH (Khaled Bhuiyan)
"""Bangla aware text normalisation."""

from __future__ import annotations

from mash_mix.lang.normalize import (
    bangla_ratio,
    canonical_nukta,
    clean_asr_text,
    is_bangla_text,
    normalize_bangla,
    normalize_for_tts,
    normalize_unicode,
    normalize_whitespace,
    script_profile,
    strip_invisible,
    to_ascii_digits,
    to_bangla_digits,
)

NUKTA = chr(0x09BC)
BASE_RA = chr(0x09A1)
BASE_RHA = chr(0x09A2)
BASE_YA = chr(0x09AF)
PRE_RA = chr(0x09DC)
PRE_RHA = chr(0x09DD)
PRE_YA = chr(0x09DF)
BENGALI_DIGITS = "".join(chr(code) for code in range(0x09E6, 0x09F0))
ZWNJ = chr(0x200C)
ZWJ = chr(0x200D)
ZWSP = chr(0x200B)


def test_normalize_bangla_decomposes_the_nukta_letters():
    """The nukta letters end up as base plus U+09BC.

    Unicode excludes these three characters from composition, so the NFC step
    of the pass rewrites the precomposed spelling into base plus NUKTA.
    """
    assert normalize_bangla(PRE_RA) == BASE_RA + NUKTA
    assert normalize_bangla(PRE_RHA) == BASE_RHA + NUKTA
    assert normalize_bangla(PRE_YA) == BASE_YA + NUKTA


def test_canonical_nukta_leaves_the_decomposed_form_alone():
    """A letter that is already base plus NUKTA stays in that form."""
    for base in (BASE_RA, BASE_RHA, BASE_YA):
        assert canonical_nukta(base + NUKTA) == base + NUKTA


def test_nukta_can_be_kept_or_recomposed():
    """The compose setting rebuilds the precomposed spelling.

    The nukta letters are in the Unicode composition exclusion list, so NFC
    alone will not build them. The package maps them explicitly, so the
    composed spelling is produced on request.
    """
    kept = normalize_bangla(BASE_RA + NUKTA, nukta="keep")
    assert kept == BASE_RA + NUKTA
    composed = normalize_bangla(BASE_RA + NUKTA, nukta="compose")
    assert composed == PRE_RA
    assert NUKTA not in composed
    assert composed == normalize_bangla(PRE_RA, nukta="compose")


def test_zwnj_survives_normalisation():
    text = "ব্যবস্" + ZWNJ + "তা"
    result = normalize_bangla(text)
    assert ZWNJ in result
    assert result == text


def test_zwj_is_removed():
    text = "ক" + ZWJ + "ক"
    assert ZWJ not in normalize_bangla(text)
    assert normalize_bangla(text) == "কক"


def test_strip_invisible_can_drop_the_zwnj():
    text = "ব্যবস্" + ZWNJ + "তা"
    assert ZWNJ in strip_invisible(text)
    assert ZWNJ not in strip_invisible(text, keep_zwnj=False)


def test_zero_width_space_becomes_a_space():
    assert strip_invisible("ক" + ZWSP + "খ") == "ক খ"


def test_bangla_digits_convert_both_ways():
    assert to_bangla_digits("0123456789") == BENGALI_DIGITS
    assert to_ascii_digits(BENGALI_DIGITS) == "0123456789"
    assert normalize_bangla("123", digits="native") == "১২৩"
    assert normalize_bangla("১২৩", digits="ascii") == "123"
    assert normalize_bangla("১২৩", digits="keep") == "১২৩"


def test_normalize_for_tts_expands_the_percent_sign():
    assert "শতাংশ" in normalize_for_tts("50%")
    assert normalize_for_tts("50%") == "৫০ শতাংশ"


def test_normalize_for_tts_expands_the_ampersand():
    result = normalize_for_tts("A & B")
    assert "এবং" in result
    assert result == "A এবং B"


def test_normalize_for_tts_uses_ascii_digits_for_indian_bengali():
    assert normalize_for_tts("25", language="bn-IN") == "25"
    assert normalize_for_tts("25", language="bn-BD") == "২৫"


def test_clean_asr_text_removes_bracket_noise():
    assert clean_asr_text("[আড়ালা] গান") == "গান"
    assert clean_asr_text("[গান] আমি") == "আমি"
    assert clean_asr_text("[music] hello") == "hello"
    assert "ইনস্ট্রুমেন্ট" not in clean_asr_text("[ইনস্ট্রুমেন্ট] গান")


def test_clean_asr_text_collapses_repeated_punctuation():
    assert clean_asr_text("হ্যাঁ!!!") == "হ্যাঁ!"
    assert clean_asr_text("কি??") == "কি?"
    assert clean_asr_text("ঠিক,,, ঠিক") == "ঠিক, ঠিক"


def test_clean_asr_text_tidies_whitespace():
    assert clean_asr_text("  আমি    ভালো  ") == "আমি ভালো"


def test_bangla_ratio():
    assert bangla_ratio("আমি ভালো আছি") == 1.0
    assert bangla_ratio("hello world") == 0.0
    assert bangla_ratio("আমি hello") == 0.375
    assert bangla_ratio("") == 0.0


def test_script_profile_on_a_mixed_string():
    profile = script_profile("আমি fine 123")
    assert set(profile) == {"bangla", "devanagari", "latin", "other"}
    assert profile["bangla"] == 0.4286
    assert profile["latin"] == 0.5714
    assert profile["devanagari"] == 0.0
    assert abs(sum(profile.values()) - 1.0) < 0.001


def test_script_profile_on_pure_script():
    profile = script_profile("আমি তোমায়")
    assert profile["bangla"] == 1.0
    assert profile["latin"] == 0.0


def test_is_bangla_text():
    assert is_bangla_text("আমি ভালো আছি")
    assert not is_bangla_text("hello world")


def test_normalize_unicode_and_whitespace():
    assert normalize_unicode("") == ""
    assert normalize_whitespace("  a   b  ") == "a b"
    assert normalize_whitespace("a\r\nb") == "a\nb"
