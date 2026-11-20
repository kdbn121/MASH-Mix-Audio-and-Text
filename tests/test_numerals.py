# SPDX-License-Identifier: MIT | Copyright (c) 2026 MASH (Khaled Bhuiyan)
"""Bangla numerals, number words, symbols, dates and times."""

from __future__ import annotations

from mash_mix.lang.numerals import (
    expand_ordinal,
    expand_symbols,
    format_currency,
    format_date_bangla,
    format_time_bangla,
    number_to_bangla_words,
    ordinal_words,
    parse_number,
)

BENGALI_DIGITS = "".join(chr(code) for code in range(0x09E6, 0x09F0))


def test_zero_to_ten():
    assert number_to_bangla_words(0) == "শূন্য"
    assert number_to_bangla_words(1) == "এক"
    assert number_to_bangla_words(2) == "দুই"
    assert number_to_bangla_words(3) == "তিন"
    assert number_to_bangla_words(4) == "চার"
    assert number_to_bangla_words(5) == "পাঁচ"
    assert number_to_bangla_words(6) == "ছয়"
    assert number_to_bangla_words(7) == "সাত"
    assert number_to_bangla_words(8) == "আট"
    assert number_to_bangla_words(9) == "নয়"
    assert number_to_bangla_words(10) == "দশ"


def test_tens_to_ninety_nine():
    assert number_to_bangla_words(11) == "এগারো"
    assert number_to_bangla_words(12) == "বারো"
    assert number_to_bangla_words(20) == "বিশ"
    assert number_to_bangla_words(21) == "একুশ"
    assert number_to_bangla_words(22) == "বাইশ"
    assert number_to_bangla_words(30) == "ত্রিশ"
    assert number_to_bangla_words(40) == "চল্লিশ"
    assert number_to_bangla_words(42) == "বল্লিশ"
    assert number_to_bangla_words(50) == "পঞ্চাশ"
    assert number_to_bangla_words(60) == "ষাট"
    assert number_to_bangla_words(70) == "সত্তর"
    assert number_to_bangla_words(80) == "আশি"
    assert number_to_bangla_words(90) == "নব্বই"
    assert number_to_bangla_words(99) == "নবানব্বই"


def test_one_to_one_hundred():
    assert number_to_bangla_words(100) == "একশো"
    assert number_to_bangla_words(101) == "একশো এক"


def test_lakh_and_crore_grouping():
    """The grouping is by lakh and crore, not by thousand and million."""
    assert number_to_bangla_words(1000) == "এক হাজার"
    assert number_to_bangla_words(100000) == "এক লক্ষ"
    assert number_to_bangla_words(10000000) == "এক কোটি"
    assert number_to_bangla_words(12345678) == (
        "এক কোটি তেইশ লক্ষ পঁয়তাল্লিশ হাজার ছয়শো আটাত্তর"
    )


def test_hundreds_use_the_joined_sho_form():
    assert number_to_bangla_words(200) == "দুইশো"
    assert number_to_bangla_words(678) == "ছয়শো আটাত্তর"
    assert number_to_bangla_words(101) == "একশো এক"


def test_negative_numbers():
    assert number_to_bangla_words(-5) == "ঋণাত্মক পাঁচ"
    assert number_to_bangla_words(-1).startswith("ঋণাত্মক")


def test_decimals():
    assert number_to_bangla_words(3.5) == "তিন দশমিক পাঁচ"
    assert "দশমিক" in number_to_bangla_words(2.25)


def test_number_to_words_accepts_a_string():
    assert number_to_bangla_words("42") == "বল্লিশ"
    assert number_to_bangla_words("not a number") == "not a number"


def test_parse_number_handles_ascii_digits():
    assert parse_number("12345678") == 12345678.0
    assert parse_number("0") == 0.0


def test_parse_number_handles_bangla_digits():
    assert parse_number(BENGALI_DIGITS[1:4]) == 123.0
    assert parse_number("১২৩৪৫৬৭৮") == 12345678.0


def test_parse_number_handles_separators_and_signs():
    assert parse_number("12,345") == 12345.0
    assert parse_number("1,234,567") == 1234567.0
    assert parse_number("-3.5") == -3.5
    assert parse_number("+7") == 7.0


def test_parse_number_rejects_nonsense():
    assert parse_number("abc") is None
    assert parse_number("") is None
    assert parse_number("12.3.4") is None


def test_expand_symbols():
    assert "শতাংশ" in expand_symbols("50%")
    assert expand_symbols("50%") == "50 শতাংশ"
    assert expand_symbols("A & B") == "A এবং B"
    assert expand_symbols("৳100") == "টাকা 100"
    assert expand_symbols("") == ""


def test_format_date_bangla():
    assert format_date_bangla(2026, 11, 20) == "বিশ নভেম্বর, দুই হাজার ছাব্বিশ"
    assert format_date_bangla(2026, 1, 1, bengali_calendar=True) == (
        "এক বৈশাখ, দুই হাজার ছাব্বিশ"
    )
    assert "জানুয়ারি" in format_date_bangla(2026, 1, 15)


def test_format_time_bangla():
    assert format_time_bangla(13, 45) == "তেরো টা পঁয়তাল্লিশ মিনিট"
    assert format_time_bangla(0, 0) == "শূন্য টা"
    assert "সেকেন্ড" in format_time_bangla(9, 5, second=7)


def test_format_time_bangla_with_am_pm():
    afternoon = format_time_bangla(13, 45, am_pm=True)
    assert afternoon.endswith("অপরাহ্ন")
    assert format_time_bangla(9, 5, am_pm=True).endswith("পূর্বাহ্ন")


def test_format_currency():
    assert format_currency(1200) == "১,২০০ টাকা"
    assert "রুপি" in format_currency(1200, currency="₹")


def test_ordinals():
    assert ordinal_words(1) == "প্রথম"
    assert ordinal_words(3) == "তৃতীয়"
    assert expand_ordinal("1st") == "প্রথম"
    assert expand_ordinal("4th") == "চতুর্থ"
    assert expand_ordinal("nothing") == "nothing"
