# SPDX-License-Identifier: MIT | Copyright (c) 2026 MASH (Khaled Bhuiyan)
"""Bangla numerals, number words and symbol expansion.

Bangla uses the lakh and crore grouping, not the western thousand and
million, so ``12345678`` reads as ``এক কোটি তেইশ লক্ষ পঁয়তাল্লিশ হাজার ছয় শত আশি``:
one crore, twenty three lakh, forty five thousand, six hundred and seventy
eight. Indian Bangla uses the same grouping, which is one more reason the two
varieties can share a number reader.
"""

from __future__ import annotations

import re
from typing import Dict, Optional, Tuple, Union

from .normalize import to_ascii_digits

__all__ = [
    "number_to_bangla_words",
    "parse_number",
    "expand_symbols",
    "expand_ordinal",
    "format_date_bangla",
    "format_time_bangla",
    "format_currency",
    "ordinal_words",
]

_ONES = (
    "শূন্য",
    "এক",
    "দুই",
    "তিন",
    "চার",
    "পাঁচ",
    "ছয়",
    "সাত",
    "আট",
    "নয়",
)

_TENS = (
    "",
    "দশ",
    "বিশ",
    "ত্রিশ",
    "চল্লিশ",
    "পঞ্চাশ",
    "ষাট",
    "সত্তর",
    "আশি",
    "নব্বই",
)

_ORDINALS = (
    "প্রথম",
    "দ্বিতীয়",
    "তৃতীয়",
    "চতুর্থ",
    "পঞ্চম",
    "ষষ্ঠ",
    "সপ্তম",
    "অষ্টম",
    "নবম",
    "দশম",
)

# Bengali does not read forty two as "forty and two". It uses the traditional
# compound word, so the reader has to know those forms.
_TEENS = (
    "এগারো",
    "বারো",
    "তেরো",
    "চৌদ্দ",
    "পনেরো",
    "ষোলো",
    "সতেরো",
    "আঠারো",
    "উনিশ",
)

_COMPOUND_TENS: Dict[int, Tuple[str, ...]] = {
    20: (
        "একুশ",
        "বাইশ",
        "তেইশ",
        "চব্বিশ",
        "পঁচিশ",
        "ছাব্বিশ",
        "সাতাশ",
        "আটাশ",
        "ঊনত্রিশ",
    ),
    30: (
        "একত্রিশ",
        "বত্রিশ",
        "তেত্রিশ",
        "চৌত্রিশ",
        "পঁয়ত্রিশ",
        "ছত্রিশ",
        "সাঁইত্রিশ",
        "আটত্রিশ",
        "ঊনত্রিশ",
    ),
    40: (
        "একচল্লিশ",
        "বল্লিশ",
        "তেতাল্লিশ",
        "চুয়াল্লিশ",
        "পঁয়তাল্লিশ",
        "ছেচল্লিশ",
        "সাতচল্লিশ",
        "আটচল্লিশ",
        "নবচল্লিশ",
    ),
    50: (
        "একান্ন",
        "বায়ান্ন",
        "তিপ্পান্ন",
        "চুয়ান্ন",
        "পঞ্চান্ন",
        "ছাপ্পান্ন",
        "সাতান্ন",
        "আটান্ন",
        "নবান্ন",
    ),
    60: (
        "একষট্টি",
        "বাষট্টি",
        "তেষট্টি",
        "চৌষট্টি",
        "পঁয়ষট্টি",
        "ছেষট্টি",
        "সাতষট্টি",
        "আটষট্টি",
        "নবষট্টি",
    ),
    70: (
        "একাত্তর",
        "বাত্তর",
        "তিয়াত্তর",
        "চুয়াত্তর",
        "পঁচাত্তর",
        "ছিয়াত্তর",
        "সাতাত্তর",
        "আটাত্তর",
        "নবাত্তর",
    ),
    80: (
        "একাশি",
        "বিরাশি",
        "তিরাশি",
        "চুরাশি",
        "পঁচাশি",
        "ছিয়াশি",
        "সাতাশি",
        "আটাশি",
        "নবাশি",
    ),
    90: (
        "একানব্বই",
        "বিরানব্বই",
        "তিরানব্বই",
        "চুরানব্বই",
        "পঁচানব্বই",
        "ছিয়ানব্বই",
        "সাতানব্বই",
        "আটানব্বই",
        "নবানব্বই",
    ),
}

_GREGORIAN_MONTHS = (
    "জানুয়ারি",
    "ফেব্রুয়ারি",
    "মার্চ",
    "এপ্রিল",
    "মে",
    "জুন",
    "জুলাই",
    "আগস্ট",
    "সেপ্টেম্বর",
    "অক্টোবর",
    "নভেম্বর",
    "ডিসেম্বর",
)

_BENGALI_MONTHS = (
    "বৈশাখ",
    "জ্যৈষ্ঠ",
    "আষাঢ়",
    "শ্রাবণ",
    "ভাদ্র",
    "আশ্বিন",
    "কার্তিক",
    "অগ্রহায়ণ",
    "পৌষ",
    "মাঘ",
    "ফাল্গুন",
    "চৈত্র",
)

_SYMBOLS = {
    "%": " শতাংশ ",
    "&": " এবং ",
    "+": " যোগ ",
    "=": " সমান ",
    "@": " অ্যাট ",
    "#": " নম্বর ",
    "$": " ডলার ",
    "€": " ইউরো ",
    "£": " পাউন্ড ",
    "¥": " ইয়েন ",
    "₹": " রুপি ",
    "৳": " টাকা ",
    "°": " ডিগ্রি ",
    "©": " কপিরাইট ",
    "®": " রেজিস্টার্ড ট্রেডমার্ক ",
    "™": " ট্রেডমার্ক ",
    "~": " প্রায় ",
    "*": " তারকা ",
    "_": " আন্ডারস্কোর ",
    "|": " পাইপ ",
    "~=": " প্রায় সমান ",
    "*=": " গুণ ",
    "/=": " ভাগ করে ",
    "->": " থেকে ",
    "=>": " থেকে ",
}

_ORDINAL_WORDS = {
    "1st": "প্রথম",
    "2nd": "দ্বিতীয়",
    "3rd": "তৃতীয়",
    "4th": "চতুর্থ",
    "5th": "পঞ্চম",
    "6th": "ষষ্ঠ",
    "7th": "সপ্তম",
    "8th": "অষ্টম",
    "9th": "নবম",
    "10th": "দশম",
    "first": "প্রথম",
    "second": "দ্বিতীয়",
    "third": "তৃতীয়",
    "fourth": "চতুর্থ",
    "fifth": "পঞ্চম",
    "sixth": "ষষ্ঠ",
    "seventh": "সপ্তম",
    "eighth": "অষ্টম",
    "ninth": "নবম",
    "tenth": "দশম",
}


def ordinal_words(number: int) -> str:
    """Bangla ordinal word for 1 to 10."""
    if 1 <= number <= 10:
        return _ORDINALS[number - 1]
    return number_to_bangla_words(number)


def expand_ordinal(token: str) -> str:
    """Expand an English ordinal token such as ``3rd`` into Bangla."""
    key = token.strip().lower()
    if key in _ORDINAL_WORDS:
        return _ORDINAL_WORDS[key]
    match = re.fullmatch(r"(\d+)(st|nd|rd|th)", key)
    if match:
        value = int(match.group(1))
        if 1 <= value <= 10:
            return _ORDINALS[value - 1]
    return token


def _two_digit_words(value: int) -> str:
    """Words for 0 to 99, using the traditional Bengali compound forms."""
    if value < 10:
        return _ONES[value]
    if value == 10:
        return _TENS[1]
    if value < 20:
        return _TEENS[value - 11]
    tens = (value // 10) * 10
    ones = value % 10
    if ones == 0:
        return _TENS[tens // 10]
    compound = _COMPOUND_TENS.get(tens)
    if compound:
        return compound[ones - 1]
    return f"{_TENS[tens // 10]} {_ONES[ones]}"


def _three_digit_words(value: int) -> str:
    """Words for 0 to 999, using the joined শো form."""
    if value < 100:
        return _two_digit_words(value)
    hundreds, rest = divmod(value, 100)
    text = f"{_ONES[hundreds]}শো"
    if rest:
        text = f"{text} {_two_digit_words(rest)}"
    return text


def number_to_bangla_words(value: Union[int, float], *, language: str = "bn-BD") -> str:
    """Read a number in Bangla words using lakh and crore grouping."""
    if value is None:
        return ""
    if isinstance(value, str):
        parsed = parse_number(value)
        if parsed is None:
            return value
        value = parsed

    negative = value < 0
    value = abs(float(value))

    integer_part = int(value)
    fraction = value - integer_part

    words = _integer_words(integer_part)
    if fraction > 1e-9:
        digits = str(f"{fraction:.6f}").split(".")[1].rstrip("0")
        if digits:
            fraction_words = " ".join(_ONES[int(d)] for d in digits)
            words = f"{words} দশমিক {fraction_words}"

    if negative:
        words = f"ঋণাত্মক {words}"
    return words.strip()


def _integer_words(integer_part: int) -> str:
    """Words for a non negative integer using crore, lakh, thousand and hundred."""
    if integer_part == 0:
        return _ONES[0]

    crore, rest = divmod(integer_part, 10_000_000)
    lakh, rest = divmod(rest, 100_000)
    thousand, rest = divmod(rest, 1_000)
    hundred = rest

    parts: list[str] = []
    if crore:
        parts.append(f"{_integer_words(crore)} কোটি")
    if lakh:
        parts.append(f"{_integer_words(lakh)} লক্ষ")
    if thousand:
        parts.append(f"{_integer_words(thousand)} হাজার")
    if hundred:
        parts.append(_three_digit_words(hundred))
    return " ".join(parts)


def parse_number(text: str) -> Optional[float]:
    """Parse a number that may use ASCII or Bangla digits, with separators."""
    if text is None:
        return None
    cleaned = to_ascii_digits(str(text)).strip().replace(" ", "")
    if not cleaned:
        return None
    cleaned = cleaned.replace(",", "")
    if not re.fullmatch(r"[-+]?\d*\.?\d+", cleaned):
        return None
    try:
        return float(cleaned)
    except ValueError:
        return None


def expand_symbols(text: str, *, language: str = "bn-BD") -> str:
    """Replace symbols with Bangla words so a speech engine reads them out."""
    if not text:
        return ""
    out = text
    for symbol, replacement in _SYMBOLS.items():
        if symbol in out:
            out = out.replace(symbol, replacement)
    out = re.sub(r"\s+", " ", out)
    return out.strip()


def format_currency(amount: Union[int, float], *, currency: str = "৳", language: str = "bn-BD") -> str:
    """Format an amount in Bangla, for example ``১২০ টাকা``."""
    value = float(amount)
    name = {"৳": "টাকা", "₹": "রুপি", "$": "ডলার", "€": "ইউরো", "£": "পাউন্ড"}.get(currency, "টাকা")
    rounded = int(round(value))
    from .normalize import to_bangla_digits

    if rounded == int(value):
        digits = to_bangla_digits(f"{rounded:,}")
        return f"{digits} {name}"
    return f"{to_bangla_digits(f'{value:,.2f}')} {name}"


def format_date_bangla(year: int, month: int, day: int, *, bengali_calendar: bool = False) -> str:
    """Format a date in Bangla words."""
    if bengali_calendar:
        month_name = _BENGALI_MONTHS[(month - 1) % 12]
    else:
        month_name = _GREGORIAN_MONTHS[(month - 1) % 12]
    return f"{number_to_bangla_words(day)} {month_name}, {number_to_bangla_words(year)}"


def format_time_bangla(hour: int, minute: int, *, second: int = 0, am_pm: bool = False) -> str:
    """Format a clock time in Bangla words."""
    hour = hour % 24
    if am_pm:
        marker = "পূর্বাহ্ন" if hour < 12 else "অপরাহ্ন"
        twelve = hour % 12 or 12
        text = f"{number_to_bangla_words(twelve)} টা"
        if minute:
            text = f"{text} {_two_digit_words(minute)} মিনিট"
        if second:
            text = f"{text} {_two_digit_words(second)} সেকেন্ড"
        return f"{text} {marker}"
    text = f"{number_to_bangla_words(hour)} টা"
    if minute:
        text = f"{text} {_two_digit_words(minute)} মিনিট"
    if second:
        text = f"{text} {_two_digit_words(second)} সেকেন্ড"
    return text
