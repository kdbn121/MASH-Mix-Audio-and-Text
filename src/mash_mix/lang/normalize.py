# SPDX-License-Identifier: MIT | Copyright (c) 2026 MASH (Khaled Bhuiyan)
"""Bangla aware text normalisation.

Bangla text needs a normalisation pass before it reaches a model or a subtitle
file, because the same sound can arrive in several byte sequences and because
a few characters carry real linguistic meaning that generic Unicode cleaners
destroy:

* ``ZWNJ`` (U+200C) marks an explicit non conjunct, for example ``ব্যবস্থা``
  against ``ব্যবস্‌তা``. Stripping it merges two different words, so it is kept.
* ``ZWJ`` (U+200D) has no standard use in Bangla because conjuncts are inherent
  in the script. It is dropped.
* The nukta letters ``ড়`` (U+09DC), ``ঢ়`` (U+09DD) and ``য়`` (U+09DF) are
  canonically decomposed to base letter plus ``NUKTA`` (U+09BC), which is what
  most Bangla tokenisers expect.
* The Bangla digits (U+09E6 to U+09EF) map onto ASCII digits.
"""

from __future__ import annotations

import re
import unicodedata
from typing import Dict, Iterable, Optional

__all__ = [
    "BANGLA_DIGITS",
    "ASCII_DIGITS",
    "normalize_unicode",
    "canonical_nukta",
    "strip_invisible",
    "to_bangla_digits",
    "to_ascii_digits",
    "normalize_bangla",
    "normalize_for_tts",
    "clean_asr_text",
    "bangla_ratio",
    "script_profile",
    "is_bangla_text",
    "normalize_whitespace",
]

BANGLA_DIGITS = "০১২৩৪৫৬৭৮৯"
ASCII_DIGITS = "0123456789"

_BANGLA_BLOCK = (0x0980, 0x09FF)
_DEVANAGARI_BLOCK = (0x0900, 0x097F)

# Canonical decompositions for the nukta letters. The codepoints are written
# out so the precomposed and the decomposed forms can never be confused.
# Precomposed: U+09DC, U+09DD, U+09DF. Decomposed: base letter plus U+09BC.
_NUKTA_MAP: Dict[str, str] = {
    "\u09dc": "\u09a1\u09bc",
    "\u09dd": "\u09a2\u09bc",
    "\u09df": "\u09af\u09bc",
}

# The same letters the other way, for the compose direction. Unicode excludes
# these three from composition, so the mapping has to be explicit.
_NUKTA_COMPOSE_MAP: Dict[str, str] = {
    value: key for key, value in _NUKTA_MAP.items()
}

_ZERO_WIDTH_DROP = {
    "",  # zero width space, handled separately
    "‌",  # ZWNJ, kept
    "‍",  # ZWJ, dropped
    "⁠",  # word joiner
    "﻿",  # BOM
    "­",  # soft hyphen
}

_BRACKET_NOISE = re.compile(
    r"[\[\(]\s*(আড়ালা|আড়ালো|গান|সঙ্গীত|music|applause|রশম|শব্দ|অশ্রু|কান্না|ইনস্ট্রুমেন্ট|"
    r"background\s*music|instrumental|laugh(?:ing)?|applauding|applause|noise|"
    r"সাউন্ড\s*ইফেক্ট|শব্দের\s*ইফেক্ট)\s*[\]\)]",
    re.IGNORECASE,
)
_MULTI_PUNCT = re.compile(r"([!?।,;:]){2,}")
_SPACE_RUN = re.compile(r"[ \t ]{2,}")


def normalize_unicode(text: str, form: str = "NFC") -> str:
    """Apply a Unicode normalisation form. NFC keeps Bangla composed, NFKC folds widths."""
    if not text:
        return ""
    return unicodedata.normalize(form, text)


def canonical_nukta(text: str) -> str:
    """Rewrite precomposed nukta letters as base letter plus NUKTA."""
    if not text:
        return ""
    for precomposed, decomposed in _NUKTA_MAP.items():
        if precomposed in text:
            text = text.replace(precomposed, decomposed)
    return text


def strip_invisible(text: str, keep_zwnj: bool = True) -> str:
    """Remove invisible formatting characters that confuse models and viewers.

    ``ZWNJ`` is kept by default because it changes Bangla words.
    """
    if not text:
        return ""
    out: list[str] = []
    for ch in text:
        code = ord(ch)
        if code == 0x200C:
            if keep_zwnj:
                out.append(ch)
            continue
        if code in (0x200D, 0x2060, 0xFEFF, 0x00AD, 0x200E, 0x200F, 0x202A, 0x202B, 0x202C, 0x202D):
            continue
        if code == 0x200B:
            out.append(" ")
            continue
        out.append(ch)
    return "".join(out)


def to_bangla_digits(text: str) -> str:
    """ASCII digits become Bangla digits."""
    if not text:
        return ""
    return text.translate(str.maketrans(ASCII_DIGITS, BANGLA_DIGITS))


def to_ascii_digits(text: str) -> str:
    """Bangla digits become ASCII digits."""
    if not text:
        return ""
    return text.translate(str.maketrans(BANGLA_DIGITS, ASCII_DIGITS))


def normalize_whitespace(text: str) -> str:
    """Collapse whitespace runs and trim the ends."""
    if not text:
        return ""
    text = text.replace("\r\n", "\n").replace("\r", "\n")
    text = _SPACE_RUN.sub(" ", text)
    return "\n".join(line.strip() for line in text.split("\n")).strip()


def bangla_ratio(text: str) -> float:
    """Share of Bangla script characters in the text, between 0.0 and 1.0."""
    if not text:
        return 0.0
    letters = [c for c in text if c.isalpha() or "\u0980" <= c <= "\u09FF"]
    if not letters:
        return 0.0
    bangla = sum(1 for c in letters if "\u0980" <= c <= "\u09FF")
    return bangla / float(len(letters))


def is_bangla_text(text: str, threshold: float = 0.4) -> bool:
    """True when the text is mostly Bangla script."""
    return bangla_ratio(text) >= threshold


def script_profile(text: str) -> Dict[str, float]:
    """Share of Bangla, Devanagari and Latin letters."""
    counts = {"bangla": 0, "devanagari": 0, "latin": 0, "other": 0}
    for ch in text:
        if not ch.isalpha() and not ("\u0980" <= ch <= "\u09FF"):
            continue
        code = ord(ch)
        if _BANGLA_BLOCK[0] <= code <= _BANGLA_BLOCK[1]:
            counts["bangla"] += 1
        elif _DEVANAGARI_BLOCK[0] <= code <= _DEVANAGARI_BLOCK[1]:
            counts["devanagari"] += 1
        elif code < 0x0250:
            counts["latin"] += 1
        else:
            counts["other"] += 1
    total = sum(counts.values())
    if total == 0:
        return {k: 0.0 for k in counts}
    return {k: round(v / total, 4) for k, v in counts.items()}


def clean_asr_text(text: str) -> str:
    """Tidy raw recognition output: drop bracket noise, repeated punctuation, stray spaces."""
    if not text:
        return ""
    text = _BRACKET_NOISE.sub(" ", text)
    text = _MULTI_PUNCT.sub(lambda m: m.group(1), text)
    text = text.replace(" ,", ",").replace(" .", ".")
    text = normalize_whitespace(text)
    return text.strip()


def normalize_bangla(
    text: str,
    *,
    unicode_form: str = "NFC",
    nukta: str = "decompose",
    keep_zwnj: bool = True,
    digits: str = "keep",
) -> str:
    """Full Bangla normalisation pass.

    Parameters
    ----------
    unicode_form:
        Unicode normalisation form, ``NFC`` by default.
    nukta:
        ``decompose`` rewrites precomposed nukta letters, ``compose`` does the
        opposite, ``keep`` leaves them alone.
    keep_zwnj:
        Keep the zero width non joiner, which is meaningful in Bangla.
    digits:
        ``keep``, ``native`` (Bangla digits) or ``ascii``.
    """
    if not text:
        return ""
    text = normalize_unicode(text, unicode_form)
    if nukta == "decompose":
        text = canonical_nukta(text)
    elif nukta == "compose":
        text = unicodedata.normalize("NFC", text)
        for decomposed, precomposed in _NUKTA_COMPOSE_MAP.items():
            text = text.replace(decomposed, precomposed)
    text = strip_invisible(text, keep_zwnj=keep_zwnj)
    if digits == "native":
        text = to_bangla_digits(text)
    elif digits == "ascii":
        text = to_ascii_digits(text)
    return normalize_whitespace(text)


def normalize_for_tts(
    text: str,
    *,
    language: str = "bn-BD",
    expand_numbers: bool = False,
    keep_zwnj: bool = False,
    numerals: Optional[str] = None,
) -> str:
    """Prepare text for a speech engine.

    Bangla speech engines mispronounce symbols, so ``%`` becomes
    ``শতাংশ``, ``&`` becomes ``এবং``, currency signs are spelled out, and
    digits are rendered in the numerals the listener expects.
    """
    from .numerals import expand_symbols, number_to_bangla_words, parse_number

    if not text:
        return ""

    style = numerals
    if style is None:
        style = "native" if language.upper().endswith("BD") else "ascii"

    text = normalize_bangla(text, unicode_form="NFC", nukta="decompose", keep_zwnj=keep_zwnj)
    text = expand_symbols(text, language=language)

    if expand_numbers:
        out: list[str] = []
        for token in re.split(r"(\d[\d,./]*\d|\d)", text):
            if token and token[0].isdigit():
                value = parse_number(token)
                if value is not None:
                    out.append(number_to_bangla_words(value, language=language))
                    continue
            out.append(token)
        text = "".join(out)

    if style == "native":
        text = to_bangla_digits(text)
    elif style == "ascii":
        text = to_ascii_digits(text)

    return normalize_whitespace(text)


def iter_bangla_chars(text: str) -> Iterable[str]:
    """Yield only the Bangla script characters of a string."""
    for ch in text:
        if _BANGLA_BLOCK[0] <= ord(ch) <= _BANGLA_BLOCK[1]:
            yield ch
