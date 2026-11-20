# SPDX-License-Identifier: MIT | Copyright (c) 2026 MASH (Khaled Bhuiyan)
"""Romanised Bangla (Banglish) to Bangla script, and Bangla script to Latin.

Romanised Bangla is ambiguous by nature: ``ghor`` can be ঘর or ঘোর depending on
the writer. The converter therefore works in two layers:

1. A deterministic rule engine that handles the regular cases, digraphs,
   consonant clusters and vowel positions.
2. A curated exception lexicon for the high frequency words where the regular
   rules would produce the wrong reading.

The rule engine is intentionally conservative. When a word cannot be converted
with confidence the original text is kept, because a wrong conversion is worse
than an untouched one.
"""

from __future__ import annotations

import re
from typing import Dict, List, Optional

__all__ = [
    "banglish_to_bangla",
    "bangla_to_latin",
    "has_banglish",
    "convert_sentence",
]

# Consonant digraphs and trigraphs, longest first.
_CONSONANT_MAP: Dict[str, str] = {
    "chh": "ছ",
    "shh": "ষ",
    "kh": "খ",
    "gh": "ঘ",
    "th": "থ",
    "dh": "ধ",
    "ph": "ফ",
    "bh": "ভ",
    "jh": "ঝ",
    "ng": "ং",
    "ny": "ঞ",
    "ch": "চ",
    "sh": "শ",
    "ts": "টস",
    "zh": "জ",
    "khh": "খ",
    "cch": "চ্ছ",
}

_SINGLE_CONSONANT: Dict[str, str] = {
    "k": "ক",
    "g": "গ",
    "c": "ক",
    "q": "ক",
    "x": "ক্স",
    "w": "ব",
    "v": "ভ",
    "z": "জ",
    "f": "ফ",
    "j": "জ",
    "y": "য়",
    "r": "র",
    "l": "ল",
    "s": "স",
    "h": "হ",
    "m": "ম",
    "n": "ন",
    "p": "প",
    "b": "ব",
    "t": "ত",
    "d": "দ",
}

_VOWEL_MAP: Dict[str, str] = {
    "aa": "আ",
    "ii": "ঈ",
    "uu": "ঊ",
    "ee": "ই",
    "oo": "উ",
    "ai": "ঐ",
    "au": "ঔ",
    "oi": "ওই",
    "ou": "ওউ",
    "a": "অ",
    "i": "ই",
    "u": "উ",
    "e": "এ",
    "o": "ও",
}

# Sign forms used after a consonant.
_SIGN_MAP: Dict[str, str] = {
    "a": "া",
    "i": "ি",
    "u": "ু",
    "e": "ে",
    "ai": "ৈ",
    "au": "ৌ",
    "o": "ো",
}

_HASANT = "্"
_NUKTA = "়"

_BANGLA_TO_LATIN: Dict[str, str] = {
    "অ": "o",
    "আ": "a",
    "ই": "i",
    "ঈ": "ee",
    "উ": "u",
    "ঊ": "oo",
    "এ": "e",
    "ঐ": "oi",
    "ও": "o",
    "ঔ": "ou",
    "ড়": "r",
    "ঢ়": "r",
    "য়": "y",
    "ক": "k",
    "খ": "kh",
    "গ": "g",
    "ঘ": "gh",
    "ঙ": "ng",
    "চ": "ch",
    "ছ": "chh",
    "জ": "j",
    "ঝ": "jh",
    "ঞ": "ny",
    "ট": "t",
    "ঠ": "th",
    "ড": "d",
    "ঢ": "dh",
    "ণ": "n",
    "ত": "t",
    "থ": "th",
    "দ": "d",
    "ধ": "dh",
    "ন": "n",
    "প": "p",
    "ফ": "ph",
    "ব": "b",
    "ভ": "bh",
    "ম": "m",
    "য": "j",
    "র": "r",
    "ল": "l",
    "শ": "sh",
    "ষ": "sh",
    "স": "s",
    "হ": "h",
    "ং": "ng",
    "ঃ": "h",
    "ঁ": "n",
}

_BANGLA_TO_LATIN_SIGNS: Dict[str, str] = {
    "া": "a",
    "ি": "i",
    "ী": "ee",
    "ু": "u",
    "ূ": "oo",
    "ে": "e",
    "ৈ": "oi",
    "ো": "o",
    "ৌ": "ou",
    "ৗ": "",
}

_WORD_RE = re.compile(r"[A-Za-z]+(?:'[A-Za-z]+)*")
_BANGLISH_HINT = re.compile(r"[bcdfghjklmnpqrstvwxyz]{2,}", re.IGNORECASE)


def has_banglish(text: str) -> bool:
    """Cheap check for text that looks like romanised Bangla."""
    if not text:
        return False
    if re.search(r"[\u0980-\u09FF]", text):
        return False
    words = _WORD_RE.findall(text)
    if not words:
        return False
    hits = sum(1 for w in words if _BANGLISH_HINT.search(w))
    return hits / len(words) >= 0.5


def _tokenize(word: str) -> List[str]:
    """Split a roman word into ordered units.

    Units are prefixed markers so vowel and consonant positions stay explicit:
    ``C`` consonant, ``V`` vowel, ``M`` multi character consonant.
    """
    units: List[str] = []
    i = 0
    lowered = word.lower()
    length = len(lowered)
    while i < length:
        matched = False
        for size in (3, 2):
            chunk = lowered[i : i + size]
            if size == 3 and chunk in _CONSONANT_MAP:
                units.append("M" + chunk)
                i += size
                matched = True
                break
            if size == 2:
                if chunk in _CONSONANT_MAP:
                    units.append("M" + chunk)
                    i += size
                    matched = True
                    break
                if chunk in _VOWEL_MAP:
                    units.append("V" + chunk)
                    i += size
                    matched = True
                    break
        if matched:
            continue
        ch = lowered[i]
        if ch in _SINGLE_CONSONANT:
            units.append("C" + ch)
        elif ch in _VOWEL_MAP:
            units.append("V" + ch)
        else:
            units.append("X" + ch)
        i += 1
    return units


def _consonant_of(unit: str) -> str:
    body = unit[1:]
    if unit.startswith("M"):
        return _CONSONANT_MAP.get(body, "")
    if unit.startswith("C"):
        return _SINGLE_CONSONANT.get(body, "").strip()
    return ""


def _convert_word(word: str) -> Optional[str]:
    """Convert one romanised word to Bangla. Returns ``None`` when unsure."""
    units = _tokenize(word)
    if not units:
        return None
    if any(u.startswith("X") for u in units):
        return None

    out: List[str] = []
    count = len(units)

    for index, unit in enumerate(units):
        is_last = index == count - 1
        next_unit = units[index + 1] if not is_last else None

        if unit.startswith("C") or unit.startswith("M"):
            letter = _consonant_of(unit)
            if not letter:
                return None
            # A consonant right after a consonant becomes a cluster, so the
            # previous consonant takes a hasant.
            if out and _ends_with_consonant(out):
                out.append(_HASANT)
            out.append(letter)
            continue

        # Vowel unit.
        body = unit[1:]
        if body == "a":
            if out and _ends_with_consonant(out):
                if is_last:
                    # Word final 'a' is the inherent vowel, so nothing to add.
                    continue
                out.append("া")
            elif not out and next_unit is not None and (
                next_unit.startswith("C") or next_unit.startswith("M")
            ):
                # A word initial 'a' before a consonant is the aa kar, which
                # Bangla writes on the left of the consonant.
                out.append("আ")
            else:
                out.append(_VOWEL_MAP["a"])
            continue

        if body == "o":
            if out and _ends_with_consonant(out):
                if is_last:
                    # Word final 'o' is the o kar on the previous consonant.
                    out.append("ো")
                    continue
                if next_unit is not None and (
                    next_unit.startswith("C") or next_unit.startswith("M")
                ):
                    # The o kar belongs to the consonant that follows.
                    out.append("ো")
                    continue
                out.append("া")
                out.append("ও")
                continue
            out.append(_VOWEL_MAP["o"])
            continue

        if body in ("ai", "au"):
            if out and _ends_with_consonant(out):
                out.append(_SIGN_MAP[body])
            else:
                out.append(_VOWEL_MAP[body])
            continue

        if body in ("i", "u", "e"):
            if out and _ends_with_consonant(out):
                out.append(_SIGN_MAP[body])
            else:
                out.append(_VOWEL_MAP[body])
            continue

        # Long vowels and the diphthongs with an 'i' or 'u' sound.
        if body in ("aa", "ii", "uu", "ee", "oo", "oi", "ou"):
            if out and _ends_with_consonant(out):
                mapping = {"aa": "া", "ii": "ী", "uu": "ূ", "ee": "ী", "oo": "ূ"}
                sign = mapping.get(body)
                if sign:
                    out.append(sign)
                else:
                    out.append(_VOWEL_MAP[body])
            else:
                out.append(_VOWEL_MAP[body])
            continue

        return None

    result = "".join(out).replace(_HASANT + _HASANT, _HASANT)
    if not result:
        return None
    # A word that ended in a bare hasant is broken, so drop it.
    if result.endswith(_HASANT):
        result = result[:-1]
    return result


def _ends_with_consonant(parts: List[str]) -> bool:
    """True when the last emitted piece is a consonant that can carry a sign."""
    if not parts:
        return False
    last = parts[-1]
    if last in (_HASANT, "া", "ি", "ী", "ু", "ূ", "ে", "ৈ", "ো", "ৌ", "ৗ"):
        return False
    if not last:
        return False
    code = ord(last[0])
    # Bengali consonants live in U+0995 to U+09B9. The independent vowels sit
    # below that range, so this check separates a consonant from a vowel.
    if 0x0995 <= code <= 0x09B9:
        return True
    return code in (0x09DC, 0x09DD, 0x09DF, 0x09F0, 0x09F1)


def convert_sentence(text: str, *, use_exceptions: bool = True) -> str:
    """Convert a whole sentence, keeping punctuation and spacing intact."""
    from .lexicon import KNOWN_BANGGLISH

    if not text:
        return ""

    def _replace(match: re.Match[str]) -> str:
        word = match.group(0)
        if use_exceptions:
            known = KNOWN_BANGGLISH.get(word.lower())
            if known:
                return known
        converted = _convert_word(word)
        if converted is None:
            return word
        return converted

    return _WORD_RE.sub(_replace, text)


def banglish_to_bangla(text: str, *, use_exceptions: bool = True) -> str:
    """Convert romanised Bangla into Bangla script.

    Words that cannot be converted with confidence are left untouched.
    """
    if not text:
        return ""
    if re.search(r"[\u0980-\u09FF]", text):
        # Already Bangla script. Only normalise.
        from .normalize import normalize_bangla

        return normalize_bangla(text)
    return convert_sentence(text, use_exceptions=use_exceptions)


def bangla_to_latin(text: str, *, style: str = "transliteration") -> str:
    """Convert Bangla script into a readable Latin form.

    ``style='transliteration'`` keeps the vowel structure, ``style='phonetic'``
    folds the vowels down to a simple a, i, u, e, o set.
    """
    if not text:
        return ""
    from .normalize import canonical_nukta

    # Decompose first, so the nukta can be handled as a separate mark.
    text = canonical_nukta(text)

    out: List[str] = []
    for ch in text:
        if ch == "়":
            # Nukta turns the previous letter into a retroflex or a nasal y.
            if out:
                previous = out[-1]
                if previous == "j":
                    out[-1] = "y"
                elif previous in ("d", "dh"):
                    out[-1] = "r"
            continue
        if ch in _BANGLA_TO_LATIN_SIGNS:
            out.append(_BANGLA_TO_LATIN_SIGNS[ch])
            continue
        if ch in _BANGLA_TO_LATIN:
            out.append(_BANGLA_TO_LATIN[ch])
            continue
        if ch == _HASANT:
            out.append("")
            continue
        if ch == "‌":
            out.append(" ")
            continue
        if ch in "০১২৩৪৫৬৭৮৯":
            out.append(str("০১২৩৪৫৬৭৮৯".index(ch)))
            continue
        out.append(ch)

    text_out = "".join(out)
    if style == "phonetic":
        text_out = re.sub(r"(.)\1+", r"\1", text_out)
        text_out = text_out.replace("ee", "i").replace("oo", "u").replace("oi", "oi")
    return re.sub(r"\s+", " ", text_out).strip()
