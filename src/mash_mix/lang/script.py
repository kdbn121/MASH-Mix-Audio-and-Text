# SPDX-License-Identifier: MIT | Copyright (c) 2026 MASH (Khaled Bhuiyan)
"""Script repair between Bangla and Devanagari.

A Whisper model that is told to recognise Bangla will sometimes answer in
Devanagari, because the two scripts are close relatives in the model's training
data and the audio is the same language. The result looks wrong on a Bangla
keyboard and reads wrong to a Bangla speaker.

Both scripts share a large part of their alphabet, so the mapping is exact for
most letters. This module converts one to the other, which means the pipeline
can guarantee Bangla script output no matter what the model drifts into.

Bangla conjuncts are inherent in the script, while Devanagari uses ZWJ to build
them. The repair therefore drops ZWJ and keeps ZWNJ, which is the opposite of
what a generic cleaner would do.

Every mapping is written as a pair of code points. Bengali and Devanagari
letters look alike, and any tool that touches this file can quietly turn one
into the other, so the numbers are used instead of the glyphs.
"""

from __future__ import annotations

from typing import Dict, List, Tuple

from .normalize import bangla_ratio

__all__ = [
    "devanagari_to_bangla",
    "bangla_to_devanagari",
    "dominant_script",
    "needs_script_repair",
    "repair_to_bangla",
]


def _pairs(items: List[Tuple[int, int]]) -> Dict[str, str]:
    """Build a character map from code point pairs."""
    return {chr(source): chr(target) for source, target in items}


# Consonants. The order follows the Devanagari alphabet.
_CONSONANTS: List[Tuple[int, int]] = [
    (0x0915, 0x0995),  # KA
    (0x0916, 0x0996),  # KHA
    (0x0917, 0x0997),  # GA
    (0x0918, 0x0998),  # GHA
    (0x0919, 0x0999),  # NGA
    (0x091A, 0x099A),  # CA
    (0x091B, 0x099B),  # CHA
    (0x091C, 0x099C),  # JA
    (0x091D, 0x099D),  # JHA
    (0x091E, 0x099E),  # NYA
    (0x091F, 0x099F),  # TTA
    (0x0920, 0x09A0),  # TTHA
    (0x0921, 0x09A1),  # DDA
    (0x0922, 0x09A2),  # DDHA
    (0x0923, 0x09A3),  # NNA
    (0x0924, 0x09A4),  # TA
    (0x0925, 0x09A5),  # THA
    (0x0926, 0x09A6),  # DA
    (0x0927, 0x09A7),  # DHA
    (0x0928, 0x09A8),  # NA
    (0x092A, 0x09AA),  # PA
    (0x092B, 0x09AB),  # PHA
    (0x092C, 0x09AC),  # BA
    (0x092D, 0x09AD),  # BHA
    (0x092E, 0x09AE),  # MA
    (0x092F, 0x09AF),  # YA
    (0x0930, 0x09B0),  # RA
    (0x0932, 0x09B2),  # LA
    (0x0935, 0x09AC),  # VA, Bangla has no separate wa so it becomes BA
    (0x0936, 0x09B6),  # SHA
    (0x0937, 0x09B7),  # SSA
    (0x0938, 0x09B8),  # SA
    (0x0939, 0x09B9),  # HA
    (0x0933, 0x09B3),  # LLA
]

# The nukta letters. Bangla keeps the precomposed form for the retroflex
# letters and the decomposed form for the ya.
_NUKTA_LETTERS: List[Tuple[int, int, int]] = [
    (0x095C, 0x09A1, 0x09BC),  # DDA with nukta, DRA
    (0x095D, 0x09A2, 0x09BC),  # DDHA with nukta, DDHA
    (0x095F, 0x09AF, 0x09BC),  # YA with nukta, YYA
]

# Independent vowels.
_VOWELS: List[Tuple[int, int]] = [
    (0x0905, 0x0985),  # A
    (0x0906, 0x0986),  # AA
    (0x0907, 0x0987),  # I
    (0x0908, 0x0988),  # II
    (0x0909, 0x0989),  # U
    (0x090A, 0x098A),  # UU
    (0x090F, 0x098F),  # E
    (0x0910, 0x0990),  # AI
    (0x0913, 0x0993),  # O
    (0x0914, 0x0994),  # AU
]

# Vowel signs.
_MATRAS: List[Tuple[int, int]] = [
    (0x093E, 0x09BE),  # AA sign
    (0x093F, 0x09BF),  # I sign
    (0x0940, 0x09C0),  # II sign
    (0x0941, 0x09C1),  # U sign
    (0x0942, 0x09C2),  # UU sign
    (0x0947, 0x09C7),  # E sign
    (0x0948, 0x09C8),  # AI sign
    (0x094B, 0x09CB),  # O sign
    (0x094C, 0x09CC),  # AU sign
]

# Other marks.
_OTHER: List[Tuple[int, int]] = [
    (0x0902, 0x0982),  # anusvara
    (0x0901, 0x0981),  # candrabindu
    (0x0903, 0x0983),  # visarga
    (0x0964, 0x0964),  # danda
    (0x0965, 0x0965),  # double danda
    (0x0951, 0x0981),  # stress, folded into the candrabindu
]

_DEVANAGARI_TO_BANGLA: Dict[str, str] = {}
_DEVANAGARI_TO_BANGLA.update(_pairs(_CONSONANTS))
_DEVANAGARI_TO_BANGLA.update(_pairs(_VOWELS))
_DEVANAGARI_TO_BANGLA.update(_pairs(_MATRAS))
_DEVANAGARI_TO_BANGLA.update(_pairs(_OTHER))

# The nukta letters map to two characters, so they are handled on their own.
_NUKTA_MAP: Dict[str, str] = {
    chr(source): chr(base) + chr(nukta) for source, base, nukta in _NUKTA_LETTERS
}

_BANGLA_TO_DEVANAGARI: Dict[str, str] = {}
for _source, _target in _CONSONANTS:
    _BANGLA_TO_DEVANAGARI.setdefault(chr(_target), chr(_source))
for _source, _target in _VOWELS:
    _BANGLA_TO_DEVANAGARI.setdefault(chr(_target), chr(_source))
for _source, _target in _OTHER:
    _BANGLA_TO_DEVANAGARI.setdefault(chr(_target), chr(_source))

# The vowel signs are set explicitly rather than reversed from the table above,
# because Devanagari has rare variants that would otherwise win the reverse
# mapping and give the wrong sign.
for _source, _target in _MATRAS:
    _BANGLA_TO_DEVANAGARI[chr(_target)] = chr(_source)

_BANGLA_TO_DEVANAGARI.update(
    {
        chr(0x09DC): chr(0x095C),
        chr(0x09DD): chr(0x095D),
        chr(0x09DF): chr(0x095F),
        chr(0x09F3): chr(0x20B9),
        chr(0x09D7): chr(0x093D),
    }
)

_DEVANAGARI_BLOCK = (0x0900, 0x097F)
_BANGLA_BLOCK = (0x0980, 0x09FF)

_ZERO_WIDTH_JOINER = "\u200d"
_ZERO_WIDTH_NON_JOINER = "\u200c"
_NUKTA = "\u09bc"

def dominant_script(text: str) -> str:
    """Name the script that dominates the text.

    Returns ``bangla``, ``devanagari``, ``latin`` or ``unknown``.
    """
    if not text:
        return "unknown"
    counts = {"bangla": 0, "devanagari": 0, "latin": 0}
    for ch in text:
        if not ch.isalpha() and not ("\u0900" <= ch <= "\u09FF"):
            continue
        code = ord(ch)
        if _BANGLA_BLOCK[0] <= code <= _BANGLA_BLOCK[1]:
            counts["bangla"] += 1
        elif _DEVANAGARI_BLOCK[0] <= code <= _DEVANAGARI_BLOCK[1]:
            counts["devanagari"] += 1
        elif code < 0x0250:
            counts["latin"] += 1
    best = max(counts, key=lambda key: counts[key])
    if counts[best] == 0:
        return "unknown"
    return best


def needs_script_repair(text: str, *, target: str = "bangla") -> bool:
    """True when the text is mostly in the wrong script for the target language."""
    if not text:
        return False
    script = dominant_script(text)
    lowered = target.lower()
    if lowered.startswith("bangla") or lowered.startswith("bn"):
        return script == "devanagari"
    if lowered.startswith("devanagari") or lowered.startswith("hi"):
        return script == "bangla"
    return False


def devanagari_to_bangla(text: str) -> str:
    """Convert Devanagari text into Bangla script."""
    if not text:
        return ""
    out: List[str] = []
    for ch in text:
        if ch == _ZERO_WIDTH_JOINER:
            # ZWJ builds Devanagari conjuncts. Bangla does not need it.
            continue
        if ch == _ZERO_WIDTH_NON_JOINER:
            out.append(ch)
            continue
        if ch == _NUKTA:
            # A stray nukta. The precomposed letters are already mapped.
            continue
        if ch in _NUKTA_MAP:
            out.append(_NUKTA_MAP[ch])
            continue
        if ch in _DEVANAGARI_TO_BANGLA:
            out.append(_DEVANAGARI_TO_BANGLA[ch])
            continue
        code = ord(ch)
        if _DEVANAGARI_BLOCK[0] <= code <= _DEVANAGARI_BLOCK[1]:
            # An unmapped Devanagari mark. Drop it rather than keep a foreign mark.
            continue
        out.append(ch)
    return "".join(out)


def bangla_to_devanagari(text: str) -> str:
    """Convert Bangla text into Devanagari script."""
    if not text:
        return ""
    out: List[str] = []
    for ch in text:
        if ch == _ZERO_WIDTH_JOINER:
            continue
        if ch == _ZERO_WIDTH_NON_JOINER:
            out.append(ch)
            continue
        if ch == _NUKTA:
            # The precomposed Bangla letters carry the nukta already.
            continue
        out.append(_BANGLA_TO_DEVANAGARI.get(ch, ch))
    return "".join(out)


def repair_to_bangla(text: str) -> str:
    """Bring a Bangla transcript back into Bangla script."""
    if not text:
        return ""
    if bangla_ratio(text) >= 0.5:
        return text
    if dominant_script(text) == "devanagari":
        return devanagari_to_bangla(text)
    return text
