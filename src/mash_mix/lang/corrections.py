# SPDX-License-Identifier: MIT | Copyright (c) 2026 MASH (Khaled Bhuiyan)
"""Fixing the words a recogniser keeps getting wrong.

A recogniser that is nearly right is wrong in a different way every time it is
tried, and a song that is sung the same way every time gives the same mistake
every time. "Colour" that comes back as "color" on every line, or a name that
is always heard as something else, is worth correcting in one place rather than
editing in the subtitle file afterwards.

The table is applied to the whole word, not to a piece of it, so "cloud" is not
changed inside "cloudy". The case of the word is kept, so a capital in the
middle of a line stays a capital, and the punctuation around the word stays
where it was.

The built in table holds only mistakes that are wrong in every language. A
caller that knows its own song can add to it, and the table it passes wins over
the built in one.
"""

from __future__ import annotations

import re
from typing import Dict, Iterable, Mapping, Optional

__all__ = [
    "apply_corrections",
    "merge_corrections",
    "DEFAULT_CORRECTIONS",
]

#: The corrections that are wrong in every language.
#:
#: The keys are the way the mistake is heard and the values are the way it is
#: written. The keys are lower case, because the lookup ignores the case.
DEFAULT_CORRECTIONS: Dict[str, str] = {
    # English spellings a recogniser hears wrong.
    "recieve": "receive",
    "recieved": "received",
    "seperate": "separate",
    "seperated": "separated",
    "occured": "occurred",
    "occuring": "occurring",
    "untill": "until",
    "beleive": "believe",
    "beleived": "believed",
    "wierd": "weird",
    "freind": "friend",
    "liason": "liaison",
    "definate": "definite",
    "definately": "definitely",
    "goverment": "government",
    "enviroment": "environment",
    "arguement": "argument",
    "independant": "independent",
    "neccessary": "necessary",
    "occassion": "occasion",
    "priviledge": "privilege",
    "publically": "publicly",
    "refered": "referred",
    "relevent": "relevant",
    "succesful": "successful",
    "tommorow": "tomorrow",
    "writting": "writing",
    "writen": "written",
}

#: The marks that may sit after a word and must be kept where they are.
_TRAILING = ".,!?;:\"'()[]{}"


def merge_corrections(
    *tables: Optional[Mapping[str, str]],
) -> Dict[str, str]:
    """Join correction tables, with the last one winning.

    A table passed by the caller is applied after the built in one, so a caller
    can correct a word the way its own song spells it.
    """
    out: Dict[str, str] = {}
    for table in tables:
        if not table:
            continue
        for wrong, right in table.items():
            key = str(wrong or "").strip()
            value = str(right or "").strip()
            if not key or not value:
                continue
            out[key.lower()] = value
    return out


def _split_trailing(word: str) -> tuple:
    """The word without the marks that sit after it, and those marks."""
    index = len(word)
    while index > 0 and word[index - 1] in _TRAILING:
        index -= 1
    return word[:index], word[index:]


def _match_case(original: str, replacement: str) -> str:
    """Give the replacement the case of the word it replaces."""
    if not original:
        return replacement
    if original.isupper():
        return replacement.upper()
    if original[:1].isupper():
        return replacement[:1].upper() + replacement[1:]
    return replacement


def apply_corrections(
    text: str,
    corrections: Optional[Mapping[str, str]] = None,
    *,
    use_defaults: bool = True,
) -> str:
    """Return ``text`` with the corrected words in it.

    The corrections are applied to whole words. The case and the punctuation of
    the word that was wrong are kept, so only the word itself changes.
    """
    if not text:
        return text or ""

    table = merge_corrections(DEFAULT_CORRECTIONS if use_defaults else None, corrections)
    if not table:
        return text

    # A word is a run of letters. The pattern finds every word in the text,
    # including the Bangla ones, and the table decides what happens to it.
    pattern = re.compile(r"[^\W\d_]+", re.UNICODE)

    def swap(match: re.Match[str]) -> str:
        found = match.group(0)
        body, trailing = _split_trailing(found)
        if not body:
            return found
        replacement = table.get(body.lower())
        if not replacement:
            return found
        return _match_case(body, replacement) + trailing

    return pattern.sub(swap, text)


def correct_words(
    words: Iterable,
    corrections: Optional[Mapping[str, str]] = None,
    *,
    use_defaults: bool = True,
) -> list:
    """Apply the corrections to a list of words, keeping their timings.

    A word that holds several words is corrected as a whole, so the timing of
    the word is kept.
    """
    out = []
    for word in words:
        text = str(getattr(word, "text", "") or "")
        fixed = apply_corrections(text, corrections, use_defaults=use_defaults)
        if fixed == text:
            out.append(word)
            continue
        out.append(
            type(word)(
                start=word.start,
                end=word.end,
                text=fixed,
                probability=word.probability,
            )
        )
    return out
