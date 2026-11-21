# SPDX-License-Identifier: MIT | Copyright (c) 2026 MASH (Khaled Bhuiyan)
"""Cue building from word level timings.

Every backend in this package that can hand back the time of each word is worth
more than a plain lyric line, because a karaoke subtitle needs to know where the
highlight is at every moment. A line of text with one start and one end can only
be read all at once. A line with the time of every word can have the word that
is being sung marked inside it.

This module turns word timings into cues and renders them three ways:

* plain, one cue for each line, which is what an ordinary subtitle wants.
* one cue for each word, which is what a word by word reveal wants.
* a karaoke cue, where the whole line is on the screen and the word being sung
  is marked inside it, which is what a karaoke wants.

The break rules matter more than they look. A cue is broken after a word that
ends a sentence, never across a silence, never over the word limit and never
over the character limit. Without those rules a long song turns into a wall of
one line, and a line that runs past the width of the screen is a line nobody
can read.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Iterable, List, Optional, Sequence

from ..types import LyricLine, Segment, Word

__all__ = [
    "Cue",
    "words_of",
    "cues_from_lines",
    "build_cues",
    "to_srt",
    "to_vtt",
    "to_ass",
    "STYLES",
]

#: The styles a cue file can be written in.
STYLES = ("plain", "word_by_word", "karaoke")

#: The character that ends a sentence in Bangla.
_BANGLA_DANDA = "।"

#: The marks that end a sentence in any script.
_SENTENCE_MARKS = (".", "!", "?", _BANGLA_DANDA)

#: The default number of words in a cue.
DEFAULT_MAX_WORDS = 8

#: The default number of characters on a line.
DEFAULT_MAX_CHARS = 42

#: A silence longer than this is a break in the speech, so a cue never crosses it.
DEFAULT_MAX_GAP = 0.8

#: A cue is held on the screen for at least this long, so a single short word
#: does not flash past.
DEFAULT_MIN_SECONDS = 0.7

#: The longest a cue is allowed to be, so one long sentence does not sit there.
DEFAULT_MAX_SECONDS = 7.0


@dataclass
class Cue:
    """One subtitle cue: a line of text and the time span it belongs to."""

    index: int
    start: float
    end: float
    text: str
    words: List[Word] = field(default_factory=list)
    speaker: Optional[str] = None

    @property
    def duration(self) -> float:
        return max(0.0, float(self.end) - float(self.start))

    def to_dict(self) -> dict:
        return {
            "index": int(self.index),
            "start": round(float(self.start), 3),
            "end": round(float(self.end), 3),
            "text": self.text,
            "speaker": self.speaker,
            "words": [word.to_dict() for word in self.words],
        }


def words_of(segments: Iterable[Segment]) -> List[Word]:
    """Every word of every segment, in the order they were spoken.

    Segments that carry no word timings are skipped. A backend that cannot say
    when each word was said has nothing to add here.
    """
    out: List[Word] = []
    for segment in segments:
        for word in segment.words:
            text = str(word.text or "").strip()
            if not text:
                continue
            out.append(
                Word(
                    start=float(word.start),
                    end=float(word.end),
                    text=text,
                    probability=word.probability,
                )
            )
    out.sort(key=lambda item: (float(item.start), float(item.end)))
    return out


def _ends_sentence(text: str) -> bool:
    """True when a word closes a sentence."""
    stripped = str(text or "").strip()
    if not stripped:
        return False
    return stripped[-1] in _SENTENCE_MARKS


def _closes_word(text: str) -> bool:
    """True when a word is joined to the next one, so the two cannot be split.

    A hyphen at the end of a word means the word continues, and a Bangla
    sentence does not end on a double danda either.
    """
    stripped = str(text or "").strip()
    if not stripped:
        return False
    if stripped.endswith("-"):
        return True
    return stripped.endswith("\u0965")


def build_cues(
    words: Sequence[Word],
    *,
    max_words: int = DEFAULT_MAX_WORDS,
    max_chars: int = DEFAULT_MAX_CHARS,
    max_gap: float = DEFAULT_MAX_GAP,
    min_seconds: float = DEFAULT_MIN_SECONDS,
    max_seconds: float = DEFAULT_MAX_SECONDS,
) -> List[Cue]:
    """Group word timings into cues.

    The rules, in the order they are applied:

    1. A cue never crosses a silence longer than ``max_gap``.
    2. A cue never holds more than ``max_words`` words or more than
       ``max_chars`` characters.
    3. A cue never runs longer than ``max_seconds``.
    4. A cue is closed early after a word that ends a sentence, so a line
       breaks where the singing breaks.
    5. A cue is never shorter than ``min_seconds`` unless it is the last one
       or the next word would overflow it anyway.

    A word that is joined to the next one by a hyphen is kept with it, because
    splitting it would put half a word on the screen.
    """
    if not words:
        return []

    limit_words = max(1, int(max_words))
    limit_chars = max(1, int(max_chars))
    limit_gap = max(0.0, float(max_gap))
    limit_min = max(0.0, float(min_seconds))
    limit_max = max(limit_min, float(max_seconds))

    groups: List[List[Word]] = []
    current: List[Word] = []

    def close() -> None:
        nonlocal current
        if current:
            groups.append(current)
            current = []

    for word in words:
        text = str(word.text or "").strip()
        if not text:
            continue
        if current:
            previous = current[-1]
            silence = float(word.start) - float(previous.end)
            candidate = " ".join(item.text for item in current) + " " + text
            span = float(word.end) - float(current[0].start)
            too_long = (
                len(current) >= limit_words
                or len(candidate) > limit_chars
                or span > limit_max
            )
            if silence > limit_gap or too_long:
                close()
            elif _ends_sentence(previous.text):
                # The singing broke here, so the line breaks here too.
                close()
        current.append(word)
    close()

    cues: List[Cue] = []
    for group in groups:
        text = " ".join(str(item.text or "").strip() for item in group).strip()
        if not text:
            continue
        start = float(group[0].start)
        end = float(group[-1].end)
        cues.append(
            Cue(
                index=len(cues) + 1,
                start=start,
                end=max(float(end), start),
                text=text,
                words=list(group),
            )
        )

    return _merge_short_cues(cues, limit_min)


def cues_from_lines(lines: Sequence[LyricLine]) -> List[Cue]:
    """Build cues out of lyric lines that carry the time of every word.

    A line that has no word timings is kept as one cue, because it is still a
    line and it still belongs on the screen. A line that has word timings is
    broken up by the rules of :func:`build_cues`, so the highlight can travel
    along it.
    """
    out: List[Cue] = []
    for line in lines:
        text = str(line.text or "").strip()
        if not text:
            continue
        if not line.words:
            out.append(
                Cue(
                    index=len(out) + 1,
                    start=float(line.start),
                    end=float(line.end),
                    text=text,
                    words=[],
                    speaker=line.speaker,
                )
            )
            continue
        for cue in build_cues(line.words):
            out.append(
                Cue(
                    index=len(out) + 1,
                    start=cue.start,
                    end=cue.end,
                    text=cue.text,
                    words=cue.words,
                    speaker=line.speaker,
                )
            )
    return out


def _merge_short_cues(cues: List[Cue], min_seconds: float) -> List[Cue]:
    """Hold every cue on the screen long enough to read it.

    A cue that is too short is not joined to its neighbour, because joining
    them would make a line that is too wide. Instead the cue is held on the
    screen for the time it needs, and the cue that comes after it is pushed
    back when it would have come too early. The words and the break between
    the cues are both left as they were.
    """
    if min_seconds <= 0.0 or len(cues) < 2:
        return cues
    out: List[Cue] = []
    for cue in cues:
        held = Cue(
            index=cue.index,
            start=cue.start,
            end=cue.end,
            text=cue.text,
            words=list(cue.words),
            speaker=cue.speaker,
        )
        if out:
            previous = out[-1]
            earliest = previous.start + min_seconds
            if held.start < earliest:
                held.start = earliest
        wanted = held.start + min_seconds
        if held.end < wanted:
            held.end = wanted
        out.append(held)
    for position, cue in enumerate(out, start=1):
        cue.index = position
    return out


def _timestamp(seconds: float, separator: str) -> str:
    """A subtitle timestamp with real millisecond rounding.

    The rounding is done on the total milliseconds first, so the carry reaches
    the seconds field instead of leaving a four digit millisecond field.
    """
    total = max(0, int(round(float(seconds) * 1000)))
    hours, remainder = divmod(total, 3_600_000)
    minutes, remainder = divmod(remainder, 60_000)
    secs, millis = divmod(remainder, 1000)
    return f"{hours:02d}:{minutes:02d}:{secs:02d}{separator}{millis:03d}"


def _escape(text: str) -> str:
    """Escape a word for a WebVTT cue."""
    return (
        str(text or "")
        .replace("&", "&amp;")
        .replace("<", "&lt;")
        .replace(">", "&gt;")
    )


def _plain_cues(cues: Sequence[Cue]) -> List[Cue]:
    """The cues as they are, for a plain subtitle."""
    return list(cues)


def _word_cues(cues: Sequence[Cue]) -> List[Cue]:
    """One cue for every word."""
    out: List[Cue] = []
    for cue in cues:
        for word in cue.words:
            out.append(
                Cue(
                    index=len(out) + 1,
                    start=float(word.start),
                    end=float(word.end),
                    text=str(word.text or "").strip(),
                    words=[word],
                    speaker=cue.speaker,
                )
            )
    return out


def _karaoke_cues(cues: Sequence[Cue]) -> List[Cue]:
    """One cue for every word, but each carries the whole line it belongs to.

    The word that is being sung is the only thing that changes between the
    cues, so the reader sees the line and the highlight travelling along it.
    """
    out: List[Cue] = []
    for cue in cues:
        for word in cue.words:
            out.append(
                Cue(
                    index=len(out) + 1,
                    start=float(word.start),
                    end=float(word.end),
                    text=str(word.text or "").strip(),
                    words=[word],
                    speaker=cue.speaker,
                )
            )
    return out


def _karaoke_body(cue: Cue, active: Word, escaper: Optional[Any] = None) -> str:
    """The line of a karaoke cue, with the active word marked.

    The words of the line are escaped when an escaper is given, so the mark
    that says which word is being sung is never escaped with them.
    """
    parts: List[str] = []
    for word in cue.words:
        text = str(word.text or "").strip()
        if not text:
            continue
        if escaper is not None:
            text = escaper(text)
        if word is active:
            parts.append(f"<b>{text}</b>")
        else:
            parts.append(text)
    return " ".join(parts)


def to_srt(cues: Sequence[Cue], style: str = "plain") -> str:
    """Render cues as an SRT file."""
    chosen = str(style or "plain").lower()
    if chosen not in STYLES:
        chosen = "plain"

    blocks: List[str] = []
    index = 1

    if chosen == "word_by_word":
        for cue in _word_cues(cues):
            blocks.append(
                f"{index}\n"
                f"{_timestamp(cue.start, ',')} --> {_timestamp(cue.end, ',')}\n"
                f"{cue.text}\n"
            )
            index += 1
        return "\n".join(blocks)

    if chosen == "karaoke":
        for cue in cues:
            if not cue.words:
                blocks.append(
                    f"{index}\n"
                    f"{_timestamp(cue.start, ',')} --> {_timestamp(cue.end, ',')}\n"
                    f"{cue.text}\n"
                )
                index += 1
                continue
            for word in cue.words:
                blocks.append(
                    f"{index}\n"
                    f"{_timestamp(word.start, ',')} --> {_timestamp(word.end, ',')}\n"
                    f"{_karaoke_body(cue, word)}\n"
                )
                index += 1
        return "\n".join(blocks)

    for cue in _plain_cues(cues):
        blocks.append(
            f"{cue.index}\n"
            f"{_timestamp(cue.start, ',')} --> {_timestamp(cue.end, ',')}\n"
            f"{cue.text}\n"
        )
    return "\n".join(blocks)


def to_vtt(cues: Sequence[Cue], style: str = "plain", title: Optional[str] = None) -> str:
    """Render cues as a WebVTT file."""
    chosen = str(style or "plain").lower()
    if chosen not in STYLES:
        chosen = "plain"

    header = "WEBVTT"
    if title:
        header += f"\nNOTE\nTitle: {title}\n"
    header += "\n"

    blocks: List[str] = []
    index = 1

    if chosen == "word_by_word":
        for cue in _word_cues(cues):
            blocks.append(
                f"{_timestamp(cue.start, '.')} --> {_timestamp(cue.end, '.')}\n"
                f"{_escape(cue.text)}\n"
            )
            index += 1
        return header + "\n".join(blocks)

    if chosen == "karaoke":
        for cue in cues:
            if not cue.words:
                blocks.append(
                    f"{_timestamp(cue.start, '.')} --> {_timestamp(cue.end, '.')}\n"
                    f"{_escape(cue.text)}\n"
                )
                index += 1
                continue
            for word in cue.words:
                blocks.append(
                    f"{_timestamp(word.start, '.')} --> {_timestamp(word.end, '.')}\n"
                    f"{_karaoke_body(cue, word, _escape)}\n"
                )
                index += 1
        return header + "\n".join(blocks)

    for cue in _plain_cues(cues):
        blocks.append(
            f"{_timestamp(cue.start, '.')} --> {_timestamp(cue.end, '.')}\n"
            f"{_escape(cue.text)}\n"
        )
    return header + "\n".join(blocks)


def to_ass(
    cues: Sequence[Cue],
    style: str = "plain",
    *,
    width: int = 1280,
    height: int = 720,
    font_name: str = "Noto Sans Bengali",
    font_size: int = 48,
) -> str:
    """Render cues as an ASS file.

    The karaoke style uses the ``\\k`` tag of the format, which is the way a
    player is told to change the colour of a syllable as it is sung. The tag
    counts hundredths of a second, so the time of each word is rounded to the
    nearest hundredth.
    """
    chosen = str(style or "plain").lower()
    if chosen not in STYLES:
        chosen = "plain"

    header = f"""[Script Info]
ScriptType: v4.00+
PlayResX: {width}
PlayResY: {height}
WrapStyle: 2
ScaledBorderAndShadow: yes

[V4+ Styles]
Format: Name, Fontname, Fontsize, PrimaryColour, SecondaryColour, OutlineColour, BackColour, Bold, Italic, Underline, StrikeOut, ScaleX, ScaleY, Spacing, Angle, BorderStyle, Outline, Shadow, Alignment, MarginL, MarginR, MarginV, Encoding
Style: Default,{font_name},{font_size},&H00FFFFFF,&H000000FF,&H00101010,&H80000000,0,0,0,0,100,100,0,0,1,3,1,2,60,60,90,1

[Events]
Format: Layer, Start, End, Style, Name, MarginL, MarginR, MarginV, Effect, Text
"""

    events: List[str] = []
    for cue in cues:
        if chosen == "karaoke" and cue.words:
            parts: List[str] = []
            for word in cue.words:
                duration = max(0.0, float(word.end) - float(word.start))
                parts.append(f"{{\\k{int(round(duration * 100))}}}{str(word.text or '').strip()}")
            body = " ".join(part for part in parts if part.strip())
            start = _timestamp(cue.start, ".")
            end = _timestamp(cue.end, ".")
            events.append(f"Dialogue: 0,{start},{end},Default,,0,0,0,,{body}")
            continue
        if chosen == "word_by_word":
            for word in cue.words:
                start = _timestamp(word.start, ".")
                end = _timestamp(word.end, ".")
                events.append(
                    f"Dialogue: 0,{start},{end},Default,,0,0,0,,{str(word.text or '').strip()}"
                )
            continue
        start = _timestamp(cue.start, ".")
        end = _timestamp(cue.end, ".")
        events.append(f"Dialogue: 0,{start},{end},Default,,0,0,0,,{cue.text}")
    return header + "\n".join(events) + ("\n" if events else "")
