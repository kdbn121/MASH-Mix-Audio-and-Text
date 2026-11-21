# SPDX-License-Identifier: MIT | Copyright (c) 2026 MASH (Khaled Bhuiyan)
"""Subtitle exporters and readers: SRT, WebVTT, LRC and ASS.

SRT is the format the lyric download button produces, so it is the one that
gets the most care: real millisecond timestamps, blank line between cues, and
text wrapped so a subtitle never runs off the screen.

The reader lives here too, because a subtitle file that arrives from a site
has to be read before it can be written back out in the format the caller
asked for. Sites publish both SRT and WebVTT, and both are read here.
"""

from __future__ import annotations

import re
from typing import Iterable, List, Optional, Sequence

from ..types import LyricLine, Segment, Word

__all__ = [
    "format_srt_timestamp",
    "format_vtt_timestamp",
    "format_lrc_timestamp",
    "to_srt",
    "to_vtt",
    "to_lrc",
    "to_ass",
    "write_subtitles",
    "wrap_text",
    "parse_timestamp",
    "parse_subtitles",
    "STYLES",
]

#: The styles a subtitle file can be written in.
#:
#: ``plain`` is one cue for each line. ``word_by_word`` is one cue for each
#: word. ``karaoke`` is the whole line on the screen with the word that is
#: being sung marked inside it.
STYLES = ("plain", "word_by_word", "karaoke")

#: A timestamp of a cue. The hours are optional, because a short clip may
#: write its cues as ``MM:SS,mmm``.
_TIMESTAMP_RE = re.compile(
    r"(?:(?P<h>\d{1,3}):)?(?P<m>\d{1,2}):(?P<s>\d{1,2})(?:[.,](?P<ms>\d{1,3}))?"
)

#: A markup tag inside a cue. Voice labels, styling and karaoke marks are all
#: removed, and a plain ``<3`` in a lyric is left alone.
_TAG_RE = re.compile(r"</?[A-Za-z][^>]*>|<\d{2}:\d{2}(?::\d{2})?[.,]\d{1,3}>")

#: The entities a subtitle file may carry.
_ENTITIES = (
    ("&amp;", "&"),
    ("&lt;", "<"),
    ("&gt;", ">"),
    ("&quot;", '"'),
    ("&apos;", "'"),
    ("&nbsp;", " "),
)


def _ms(value: float) -> int:
    return max(0, int(round(float(value) * 1000)))


def format_srt_timestamp(seconds: float) -> str:
    """``HH:MM:SS,mmm`` as SRT requires."""
    total = _ms(seconds)
    hours, remainder = divmod(total, 3_600_000)
    minutes, remainder = divmod(remainder, 60_000)
    secs, millis = divmod(remainder, 1000)
    return f"{hours:02d}:{minutes:02d}:{secs:02d},{millis:03d}"


def format_vtt_timestamp(seconds: float) -> str:
    """``HH:MM:SS.mmm`` as WebVTT requires."""
    return format_srt_timestamp(seconds).replace(",", ".")


def format_lrc_timestamp(seconds: float) -> str:
    """``[mm:ss.xx]`` as LRC requires."""
    total = _ms(seconds)
    minutes, remainder = divmod(total, 60_000)
    secs, centis = divmod(remainder, 1000)
    return f"[{minutes:02d}:{secs:02d}.{centis // 10:02d}]"


def wrap_text(text: str, *, max_chars: int = 42, max_lines: int = 2) -> List[str]:
    """Wrap a line for subtitle display, breaking on word boundaries.

    Bangla words can be long, so a word longer than the limit is split by
    character rather than allowed to overflow.
    """
    text = " ".join(text.split())
    if not text:
        return []
    words = text.split(" ")
    lines: List[str] = []
    current = ""
    for word in words:
        candidate = f"{current} {word}".strip()
        if len(candidate) <= max_chars or not current:
            current = candidate
            if len(current) > max_chars and max_lines <= 1:
                lines.append(current)
                current = ""
            continue
        lines.append(current)
        current = word
    if current:
        lines.append(current)

    if len(lines) > max_lines:
        merged = " ".join(lines)
        lines = [merged]

    out: List[str] = []
    for line in lines:
        while len(line) > max_chars:
            cut = line.rfind(" ", 0, max_chars)
            if cut <= 0:
                cut = max_chars
            out.append(line[:cut].strip())
            line = line[cut:].strip()
        if line:
            out.append(line)
    return out or [text]


def _lines_to_srt(lines: Sequence[LyricLine], *, max_chars: int, max_lines: int) -> str:
    blocks: List[str] = []
    index = 1
    for line in lines:
        text = " ".join(line.text.split())
        if not text:
            continue
        body = "\n".join(wrap_text(text, max_chars=max_chars, max_lines=max_lines))
        blocks.append(
            f"{index}\n"
            f"{format_srt_timestamp(line.start)} --> {format_srt_timestamp(line.end)}\n"
            f"{body}\n"
        )
        index += 1
    return "\n".join(blocks)


def _styled(lines: Sequence[LyricLine], style: str) -> Optional[List]:
    """The cues of the lines, when a style needs the time of every word.

    Nothing is returned when the style is plain, or when the lines carry no
    word timings, because in those two cases the plain writer is the right one.
    """
    chosen = str(style or "plain").lower()
    if chosen not in ("word_by_word", "karaoke"):
        return None
    if not any(line.words for line in lines):
        return None
    from .word_cues import cues_from_lines

    return cues_from_lines(lines)


def to_srt(
    lines: Sequence[LyricLine],
    *,
    max_chars: int = 42,
    max_lines: int = 2,
    style: str = "plain",
    **_: object,
) -> str:
    """Render lyric lines as SRT.

    The ``style`` argument asks for the words to be marked. ``karaoke`` puts
    the whole line on the screen and marks the word that is being sung, and
    ``word_by_word`` gives every word a cue of its own. A line that carries no
    word timings cannot be marked, so it is written the plain way.
    """
    cues = _styled(lines, style)
    if cues is not None:
        from .word_cues import to_srt as cues_to_srt

        return cues_to_srt(cues, style)
    return _lines_to_srt(lines, max_chars=max_chars, max_lines=max_lines)


def to_vtt(
    lines: Sequence[LyricLine],
    *,
    max_chars: int = 42,
    max_lines: int = 2,
    title: Optional[str] = None,
    style: str = "plain",
    **_: object,
) -> str:
    """Render lyric lines as WebVTT."""
    cues = _styled(lines, style)
    if cues is not None:
        from .word_cues import to_vtt as cues_to_vtt

        return cues_to_vtt(cues, style, title=title)
    header = "WEBVTT"
    if title:
        header += f"\nNOTE\nTitle: {title}\n"
    header += "\n"
    return header + _lines_to_srt(lines, max_chars=max_chars, max_lines=max_lines).replace(",", ".")


def to_lrc(lines: Sequence[LyricLine], *, offset: float = 0.0, **_: object) -> str:
    """Render lyric lines as an LRC file."""
    blocks: List[str] = []
    for line in lines:
        text = " ".join(line.text.split())
        if not text:
            continue
        blocks.append(f"{format_lrc_timestamp(max(0.0, line.start - offset))}{text}")
    return "\n".join(blocks) + ("\n" if blocks else "")


def to_ass(
    lines: Sequence[LyricLine],
    *,
    width: int = 1280,
    height: int = 720,
    font_name: str = "Noto Sans Bengali",
    font_size: int = 48,
    style: str = "plain",
    **_: object,
) -> str:
    """Render lyric lines as an ASS subtitle file with Bangla friendly defaults."""
    cues = _styled(lines, style)
    if cues is not None:
        from .word_cues import to_ass as cues_to_ass

        return cues_to_ass(
            cues,
            style,
            width=width,
            height=height,
            font_name=font_name,
            font_size=font_size,
        )
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
    for line in lines:
        text = " ".join(line.text.split())
        if not text:
            continue
        start = format_srt_timestamp(line.start).replace(",", ".")
        end = format_srt_timestamp(line.end).replace(",", ".")
        events.append(f"Dialogue: 0,{start},{end},Default,,0,0,0,,{text}")
    return header + "\n".join(events) + ("\n" if events else "")


def write_subtitles(
    lines: Iterable[LyricLine],
    path: str,
    *,
    fmt: Optional[str] = None,
    max_chars: int = 42,
    max_lines: int = 2,
    title: Optional[str] = None,
    style: str = "plain",
) -> str:
    """Write lyric lines to a file.

    The format comes from the ``fmt`` argument, and falls back to the file
    suffix when it is not given. The format ``karaoke`` is an SRT file with the
    word that is being sung marked inside the line.
    """
    from pathlib import Path

    target = Path(path)
    suffix = (fmt or target.suffix.lstrip(".") or "srt").lower()
    items = list(lines)

    if suffix == "karaoke":
        content = to_srt(items, max_chars=max_chars, max_lines=max_lines, style="karaoke")
    elif suffix == "srt":
        content = to_srt(
            items,
            max_chars=max_chars,
            max_lines=max_lines,
            style=style,
        )
    elif suffix in ("vtt", "webvtt"):
        content = to_vtt(
            items,
            max_chars=max_chars,
            max_lines=max_lines,
            title=title,
            style=style,
        )
    elif suffix == "lrc":
        content = to_lrc(items)
    elif suffix == "ass":
        content = to_ass(items, style=style)
    else:
        raise ValueError(f"Unsupported subtitle format: {suffix}")

    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_text(content, encoding="utf-8")
    return str(target)


def segments_to_lines(segments: Sequence[Segment]) -> List[LyricLine]:
    """Turn transcription segments into lyric lines.

    The time of every word is kept, because a karaoke subtitle needs it. A
    segment that has no word timings gives a line with no word timings.
    """
    return [
        LyricLine(
            start=float(seg.start),
            end=float(seg.end),
            text=" ".join(seg.text.split()),
            speaker=seg.speaker,
            probability=seg.probability,
            words=[
                Word(
                    start=float(word.start),
                    end=float(word.end),
                    text=str(word.text or "").strip(),
                    probability=word.probability,
                )
                for word in seg.words
                if str(word.text or "").strip()
            ],
        )
        for seg in segments
        if seg.text and seg.text.strip()
    ]


def parse_timestamp(value: str) -> Optional[float]:
    """Read one subtitle timestamp and return it in seconds.

    Both spellings are read: ``00:01:02,500`` as SRT writes it and
    ``00:01:02.500`` as WebVTT writes it. The hours may be left out, and a
    fraction of a second may be written with one or two digits.
    """
    match = _TIMESTAMP_RE.search(value or "")
    if not match:
        return None
    try:
        hours = int(match.group("h") or 0)
        minutes = int(match.group("m") or 0)
        seconds = int(match.group("s") or 0)
        raw_ms = match.group("ms") or ""
    except (TypeError, ValueError):  # pragma: no cover - defensive
        return None
    # A fraction written as ".5" means half a second, so the digits are read
    # from the right.
    millis = int(raw_ms.ljust(3, "0")) if raw_ms else 0
    return hours * 3600.0 + minutes * 60.0 + seconds + millis / 1000.0


def _cue_times(row: str) -> tuple:
    """The start and the end of a cue timing line."""
    if "-->" not in row:
        return None, None
    left, _, right = row.partition("-->")
    start = parse_timestamp(left)
    end = parse_timestamp(right)
    return start, end


def _clean_cue_text(raw: str) -> str:
    """The readable text of a cue, without the markup a site may wrap it in."""
    text = _TAG_RE.sub("", raw)
    for entity, character in _ENTITIES:
        text = text.replace(entity, character)
    rows = [" ".join(row.split()) for row in text.split("\n")]
    return " ".join(row for row in rows if row)


def parse_subtitles(text: str) -> List[LyricLine]:
    """Read SRT or WebVTT text and return the timed lines.

    Both formats are read by the same pass, because both are built out of the
    same thing: a timing line that holds an arrow, and the words that follow
    it. The cue number of an SRT file, the identifier of a WebVTT cue, the
    voice labels, the styling and the karaoke marks are all removed, and the
    words are left.

    A file that is not a subtitle file at all gives an empty list rather than
    an error, because the caller can always fall back on reading the audio.
    """
    raw = (text or "").replace("\r\n", "\n").replace("\r", "\n").lstrip("\ufeff")
    if not raw.strip():
        return []

    rows = raw.split("\n")
    total = len(rows)
    out: List[LyricLine] = []
    index = 0
    while index < total:
        row = rows[index]
        if "-->" not in row:
            index += 1
            continue
        start, end = _cue_times(row)
        if start is None:
            index += 1
            continue
        body: List[str] = []
        index += 1
        while index < total:
            following = rows[index]
            if not following.strip():
                break
            if "-->" in following:
                # The next cue starts here, so this one ends here.
                break
            body.append(following)
            index += 1
        words = _clean_cue_text("\n".join(body))
        if not words:
            continue
        out.append(LyricLine(start=start, end=max(float(end or start), start), text=words))
    return out
