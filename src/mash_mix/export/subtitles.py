# SPDX-License-Identifier: MIT | Copyright (c) 2026 MASH (Khaled Bhuiyan)
"""Subtitle exporters: SRT, WebVTT, LRC and ASS.

SRT is the format the lyric download button produces, so it is the one that
gets the most care: real millisecond timestamps, blank line between cues, and
text wrapped so a subtitle never runs off the screen.
"""

from __future__ import annotations

from typing import Iterable, List, Optional, Sequence

from ..types import LyricLine, Segment

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
]


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


def to_srt(
    lines: Sequence[LyricLine],
    *,
    max_chars: int = 42,
    max_lines: int = 2,
    **_: object,
) -> str:
    """Render lyric lines as SRT."""
    return _lines_to_srt(lines, max_chars=max_chars, max_lines=max_lines)


def to_vtt(
    lines: Sequence[LyricLine],
    *,
    max_chars: int = 42,
    max_lines: int = 2,
    title: Optional[str] = None,
    **_: object,
) -> str:
    """Render lyric lines as WebVTT."""
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
    **_: object,
) -> str:
    """Render lyric lines as an ASS subtitle file with Bangla friendly defaults."""
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
) -> str:
    """Write lyric lines to a file.

    The format comes from the ``fmt`` argument, and falls back to the file
    suffix when it is not given.
    """
    from pathlib import Path

    target = Path(path)
    suffix = (fmt or target.suffix.lstrip(".") or "srt").lower()
    items = list(lines)

    if suffix == "srt":
        content = to_srt(items, max_chars=max_chars, max_lines=max_lines)
    elif suffix in ("vtt", "webvtt"):
        content = to_vtt(items, max_chars=max_chars, max_lines=max_lines, title=title)
    elif suffix == "lrc":
        content = to_lrc(items)
    elif suffix == "ass":
        content = to_ass(items)
    else:
        raise ValueError(f"Unsupported subtitle format: {suffix}")

    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_text(content, encoding="utf-8")
    return str(target)


def segments_to_lines(segments: Sequence[Segment]) -> List[LyricLine]:
    """Turn transcription segments into lyric lines."""
    return [
        LyricLine(
            start=float(seg.start),
            end=float(seg.end),
            text=" ".join(seg.text.split()),
            speaker=seg.speaker,
            probability=seg.probability,
        )
        for seg in segments
        if seg.text and seg.text.strip()
    ]
