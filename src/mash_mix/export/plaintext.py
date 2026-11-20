# SPDX-License-Identifier: MIT | Copyright (c) 2026 MASH (Khaled Bhuiyan)
"""Plain text and structured exports for transcripts and lyrics."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict, List, Optional

from ..types import LyricsResult, Transcription

__all__ = [
    "transcript_to_text",
    "lyrics_to_text",
    "to_json",
    "write_text",
    "write_json",
]


def transcript_to_text(
    transcription: Transcription,
    *,
    timestamps: bool = False,
    speakers: bool = False,
    header: Optional[Dict[str, Any]] = None,
) -> str:
    """Render a transcript as readable plain text."""
    parts: List[str] = []
    if header:
        for key, value in header.items():
            if value:
                parts.append(f"{key}: {value}")
        if parts:
            parts.append("")

    for seg in transcription.segments:
        text = " ".join(seg.text.split())
        if not text:
            continue
        line = text
        if speakers and seg.speaker:
            line = f"{seg.speaker}: {line}"
        if timestamps:
            line = f"[{_clock(seg.start)}] {line}"
        parts.append(line)

    return "\n".join(parts).strip() + ("\n" if parts else "")


def _clock(seconds: float) -> str:
    total = int(max(0.0, float(seconds)))
    hours, remainder = divmod(total, 3600)
    minutes, secs = divmod(remainder, 60)
    if hours:
        return f"{hours:02d}:{minutes:02d}:{secs:02d}"
    return f"{minutes:02d}:{secs:02d}"


def lyrics_to_text(
    result: LyricsResult,
    *,
    header: bool = True,
    timestamps: bool = False,
) -> str:
    """Render lyrics as plain text, one line per lyric line."""
    parts: List[str] = []
    if header:
        meta: List[str] = []
        if result.title:
            meta.append(f"Title: {result.title}")
        if result.artist:
            meta.append(f"Artist: {result.artist}")
        if result.language:
            meta.append(f"Language: {result.language}")
        if meta:
            parts.append("\n".join(meta))
            parts.append("")

    for line in result.lines:
        text = " ".join(line.text.split())
        if not text:
            continue
        if timestamps:
            parts.append(f"[{_clock(line.start)}] {text}")
        else:
            parts.append(text)

    return "\n".join(parts).strip() + ("\n" if parts else "")


def to_json(payload: Any, *, indent: int = 2) -> str:
    """Serialise a result object to JSON."""
    if hasattr(payload, "to_dict"):
        payload = payload.to_dict()
    return json.dumps(payload, ensure_ascii=False, indent=indent)


def write_text(path: str, content: str) -> str:
    """Write UTF-8 text to disk, creating parent folders."""
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_text(content, encoding="utf-8")
    return str(target)


def write_json(path: str, payload: Any, *, indent: int = 2) -> str:
    """Write a result object as JSON."""
    return write_text(path, to_json(payload, indent=indent))
