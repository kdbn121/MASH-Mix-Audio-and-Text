# SPDX-License-Identifier: MIT | Copyright (c) 2026 MASH (Khaled Bhuiyan)
"""MASH-Mix-Audio-and-Text.

A cross platform speech and text engine. One API for analysis, extraction,
song to lyric, text to speech, speech to text and speech to speech, running on
Windows, Linux, macOS, Android and iOS.

Bangla is a first class citizen here, with separate handling for Bangladeshi
Bangla (bn-BD) and Indian Bengali (bn-IN).
"""

from __future__ import annotations

__version__ = "0.3.0"
__all__ = [
    "__version__",
    "transcribe",
    "extract_lyrics",
    "synthesize",
    "speak",
    "analyze",
    "available_backends",
]


def __getattr__(name: str):  # pragma: no cover - thin lazy re-export
    """Expose the main entry points without importing heavy modules at start up."""
    if name in ("transcribe", "extract_lyrics", "synthesize", "speak", "analyze"):
        from . import tasks

        return getattr(tasks, name)
    if name == "available_backends":
        from .backends.registry import backend_infos

        return backend_infos
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
