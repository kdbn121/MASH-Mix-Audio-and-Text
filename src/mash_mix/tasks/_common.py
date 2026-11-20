# SPDX-License-Identifier: MIT | Copyright (c) 2026 MASH (Khaled Bhuiyan)
"""Shared helpers for the task layer."""

from __future__ import annotations

import time
from typing import Any, List, Optional, Tuple

from ..audio.io import load as load_audio_file
from ..types import AudioBuffer

__all__ = [
    "pick_backend",
    "to_buffer",
    "to_stereo_buffer",
    "Timer",
    "chunk_offsets",
]

# Preference order when the caller asks for the automatic backend. The first
# entry that is installed and available wins.
_STT_PREFERENCE = (
    "faster-whisper",
    "whisper",
    "funasr",
    "transformers",
    "sherpa-onnx",
    "whisper.cpp",
)

_TTS_PREFERENCE = (
    "mms-tts",
    "vits",
    "indic-tts",
    "parler-tts",
    "piper",
    "xtts",
    "chatterbox",
    "edge-tts",
)


def pick_backend(kind: str, preferred: str = "auto") -> str:
    """Choose a backend name.

    ``auto`` walks the preference list for the kind and returns the first
    backend that reports itself available.
    """
    from ..backends.registry import backend_infos

    infos = backend_infos()
    by_kind = [info for info in infos if info["kind"] == kind]

    if preferred and preferred != "auto":
        names = {info["name"] for info in by_kind}
        if preferred in names:
            return preferred
        raise ValueError(f"Unknown {kind} backend '{preferred}'. Known: {', '.join(sorted(names))}")

    order = _STT_PREFERENCE if kind == "stt" else _TTS_PREFERENCE
    available = {info["name"] for info in by_kind if info.get("available")}
    for name in order:
        if name in available:
            return name
    for info in by_kind:
        if info.get("available"):
            return info["name"]
    raise RuntimeError(
        f"No {kind} backend is installed. Install one with: "
        f"pip install 'mash-mix-audio-and-text[{'stt' if kind == 'stt' else 'tts'}]'"
    )


def to_buffer(source: Any, *, sample_rate: Optional[int] = 16000, mono: bool = True) -> AudioBuffer:
    """Turn a path, file object or AudioBuffer into an AudioBuffer.

    ``mono`` asks for the downmix. A measurement of a recording wants the
    channels as they are, because the peak of a stereo master is the loudest
    channel and not the average of the two.
    """
    if isinstance(source, AudioBuffer):
        buffer = source
    else:
        buffer = load_audio_file(source, sr=sample_rate, mono=mono)
    if sample_rate and buffer.sample_rate != sample_rate:
        from ..audio.resample import resample_buffer

        buffer = resample_buffer(buffer, sample_rate)
    return buffer


def to_stereo_buffer(source: Any) -> AudioBuffer:
    """Turn a path, file object or AudioBuffer into a buffer that keeps its channels.

    A source separator needs the stereo mix at the sample rate it was recorded
    at. The mono buffer the recogniser works on has already thrown both of
    those away, so the file is opened a second time when separation is asked
    for.
    """
    if isinstance(source, AudioBuffer):
        return source
    return load_audio_file(source, sr=None, mono=False)


class Timer:
    """Small context manager that measures elapsed seconds."""

    def __init__(self) -> None:
        self.start = 0.0
        self.elapsed = 0.0

    def __enter__(self) -> Timer:
        self.start = time.perf_counter()
        return self

    def __exit__(self, *_: object) -> None:
        self.elapsed = time.perf_counter() - self.start


def chunk_offsets(total_frames: int, chunk_seconds: float, sample_rate: int) -> List[Tuple[int, int]]:
    """Frame ranges for fixed size windows over a buffer."""
    step = max(1, int(chunk_seconds * sample_rate))
    ranges: List[Tuple[int, int]] = []
    start = 0
    while start < total_frames:
        end = min(total_frames, start + step)
        ranges.append((start, end))
        start = end
    return ranges
