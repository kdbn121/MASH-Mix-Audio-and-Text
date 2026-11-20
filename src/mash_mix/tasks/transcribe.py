# SPDX-License-Identifier: MIT | Copyright (c) 2026 MASH (Khaled Bhuiyan)
"""Speech to text task."""

from __future__ import annotations

from typing import Any, Optional, Sequence

from ..types import Transcription
from ._common import Timer, pick_backend, to_buffer

__all__ = ["transcribe", "transcribe_file"]


def transcribe(
    audio: Any,
    *,
    backend: str = "auto",
    model: Optional[str] = None,
    language: Optional[str] = None,
    task: str = "transcribe",
    word_timestamps: bool = True,
    vad: bool = True,
    hotwords: Optional[Sequence[str]] = None,
    initial_prompt: Optional[str] = None,
    sample_rate: int = 16000,
    progress: bool = False,
    script: str = "bangla",
    **kwargs: Any,
) -> Transcription:
    """Transcribe speech from a file path, file object or AudioBuffer.

    ``language`` accepts any Bangla tag, for example ``bn``, ``bn-BD`` or
    ``bn-IN``. When it is not given the backend detects the language.
    ``script`` picks the writing system of the returned text: ``bangla`` keeps
    the native script and writes out romanised Bangla, ``banglish`` returns
    Latin letters.
    """
    from ..backends.registry import get_stt

    buffer = to_buffer(audio, sample_rate=sample_rate)
    name = pick_backend("stt", backend)

    with Timer() as timer:
        engine = get_stt(name, model=model)
        result = engine.transcribe(
            buffer.mono(),
            buffer.sample_rate,
            language=language,
            task=task,
            initial_prompt=initial_prompt,
            hotwords=list(hotwords) if hotwords else None,
            word_timestamps=word_timestamps,
            vad=vad,
            progress=progress,
            **kwargs,
        )

    from ..lang.translit import apply_script, resolve_script_target

    target_script = resolve_script_target(result.text, script)
    if target_script:
        result.text = apply_script(result.text, target_script)
        for segment in result.segments:
            segment.text = apply_script(segment.text, target_script)

    result.duration = buffer.duration
    result.backend = result.backend or name
    result.elapsed = timer.elapsed
    result.meta.setdefault("sample_rate", buffer.sample_rate)
    result.meta.setdefault("channels", buffer.channels)
    result.meta.setdefault("script", target_script or "as recognised")
    return result


def transcribe_file(path: str, **kwargs: Any) -> Transcription:
    """Transcribe a file on disk."""
    return transcribe(path, **kwargs)
