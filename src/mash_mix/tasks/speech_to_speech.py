# SPDX-License-Identifier: MIT | Copyright (c) 2026 MASH (Khaled Bhuiyan)
"""Speech to speech task."""

from __future__ import annotations

from typing import Any, Optional

from ..types import SynthesisResult
from ._common import Timer, pick_backend, to_buffer

__all__ = ["speak"]


def speak(
    audio: Any,
    *,
    backend: str = "auto",
    model: Optional[str] = None,
    target_language: str = "bn-BD",
    target_voice: Optional[str] = None,
    reference_audio: Optional[Any] = None,
    stt_backend: str = "auto",
    tts_backend: str = "auto",
    translator: Optional[Any] = None,
    sample_rate: int = 16000,
    **kwargs: Any,
) -> SynthesisResult:
    """Speak the content of an audio file in another language or another voice.

    The default path recognises the speech, then speaks it again with a
    synthesis backend. Pass a ``translator`` callable, which takes the
    recognised text and the target language tag and returns the translated
    text, to change language as well.
    """
    from ..backends.registry import get_sts

    buffer = to_buffer(audio, sample_rate=sample_rate)

    name = pick_backend("sts", backend) if backend != "auto" else "pipeline"
    with Timer() as timer:
        engine = get_sts(name)
        result = engine.speak(
            buffer,
            target_language=target_language,
            target_voice=target_voice,
            reference_audio=(
                to_buffer(reference_audio, sample_rate=sample_rate)
                if reference_audio is not None
                else None
            ),
            stt_backend=stt_backend,
            tts_backend=tts_backend,
            translator=translator,
            **kwargs,
        )

    result.backend = result.backend or name
    result.elapsed = timer.elapsed
    result.meta.setdefault("target_language", target_language)
    return result
