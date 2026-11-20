# SPDX-License-Identifier: MIT | Copyright (c) 2026 MASH (Khaled Bhuiyan)
"""Text to speech task."""

from __future__ import annotations

from typing import Any, Optional

from ..types import SynthesisResult
from ._common import Timer, pick_backend, to_buffer

__all__ = ["synthesize", "synthesize_to_file"]


def synthesize(
    text: str,
    *,
    backend: str = "auto",
    model: Optional[str] = None,
    language: str = "bn-BD",
    voice: Optional[str] = None,
    speed: float = 1.0,
    reference_audio: Optional[Any] = None,
    normalize: bool = True,
    script: str = "bangla",
    **kwargs: Any,
) -> SynthesisResult:
    """Turn text into speech.

    ``language`` accepts ``bn-BD``, ``bn-IN`` or ``bn``. When ``normalize`` is
    on, the text is prepared for a Bangla speech engine: symbols are spelled
    out, digits are rendered in the numerals the target region expects, and the
    nukta letters are normalised.

    ``script`` decides how the text is written before it is spoken.
    ``bangla`` writes romanised Bangla out in Bangla, because a Bangla voice
    reads Latin letters as English. ``banglish`` leaves the text as it was
    typed.
    """
    from ..backends.registry import get_tts
    from ..lang.normalize import normalize_for_tts
    from ..lang.translit import apply_script, resolve_script_target

    if not text or not text.strip():
        raise ValueError("Nothing to synthesise: the text is empty.")

    prepared = normalize_for_tts(text, language=language) if normalize else text

    target_script = resolve_script_target(prepared, script)
    if target_script == "bangla":
        prepared = apply_script(prepared, "bangla")

    name = pick_backend("tts", backend)
    reference = to_buffer(reference_audio, sample_rate=16000) if reference_audio is not None else None

    with Timer() as timer:
        engine = get_tts(name, model=model)
        result = engine.synthesize(
            prepared,
            language=language,
            voice=voice,
            speed=speed,
            reference_audio=reference,
            **kwargs,
        )

    result.backend = result.backend or name
    result.elapsed = timer.elapsed
    result.meta.setdefault("language", language)
    result.meta.setdefault("normalized_text", prepared)
    result.meta.setdefault("script", target_script or "as typed")
    return result


def synthesize_to_file(
    text: str,
    path: str,
    *,
    format: Optional[str] = None,
    **kwargs: Any,
) -> dict:
    """Synthesize text and write the audio to disk."""
    from ..audio.io import save

    result = synthesize(text, **kwargs)
    written = save(path, result.audio, result.sample_rate, format=format)
    return {"audio": written, "result": result}
