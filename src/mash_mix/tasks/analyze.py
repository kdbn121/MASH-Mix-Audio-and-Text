# SPDX-License-Identifier: MIT | Copyright (c) 2026 MASH (Khaled Bhuiyan)
"""Audio analysis and stem extraction tasks."""

from __future__ import annotations

from typing import Any, List, Optional

from ..types import AnalysisResult, SeparationResult
from ._common import to_buffer

__all__ = ["analyze", "separate"]


def analyze(audio: Any, *, sample_rate: Optional[int] = None, **kwargs: Any) -> AnalysisResult:
    """Measure a track: tempo, key, loudness, spectral shape and sections."""
    from ..audio.analyze import analyze as _analyze

    buffer = to_buffer(audio, sample_rate=sample_rate)
    return _analyze(buffer.mono(), buffer.sample_rate, **kwargs)


def separate(
    audio: Any,
    *,
    model: str = "htdemucs",
    stems: Optional[List[str]] = None,
    output_dir: Optional[str] = None,
    sample_rate: Optional[int] = None,
    **kwargs: Any,
) -> SeparationResult:
    """Split a track into stems such as vocals and instrumentals."""
    from ..audio.separate import separate as _separate

    buffer = to_buffer(audio, sample_rate=sample_rate)
    result = _separate(
        buffer.mono(),
        buffer.sample_rate,
        model=model,
        stems=tuple(stems) if stems else ("vocals", "no_vocals"),
        **kwargs,
    )

    if output_dir:
        from pathlib import Path

        from ..audio.io import save

        folder = Path(output_dir)
        folder.mkdir(parents=True, exist_ok=True)
        for name, wave in result.stems.items():
            save(str(folder / f"{name}.wav"), wave, result.sample_rate)

    return result
