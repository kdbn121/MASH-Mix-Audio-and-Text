# SPDX-License-Identifier: MIT | Copyright (c) 2026 MASH (Khaled Bhuiyan)
"""Split long audio into overlapping chunks.

The chunker is designed for very long inputs such as a full album or a long
interview. It can be told where the speech pauses are so a chunk boundary
never lands in the middle of a word. When no speech information is available
it falls back to a fixed sliding window.

Every chunk is a plain dictionary so the result can be serialised without any
extra conversion:

``index``        position of the chunk in the list
``start``        start time in seconds
``end``          end time in seconds
``samples``      float32 mono samples
``start_sample`` first sample index in the source
``end_sample``   last sample index in the source
"""

from __future__ import annotations

from collections.abc import Sequence
from typing import Any, Callable

import numpy as np

__all__ = ["chunk_audio", "offset_chunks"]

SegmentList = list[tuple[float, float]]


def _to_mono(audio: np.ndarray) -> np.ndarray:
    """Return a mono float32 view of the given samples."""
    data = np.asarray(audio, dtype=np.float32)
    if data.ndim == 1:
        return data
    if data.ndim == 2:
        if data.shape[0] == 1:
            return data[0]
        return data.mean(axis=0).astype(np.float32)
    return data.reshape(-1)


def _resolve_speech(
    audio: np.ndarray,
    sr: int,
    vad: None | Sequence[tuple[float, float]] | Callable[[np.ndarray, int], SegmentList],
) -> SegmentList:
    """Turn the ``vad`` argument into a list of speech segments."""
    if vad is None:
        try:
            from . import vad as _vad

            return list(_vad.detect_speech(audio, sr))
        except Exception:  # noqa: BLE001 - chunking must work without a VAD
            return []
    if callable(vad):
        try:
            return list(vad(audio, sr))
        except Exception:  # noqa: BLE001 - a broken callback must not break us
            return []
    try:
        return [(float(a), float(b)) for a, b in vad]
    except (TypeError, ValueError):
        return []


def _pause_points(segments: SegmentList, sr: int) -> list[float]:
    """Return the mid points of the gaps between speech segments."""
    pauses: list[float] = []
    for index in range(len(segments) - 1):
        gap_start = segments[index][1]
        gap_end = segments[index + 1][0]
        if gap_end > gap_start:
            pauses.append((gap_start + gap_end) / 2.0)
    return pauses


def _nearest_pause(
    pauses: Sequence[float], target: float, tolerance: float
) -> float | None:
    """Return the pause closest to the target within the tolerance."""
    if not pauses:
        return None
    best = min(pauses, key=lambda value: abs(value - target))
    if abs(best - target) <= tolerance:
        return best
    return None


def chunk_audio(
    audio: np.ndarray,
    sr: int,
    *,
    chunk_seconds: float = 30.0,
    overlap_seconds: float = 2.0,
    vad: None | Sequence[tuple[float, float]] | Callable[[np.ndarray, int], SegmentList] = None,
    respect_silence: bool = True,
) -> list[dict[str, Any]]:
    """Split audio into overlapping chunks.

    Parameters
    ----------
    audio:
        Mono samples.
    sr:
        Sample rate of the samples.
    chunk_seconds:
        Wanted length of each chunk.
    overlap_seconds:
        How much consecutive chunks overlap.
    vad:
        Optional speech segments, either precomputed as ``(start, end)`` pairs
        or a callable that returns them. ``None`` asks the VAD module.
    respect_silence:
        Try to place chunk boundaries inside a speech pause.

    Returns
    -------
    list of dict
        One dictionary per chunk, as described in the module docstring.
    """
    samples = _to_mono(audio)
    total = int(samples.size)
    rate = int(sr) if sr else 0
    if total == 0 or rate <= 0:
        return []
    length = float(chunk_seconds) if chunk_seconds and chunk_seconds > 0 else 30.0
    overlap = float(overlap_seconds) if overlap_seconds and overlap_seconds > 0 else 0.0
    if overlap >= length:
        overlap = max(0.0, length / 2.0)
    step = max(1, length - overlap)
    duration = total / float(rate)

    segments: SegmentList = []
    if respect_silence:
        segments = _resolve_speech(samples, rate, vad)
    pauses = _pause_points(segments, rate)

    chunks: list[dict[str, Any]] = []
    position = 0.0
    index = 0
    tolerance = max(1.0, min(5.0, length / 4.0))
    while position < duration:
        limit = position + length
        target = limit
        if target >= duration:
            boundary = duration
        else:
            # Prefer a speech pause close to the target so a word is not cut.
            # The pause has to sit inside the window, because a speech model
            # that was trained on fixed windows only hears the first window of
            # anything longer.
            pause = _nearest_pause(pauses, target, tolerance) if pauses else None
            if pause is not None and pause <= limit:
                boundary = max(pause, position + length / 2.0)
            else:
                boundary = limit
        start_sample = int(round(position * rate))
        end_sample = int(round(boundary * rate))
        start_sample = max(0, min(start_sample, total))
        end_sample = max(start_sample + 1, min(end_sample, total))
        block = samples[start_sample:end_sample]
        if block.size == 0:
            break
        chunks.append(
            {
                "index": index,
                "start": start_sample / float(rate),
                "end": end_sample / float(rate),
                "samples": np.ascontiguousarray(block, dtype=np.float32),
                "start_sample": start_sample,
                "end_sample": end_sample,
            }
        )
        index += 1
        if end_sample >= total:
            break
        next_position = end_sample / float(rate) - overlap
        if next_position <= position:
            next_position = position + step
        position = next_position
    return chunks


def offset_chunks(chunks: Sequence[dict[str, Any]], index: int, sr: int) -> list[dict[str, Any]]:
    """Shift the sample indices of chunks by ``index`` seconds.

    The audio itself is left untouched, only the bookkeeping moves. This is
    used when a chunk of a larger file is analysed on its own and the word
    timings have to be reported against the original timeline.
    """
    shift_samples = int(round(float(index) * float(sr)))
    shifted: list[dict[str, Any]] = []
    for chunk in chunks or []:
        item = dict(chunk)
        item["start_sample"] = int(chunk.get("start_sample", 0)) + shift_samples
        item["end_sample"] = int(chunk.get("end_sample", 0)) + shift_samples
        if sr:
            item["start"] = item["start_sample"] / float(sr)
            item["end"] = item["end_sample"] / float(sr)
        shifted.append(item)
    return shifted
