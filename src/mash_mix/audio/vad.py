# SPDX-License-Identifier: MIT | Copyright (c) 2026 MASH (Khaled Bhuiyan)
"""Voice activity detection.

Two detectors are available:

* A Silero VAD ONNX model, used through ``onnxruntime`` when the model file is
  present. The loader returns ``None`` when neither the model nor the runtime
  is available so callers can fall back without raising.
* A pure numpy energy and zero crossing rate detector. It frames the signal at
  30 ms with a 10 ms hop, estimates a noise floor, applies hysteresis and then
  merges the segments. This fallback needs no model and no extra package.

The numpy detector is deliberately conservative: it estimates the noise floor
from the lower percentile of the frame energies and only calls a frame speech
when the energy clearly rises above that floor. A stationary signal therefore
stays a single continuous segment instead of collapsing into noise.
"""

from __future__ import annotations

import os
import warnings
from collections.abc import Sequence
from pathlib import Path
from typing import Any

import numpy as np

__all__ = [
    "silero_vad",
    "detect_speech",
    "merge_segments",
    "speech_ratio",
    "trim_silence",
    "HAVE_ONNX",
]

try:  # pragma: no cover - depends on the host environment
    import onnxruntime as _onnxruntime

    HAVE_ONNX = True
except Exception:  # pragma: no cover - optional dependency
    _onnxruntime = None  # type: ignore[assignment]
    HAVE_ONNX = False

#: Frame length of the numpy detector in seconds.
_FRAME_SECONDS = 0.030

#: Hop length of the numpy detector in seconds.
_HOP_SECONDS = 0.010

#: Absolute silence floor in dBFS used when the signal is stationary.
_ABSOLUTE_FLOOR_DB = -55.0

#: Model file names looked for when no explicit path is given.
_MODEL_FILENAMES = ("silero_vad.onnx", "silero_vad_v5.onnx", "silero_vad.onnx.gz")


def _model_search_paths() -> list[Path]:
    """Return the folders scanned for a cached Silero VAD model."""
    paths: list[Path] = []
    env_dir = os.environ.get("MASH_MIX_SILERO_MODEL_DIR")
    if env_dir:
        paths.append(Path(env_dir))
    env_file = os.environ.get("MASH_MIX_SILERO_MODEL")
    if env_file:
        paths.append(Path(env_file).parent)
    cache = os.environ.get("XDG_CACHE_HOME") or os.path.join(
        os.path.expanduser("~"), ".cache"
    )
    paths.append(Path(cache) / "mash-mix" / "vad")
    torch_home = os.environ.get("TORCH_HOME")
    if torch_home:
        paths.append(Path(torch_home) / "hub" / "checkpoints")
    paths.append(Path.home() / ".cache" / "mash-mix" / "vad")
    return paths


def _find_model(model_path: str | None) -> Path | None:
    """Locate a Silero VAD ONNX file on disk."""
    if model_path:
        candidate = Path(model_path)
        if candidate.is_file():
            return candidate
        return None
    for folder in _model_search_paths():
        if not folder.is_dir():
            continue
        for name in _MODEL_FILENAMES:
            candidate = folder / name
            if candidate.is_file():
                return candidate
    return None


class _SileroOnnx:
    """Thin wrapper around the Silero VAD ONNX graph."""

    def __init__(self, session: Any, model_file: Path) -> None:
        self.session = session
        self.model_file = model_file
        inputs = {i.name for i in session.get_inputs()}
        self._input_name = "input" if "input" in inputs else sorted(inputs)[0]
        self._state_name = "state" if "state" in inputs else None
        self._sr_name = "sr" if "sr" in inputs else None

    def probabilities(self, audio: np.ndarray, sr: int) -> np.ndarray:
        """Return one speech probability per 512 sample hop at 16 kHz."""
        from ..audio import resample as _resample

        rate = int(sr)
        if rate != 16000:
            audio = _resample.resample(audio, rate, 16000)
            rate = 16000
        samples = np.asarray(audio, dtype=np.float32).reshape(-1)
        window = 512
        state = np.zeros((2, 1, 128), dtype=np.float32)
        outputs: list[float] = []
        for start in range(0, len(samples), window):
            block = samples[start : start + window]
            if block.size < window:
                block = np.pad(block, (0, window - block.size))
            feed: dict = {self._input_name: block.reshape(1, -1)}
            if self._state_name is not None:
                feed[self._state_name] = state
            if self._sr_name is not None:
                feed[self._sr_name] = np.array(rate, dtype=np.int64)
            result = self.session.run(None, feed)
            outputs.append(float(np.asarray(result[0]).reshape(-1)[0]))
            if len(result) > 1 and isinstance(result[1], np.ndarray):
                state = result[1].astype(np.float32)
        return np.asarray(outputs, dtype=np.float32)

    def __call__(self, audio: np.ndarray, sr: int) -> np.ndarray:
        return self.probabilities(audio, sr)


def silero_vad(model_path: str | None = None, force_onnx: bool = False) -> Any | None:
    """Load the Silero VAD model.

    Parameters
    ----------
    model_path:
        Explicit path to the ONNX model. When omitted a few well known cache
        folders are scanned.
    force_onnx:
        Kept for API symmetry. The ONNX runtime is the only supported path, so
        the flag never changes the outcome.

    Returns
    -------
    object or None
        A callable returning per hop speech probabilities, or ``None`` when the
        runtime or the model file is unavailable.
    """
    if not HAVE_ONNX:
        return None
    model_file = _find_model(model_path)
    if model_file is None:
        return None
    try:
        session = _onnxruntime.InferenceSession(
            str(model_file), providers=["CPUExecutionProvider"]
        )
    except Exception:  # noqa: BLE001 - a broken model must not break the import
        return None
    return _SileroOnnx(session, model_file)


def _frame_signal(audio: np.ndarray, sr: int) -> tuple[np.ndarray, np.ndarray, float]:
    """Frame the signal and return ``(rms, zcr, hop_seconds)`` per frame."""
    samples = np.asarray(audio, dtype=np.float64).reshape(-1)
    hop = max(1, int(round(_HOP_SECONDS * sr)))
    width = max(hop, int(round(_FRAME_SECONDS * sr)))
    if samples.size < width:
        width = max(1, samples.size)
        hop = width
    count = 1 + max(0, (samples.size - width) // hop)
    if samples.size == 0:
        return np.zeros(0), np.zeros(0), hop / float(sr)
    indices = np.arange(width)[None, :] + hop * np.arange(count)[:, None]
    frames = samples[indices]
    rms = np.sqrt(np.mean(frames * frames, axis=1))
    signs = np.sign(frames)
    zcr = np.mean(signs[:, 1:] * signs[:, :-1] < 0, axis=1) if width > 1 else np.zeros(count)
    return rms, zcr, hop / float(sr)


def _energy_detector(
    audio: np.ndarray,
    sr: int,
    min_speech_ms: int,
    min_silence_ms: int,
) -> list[tuple[float, float]]:
    """Detect speech using frame energy and zero crossing rate."""
    rms, zcr, hop_seconds = _frame_signal(audio, sr)
    if rms.size == 0:
        return []
    eps = 1e-12
    rms_db = 20.0 * np.log10(np.maximum(rms, eps))
    peak_db = float(np.percentile(rms_db, 95))
    floor_db = float(np.percentile(rms_db, 10))
    spread = peak_db - floor_db
    if spread < 6.0:
        # The signal is stationary. Either it is above the absolute floor and
        # is treated as one continuous region, or it is silence.
        if peak_db <= _ABSOLUTE_FLOOR_DB:
            return []
        threshold_db = max(floor_db, _ABSOLUTE_FLOOR_DB)
        active = rms_db >= threshold_db
    else:
        threshold_db = floor_db + max(6.0, 0.25 * spread)
        active = rms_db >= threshold_db
    # A frame with almost no crossings is noise rather than speech.
    voiced = (zcr > 0.01) | (rms_db > threshold_db + 8.0)
    active = active & voiced
    if not active.any():
        return []
    high = threshold_db
    low = threshold_db - 4.0
    segments: list[tuple[float, float]] = []
    start = None
    for index, flag in enumerate(active):
        db = rms_db[index]
        if start is None:
            if flag or db >= high:
                start = index
        else:
            if db < low:
                segments.append((start * hop_seconds, index * hop_seconds))
                start = None
    if start is not None:
        segments.append((start * hop_seconds, len(active) * hop_seconds))
    segments = [
        (float(a), float(b)) for a, b in segments if b > a
    ]
    segments = merge_segments(segments, gap=max(0.0, min_silence_ms / 1000.0))
    min_speech = max(0.0, min_speech_ms / 1000.0)
    kept = [(a, b) for a, b in segments if (b - a) >= min_speech]
    return kept


def _silero_segments(
    model: Any,
    audio: np.ndarray,
    sr: int,
    min_speech_ms: int,
    min_silence_ms: int,
) -> list[tuple[float, float]]:
    """Turn Silero probabilities into speech segments."""
    probabilities = np.asarray(model(audio, sr), dtype=np.float32).reshape(-1)
    if probabilities.size == 0:
        return []
    hop_seconds = 512.0 / 16000.0
    active = probabilities >= 0.5
    segments: list[tuple[float, float]] = []
    start = None
    for index, flag in enumerate(active):
        if flag and start is None:
            start = index
        elif not flag and start is not None:
            segments.append((start * hop_seconds, index * hop_seconds))
            start = None
    if start is not None:
        segments.append((start * hop_seconds, len(active) * hop_seconds))
    segments = merge_segments(segments, gap=max(0.0, min_silence_ms / 1000.0))
    min_speech = max(0.0, min_speech_ms / 1000.0)
    return [(a, b) for a, b in segments if (b - a) >= min_speech]


def detect_speech(
    audio: np.ndarray,
    sr: int,
    *,
    min_speech_ms: int = 250,
    min_silence_ms: int = 300,
    method: str = "auto",
) -> list[tuple[float, float]]:
    """Return the speech regions of an audio signal as ``(start, end)`` pairs.

    Parameters
    ----------
    audio:
        Mono samples.
    sr:
        The sample rate of the samples.
    min_speech_ms:
        Regions shorter than this are dropped.
    min_silence_ms:
        Gaps shorter than this are treated as continuous speech.
    method:
        ``"auto"`` uses Silero when it is available and the numpy detector
        otherwise. ``"energy"`` forces the numpy detector. ``"silero"`` asks for
        Silero and warns when it is unavailable before falling back.
    """
    samples = np.asarray(audio, dtype=np.float32).reshape(-1)
    if samples.size == 0 or int(sr) <= 0:
        return []
    wanted = str(method or "auto").lower()
    if wanted not in ("auto", "energy", "silero", "numpy", "onnx"):
        raise ValueError(f"Unknown VAD method: {method!r}")
    if wanted in ("silero", "onnx"):
        model = silero_vad()
        if model is None:
            warnings.warn(
                "Silero VAD is not available, falling back to the numpy "
                "energy detector.",
                RuntimeWarning,
                stacklevel=2,
            )
        else:
            try:
                return _silero_segments(model, samples, int(sr), min_speech_ms, min_silence_ms)
            except Exception:  # noqa: BLE001 - never fail on the model
                pass
    if wanted == "auto":
        model = silero_vad()
        if model is not None:
            try:
                return _silero_segments(model, samples, int(sr), min_speech_ms, min_silence_ms)
            except Exception:  # noqa: BLE001 - never fail on the model
                pass
    return _energy_detector(samples, int(sr), min_speech_ms, min_silence_ms)


def merge_segments(
    segments: Sequence[tuple[float, float]], gap: float = 0.3
) -> list[tuple[float, float]]:
    """Merge segments that are separated by less than ``gap`` seconds."""
    items = []
    for pair in segments or []:
        try:
            start, end = float(pair[0]), float(pair[1])
        except (TypeError, ValueError, IndexError):
            continue
        if end > start:
            items.append((start, end))
    if not items:
        return []
    items.sort()
    merged: list[list[float]] = [list(items[0])]
    for start, end in items[1:]:
        if start - merged[-1][1] <= float(gap):
            merged[-1][1] = max(merged[-1][1], end)
        else:
            merged.append([start, end])
    return [(float(a), float(b)) for a, b in merged]


def speech_ratio(audio: np.ndarray, sr: int) -> float:
    """Return the fraction of the signal that is speech, between 0 and 1."""
    samples = np.asarray(audio, dtype=np.float32).reshape(-1)
    if samples.size == 0 or int(sr) <= 0:
        return 0.0
    total = samples.size / float(sr)
    segments = detect_speech(samples, int(sr))
    spoken = sum(max(0.0, b - a) for a, b in segments)
    if total <= 0:
        return 0.0
    return float(min(1.0, spoken / total))


def trim_silence(
    audio: np.ndarray, sr: int, **kwargs: Any
) -> tuple[np.ndarray, int]:
    """Remove leading and trailing silence.

    Returns
    -------
    tuple
        The trimmed samples and the number of samples removed from the front,
        so the caller can realign word timings.
    """
    samples = np.asarray(audio, dtype=np.float32).reshape(-1)
    if samples.size == 0 or int(sr) <= 0:
        return samples, 0
    kwargs.setdefault("min_silence_ms", 300)
    kwargs.setdefault("min_speech_ms", 250)
    segments = detect_speech(samples, int(sr), **kwargs)
    if not segments:
        return samples, 0
    pad = float(kwargs.get("pad_seconds", 0.05))
    start = max(0, int((segments[0][0] - pad) * sr))
    end = min(samples.size, int((segments[-1][1] + pad) * sr) + 1)
    if end <= start:
        return samples, 0
    return samples[start:end].copy(), int(start)
