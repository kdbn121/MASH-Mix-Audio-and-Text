# SPDX-License-Identifier: MIT | Copyright (c) 2026 MASH (Khaled Bhuiyan)
"""High quality sample rate conversion.

When SciPy is available the polyphase filter from ``scipy.signal.resample_poly``
is used, which is the reference implementation for rational rate conversion.
Without SciPy the module falls back to a band limited windowed sinc
interpolation written with numpy only. The fallback is not a nearest neighbour
or a linear interpolation: it applies a Kaiser windowed sinc kernel whose
cutoff tracks the output Nyquist rate, so it prevents aliasing when
downsampling and stays accurate when upsampling.

Both paths produce a length accurate result of
``round(frames * target_sr / orig_sr)`` samples.
"""

from __future__ import annotations

import math

import numpy as np

from ..types import AudioBuffer

__all__ = ["resample", "resample_buffer", "HAVE_SCIPY"]

try:  # pragma: no cover - depends on the host environment
    from scipy.signal import resample_poly as _resample_poly

    HAVE_SCIPY = True
except Exception:  # pragma: no cover - optional dependency
    _resample_poly = None  # type: ignore[assignment]
    HAVE_SCIPY = False

#: Number of zero crossings kept on each side of the sinc kernel.
_KERNEL_HALF = 16

#: Kaiser shape factor, roughly -80 dB stop band attenuation.
_KERNEL_BETA = 8.6


def _target_length(frames: int, orig_sr: int, target_sr: int) -> int:
    """Return the exact number of output samples for a given conversion."""
    if frames <= 0:
        return 0
    return int(round(frames * float(target_sr) / float(orig_sr)))


def _fit_length(data: np.ndarray, wanted: int, axis: int = -1) -> np.ndarray:
    """Trim or zero pad the last axis so the array has exactly ``wanted`` samples."""
    current = data.shape[axis]
    if current == wanted:
        return data
    if current > wanted:
        slicer = [slice(None)] * data.ndim
        slicer[axis] = slice(0, wanted)
        return data[tuple(slicer)]
    pad = wanted - current
    pad_width = [(0, 0)] * data.ndim
    pad_width[axis] = (0, pad)
    return np.pad(data, pad_width, mode="constant")


def _sinc_resample(x: np.ndarray, orig_sr: int, target_sr: int) -> np.ndarray:
    """Band limited windowed sinc interpolation implemented with numpy only."""
    frames = x.shape[-1]
    ratio = float(target_sr) / float(orig_sr)
    wanted = _target_length(frames, orig_sr, target_sr)
    if wanted <= 0:
        return np.zeros(x.shape[:-1] + (0,), dtype=np.float64)
    # Cutoff in cycles per input sample. It is the input Nyquist rate when
    # upsampling and the output Nyquist rate when downsampling.
    cutoff = 0.5 * min(1.0, ratio)
    # The kernel half width must cover enough zero crossings at the cutoff.
    half = max(_KERNEL_HALF, int(math.ceil(_KERNEL_BETA / (2.0 * cutoff))))
    half = min(half, 512)
    taps = np.arange(-half, half + 1, dtype=np.float64)
    window = np.kaiser(2 * half + 1, _KERNEL_BETA)
    padded = np.concatenate(
        [
            np.zeros(x.shape[:-1] + (half + 1,), dtype=np.float64),
            x.astype(np.float64, copy=False),
            np.zeros(x.shape[:-1] + (half + 1,), dtype=np.float64),
        ],
        axis=-1,
    )
    flat = padded.reshape(-1, padded.shape[-1])
    out = np.zeros((flat.shape[0], wanted), dtype=np.float64)
    block = 32768
    for start in range(0, wanted, block):
        stop = min(start + block, wanted)
        positions = (np.arange(start, stop, dtype=np.float64) + 0.5) / ratio - 0.5
        base = np.floor(positions).astype(np.int64)
        distance = (base[:, None] + taps[None, :]) - positions[:, None]
        kernel = cutoff * np.sinc(cutoff * distance) * window[None, :]
        indices = (base[:, None] + taps[None, :]).astype(np.int64) + half + 1
        indices = np.clip(indices, 0, flat.shape[1] - 1)
        gathered = np.take_along_axis(flat, indices, axis=1)
        out[:, start:stop] = np.sum(gathered * kernel, axis=1)
    return out.reshape(x.shape[:-1] + (wanted,))


def resample(audio: np.ndarray, orig_sr: int, target_sr: int) -> np.ndarray:
    """Resample a waveform to a new rate.

    Parameters
    ----------
    audio:
        A 1-D mono array or a 2-D ``(channels, frames)`` array.
    orig_sr:
        The sample rate of the input.
    target_sr:
        The wanted sample rate.

    Returns
    -------
    numpy.ndarray
        A float32 array with the same channel layout as the input and exactly
        ``round(frames * target_sr / orig_sr)`` samples per channel.
    """
    data = np.asarray(audio)
    if data.ndim == 0:
        raise ValueError("resample expects an array of samples")
    if int(orig_sr) <= 0 or int(target_sr) <= 0:
        raise ValueError("Sample rates must be positive")
    if int(orig_sr) == int(target_sr):
        return np.asarray(data, dtype=np.float32)
    wanted = _target_length(data.shape[-1], int(orig_sr), int(target_sr))
    if HAVE_SCIPY:
        from math import gcd

        divisor = gcd(int(target_sr), int(orig_sr))
        up = int(target_sr) // divisor
        down = int(orig_sr) // divisor
        converted = _resample_poly(data.astype(np.float64), up, down, axis=-1)
        converted = _fit_length(converted, wanted)
    else:
        converted = _sinc_resample(data, int(orig_sr), int(target_sr))
        converted = _fit_length(converted, wanted)
    return np.ascontiguousarray(converted, dtype=np.float32)


def resample_buffer(buffer: AudioBuffer, target_sr: int) -> AudioBuffer:
    """Resample an :class:`AudioBuffer` and return a new buffer."""
    if int(target_sr) <= 0:
        raise ValueError("Target sample rate must be positive")
    samples = resample(buffer.samples, buffer.sample_rate, int(target_sr))
    return AudioBuffer(
        samples=samples, sample_rate=int(target_sr), channels=samples.shape[0]
    )


def resample_to(
    audio: np.ndarray, orig_sr: int, target_sr: int
) -> np.ndarray | None:
    """Convenience wrapper returning ``None`` when the rates already match."""
    if int(orig_sr) == int(target_sr):
        return np.asarray(audio, dtype=np.float32)
    return resample(audio, orig_sr, target_sr)
