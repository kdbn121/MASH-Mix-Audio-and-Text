# SPDX-License-Identifier: MIT | Copyright (c) 2026 MASH (Khaled Bhuiyan)
"""Musical and acoustic measurements.

Everything in this module is computed with numpy so the package stays usable
with numpy as the only dependency. When librosa happens to be installed the
tempo estimate is taken from it because it is better tested, otherwise a numpy
autocorrelation beat tracker is used.

The loudness measurement follows the ITU-R BS.1770 recipe: K weighting with a
high shelf and an RLB high pass, 400 ms blocks with 75 percent overlap, an
absolute gate at -70 LUFS and a relative gate 10 LU below the integrated
value.

No measurement raises. Silence, empty input and very short input simply
produce an :class:`AnalysisResult` where the fields that cannot be computed
stay ``None``.
"""

from __future__ import annotations

import math
from collections.abc import Sequence
from typing import Any

import numpy as np

from ..types import AnalysisResult

__all__ = ["analyze", "HAVE_LIBROSA", "HAVE_SCIPY"]

try:  # pragma: no cover - depends on the host environment
    import librosa as _librosa

    HAVE_LIBROSA = True
except Exception:  # pragma: no cover - optional dependency
    _librosa = None  # type: ignore[assignment]
    HAVE_LIBROSA = False

try:  # pragma: no cover - depends on the host environment
    from scipy.signal import lfilter as _lfilter

    HAVE_SCIPY = True
except Exception:  # pragma: no cover - optional dependency
    _lfilter = None  # type: ignore[assignment]
    HAVE_SCIPY = False


_KEY_NAMES = ("C", "C#", "D", "D#", "E", "F", "F#", "G", "G#", "A", "A#", "B")

_MAJOR_PROFILE = np.array(
    [6.35, 2.23, 3.48, 2.33, 4.38, 4.09, 2.52, 5.19, 2.39, 3.66, 2.29, 2.88]
)
_MINOR_PROFILE = np.array(
    [6.33, 2.68, 3.52, 5.38, 2.60, 3.53, 2.54, 4.75, 3.98, 2.69, 3.34, 3.17]
)

_MIN_DB = -120.0
_MAX_BEATS = 2000

#: Half width of the triangular window that folds a frequency bin onto a
#: semitone, in semitones.
_CHROMA_WIDTH = 0.5

#: How close the autocorrelation at half the chosen period has to be before the
#: tempo is read as twice as fast. A beat tracker that loses the plot often
#: finds the period of two beats instead of one.
_TEMPO_OCTAVE_RATIO = 0.8

#: The tempo a beat tracker expects a track to be at, in beats a minute.
_TEMPO_PRIOR_BPM = 120.0

#: How many octaves around that expectation the prior reaches, as a standard
#: deviation on a log scale.
_TEMPO_PRIOR_WIDTH = 1.0

#: The step of the beat period search, in frames of the onset envelope. The
#: real beat rarely falls on a whole frame, and a search that only looks at
#: whole frames reads the wrong tempo when the beat lands between two of them.
_PERIOD_STEP_FRAMES = 0.05


# ---------------------------------------------------------------------------
# small helpers
# ---------------------------------------------------------------------------


def _mono(audio: np.ndarray) -> tuple[np.ndarray, int]:
    """Return mono float32 samples and the channel count of the input."""
    data = np.asarray(audio, dtype=np.float32)
    if data.ndim == 1:
        return data, 1
    if data.ndim == 2:
        channels = int(data.shape[0])
        if channels == 1:
            return data[0], 1
        return data.mean(axis=0).astype(np.float32), channels
    return data.reshape(-1), 1


def _db(value: float) -> float:
    """Convert a linear magnitude to dBFS with a floor."""
    if value <= 0:
        return _MIN_DB
    return max(_MIN_DB, float(20.0 * math.log10(value)))


def _frames(x: np.ndarray, n_fft: int, hop: int) -> np.ndarray:
    """Slice a signal into overlapping frames."""
    if x.size < n_fft:
        x = np.pad(x, (0, n_fft - x.size))
    count = 1 + (x.size - n_fft) // hop
    indices = np.arange(n_fft)[None, :] + hop * np.arange(count)[:, None]
    return x[indices]


def _hann(n: int) -> np.ndarray:
    return np.hanning(n) if n > 1 else np.ones(1)


def _stft_magnitude(x: np.ndarray, n_fft: int, hop: int) -> np.ndarray:
    """Magnitude spectrogram shaped ``(frames, bins)``."""
    frames = _frames(x, n_fft, hop)
    spectrum = np.fft.rfft(frames * _hann(n_fft), axis=1)
    return np.abs(spectrum).astype(np.float64)


def _autocorrelation(signal: np.ndarray) -> np.ndarray:
    """Normalised autocorrelation computed with the FFT."""
    if signal.size == 0:
        return np.zeros(0)
    n = int(1 << (2 * signal.size - 1).bit_length())
    spectrum = np.fft.rfft(signal, n)
    ac = np.fft.irfft(np.abs(spectrum) ** 2, n)[: signal.size]
    if ac[0] > 0:
        ac = ac / ac[0]
    return ac


# ---------------------------------------------------------------------------
# tempo and beats
# ---------------------------------------------------------------------------


def _onset_envelope(x: np.ndarray, sr: int) -> tuple[np.ndarray, float]:
    """Return the onset strength envelope and the hop in seconds."""
    n_fft = 1024
    hop = 512
    magnitude = _stft_magnitude(x, n_fft, hop)
    if magnitude.shape[0] < 2:
        return np.zeros(0), hop / float(sr)
    diff = np.diff(magnitude, axis=0)
    envelope = np.sum(np.maximum(diff, 0.0), axis=1)
    if envelope.size:
        peak = float(np.max(envelope))
        if peak > 0:
            envelope = envelope / peak
    return envelope, hop / float(sr)


def _ac_at(ac: np.ndarray, lag: float) -> float:
    """The autocorrelation read at a lag that falls between two frames."""
    if lag < 0.0 or lag >= float(ac.size) - 1.0:
        return 0.0
    low = int(math.floor(lag))
    fraction = lag - low
    return float(ac[low] * (1.0 - fraction) + ac[low + 1] * fraction)


def _phase_concentration(times: np.ndarray, weights: np.ndarray, period: float) -> float:
    """How tightly the onsets sit on the beat grid of a given period.

    Every onset is folded onto a circle of circumference ``period``. When the
    period is the real beat, the onsets all land on the same point of that
    circle and the length of their weighted average is close to one. When the
    period is twice the beat, every second onset lands on the opposite point
    and the two halves cancel each other out.

    This is what tells a beat apart from two beats. Unlike a plain
    autocorrelation peak it keeps working when the beat falls between two
    frames of the analysis grid, which is the normal case for a real tempo.
    """
    if times.size == 0 or period <= 0:
        return 0.0
    total = float(np.sum(weights))
    if total <= 0:
        return 0.0
    angles = 2.0 * math.pi * times / period
    real = float(np.sum(weights * np.cos(angles)))
    imag = float(np.sum(weights * np.sin(angles)))
    return math.sqrt(real * real + imag * imag) / total


def _pick_period(
    ac: np.ndarray,
    min_lag: int,
    max_lag: int,
    frame_rate: float,
    onset_times: np.ndarray,
    onset_weights: np.ndarray,
    hop_seconds: float,
) -> float | None:
    """The beat period in frames, or ``None`` when nothing stands out.

    The strongest single peak of the autocorrelation is not always the beat. A
    track that beats twice as fast has a strong peak at twice the period as
    well, and the slower one can win by a hair, which is how a tempo ends up
    reported at half its real value.

    Every candidate period is therefore scored on three things: the
    autocorrelation at its multiples, so a period whose double also carries
    support wins over a period that stands alone; how tightly the onsets sit
    on its grid, which is what tells a beat apart from two beats; and a tempo
    prior, which pulls the result towards the range a listener expects.

    The search moves in steps of a fraction of a frame, because the beat of a
    real track does not land on a whole frame and a search that only looks at
    whole frames picks the wrong tempo whenever it does not.
    """
    if max_lag <= min_lag:
        return None
    best_frame: float | None = None
    best_score = -1.0
    frame = float(min_lag)
    while frame <= float(max_lag):
        period = frame * hop_seconds
        score = 0.0
        weights = 0.0
        for multiple, weight in ((1, 1.0), (2, 0.5), (3, 0.33), (4, 0.25)):
            score += weight * _ac_at(ac, frame * multiple)
            weights += weight
        if weights > 0:
            bpm = 60.0 / period
            lock = _phase_concentration(onset_times, onset_weights, period)
            combined = (score / weights) * (0.25 + 0.75 * lock) * _tempo_prior(bpm)
            if combined > best_score:
                best_score = combined
                best_frame = frame
        frame += _PERIOD_STEP_FRAMES
    return best_frame


def _tempo_prior(bpm: float) -> float:
    """A log normal prior over the tempo, centred on 120 beats a minute."""
    if bpm <= 0:
        return 0.0
    distance = math.log2(bpm / _TEMPO_PRIOR_BPM)
    return math.exp(-0.5 * (distance / _TEMPO_PRIOR_WIDTH) ** 2)


def _numpy_beats(x: np.ndarray, sr: int) -> tuple[float | None, list[float]]:
    """Beat tracker built on the numpy autocorrelation of the onset envelope."""
    envelope, hop_seconds = _onset_envelope(x, sr)
    if envelope.size < 8:
        return None, []
    envelope = envelope - float(np.mean(envelope))
    if float(np.max(np.abs(envelope))) <= 1e-9:
        return None, []
    ac = _autocorrelation(envelope)
    frame_rate = 1.0 / hop_seconds
    min_lag = max(2, int(frame_rate * 60.0 / 200.0))
    max_lag = min(ac.size - 1, int(frame_rate * 60.0 / 40.0))
    if max_lag <= min_lag:
        return None, []
    # The first meaningful onset anchors the beat grid.
    positive = np.maximum(envelope, 0.0)
    threshold = float(np.mean(positive) + 0.5 * float(np.std(positive)))
    anchors = np.flatnonzero(positive > threshold)
    if anchors.size == 0:
        return None, []
    offset = float(anchors[0]) * hop_seconds
    onset_times = anchors.astype(np.float64) * hop_seconds
    onset_weights = positive[anchors]
    best = _pick_period(
        ac, min_lag, max_lag, frame_rate, onset_times, onset_weights, hop_seconds
    )
    if best is None:
        return None, []
    # Refine the peak with a parabolic fit on the three points around it.
    lag = float(best)
    left = _ac_at(ac, lag - 1.0)
    middle = _ac_at(ac, lag)
    right = _ac_at(ac, lag + 1.0)
    denominator = left - 2.0 * middle + right
    if abs(denominator) > 1e-12:
        refined = lag + 0.5 * (left - right) / denominator
        if min_lag <= refined <= max_lag:
            lag = refined
    confidence = float(_ac_at(ac, lag))
    if confidence < 0.05:
        return None, []
    period = lag / frame_rate
    if period <= 0:
        return None, []
    bpm = 60.0 / period
    if bpm < 40.0 or bpm > 200.0:
        return None, []
    total = x.size / float(sr)
    beats: list[float] = []
    time = offset
    while time <= total and len(beats) < _MAX_BEATS:
        if time >= 0:
            beats.append(float(time))
        time += period
    return float(bpm), beats


def _librosa_beats(x: np.ndarray, sr: int) -> tuple[float | None, list[float]]:
    """Tempo and beats from librosa when it is installed."""
    try:
        tempo, frames = _librosa.beat.beat_track(y=x, sr=sr)
        bpm = float(np.asarray(tempo).reshape(-1)[0])
        times = _librosa.frames_to_time(frames, sr=sr)
        beats = [float(t) for t in np.asarray(times).reshape(-1)][:_MAX_BEATS]
        return bpm, beats
    except Exception:  # noqa: BLE001 - never fail on the optional backend
        return None, []


# ---------------------------------------------------------------------------
# key and mode
# ---------------------------------------------------------------------------


def _chroma(x: np.ndarray, sr: int) -> np.ndarray | None:
    """Average chroma vector built from an FFT magnitude spectrogram.

    Only the peaks of the average spectrum are read, and each peak is folded
    onto the nearest semitone with a triangular window, so a bin that sits
    between two semitones counts for less than a peak that sits on one.

    Reading the peaks instead of every bin is what keeps the answer honest.
    A loud note spreads its energy over the bins around it, and the broadband
    skirt of that energy leaks into the pitch classes next door. The harmonic
    content of a chord does the same, and it pushes the answer towards the
    wrong key: a C major triad that carries a few harmonics reads as E minor
    when the broadband energy is counted, and as C major when only the peaks
    are counted.
    """
    n_fft = 4096
    hop = 2048
    magnitude = _stft_magnitude(x, n_fft, hop)
    if magnitude.shape[0] == 0:
        return None
    spectrum = magnitude.mean(axis=0)
    freqs = np.fft.rfftfreq(n_fft, 1.0 / float(sr))
    chroma = np.zeros(12)
    for index in range(1, spectrum.size - 1):
        value = float(spectrum[index])
        if value <= 0.0:
            continue
        if value <= spectrum[index - 1] or value < spectrum[index + 1]:
            continue
        frequency = freqs[index]
        if frequency < 55.0 or frequency > 2200.0:
            continue
        # The peak of a tone is worked out from the three bins around it, so
        # the frequency of the tone does not depend on where it happens to
        # fall between two bins. At the bottom of the range a semitone is
        # narrower than a bin, and the raw bin of a peak can be a quarter of a
        # semitone away from the tone it belongs to.
        left = math.log(float(spectrum[index - 1]))
        middle = math.log(value)
        right = math.log(float(spectrum[index + 1]))
        denominator = left - 2.0 * middle + right
        shift = 0.0
        if abs(denominator) > 1e-18:
            shift = 0.5 * (left - right) / denominator
            shift = max(-1.0, min(1.0, shift))
        hz = (index + shift) * (float(sr) / n_fft)
        peak = math.exp(middle - 0.25 * (left - right) * shift)
        midi = 69.0 + 12.0 * math.log2(hz / 440.0)
        centre = float(round(midi))
        distance = abs(midi - centre)
        if distance > _CHROMA_WIDTH:
            continue
        weight = 1.0 - (distance / _CHROMA_WIDTH)
        chroma[int(centre) % 12] += peak * weight
    total = float(chroma.sum())
    if total <= 0:
        return None
    return chroma / total


def _correlate(a: np.ndarray, b: np.ndarray) -> float:
    """Pearson correlation between two short vectors."""
    x = np.asarray(a, dtype=np.float64)
    y = np.asarray(b, dtype=np.float64)
    if x.size < 2 or y.size < 2:
        return 0.0
    x = x - x.mean()
    y = y - y.mean()
    denominator = float(np.sqrt(float(np.sum(x * x)) * float(np.sum(y * y))))
    if denominator <= 1e-12:
        return 0.0
    return float(np.sum(x * y) / denominator)


def _estimate_key(chroma: np.ndarray | None) -> tuple[str | None, str | None]:
    """Estimate the musical key from a chroma vector."""
    if chroma is None:
        return None, None
    best_score = -1.0
    best_key: str | None = None
    best_mode: str | None = None
    for rotation in range(12):
        major = _correlate(chroma, np.roll(_MAJOR_PROFILE, rotation))
        minor = _correlate(chroma, np.roll(_MINOR_PROFILE, rotation))
        for score, mode in ((major, "major"), (minor, "minor")):
            if score > best_score:
                best_score = score
                best_key = _KEY_NAMES[rotation]
                best_mode = mode
    if best_key is None or best_score < 0.30:
        return None, None
    return best_key, best_mode


# ---------------------------------------------------------------------------
# loudness
# ---------------------------------------------------------------------------


def _biquad_high_shelf(fs: float) -> tuple[list[float], list[float]]:
    """The BS.1770 pre filter, a high shelf at about 1681 Hz."""
    f0 = 1681.974450955533
    gain_db = 3.999843853973347
    q = 0.7071752369554196
    k = math.tan(math.pi * f0 / fs)
    vh = 10.0 ** (gain_db / 20.0)
    vb = vh ** 0.4996667741545416
    a0 = 1.0 + k / q + k * k
    b = [(vh + vb * k / q + k * k) / a0, 2.0 * (k * k - vh) / a0, (vh - vb * k / q + k * k) / a0]
    a = [1.0, 2.0 * (k * k - 1.0) / a0, (1.0 - k / q + k * k) / a0]
    return b, a


def _biquad_high_pass(fs: float) -> tuple[list[float], list[float]]:
    """The BS.1770 RLB filter, a high pass at about 38 Hz."""
    f0 = 38.13547087602444
    q = 0.5003270373238773
    k = math.tan(math.pi * f0 / fs)
    a0 = 1.0 + k / q + k * k
    b = [1.0, -2.0, 1.0]
    a = [1.0, 2.0 * (k * k - 1.0) / a0, (1.0 - k / q + k * k) / a0]
    return b, a


def _apply_biquad(x: np.ndarray, b: Sequence[float], a: Sequence[float]) -> np.ndarray:
    """Apply a second order IIR filter, with a numpy fallback for SciPy."""
    if HAVE_SCIPY:
        return _lfilter(list(b), list(a), x)
    length = min(x.size, 8192)
    impulse = np.zeros(length)
    impulse[0] = 1.0
    out = np.zeros(length)
    b0, b1, b2 = b[0], b[1], b[2]
    a1, a2 = a[1], a[2]
    for n in range(length):
        value = b0 * impulse[n]
        if n >= 1:
            value += b1 * impulse[n - 1] - a1 * out[n - 1]
        if n >= 2:
            value += b2 * impulse[n - 2] - a2 * out[n - 2]
        out[n] = value
    if x.size <= length:
        return out
    # A biquad keeps the signal length, exactly like scipy.signal.lfilter.
    return _fft_convolve(x, out)[: x.size]


def _fft_convolve(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    """Linear convolution through the FFT with a block size that keeps memory low."""
    total = a.size + b.size - 1
    result = np.zeros(total, dtype=np.float64)
    block = 1 << 18
    kernel_fft = None
    for start in range(0, a.size, block):
        stop = min(start + block, a.size)
        size = 1 << int(math.ceil(math.log2((stop - start) + b.size - 1)))
        if kernel_fft is None or kernel_fft.shape[0] != size:
            kernel_fft = np.fft.rfft(b, size)
        chunk = np.fft.rfft(a[start:stop], size) * kernel_fft
        result[start : start + (stop - start) + b.size - 1] += np.fft.irfft(chunk, size)[
            : (stop - start) + b.size - 1
        ]
    return result[:total]


def _block_mean_square(x: np.ndarray, block: int, hop: int) -> np.ndarray:
    """Mean square of every block with the given length and hop."""
    if x.size < block:
        return np.zeros(0)
    cumulative = np.concatenate(([0.0], np.cumsum(x * x, dtype=np.float64)))
    starts = np.arange(0, x.size - block + 1, hop)
    return (cumulative[starts + block] - cumulative[starts]) / float(block)


def _lufs(channels: np.ndarray, sr: int) -> float | None:
    """Gated loudness in LUFS following the ITU-R BS.1770 recipe.

    Every channel is K weighted on its own, the block mean squares are then
    summed with the standard channel weights, so a stereo file is not
    downmixed before it is measured.
    """
    data = np.asarray(channels, dtype=np.float64)
    if data.ndim == 1:
        data = data[None, :]
    if data.size == 0 or sr <= 0:
        return None
    fs = float(sr)
    shelf_b, shelf_a = _biquad_high_shelf(fs)
    rlb_b, rlb_a = _biquad_high_pass(fs)
    block = max(1, int(round(0.4 * fs)))
    hop = max(1, int(round(0.1 * fs)))
    # Left and right carry a weight of 1.0, surround channels use 1.41.
    weights = np.array([1.0, 1.0] + [1.41] * max(0, data.shape[0] - 2))
    powers: list[np.ndarray] = []
    for index in range(data.shape[0]):
        weighted = _apply_biquad(_apply_biquad(data[index], shelf_b, shelf_a), rlb_b, rlb_a)
        block_power = _block_mean_square(weighted, block, hop)
        if block_power.size:
            powers.append(block_power)
    if not powers:
        return None
    length = min(item.size for item in powers)
    if length == 0:
        return None
    stacked = np.vstack([item[:length] for item in powers])
    power = np.sum(weights[: stacked.shape[0], None] * stacked, axis=0)
    with np.errstate(divide="ignore"):
        levels = -0.691 + 10.0 * np.log10(np.maximum(power, 1e-12))
    absolute = levels > -70.0
    if not absolute.any():
        return None
    mean_absolute = float(np.mean(power[absolute]))
    relative_threshold = -0.691 + 10.0 * math.log10(max(mean_absolute, 1e-12)) - 10.0
    gate = absolute & (levels > relative_threshold)
    if not gate.any():
        gate = absolute
    mean_gated = float(np.mean(power[gate]))
    if mean_gated <= 0:
        return None
    return float(-0.691 + 10.0 * math.log10(mean_gated))


# ---------------------------------------------------------------------------
# spectral centroid
# ---------------------------------------------------------------------------


def _spectral_centroid(x: np.ndarray, sr: int) -> float | None:
    """Mean frequency of the magnitude spectrum of the whole signal."""
    if x.size < 8 or sr <= 0:
        return None
    window = _hann(x.size)
    spectrum = np.abs(np.fft.rfft(x * window))
    freqs = np.fft.rfftfreq(x.size, 1.0 / float(sr))
    total = float(spectrum.sum())
    if total <= 0:
        return None
    return float(np.sum(freqs * spectrum) / total)


# ---------------------------------------------------------------------------
# structural sections
# ---------------------------------------------------------------------------


def _mel_filterbank(sr: int, n_fft: int, n_mels: int = 24) -> np.ndarray:
    """A small triangular mel filterbank built with numpy."""
    fmax = float(sr) / 2.0
    fmin = 50.0

    def hz_to_mel(value: float) -> float:
        return 2595.0 * math.log10(1.0 + value / 700.0)

    def mel_to_hz(value: float) -> float:
        return 700.0 * (10.0 ** (value / 2595.0) - 1.0)

    mels = np.linspace(hz_to_mel(fmin), hz_to_mel(fmax), n_mels + 2)
    hz = mel_to_hz(mels)
    bins = np.clip(np.floor((n_fft + 1) * hz / float(sr)).astype(int), 0, n_fft // 2)
    bank = np.zeros((n_mels, n_fft // 2 + 1))
    for index in range(1, n_mels + 1):
        left, center, right = bins[index - 1], bins[index], bins[index + 1]
        if center <= left:
            center = min(left + 1, bins[-1])
        if right <= center:
            right = min(center + 1, bins[-1])
        for k in range(left, center):
            bank[index - 1, k] = (k - left) / float(max(1, center - left))
        for k in range(center, right):
            bank[index - 1, k] = (right - k) / float(max(1, right - center))
    return bank


def _checkerboard_kernel(half: int) -> np.ndarray:
    """The checkerboard kernel used for structure analysis."""
    size = 2 * half + 1
    kernel = np.zeros((size, size))
    for i in range(half + 1):
        kernel[i, i] = 1.0
        kernel[size - 1 - i, size - 1 - i] = 1.0
    for i in range(half + 1):
        for j in range(half + 1):
            if i == j:
                continue
            kernel[i, size - 1 - j] = -1.0
            kernel[size - 1 - i, j] = -1.0
    return kernel


def _sections(x: np.ndarray, sr: int) -> list[dict[str, Any]]:
    """Detect structural boundaries with a self similarity matrix."""
    duration = x.size / float(sr)
    if duration < 2.0:
        if duration > 0:
            return [{"start": 0.0, "end": round(duration, 3), "label": "section_1"}]
        return []
    frame_seconds = 1.0
    hop_seconds = 0.5
    n_fft = 2048
    stft_hop = max(1, int(round(0.05 * sr)))
    magnitude = _stft_magnitude(x, n_fft, stft_hop)
    if magnitude.shape[0] < 2:
        return [{"start": 0.0, "end": round(duration, 3), "label": "section_1"}]
    bank = _mel_filterbank(int(sr), n_fft)
    energies = np.log1p(magnitude @ bank.T)
    # One feature vector per one second window, advanced every half second.
    window = max(1, int(round(frame_seconds * sr / stft_hop)))
    step = max(1, window // 2)
    if energies.shape[0] < window:
        return [{"start": 0.0, "end": round(duration, 3), "label": "section_1"}]
    cumulative = np.concatenate(
        [np.zeros((1, energies.shape[1])), np.cumsum(energies, axis=0)], axis=0
    )
    starts = np.arange(0, energies.shape[0] - window + 1, step)
    features = (cumulative[starts + window] - cumulative[starts]) / float(window)
    norms = np.linalg.norm(features, axis=1, keepdims=True)
    norms[norms <= 0] = 1.0
    features = features / norms
    similarity = features @ features.T
    half = min(16, max(2, similarity.shape[0] // 4))
    kernel = _checkerboard_kernel(half)
    novelty = _convolve_diagonal(similarity, kernel, half)
    if novelty.size < 3:
        return [{"start": 0.0, "end": round(duration, 3), "label": "section_1"}]
    mean = float(np.mean(novelty))
    std = float(np.std(novelty))
    threshold = mean + 0.5 * std
    boundaries: list[float] = []
    min_distance = max(1, int(2.0 / hop_seconds))
    last = -min_distance
    for index in range(1, novelty.size - 1):
        value = novelty[index]
        if value >= threshold and value >= novelty[index - 1] and value >= novelty[index + 1]:
            time = starts[index] * stft_hop / float(sr)
            if index - last >= min_distance:
                boundaries.append(time)
                last = index
    edges = [0.0] + [b for b in boundaries if 0.0 < b < duration] + [duration]
    sections: list[dict[str, Any]] = []
    for index in range(len(edges) - 1):
        start = edges[index]
        end = edges[index + 1]
        if end - start < 4.0 and sections:
            # Fold a very short region into the section before it.
            sections[-1]["end"] = round(end, 3)
            continue
        sections.append(
            {"start": round(start, 3), "end": round(end, 3), "label": f"section_{len(sections) + 1}"}
        )
    if not sections:
        sections = [{"start": 0.0, "end": round(duration, 3), "label": "section_1"}]
    for index, section in enumerate(sections, start=1):
        section["label"] = f"section_{index}"
    return sections


def _convolve_diagonal(matrix: np.ndarray, kernel: np.ndarray, half: int) -> np.ndarray:
    """Convolve a matrix and return the diagonal of the result as a novelty curve.

    The novelty at position ``t`` is the sum of the kernel weighted self
    similarity values around ``(t, t)``. The kernel is small so the sum is
    evaluated directly with numpy instead of a full two dimensional FFT.
    """
    size = matrix.shape[0]
    if size == 0:
        return np.zeros(0)
    out = np.zeros(size, dtype=np.float64)
    positions = np.arange(size)
    for i in range(kernel.shape[0]):
        rows = positions + i - half
        row_mask = (rows >= 0) & (rows < size)
        if not row_mask.any():
            continue
        for j in range(kernel.shape[1]):
            weight = float(kernel[i, j])
            if weight == 0.0:
                continue
            cols = positions + j - half
            mask = row_mask & (cols >= 0) & (cols < size)
            if not mask.any():
                continue
            out[mask] += weight * matrix[rows[mask], cols[mask]]
    return out


# ---------------------------------------------------------------------------
# signal health
# ---------------------------------------------------------------------------

#: A sample at or above this level counts as clipped.
_CLIP_LEVEL = 0.999

#: The level below which a sample counts as silence, in dBFS.
_SILENCE_DBFS = -60.0


def _clipped_samples(raw: np.ndarray) -> int:
    """Count the samples that sit at or above full scale.

    A clipped sample has reached the ceiling of the format and lost its shape,
    so it is the one fault in a recording that cannot be repaired afterwards.
    """
    if raw.size == 0:
        return 0
    return int(np.count_nonzero(np.abs(raw) >= _CLIP_LEVEL))


def _dc_offset(raw: np.ndarray) -> float | None:
    """The direct current offset of the signal, as a percentage of full scale.

    A small offset is normal in a recorded file. A large one means the signal
    has drifted and will eat headroom.
    """
    if raw.size == 0:
        return None
    return float(abs(np.mean(raw)) * 100.0)


def _head_silence(samples: np.ndarray, sr: int) -> float | None:
    """Seconds of silence before the first sound in the file."""
    if samples.size == 0 or sr <= 0:
        return None
    floor = 10.0 ** (_SILENCE_DBFS / 20.0)
    loud = np.flatnonzero(np.abs(samples) > floor)
    if loud.size == 0:
        return float(samples.size / float(sr))
    return float(loud[0] / float(sr))


def _crest_db(peak_dbfs: float | None, rms_dbfs: float | None) -> float | None:
    """The crest factor, the gap between the peak and the average level.

    A small crest means a squashed recording with no headroom left. A large
    crest means a wide dynamic range, which is what a live recording of a band
    usually has.
    """
    if peak_dbfs is None or rms_dbfs is None:
        return None
    return float(peak_dbfs - rms_dbfs)


# ---------------------------------------------------------------------------
# public API
# ---------------------------------------------------------------------------


def analyze(audio: np.ndarray, sr: int, *, backend: str = "auto") -> AnalysisResult:
    """Measure the musical and acoustic properties of an audio signal.

    Parameters
    ----------
    audio:
        Mono or multi channel samples.
    sr:
        Sample rate of the samples.
    backend:
        ``"auto"`` uses librosa when it is installed, ``"librosa"`` forces it
        and ``"numpy"`` forces the built in implementation.

    Returns
    -------
    AnalysisResult
        Duration, sample rate and channel count are always filled. Every other
        field is ``None`` when it could not be computed.
    """
    raw = np.asarray(audio, dtype=np.float64)
    samples, channels = _mono(raw)
    rate = int(sr) if sr else 0
    duration = samples.size / float(rate) if rate > 0 else 0.0
    result = AnalysisResult(
        duration=float(duration),
        sample_rate=rate,
        channels=int(channels),
        backend="numpy",
        meta={},
    )
    if samples.size == 0 or rate <= 0:
        return result
    peak = float(np.max(np.abs(raw))) if raw.size else 0.0
    rms = float(np.sqrt(np.mean(np.square(raw))))
    result.peak_dbfs = _db(peak)
    result.rms_dbfs = _db(rms)
    result.clipped_samples = _clipped_samples(raw)
    result.dc_offset_percent = _dc_offset(raw)
    result.head_silence_seconds = _head_silence(samples, rate)
    result.crest_db = _crest_db(result.peak_dbfs, result.rms_dbfs)
    result.spectral_centroid = _spectral_centroid(samples.astype(np.float64), rate)
    try:
        result.loudness_lufs = _lufs(raw, rate)
    except Exception:  # noqa: BLE001 - a measurement must never break the call
        result.loudness_lufs = None
    try:
        result.sections = _sections(samples.astype(np.float64), rate)
    except Exception:  # noqa: BLE001 - a measurement must never break the call
        result.sections = []
    try:
        chroma = _chroma(samples.astype(np.float64), rate)
        result.key, result.mode = _estimate_key(chroma)
    except Exception:  # noqa: BLE001 - a measurement must never break the call
        result.key, result.mode = None, None
    wanted = str(backend or "auto").lower()
    bpm: float | None = None
    beats: list[float] = []
    used = "numpy"
    if wanted in ("auto", "librosa") and HAVE_LIBROSA:
        bpm, beats = _librosa_beats(samples.astype(np.float64), rate)
        used = "librosa" if bpm is not None else "numpy"
    if bpm is None:
        bpm, beats = _numpy_beats(samples.astype(np.float64), rate)
    result.bpm = bpm
    result.beats = [float(b) for b in beats[:_MAX_BEATS]]
    result.backend = used
    result.meta = {
        "backend": used,
        "librosa_available": HAVE_LIBROSA,
        "scipy_available": HAVE_SCIPY,
        "peak_linear": peak,
        "rms_linear": rms,
        "clipped_samples": result.clipped_samples,
        "dc_offset_percent": result.dc_offset_percent,
        "head_silence_seconds": result.head_silence_seconds,
        "crest_db": result.crest_db,
    }
    return result
