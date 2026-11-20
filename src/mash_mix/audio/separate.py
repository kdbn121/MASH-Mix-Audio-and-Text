# SPDX-License-Identifier: MIT | Copyright (c) 2026 MASH (Khaled Bhuiyan)
"""Source separation.

Three routes are supported, tried in this order:

1. ``demucs`` when the package is installed. The ``htdemucs`` model gives the
   usual vocals, drums, bass and other stems.
2. ``audio-separator`` when that package is installed instead.
3. The built in ``mid_side`` model. It needs nothing but numpy and performs a
   centre channel extraction plus a mid side split. This gives a usable
   karaoke style vocal removal on any machine, which is why it is exposed as a
   selectable model rather than only as a fallback.

Any other model name raises :class:`DependencyMissingError` with the install
hint for the ``analysis`` extra.
"""

from __future__ import annotations

import tempfile
import time
from collections.abc import Sequence
from pathlib import Path
from typing import Any

import numpy as np

from ..errors import DependencyMissingError
from ..types import SeparationResult

__all__ = ["separate", "mid_side_separation", "HAVE_DEMUCS", "HAVE_AUDIO_SEPARATOR"]

try:  # pragma: no cover - depends on the host environment
    import demucs.api as _demucs_api

    HAVE_DEMUCS = True
except Exception:  # pragma: no cover - optional dependency
    _demucs_api = None  # type: ignore[assignment]
    HAVE_DEMUCS = False

try:  # pragma: no cover - depends on the host environment
    import audio_separator  # noqa: F401

    HAVE_AUDIO_SEPARATOR = True
except Exception:  # pragma: no cover - optional dependency
    HAVE_AUDIO_SEPARATOR = False

#: The sample rate every demucs model works at.
_MODEL_RATE = 44100

_MID_SIDE_ALIASES = {
    "vocals": "vocals",
    "vocal": "vocals",
    "center": "vocals",
    "centre": "vocals",
    "no_vocals": "no_vocals",
    "instrumental": "no_vocals",
    "side": "no_vocals",
    "backing": "no_vocals",
}


def _as_stereo(audio: np.ndarray) -> tuple[np.ndarray, int, bool]:
    """Return a ``(2, frames)`` array, the original channel count and a mono flag."""
    data = np.asarray(audio, dtype=np.float32)
    if data.ndim == 1:
        return np.vstack([data, data]), 1, True
    if data.ndim == 2:
        channels = int(data.shape[0])
        if channels >= 2:
            return data[:2].astype(np.float32), channels, False
        if channels == 1:
            return np.vstack([data[0], data[0]]), 1, True
        return np.zeros((2, data.shape[1]), dtype=np.float32), channels, False
    flat = data.reshape(-1)
    return np.vstack([flat, flat]), 1, True


def mid_side_separation(audio: np.ndarray) -> dict[str, np.ndarray]:
    """Split a signal with a centre channel extraction and a mid side split.

    Parameters
    ----------
    audio:
        Mono or stereo samples.

    Returns
    -------
    dict
        ``"vocals"`` holds the centre channel and ``"no_vocals"`` holds the
        side signal, which is what is left once the centre is removed. A mono
        input carries no spatial information, so the centre is the full signal
        and the side signal is silent.
    """
    stereo, _channels, _is_mono = _as_stereo(audio)
    left = stereo[0].astype(np.float64)
    right = stereo[1].astype(np.float64)
    centre = (left + right) / 2.0
    side = (left - right) / 2.0
    return {
        "vocals": centre.astype(np.float32),
        "no_vocals": side.astype(np.float32),
    }


def _filter_stems(
    stems: dict[str, np.ndarray], wanted: Sequence[str]
) -> dict[str, np.ndarray]:
    """Keep only the requested stems, filling the usual aliases."""
    selected: dict[str, np.ndarray] = {}
    for name in wanted or ():
        key = str(name).strip().lower()
        if key in stems:
            selected[str(name)] = np.asarray(stems[key], dtype=np.float32)
            continue
        alias = _MID_SIDE_ALIASES.get(key)
        if alias and alias in stems:
            selected[str(name)] = np.asarray(stems[alias], dtype=np.float32)
    return selected


def _release_memory() -> None:
    """Ask the runtime and the accelerator to give back the memory they hold."""
    import gc

    gc.collect()
    try:
        import torch

        if torch.cuda.is_available():
            torch.cuda.empty_cache()
    except Exception:  # pragma: no cover - torch is optional here
        pass


def _demucs_stems(separator: Any, tensor: Any) -> dict[str, np.ndarray] | None:
    """Call demucs and return its stems for either version of the API.

    demucs 4.x takes a ``(channels, samples)`` tensor and returns a tuple of
    the original wave and a dictionary of stems. demucs 3.x took a batched
    tensor and returned the dictionary directly. Both are supported here
    because the call signature changed between the two releases.
    """
    try:
        result = separator.separate_tensor(tensor)
    except TypeError:
        # demucs 3.x expects a batched tensor and has no sample rate argument.
        result = separator.separate_tensor(tensor.unsqueeze(0))
    if isinstance(result, tuple) and len(result) == 2:
        produced = result[1]
    else:
        produced = result
    if not isinstance(produced, dict) or not produced:
        return None
    stems: dict[str, np.ndarray] = {}
    for name, value in produced.items():
        array = value
        if hasattr(array, "detach"):
            array = array.detach().cpu().numpy()
        array = np.asarray(array, dtype=np.float32)
        while array.ndim > 2:
            array = array[0]
        stems[str(name).lower()] = array
    return stems or None


def _demucs_separate(
    audio: np.ndarray,
    sr: int,
    model: str,
    stems: Sequence[str],
    device: str | None,
) -> dict[str, np.ndarray] | None:
    """Run demucs and return its stems, or ``None`` when demucs is unusable."""
    if not HAVE_DEMUCS:
        return None
    try:
        import torch

        from ..audio import resample as _resample

        stereo, _channels, _is_mono = _as_stereo(audio)
        work = stereo
        work_rate = int(sr)
        if work_rate != _MODEL_RATE:
            work = _resample.resample(stereo, work_rate, _MODEL_RATE)
            work_rate = _MODEL_RATE
        tensor = torch.from_numpy(np.ascontiguousarray(work, dtype=np.float32))
        separator = _demucs_api.Separator(
            model=model, device=device if device else "cpu"
        )
        try:
            produced = _demucs_stems(separator, tensor)
        finally:
            # The separator holds a whole neural network. It is released before
            # the caller loads a speech model, so a song with separation does
            # not need both networks in memory at the same time.
            del separator
            del tensor
            _release_memory()
        if produced is None:
            return None
        if work_rate != int(sr):
            from ..audio import resample as _resample

            for name, array in list(produced.items()):
                produced[name] = _resample.resample(array, work_rate, int(sr))
        return produced
    except Exception:  # noqa: BLE001 - a broken model must not crash the call
        return None


def _audio_separator_separate(
    audio: np.ndarray,
    sr: int,
    model: str,
    device: str | None,
) -> dict[str, np.ndarray] | None:
    """Run audio-separator and return its stems, or ``None`` on any failure."""
    if not HAVE_AUDIO_SEPARATOR:
        return None
    try:
        from audio_separator.separator import Separator

        from . import io as _io

        stereo, _channels, _is_mono = _as_stereo(audio)
        with tempfile.TemporaryDirectory() as folder:
            source = Path(folder) / "input.wav"
            _io.save(str(source), stereo, int(sr))
            separator = Separator()
            separator.load_model(model)
            outputs = separator.separate(str(source), output_dir=folder)
            produced: dict[str, np.ndarray] = {}
            for item in outputs or []:
                path = Path(item)
                if not path.is_file():
                    continue
                name = path.stem.lower()
                for prefix in ("modelname_", "vocals_", "other_", "instrumental_"):
                    if name.startswith(prefix):
                        name = name[len(prefix) :]
                        break
                buffer = _io.load(str(path))
                produced[name] = buffer.mono()
            if not produced:
                return None
            return produced
    except Exception:  # noqa: BLE001 - a broken model must not crash the call
        return None


def _build_result(
    produced: dict[str, np.ndarray],
    wanted: Sequence[str],
    sr: int,
    backend: str,
    model: str,
    elapsed: float,
    extra_meta: dict[str, Any] | None = None,
) -> SeparationResult:
    """Turn raw model output into a :class:`SeparationResult`."""
    stems: dict[str, np.ndarray] = {}
    names = [str(name).strip().lower() for name in (wanted or ())]
    for name, array in produced.items():
        key = str(name).strip().lower()
        if key in names:
            stems[key] = np.asarray(array, dtype=np.float32)
    if "no_vocals" in names and "no_vocals" not in stems:
        leftovers = [
            array
            for key, array in produced.items()
            if key not in ("vocals", "vocal") and key not in names
        ]
        if leftovers:
            total = np.zeros(leftovers[0].shape, dtype=np.float32)
            for array in leftovers:
                total = total + np.asarray(array, dtype=np.float32)
            stems["no_vocals"] = total.astype(np.float32)
    if "vocals" in names and "vocals" not in stems:
        for alias in ("vocals", "vocal"):
            if alias in produced:
                stems["vocals"] = np.asarray(produced[alias], dtype=np.float32)
                break
    meta: dict[str, Any] = {"backend": backend, "available": sorted(produced.keys())}
    if extra_meta:
        meta.update(extra_meta)
    return SeparationResult(
        stems=stems,
        sample_rate=int(sr),
        backend=backend,
        model=model,
        elapsed=float(elapsed),
        meta=meta,
    )


def separate(
    audio: np.ndarray,
    sr: int,
    *,
    model: str = "htdemucs",
    stems: Sequence[str] = ("vocals", "no_vocals"),
    device: str | None = None,
) -> SeparationResult:
    """Split an audio signal into stems.

    Parameters
    ----------
    audio:
        Mono or stereo samples.
    sr:
        Sample rate of the samples.
    model:
        ``"mid_side"`` selects the built in numpy only splitter. Any other name
        is looked up in demucs and then in audio-separator.
    stems:
        The stem names to return. ``"vocals"`` and ``"no_vocals"`` are always
        understood, model specific names such as ``"drums"`` are passed
        through when the model provides them.
    device:
        Torch device for demucs, for example ``"cpu"`` or ``"cuda"``.

    Raises
    ------
    DependencyMissingError
        When the requested model needs a package that is not installed.
    """
    started = time.time()
    wanted = tuple(stems or ())
    name = str(model or "htdemucs").strip().lower()
    if name in ("mid_side", "midside", "mid-side"):
        produced = mid_side_separation(audio)
        selected = _filter_stems(produced, wanted)
        result = SeparationResult(
            stems=selected,
            sample_rate=int(sr),
            backend="mid_side",
            model="mid_side",
            elapsed=float(time.time() - started),
            meta={
                "backend": "mid_side",
                "available": sorted(produced.keys()),
                "method": "centre channel extraction and mid side split",
            },
        )
        return result
    produced = _demucs_separate(audio, sr, name, wanted, device)
    if produced is not None:
        return _build_result(
            produced, wanted, sr, "demucs", name, time.time() - started
        )
    produced = _audio_separator_separate(audio, sr, name, device)
    if produced is not None:
        return _build_result(
            produced, wanted, sr, "audio-separator", name, time.time() - started
        )
    if name in ("mid_side",):
        raise DependencyMissingError("numpy", "analysis", "mid side separation")
    raise DependencyMissingError(
        "demucs", "analysis", f"stem separation with model {name!r}"
    )
