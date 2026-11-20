# SPDX-License-Identifier: MIT | Copyright (c) 2026 MASH (Khaled Bhuiyan)
"""Local VITS text to speech backend.

A VITS checkpoint is a folder that holds a ``config.json`` and one or more
``.pth`` weight files. That is the route for the Bangladeshi Bangla VITS fine
tunes and for the AI4Bharat Bangla VITS release, so this backend takes a
folder path instead of a hub id.

The coqui ``TTS`` package is used when it is installed. When it is not, the
backend falls back to a direct torch run of a checkpoint that was saved as a
whole model, and says clearly what it cannot do.
"""

from __future__ import annotations

import json
import os
import time
from typing import Any, Optional

import numpy as np

from ...errors import DependencyMissingError, MashMixError, ModelNotFoundError
from ...lang.normalize import normalize_for_tts
from ...types import AudioBuffer, SynthesisResult
from ..base import BackendInfo, TTSBackend
from ..registry import register_tts

try:  # pragma: no cover - the import guard keeps this module importable
    from TTS.api import TTS as _CoquiApiTTS

    _HAS_COQUI_API = True
except ImportError:  # pragma: no cover
    _CoquiApiTTS = None  # type: ignore[assignment]
    _HAS_COQUI_API = False
except Exception:  # pragma: no cover - a broken coqui install must not break the registry
    _CoquiApiTTS = None  # type: ignore[assignment]
    _HAS_COQUI_API = False

try:  # pragma: no cover
    from TTS.utils.synthesizer import Synthesizer as _CoquiSynthesizer

    _HAS_COQUI_SYNTHESIZER = True
except ImportError:  # pragma: no cover
    _CoquiSynthesizer = None  # type: ignore[assignment]
    _HAS_COQUI_SYNTHESIZER = False
except Exception:  # pragma: no cover
    _CoquiSynthesizer = None  # type: ignore[assignment]
    _HAS_COQUI_SYNTHESIZER = False


_DEFAULT_MODEL = ""
_DEFAULT_SAMPLE_RATE = 22050


def _time_stretch_simple(wave: np.ndarray, factor: float) -> np.ndarray:
    """Change the duration of ``wave`` by ``factor`` and keep the pitch.

    A factor below 1 makes the speech shorter, so it plays faster, and the
    pitch stays where it was. The work is an overlap add pass: frames are cut
    from the input at a steady spacing, each frame is crossfaded with linear
    interpolation ramps, and the frames are laid down at the new spacing. The
    samples inside a frame are untouched, which is what holds the pitch in
    place, and a short search aligns the frames so the overlaps do not click.
    """
    if wave is None or wave.size == 0:
        return np.asarray(wave, dtype=np.float32)
    factor = float(factor)
    if factor <= 0:
        raise ValueError("time stretch factor must be positive")
    source = wave.astype(np.float64, copy=False)
    if abs(factor - 1.0) < 1e-6:
        return source.astype(np.float32)
    n_in = source.size
    n_out = max(1, int(round(n_in * factor)))
    frame = 1024
    hop = frame // 2
    if n_in < 2 * frame:
        # Too short for an overlap add pass, so fall back to a plain resample.
        old_x = np.arange(n_in, dtype=np.float64)
        new_x = np.linspace(0.0, n_in - 1, n_out, dtype=np.float64)
        return np.interp(new_x, old_x, source).astype(np.float32)

    out_hop = max(1, int(round(hop * factor)))
    search = max(1, hop // 2)
    overlap = max(1, frame - out_hop)
    ramp = np.linspace(0.0, 1.0, hop, endpoint=False)
    window = np.concatenate([ramp, ramp[::-1]])
    out = np.zeros(n_out + frame, dtype=np.float64)
    weight = np.zeros_like(out)

    previous = None
    template = None
    out_pos = 0
    while out_pos < n_out:
        if previous is None:
            start = 0
        else:
            ideal = previous + hop
            low = max(0, ideal - search)
            high = min(n_in - frame, ideal + search)
            start = max(0, min(ideal, n_in - frame))
            if high > low and template is not None and template.size == overlap:
                region = source[low : high + overlap]
                if region.size >= overlap:
                    corr = np.correlate(region, template, mode="valid")
                    start = low + int(np.argmax(corr))
        segment = source[start : start + frame]
        if segment.size < frame:
            break
        out[out_pos : out_pos + frame] += segment * window
        weight[out_pos : out_pos + frame] += window
        template = source[start + out_hop : start + frame]
        previous = start
        out_pos += out_hop

    result = out[:n_out]
    norm = np.where(weight[:n_out] > 1e-6, weight[:n_out], 1.0)
    return (result / norm).astype(np.float32)


class _TorchVitsRunner:
    """Direct torch runner for a checkpoint saved as a whole model."""

    def __init__(self, checkpoint: str, config: dict[str, Any]) -> None:
        import torch  # imported late so the module stays light

        self._torch = torch
        self._config = config
        try:
            self._model = torch.load(checkpoint, map_location="cpu", weights_only=False)
        except TypeError:
            self._model = torch.load(checkpoint, map_location="cpu")

    def synth(self, text: str) -> tuple[np.ndarray, int]:
        model = self._model
        for name in ("inference", "synthesize", "generate"):
            method = getattr(model, name, None)
            if callable(method):
                out = method(text)
                break
        else:
            raise DependencyMissingError(
                "TTS",
                "tts",
                "local VITS inference from a state dict checkpoint",
            )
        return _as_audio(out, int(self._config.get("output_sample_rate", _DEFAULT_SAMPLE_RATE)))


def _as_audio(value: Any, sample_rate: int) -> tuple[np.ndarray, int]:
    """Pull a float32 waveform and a sample rate out of a model output."""
    if isinstance(value, (tuple, list)) and len(value) == 2:
        audio, rate = value
        return np.asarray(audio, dtype=np.float32).reshape(-1), int(rate)
    if hasattr(value, "detach"):
        value = value.detach().to("cpu").numpy()
    return np.asarray(value, dtype=np.float32).reshape(-1), sample_rate


@register_tts("vits")
class VITSLocalBackend(TTSBackend):
    """Local VITS checkpoint folders, the route for Bangla fine tunes."""

    info = BackendInfo(
        name="vits",
        kind="tts",
        summary="Local VITS checkpoint folders, including Bangladeshi and AI4Bharat Bangla VITS.",
        package="TTS",
        extra="tts",
        languages=["bn", "bn-BD", "bn-IN", "hi", "hi-IN", "en", "en-US", "ja", "ko", "zh"],
        platforms=["windows", "linux", "macos", "android-termux", "ios"],
        notes=(
            "Point the backend at the folder that holds config.json and the .pth file. This is the "
            "path for the Bangladeshi Bangla VITS fine tune and for the AI4Bharat Bangla VITS "
            "release, so both bn-BD and bn-IN are available from the same checkpoint. The coqui "
            "TTS package gives the best result."
        ),
    )

    def __init__(self) -> None:
        super().__init__()
        self._runner: Any = None
        self._checkpoint = ""
        self._sample_rate = _DEFAULT_SAMPLE_RATE
        self._speakers: list[str] = []

    # -- lifecycle ------------------------------------------------------
    def is_available(self) -> bool:
        """True when a VITS stack that can run the checkpoint is importable."""
        return _HAS_COQUI_API or _HAS_COQUI_SYNTHESIZER or self._torch_available()

    @staticmethod
    def _torch_available() -> bool:
        try:
            import torch  # noqa: F401
        except ImportError:
            return False
        return True

    def load(self, model: Optional[str] = None) -> None:
        if self._loaded:
            return
        checkpoint, config_path = self._resolve_model(model)
        config = self._read_config(config_path)
        self._sample_rate = int(config.get("output_sample_rate", _DEFAULT_SAMPLE_RATE))

        if not (_HAS_COQUI_API or _HAS_COQUI_SYNTHESIZER or self._torch_available()):
            raise DependencyMissingError("TTS", "tts", "local VITS inference")

        if _HAS_COQUI_API:
            self._runner = _CoquiApiTTS(
                model_path=checkpoint,
                config_path=config_path,
                progress_bar=False,
            )
        elif _HAS_COQUI_SYNTHESIZER:
            self._runner = _CoquiSynthesizer(model_path=checkpoint, config_path=config_path)
        else:
            self._runner = _TorchVitsRunner(checkpoint, config)

        speakers = config.get("speakers") or {}
        if isinstance(speakers, dict):
            self._speakers = [str(key) for key in speakers]
        self._checkpoint = checkpoint
        self._model_id = os.path.dirname(checkpoint) or checkpoint
        self._loaded = True

    def unload(self) -> None:
        self._runner = None
        self._loaded = False

    # -- inference ------------------------------------------------------
    def synthesize(
        self,
        text: str,
        *,
        language: str = "bn-BD",
        voice: Optional[str] = None,
        speed: float = 1.0,
        reference_audio: Optional[AudioBuffer] = None,
        **kwargs: Any,
    ) -> SynthesisResult:
        started = time.perf_counter()
        if not self._loaded:
            self.load(kwargs.get("model"))
        prepared = normalize_for_tts(text, language=language, expand_numbers=False)
        meta: dict[str, Any] = {"normalised": prepared != text}
        if reference_audio is not None:
            meta["reference_audio"] = "ignored"
            meta["note"] = "This VITS checkpoint has no cloning speaker, so the reference was ignored."

        if not prepared.strip():
            return SynthesisResult(
                audio=np.zeros(0, dtype=np.float32),
                sample_rate=self._sample_rate,
                backend=self.name,
                model=self._model_id,
                voice=voice,
                elapsed=time.perf_counter() - started,
                meta=meta,
            )

        audio, sample_rate = self._run(prepared, voice)
        audio = _time_stretch_simple(audio, 1.0 / max(float(speed), 0.05))

        return SynthesisResult(
            audio=audio,
            sample_rate=sample_rate,
            backend=self.name,
            model=self._model_id,
            voice=voice,
            elapsed=time.perf_counter() - started,
            meta=meta,
        )

    # -- helpers --------------------------------------------------------
    def _run(self, text: str, voice: Optional[str]) -> tuple[np.ndarray, int]:
        runner = self._runner
        if _HAS_COQUI_API and isinstance(runner, _CoquiApiTTS):
            out = runner.tts(text, speaker=voice) if voice else runner.tts(text)
            return _as_audio(out, self._sample_rate)
        if _HAS_COQUI_SYNTHESIZER and isinstance(runner, _CoquiSynthesizer):
            out = runner.tts(text)
            rate = getattr(getattr(runner, "ap", None), "output_sample_rate", self._sample_rate)
            return _as_audio(out, int(rate or self._sample_rate))
        return runner.synth(text)

    def _resolve_model(self, model: Optional[str]) -> tuple[str, str]:
        """Find the checkpoint file and its config inside the model folder."""
        if not model:
            raise ModelNotFoundError(
                "Local VITS needs a folder path. Pass the folder that holds config.json and the "
                ".pth file, for example 'models/bangla-vits'."
            )
        folder = os.path.abspath(os.path.expanduser(str(model)))
        if not os.path.isdir(folder):
            raise ModelNotFoundError(
                f"The VITS folder '{folder}' does not exist. Download the checkpoint and point the "
                "backend at its folder."
            )
        checkpoints = sorted(
            os.path.join(folder, name)
            for name in os.listdir(folder)
            if name.endswith((".pth", ".pt"))
        )
        if not checkpoints:
            raise ModelNotFoundError(
                f"The VITS folder '{folder}' holds no .pth checkpoint file."
            )
        config_path = os.path.join(folder, "config.json")
        if not os.path.isfile(config_path):
            raise ModelNotFoundError(
                f"The VITS folder '{folder}' holds no config.json, so the model cannot be built."
            )
        return checkpoints[0], config_path

    @staticmethod
    def _read_config(config_path: str) -> dict[str, Any]:
        try:
            with open(config_path, encoding="utf-8") as handle:
                data = json.load(handle)
        except (OSError, ValueError) as exc:
            raise MashMixError(f"The VITS config '{config_path}' could not be read: {exc}") from exc
        return data if isinstance(data, dict) else {}

    def list_voices(self) -> list[str]:
        """Speaker ids declared by the loaded checkpoint."""
        return list(self._speakers)
