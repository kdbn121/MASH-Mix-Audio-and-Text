# SPDX-License-Identifier: MIT | Copyright (c) 2026 MASH (Khaled Bhuiyan)
"""Piper text to speech backend.

Piper is an ONNX runtime voice engine. It is the lightest option in this
package and the only one that runs comfortably on Android through Termux and
on iOS, because a voice is a single onnx file of a few tens of megabytes.

A voice lives on disk as ``voice.onnx`` with a matching ``voice.onnx.json``
configuration file, so the model argument of this backend is a path and not a
hub id.
"""

from __future__ import annotations

import os
import tempfile
import time
import wave
from typing import Any, Optional

import numpy as np

from ...errors import DependencyMissingError, MashMixError, ModelNotFoundError
from ...lang.normalize import normalize_for_tts
from ...types import AudioBuffer, SynthesisResult
from ..base import BackendInfo, TTSBackend
from ..registry import register_tts

try:  # pragma: no cover - the import guard keeps this module importable
    from piper import PiperVoice as _PiperVoice

    _HAS_PIPER = True
except ImportError:  # pragma: no cover
    try:
        from piper.voice import PiperVoice as _PiperVoice  # type: ignore[no-redef]

        _HAS_PIPER = True
    except ImportError:
        _PiperVoice = None  # type: ignore[assignment]
        _HAS_PIPER = False

try:  # pragma: no cover - newer piper releases expose a synthesis config
    from piper import SynthesisConfig as _SynthesisConfig

    _HAS_SYNTH_CONFIG = True
except ImportError:  # pragma: no cover
    _SynthesisConfig = None  # type: ignore[assignment]
    _HAS_SYNTH_CONFIG = False


_DEFAULT_MODEL = ""
_VOICE_SUFFIX = ".onnx"


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


def _read_wav(path: str) -> tuple[np.ndarray, int]:
    """Read a 16 bit PCM wav file into mono float32 samples."""
    with wave.open(path, "rb") as handle:
        channels = handle.getnchannels()
        sample_width = handle.getsampwidth()
        sample_rate = handle.getframerate()
        frames = handle.readframes(handle.getnframes())
    if sample_width != 2:
        raise MashMixError(
            f"Piper wrote {sample_width * 8} bit audio, which this reader does not handle."
        )
    samples = np.frombuffer(frames, dtype=np.int16).astype(np.float32) / 32768.0
    if channels > 1:
        samples = samples.reshape(-1, channels).mean(axis=1)
    return samples.astype(np.float32), int(sample_rate)


@register_tts("piper")
class PiperBackend(TTSBackend):
    """Piper ONNX voices, the lightest local text to speech."""

    info = BackendInfo(
        name="piper",
        kind="tts",
        summary="Piper ONNX voices, the lightest local TTS and the best fit for mobile.",
        package="piper-tts",
        extra="tts",
        languages=["bn", "bn-BD", "bn-IN", "en", "en-US", "en-GB", "de", "fr", "es", "hi"],
        platforms=["windows", "linux", "macos", "android-termux", "ios"],
        notes=(
            "Piper ships no official Bangla voice in the main voice list, so Bengali here means "
            "a community Bangla onnx voice loaded by path, for example a bn-BD or bn-IN voice. "
            "A voice is one onnx file plus its json configuration, which keeps the download small "
            "enough for a phone."
        ),
    )

    def __init__(self) -> None:
        super().__init__()
        self._voice: Any = None
        self._model_path = ""

    # -- lifecycle ------------------------------------------------------
    def is_available(self) -> bool:
        return _HAS_PIPER

    def load(self, model: Optional[str] = None) -> None:
        if self._loaded:
            return
        onnx_path, config_path = self._resolve_model(model)
        if not _HAS_PIPER:
            raise DependencyMissingError("piper-tts", "tts", "Piper text to speech")
        voice = _PiperVoice.load(onnx_path, config_path=config_path)
        self._voice = voice
        self._model_path = onnx_path
        self._model_id = onnx_path
        self._loaded = True

    def unload(self) -> None:
        self._voice = None
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
            meta["note"] = "Piper voices are fixed, so the reference audio was ignored."

        if not prepared.strip():
            return SynthesisResult(
                audio=np.zeros(0, dtype=np.float32),
                sample_rate=22050,
                backend=self.name,
                model=self._model_id,
                voice=voice,
                elapsed=time.perf_counter() - started,
                meta=meta,
            )

        handle, temp_path = tempfile.mkstemp(suffix=".wav", prefix="mashmix-piper-")
        os.close(handle)
        native = False
        try:
            if _HAS_SYNTH_CONFIG and abs(float(speed) - 1.0) > 1e-6:
                # Newer piper releases change the tempo natively, which keeps the pitch.
                config = _SynthesisConfig(length_scale=1.0 / float(speed))
                self._voice.synthesize(prepared, wav_file=temp_path, syn_config=config)
                native = True
            else:
                self._voice.synthesize(prepared, wav_file=temp_path)
            wave_data, sample_rate = _read_wav(temp_path)
        except TypeError:
            self._voice.synthesize(prepared, wav_file=temp_path)
            wave_data, sample_rate = _read_wav(temp_path)
            native = False
        finally:
            if os.path.exists(temp_path):
                os.remove(temp_path)

        if not native:
            wave_data = _time_stretch_simple(wave_data, 1.0 / max(float(speed), 0.05))

        return SynthesisResult(
            audio=wave_data,
            sample_rate=sample_rate,
            backend=self.name,
            model=self._model_id,
            voice=voice or os.path.basename(self._model_path),
            elapsed=time.perf_counter() - started,
            meta=meta,
        )

    # -- helpers --------------------------------------------------------
    def _resolve_model(self, model: Optional[str]) -> tuple[str, Optional[str]]:
        """Turn the model argument into an onnx path and its config path."""
        if not model:
            raise ModelNotFoundError(
                "Piper needs a path to a voice file. Pass the .onnx path, for example "
                "'voices/bn_BD.onnx', or a folder that holds one."
            )
        path = os.path.abspath(os.path.expanduser(str(model)))
        if os.path.isdir(path):
            found = sorted(
                os.path.join(path, name)
                for name in os.listdir(path)
                if name.endswith(_VOICE_SUFFIX)
            )
            if not found:
                raise ModelNotFoundError(
                    f"The Piper voice folder '{path}' holds no .onnx voice file."
                )
            path = found[0]
        if not os.path.isfile(path):
            raise ModelNotFoundError(
                f"The Piper voice file '{path}' does not exist. Download a voice and point the "
                "backend at the .onnx file."
            )
        config_path = path + ".json"
        if not os.path.isfile(config_path):
            alternative = os.path.join(os.path.dirname(path), "config.json")
            config_path = alternative if os.path.isfile(alternative) else None
        return path, config_path

    def list_voices(self) -> list[str]:
        """Piper voices are local files, so there is no fixed voice list."""
        return []
