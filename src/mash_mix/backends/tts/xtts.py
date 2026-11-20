# SPDX-License-Identifier: MIT | Copyright (c) 2026 MASH (Khaled Bhuiyan)
"""Coqui XTTS v2 text to speech backend.

XTTS v2 is a multilingual model that speaks Bangla, and it clones a voice
from a short wav sample. The model name can be a hub model id or the path of
a speaker wav file.
"""

from __future__ import annotations

import os
import tempfile
import time
import wave
from typing import Any, Optional

import numpy as np

from ...errors import DependencyMissingError, MashMixError
from ...lang.normalize import normalize_for_tts
from ...types import AudioBuffer, SynthesisResult
from ..base import BackendInfo, TTSBackend
from ..registry import register_tts

try:  # pragma: no cover - the import guard keeps this module importable
    from TTS.api import TTS as _CoquiTTS

    _HAS_COQUI = True
except ImportError:  # pragma: no cover
    _CoquiTTS = None  # type: ignore[assignment]
    _HAS_COQUI = False
except Exception:  # pragma: no cover - a broken coqui install must not break the registry
    _CoquiTTS = None  # type: ignore[assignment]
    _HAS_COQUI = False


_DEFAULT_MODEL = "tts_models/multilingual/multi-dataset/xtts_v2"
_DEFAULT_SAMPLE_RATE = 24000

# XTTS language codes, which differ from the ISO tags for a few languages.
_LANGUAGE = {
    "bn": "bn",
    "bn-bd": "bn",
    "bn-in": "bn",
    "en": "en",
    "en-us": "en",
    "en-gb": "en",
    "es": "es",
    "fr": "fr",
    "de": "de",
    "it": "it",
    "pt": "pt",
    "pl": "pl",
    "tr": "tr",
    "ru": "ru",
    "nl": "nl",
    "cs": "cs",
    "ar": "ar",
    "zh-cn": "zh-cn",
    "ja": "ja",
    "hu": "hu",
    "ko": "ko",
    "hi": "hi",
}

_SPEAKERS = [
    "Claribel Dervla",
    "Daisy Studious",
    "Gracie Wise",
    "Tammie Ema",
    "Alison Dietlinde",
    "Ana Florence",
    "Annmarie Nele",
    "Asya Anara",
    "Brenda Stern",
    "Gitta Nikolina",
]


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


@register_tts("xtts")
class XTTSBackend(TTSBackend):
    """Coqui XTTS v2, multilingual with Bangla support and voice cloning."""

    info = BackendInfo(
        name="xtts",
        kind="tts",
        summary="Coqui XTTS v2, a multilingual model that speaks Bangla and clones voices.",
        package="TTS",
        extra="tts",
        languages=["bn", "bn-BD", "bn-IN", "en", "en-US", "es", "fr", "de", "it", "pt", "hi"],
        platforms=["windows", "linux", "macos"],
        notes=(
            "Bangla is supported, with bn-BD and bn-IN both spoken by the same model. The model "
            "argument accepts a model id, and a path to a wav file is treated as the speaker "
            "sample for cloning."
        ),
    )

    def __init__(self) -> None:
        super().__init__()
        self._tts: Any = None
        self._speaker_wav: Optional[str] = None
        self._sample_rate = _DEFAULT_SAMPLE_RATE

    # -- lifecycle ------------------------------------------------------
    def is_available(self) -> bool:
        return _HAS_COQUI

    def load(self, model: Optional[str] = None) -> None:
        if self._loaded:
            return
        if not _HAS_COQUI:
            raise DependencyMissingError("TTS", "tts", "XTTS text to speech")
        model_id = model or _DEFAULT_MODEL
        if str(model_id).lower().endswith((".wav", ".flac", ".mp3")):
            self._speaker_wav = os.path.abspath(os.path.expanduser(str(model_id)))
            model_id = _DEFAULT_MODEL
        self._tts = _CoquiTTS(model_id, progress_bar=False)
        self._model_id = model_id
        self._loaded = True

    def unload(self) -> None:
        self._tts = None
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

        language_code = _LANGUAGE.get((language or "").lower(), (language or "en").lower()[:2])
        temp_path = None
        speaker_wav = self._speaker_wav
        if reference_audio is not None:
            temp_path = self._write_reference(reference_audio)
            speaker_wav = temp_path
            meta["reference_audio"] = "cloned"
        elif speaker_wav:
            meta["reference_audio"] = "cloned"

        native = False
        try:
            audio = self._tts.tts(
                prepared,
                speaker=voice,
                language=language_code,
                speaker_wav=speaker_wav,
                speed=float(speed),
            )
            native = True
        except TypeError:
            try:
                audio = self._tts.tts(
                    prepared,
                    speaker=voice,
                    language=language_code,
                    speaker_wav=speaker_wav,
                )
            except Exception as exc:
                raise MashMixError(f"The XTTS model rejected the synthesis call: {exc}") from exc
            native = False
        except Exception as exc:
            raise MashMixError(f"The XTTS model rejected the synthesis call: {exc}") from exc
        finally:
            if temp_path and os.path.exists(temp_path):
                os.remove(temp_path)

        wave_data = np.asarray(audio, dtype=np.float32).reshape(-1)
        if not native:
            wave_data = _time_stretch_simple(wave_data, 1.0 / max(float(speed), 0.05))

        return SynthesisResult(
            audio=wave_data,
            sample_rate=self._sample_rate,
            backend=self.name,
            model=self._model_id,
            voice=voice,
            elapsed=time.perf_counter() - started,
            meta=meta,
        )

    # -- helpers --------------------------------------------------------
    @staticmethod
    def _write_reference(reference_audio: AudioBuffer) -> str:
        """Write the reference sample to a wav file XTTS can read."""
        handle, path = tempfile.mkstemp(suffix=".wav", prefix="mashmix-xtts-")
        os.close(handle)
        samples = reference_audio.mono()
        peak = float(np.max(np.abs(samples))) if samples.size else 0.0
        scale = 32767.0 / peak if peak > 1.0 else 32767.0
        data = np.clip(samples * scale, -32768, 32767).astype(np.int16)
        with wave.open(path, "wb") as out:
            out.setnchannels(1)
            out.setsampwidth(2)
            out.setframerate(int(reference_audio.sample_rate))
            out.writeframes(data.tobytes())
        return path

    def list_voices(self) -> list[str]:
        return list(_SPEAKERS)
