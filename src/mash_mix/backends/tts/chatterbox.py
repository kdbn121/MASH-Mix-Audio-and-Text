# SPDX-License-Identifier: MIT | Copyright (c) 2026 MASH (Khaled Bhuiyan)
"""Resemble AI Chatterbox text to speech backend.

Chatterbox is a zero shot cloning model. It takes a short reference sample
and continues in that voice, which makes it useful for the voice cloning
half of this package.

The multilingual release is a general purpose fallback for other languages.
It does not ship a Bangla voice, so bn, bn-BD and bn-IN are not offered here.
Use the mms-tts, indic-tts, parler-tts or xtts backend for Bangla.
"""

from __future__ import annotations

import os
import tempfile
import time
from typing import Any, Optional

import numpy as np

from ...errors import DependencyMissingError, UnsupportedLanguageError
from ...lang.normalize import normalize_for_tts
from ...types import AudioBuffer, SynthesisResult
from ..base import BackendInfo, TTSBackend
from ..registry import register_tts

try:  # pragma: no cover - the import guard keeps this module importable
    from chatterbox.tts import ChatterboxMultilingualTTS as _ChatterboxMultilingual

    _HAS_MULTILINGUAL = True
except ImportError:  # pragma: no cover
    _ChatterboxMultilingual = None  # type: ignore[assignment]
    _HAS_MULTILINGUAL = False

try:  # pragma: no cover
    from chatterbox import ChatterboxTTS as _ChatterboxEnglish

    _HAS_ENGLISH = True
except ImportError:  # pragma: no cover
    _ChatterboxEnglish = None  # type: ignore[assignment]
    _HAS_ENGLISH = False


_DEFAULT_MODEL = "chatterbox-multilingual"
_DEFAULT_SAMPLE_RATE = 24000

# The language ids of the Chatterbox multilingual release. Bangla is not in it.
_LANGUAGE_IDS = {
    "ar": "ar",
    "cs": "cs",
    "da": "da",
    "de": "de",
    "el": "el",
    "en": "en",
    "en-us": "en",
    "en-gb": "en",
    "es": "es",
    "fi": "fi",
    "fr": "fr",
    "he": "he",
    "hi": "hi",
    "hi-in": "hi",
    "hu": "hu",
    "id": "id",
    "it": "it",
    "ja": "ja",
    "ko": "ko",
    "ms": "ms",
    "nl": "nl",
    "no": "no",
    "pl": "pl",
    "pt": "pt",
    "ro": "ro",
    "ru": "ru",
    "sv": "sv",
    "th": "th",
    "tr": "tr",
    "uk": "uk",
    "vi": "vi",
    "zh": "zh",
    "zh-cn": "zh",
}

_BANGLA_PREFIX = ("bn", "bn-bd", "bn-in")


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


@register_tts("chatterbox")
class ChatterboxBackend(TTSBackend):
    """Resemble AI Chatterbox, with zero shot voice cloning."""

    info = BackendInfo(
        name="chatterbox",
        kind="tts",
        summary="Resemble AI Chatterbox, a zero shot cloning model for many languages.",
        package="chatterbox-tts",
        extra="tts",
        languages=sorted(_LANGUAGE_IDS),
        platforms=["windows", "linux", "macos"],
        notes=(
            "The current Chatterbox multilingual release has NO Bangla voice, so bn, bn-BD and "
            "bn-IN are not supported here and a Bangla request raises an error. Use mms-tts, "
            "indic-tts, parler-tts or xtts for Bangla. Zero shot cloning works through the "
            "reference_audio argument."
        ),
    )

    def __init__(self) -> None:
        super().__init__()
        self._model: Any = None
        self._multilingual = False
        self._sample_rate = _DEFAULT_SAMPLE_RATE

    # -- lifecycle ------------------------------------------------------
    def is_available(self) -> bool:
        return _HAS_MULTILINGUAL or _HAS_ENGLISH

    def load(self, model: Optional[str] = None) -> None:
        if self._loaded:
            return
        if not self.is_available():
            raise DependencyMissingError("chatterbox-tts", "tts", "Chatterbox text to speech")
        device = self._device()
        if _HAS_MULTILINGUAL:
            loaded = _ChatterboxMultilingual.from_pretrained(device=device)
            self._multilingual = True
        else:
            loaded = _ChatterboxEnglish.from_pretrained(device=device)
            self._multilingual = False
        self._model = loaded
        self._model_id = model or _DEFAULT_MODEL
        self._loaded = True

    def unload(self) -> None:
        self._model = None
        self._loaded = False

    @staticmethod
    def _device() -> str:
        try:
            import torch
        except ImportError:
            return "cpu"
        return "cuda" if torch.cuda.is_available() else "cpu"

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
        tag = (language or "").lower()
        if tag.startswith(_BANGLA_PREFIX):
            raise UnsupportedLanguageError(
                "Chatterbox has no Bangla voice in its current multilingual release. Use the "
                "mms-tts, indic-tts, parler-tts or xtts backend for bn-BD and bn-IN."
            )
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

        prompt_path = None
        temp_path = None
        if reference_audio is not None:
            temp_path, prompt_path = self._write_reference(reference_audio)
            meta["reference_audio"] = "cloned"

        try:
            call: dict[str, Any] = {}
            if prompt_path:
                call["audio_prompt_path"] = prompt_path
            if self._multilingual:
                language_id = _LANGUAGE_IDS.get(tag, tag[:2])
                call["language_id"] = language_id
                meta["language_id"] = language_id
            for name in ("exaggeration", "cfg_weight", "temperature"):
                if name in kwargs:
                    call[name] = kwargs[name]
            output = self._model.generate(prepared, **call)
        finally:
            if temp_path and os.path.exists(temp_path):
                os.remove(temp_path)

        audio = self._to_numpy(output)
        audio = _time_stretch_simple(audio, 1.0 / max(float(speed), 0.05))

        return SynthesisResult(
            audio=audio,
            sample_rate=self._sample_rate,
            backend=self.name,
            model=self._model_id,
            voice=voice,
            elapsed=time.perf_counter() - started,
            meta=meta,
        )

    # -- helpers --------------------------------------------------------
    @staticmethod
    def _to_numpy(output: Any) -> np.ndarray:
        if hasattr(output, "detach"):
            output = output.detach().to("cpu").numpy()
        return np.asarray(output, dtype=np.float32).reshape(-1)

    @staticmethod
    def _write_reference(reference_audio: AudioBuffer) -> tuple[str, str]:
        """Write the reference sample to a wav file Chatterbox can read."""
        import wave

        samples = reference_audio.mono()
        handle, path = tempfile.mkstemp(suffix=".wav", prefix="mashmix-chatterbox-")
        os.close(handle)
        peak = float(np.max(np.abs(samples))) if samples.size else 0.0
        scale = 32767.0 / peak if peak > 1.0 else 32767.0
        data = np.clip(samples * scale, -32768, 32767).astype(np.int16)
        with wave.open(path, "wb") as out:
            out.setnchannels(1)
            out.setsampwidth(2)
            out.setframerate(int(reference_audio.sample_rate))
            out.writeframes(data.tobytes())
        return path, path
