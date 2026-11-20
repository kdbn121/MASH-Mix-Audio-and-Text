# SPDX-License-Identifier: MIT | Copyright (c) 2026 MASH (Khaled Bhuiyan)
"""AI4Bharat IndicTTS backend.

IndicTTS, and the IndicF5 style models that follow it, cover 22 Indic
languages including Bangla, and they read Bangla well in both the
Bangladeshi and the Indian register once the text has been normalised.

The model id is configurable. The default is the AI4Bharat hub id, and the
local checkpoint name of the IndicTTS package is used as a fallback when the
hub id cannot be resolved.
"""

from __future__ import annotations

import time
from typing import Any, Optional

import numpy as np

from ...errors import DependencyMissingError, MashMixError
from ...lang.normalize import normalize_for_tts
from ...types import AudioBuffer, SynthesisResult
from ..base import BackendInfo, TTSBackend
from ..registry import register_tts

try:  # pragma: no cover - the import guard keeps this module importable
    from indic_tts import TTS as _IndicTTS

    _HAS_INDIC_TTS = True
except ImportError:  # pragma: no cover
    _IndicTTS = None  # type: ignore[assignment]
    _HAS_INDIC_TTS = False

try:  # pragma: no cover
    from transformers import AutoModel as _AutoModel

    _HAS_TRANSFORMERS = True
except ImportError:  # pragma: no cover
    _AutoModel = None  # type: ignore[assignment]
    _HAS_TRANSFORMERS = False


_DEFAULT_MODEL = "ai4bharat/IndicTTS"
_FALLBACK_CHECKPOINT = "IndicTTS/IndicTTS.m1"
_DEFAULT_SAMPLE_RATE = 22050

# The 22 Indic languages of the IndicTTS family, plus the two Bangla tags.
_INDIC_LANGUAGES = [
    "bn",
    "bn-BD",
    "bn-IN",
    "as",
    "bho",
    "gu",
    "hi",
    "hi-IN",
    "kn",
    "kok",
    "mai",
    "ml",
    "mni",
    "mr",
    "ne",
    "or",
    "pa",
    "sa",
    "sat",
    "sd",
    "si",
    "ta",
    "te",
    "ur",
    "fy",
    "ks",
]

# IndicTTS uses short language codes.
_LANG_CODE = {
    "bn": "bn",
    "bn-bd": "bn",
    "bn-in": "bn",
    "as": "as",
    "bho": "bho",
    "gu": "gu",
    "hi": "hi",
    "hi-in": "hi",
    "kn": "kn",
    "kok": "kok",
    "mai": "mai",
    "ml": "ml",
    "mni": "mni",
    "mr": "mr",
    "ne": "ne",
    "or": "or",
    "pa": "pa",
    "sa": "sa",
    "sat": "sat",
    "sd": "sd",
    "si": "si",
    "ta": "ta",
    "te": "te",
    "ur": "ur",
    "fy": "fy",
    "ks": "ks",
    "en": "en",
}


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


@register_tts("indic-tts")
class IndicTTSBackend(TTSBackend):
    """AI4Bharat IndicTTS, covering Bangla and 21 more Indic languages."""

    info = BackendInfo(
        name="indic-tts",
        kind="tts",
        summary="AI4Bharat IndicTTS models for 22 Indic languages, Bangla included.",
        package="indic-tts",
        extra="tts",
        languages=list(_INDIC_LANGUAGES),
        platforms=["windows", "linux", "macos", "android-termux", "ios"],
        notes=(
            "Bangla is a first class language here, with bn-BD and bn-IN both supported from the "
            "same model. The default model id is ai4bharat/IndicTTS and the IndicTTS package "
            "falls back to its own IndicTTS.m1 checkpoint. IndicTTS has no cloning speaker."
        ),
    )

    def __init__(self) -> None:
        super().__init__()
        self._impl: Any = None
        self._sample_rate = _DEFAULT_SAMPLE_RATE

    # -- lifecycle ------------------------------------------------------
    def is_available(self) -> bool:
        return _HAS_INDIC_TTS or _HAS_TRANSFORMERS

    def load(self, model: Optional[str] = None) -> None:
        if self._loaded:
            return
        if not (_HAS_INDIC_TTS or _HAS_TRANSFORMERS):
            raise DependencyMissingError("indic-tts", "tts", "IndicTTS text to speech")
        model_id = model or _DEFAULT_MODEL
        if _HAS_INDIC_TTS:
            self._impl = self._load_package(model_id)
        else:
            self._impl = _AutoModel.from_pretrained(model_id, trust_remote_code=True)
        self._model_id = model_id
        self._loaded = True

    def unload(self) -> None:
        self._impl = None
        self._loaded = False

    @staticmethod
    def _load_package(model_id: str) -> Any:
        """Load the IndicTTS package, with its own checkpoint as a fallback."""
        try:
            engine = _IndicTTS()
            engine.load(model_id)
            return engine
        except Exception:
            engine = _IndicTTS()
            engine.load(_FALLBACK_CHECKPOINT)
            return engine

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
            meta["note"] = "IndicTTS has no cloning speaker, so the reference audio was ignored."

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

        lang_code = _LANG_CODE.get((language or "").lower(), (language or "bn").lower()[:2])
        audio, sample_rate = self._run(prepared, lang_code, voice)
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
    def _run(self, text: str, lang_code: str, voice: Optional[str]) -> tuple[np.ndarray, int]:
        impl = self._impl
        if _HAS_INDIC_TTS and isinstance(impl, _IndicTTS):
            out = self._call_package(impl, text, lang_code, voice)
            return self._interpret(out)
        for name in ("generate", "infer", "synthesize"):
            method = getattr(impl, name, None)
            if callable(method):
                return self._interpret(method(text, lang=lang_code))
        callable_impl = impl if callable(impl) else None
        if callable_impl:
            return self._interpret(callable_impl(text=text, lang=lang_code))
        raise MashMixError(
            "The loaded IndicTTS model exposes no text to speech entry point, so nothing can be "
            "synthesised."
        )

    @staticmethod
    def _call_package(impl: Any, text: str, lang_code: str, voice: Optional[str]) -> Any:
        """Call the IndicTTS package across its known signatures."""
        attempts = []
        if voice:
            attempts.append(lambda: impl.text_to_speech(text, lang=lang_code, speaker=voice))
        attempts.append(lambda: impl.text_to_speech(text, lang=lang_code))
        attempts.append(lambda: impl.text_to_speech(text, voice, lang_code))
        attempts.append(lambda: impl.text_to_speech(text, lang_code))
        attempts.append(lambda: impl.text_to_speech(text))
        last_error: Optional[Exception] = None
        for attempt in attempts:
            try:
                return attempt()
            except TypeError as exc:
                last_error = exc
                continue
        raise MashMixError(f"The IndicTTS model rejected the text to speech call: {last_error}")

    @staticmethod
    def _interpret(out: Any) -> tuple[np.ndarray, int]:
        if isinstance(out, (tuple, list)) and len(out) == 2:
            audio, rate = out
            return np.asarray(audio, dtype=np.float32).reshape(-1), int(rate)
        if hasattr(out, "detach"):
            out = out.detach().to("cpu").numpy()
        return np.asarray(out, dtype=np.float32).reshape(-1), _DEFAULT_SAMPLE_RATE

    def list_voices(self) -> list[str]:
        """IndicTTS speakers are named by the model, not by a fixed list."""
        return ["male", "female"]
