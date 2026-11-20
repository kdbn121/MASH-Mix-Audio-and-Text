# SPDX-License-Identifier: MIT | Copyright (c) 2026 MASH (Khaled Bhuiyan)
"""AI4Bharat Indic Parler TTS backend.

Indic Parler TTS is a large multilingual text to speech model for 23 Indic
languages, Bangla among them. It is driven by a natural language description
of the voice rather than by a speaker id, so the ``voice`` argument of this
backend is a sentence such as
``A warm female Bangladeshi voice speaking slowly.``
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
    from transformers import AutoModelForCausalLM, AutoTokenizer

    _HAS_TRANSFORMERS = True
except ImportError:  # pragma: no cover
    AutoModelForCausalLM = None  # type: ignore[assignment]
    AutoTokenizer = None  # type: ignore[assignment]
    _HAS_TRANSFORMERS = False


_DEFAULT_MODEL = "ai4bharat/indic-parler-tts"
_DEFAULT_DESCRIPTION = "A clear, warm Bangla voice reading at a natural pace."
_DEFAULT_SAMPLE_RATE = 24000
_MAX_NEW_TOKENS = 1200

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

_LANG_CODE = {
    "bn": "bn",
    "bn-bd": "bn",
    "bn-in": "bn",
    "hi": "hi",
    "en": "en",
}

_SAMPLE_DESCRIPTIONS = [
    "A warm female Bangladeshi voice speaking slowly.",
    "A clear male Indian Bengali voice with a calm news reader tone.",
    "A young Bangla voice speaking with bright energy.",
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


@register_tts("parler-tts")
class ParlerTTSBackend(TTSBackend):
    """AI4Bharat Indic Parler TTS, driven by a voice description."""

    info = BackendInfo(
        name="parler-tts",
        kind="tts",
        summary="AI4Bharat Indic Parler TTS, a large multilingual model driven by a voice prompt.",
        package="transformers",
        extra="tts",
        languages=list(_INDIC_LANGUAGES),
        platforms=["windows", "linux", "macos"],
        notes=(
            "Bangla is covered, so bn-BD and bn-IN are both available. The voice argument is a "
            "natural language description of the voice, for example 'A warm female Bangladeshi "
            "voice speaking slowly.' The model is large, so a GPU is strongly recommended and "
            "phones are not a target."
        ),
    )

    def __init__(self) -> None:
        super().__init__()
        self._model: Any = None
        self._tokenizer: Any = None
        self._sample_rate = _DEFAULT_SAMPLE_RATE

    # -- lifecycle ------------------------------------------------------
    def is_available(self) -> bool:
        return _HAS_TRANSFORMERS

    def load(self, model: Optional[str] = None) -> None:
        if self._loaded:
            return
        if not _HAS_TRANSFORMERS:
            raise DependencyMissingError("transformers", "tts", "Indic Parler TTS")
        model_id = model or _DEFAULT_MODEL
        tokenizer = AutoTokenizer.from_pretrained(model_id, trust_remote_code=True)
        loaded = AutoModelForCausalLM.from_pretrained(model_id, trust_remote_code=True)
        loaded.eval()
        self._tokenizer = tokenizer
        self._model = loaded
        self._model_id = model_id
        self._loaded = True

    def unload(self) -> None:
        self._model = None
        self._tokenizer = None
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
        description = voice or kwargs.get("description") or _DEFAULT_DESCRIPTION
        meta: dict[str, Any] = {"normalised": prepared != text, "description": description}
        if reference_audio is not None:
            meta["reference_audio"] = "ignored"
            meta["note"] = "Parler TTS styles the voice from the description, not from a sample."

        if not prepared.strip():
            return SynthesisResult(
                audio=np.zeros(0, dtype=np.float32),
                sample_rate=self._sample_rate,
                backend=self.name,
                model=self._model_id,
                voice=description,
                elapsed=time.perf_counter() - started,
                meta=meta,
            )

        lang_code = _LANG_CODE.get((language or "").lower(), (language or "bn").lower()[:2])
        audio, sample_rate = self._run(prepared, description, lang_code)
        audio = _time_stretch_simple(audio, 1.0 / max(float(speed), 0.05))

        return SynthesisResult(
            audio=audio,
            sample_rate=sample_rate,
            backend=self.name,
            model=self._model_id,
            voice=description,
            elapsed=time.perf_counter() - started,
            meta=meta,
        )

    # -- helpers --------------------------------------------------------
    def _run(self, text: str, description: str, lang_code: str) -> tuple[np.ndarray, int]:
        generator = getattr(self._model, "generate_audio", None)
        if callable(generator):
            out = self._call_generator(generator, text, description, lang_code)
            return self._interpret(out)
        return self._generate_with_transformers(text, description, lang_code)

    @staticmethod
    def _call_generator(generator: Any, text: str, description: str, lang_code: str) -> Any:
        attempts = [
            lambda: generator(text, description=description, language=lang_code),
            lambda: generator(text, description=description, lang=lang_code),
            lambda: generator(text, description, lang_code),
            lambda: generator(text=text, description=description, language=lang_code),
        ]
        last_error: Optional[Exception] = None
        for attempt in attempts:
            try:
                return attempt()
            except TypeError as exc:
                last_error = exc
                continue
        raise MashMixError(f"The Indic Parler TTS model rejected the audio call: {last_error}")

    def _generate_with_transformers(
        self, text: str, description: str, lang_code: str
    ) -> tuple[np.ndarray, int]:
        import torch  # imported late so the module stays light

        inputs = self._tokenizer(text, return_tensors="pt")
        attempts = [
            lambda: self._model.generate(
                **inputs,
                description=description,
                language=lang_code,
                do_sample=True,
                max_new_tokens=_MAX_NEW_TOKENS,
            ),
            lambda: self._model.generate(
                **inputs,
                description=description,
                language=lang_code,
            ),
            lambda: self._model.generate(**inputs),
        ]
        last_error: Optional[Exception] = None
        outputs = None
        for attempt in attempts:
            try:
                outputs = attempt()
                break
            except TypeError as exc:
                last_error = exc
                continue
        if outputs is None:
            raise MashMixError(
                f"The Indic Parler TTS model could not be run: {last_error}"
            )
        with torch.no_grad():
            audio = self._decode(outputs)
        return audio, self._sample_rate

    def _decode(self, outputs: Any) -> np.ndarray:
        decoder = getattr(self._model, "decode_audio", None)
        if callable(decoder):
            value = decoder(outputs)
        else:
            value = None
            for name in ("waveform", "audio_values", "audio"):
                candidate = getattr(outputs, name, None)
                if candidate is not None:
                    value = candidate
                    break
        if value is None:
            raise MashMixError(
                "The Indic Parler TTS model returned no waveform. The audio decoder of this "
                "release is not available, so the audio cannot be rebuilt."
            )
        if hasattr(value, "detach"):
            value = value.detach().to("cpu").numpy()
        return np.asarray(value, dtype=np.float32).reshape(-1)

    def list_voices(self) -> list[str]:
        """Voices are free text descriptions, so only examples are listed."""
        return list(_SAMPLE_DESCRIPTIONS)
