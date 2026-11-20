# SPDX-License-Identifier: MIT | Copyright (c) 2026 MASH (Khaled Bhuiyan)
"""Facebook MMS text to speech backend.

MMS TTS is a family of small VITS models published by Meta for more than a
thousand languages. The Bengali model ``facebook/mms-tts-ben`` is the cheapest
reliable Bangla voice that runs on a laptop or a phone without a GPU, which
makes it the default Bangla engine of this package.

The MMS tokenizer needs a language token for every non English model. This
backend sets it through ``set_lang`` when the tokenizer offers that method,
and otherwise prepends the language token by hand.
"""

from __future__ import annotations

import time
from typing import Any, Optional

import numpy as np

from ...errors import DependencyMissingError
from ...lang.normalize import normalize_for_tts
from ...types import AudioBuffer, SynthesisResult
from ..base import BackendInfo, TTSBackend
from ..registry import register_tts

try:  # pragma: no cover - the import guard keeps this module importable
    from transformers import AutoTokenizer, VitsModel, VitsTokenizer

    _HAS_TRANSFORMERS = True
except ImportError:  # pragma: no cover
    AutoTokenizer = None  # type: ignore[assignment]
    VitsModel = None  # type: ignore[assignment]
    VitsTokenizer = None  # type: ignore[assignment]
    _HAS_TRANSFORMERS = False


_DEFAULT_MODEL = "facebook/mms-tts-ben"

# MMS language codes, which are not always the ISO 639-1 code.
_MMS_CODE = {
    "bn": "ben",
    "bn-bd": "ben",
    "bn-in": "ben",
    "hi": "hin",
    "hi-in": "hin",
    "en": "eng",
    "en-us": "eng",
    "en-gb": "eng",
}

_MODEL_FOR_CODE = {
    "ben": "facebook/mms-tts-ben",
    "hin": "facebook/mms-tts-hin",
    "eng": "facebook/mms-tts-eng",
}

_KNOWN_VOICES = [
    "facebook/mms-tts-ben",
    "facebook/mms-tts-hin",
    "facebook/mms-tts-eng",
    "facebook/mms-tts-arb",
    "facebook/mms-tts-deu",
    "facebook/mms-tts-fra",
    "facebook/mms-tts-ita",
    "facebook/mms-tts-spa",
    "facebook/mms-tts-rus",
    "facebook/mms-tts-zho",
    "facebook/mms-tts-yue",
    "facebook/mms-tts-jpn",
    "facebook/mms-tts-kor",
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


@register_tts("mms-tts")
class MMSTTSBackend(TTSBackend):
    """Meta MMS TTS, one small VITS model per language."""

    info = BackendInfo(
        name="mms-tts",
        kind="tts",
        summary="Meta MMS VITS models with a dedicated Bengali voice (mms-tts-ben).",
        package="transformers",
        extra="tts",
        languages=["bn", "bn-BD", "bn-IN", "hi", "hi-IN", "en", "en-US", "en-GB"],
        platforms=["windows", "linux", "macos", "android-termux", "ios"],
        notes=(
            "Bengali is covered by facebook/mms-tts-ben and the same model reads both bn-BD "
            "and bn-IN text after the Bangla normalisation pass. The models are small, so this "
            "backend suits phones and laptops without a GPU. No voice cloning is available."
        ),
    )

    def __init__(self) -> None:
        super().__init__()
        self._model: Any = None
        self._tokenizer: Any = None
        self._sample_rate = 16000
        self._prefix = ""
        self._lang_applied = ""

    # -- lifecycle ------------------------------------------------------
    def is_available(self) -> bool:
        return _HAS_TRANSFORMERS

    def load(self, model: Optional[str] = None) -> None:
        if self._loaded:
            return
        if not _HAS_TRANSFORMERS:
            raise DependencyMissingError("transformers", "tts", "MMS text to speech")
        model_id = model or _DEFAULT_MODEL
        tokenizer = None
        if VitsTokenizer is not None:
            try:
                tokenizer = VitsTokenizer.from_pretrained(model_id)
            except Exception:  # a model without a VITS vocab falls back to AutoTokenizer
                tokenizer = None
        if tokenizer is None:
            tokenizer = AutoTokenizer.from_pretrained(model_id)
        loaded = VitsModel.from_pretrained(model_id)
        loaded.eval()
        self._tokenizer = tokenizer
        self._model = loaded
        config = getattr(loaded, "config", None)
        self._sample_rate = int(getattr(config, "sampling_rate", 16000) or 16000)
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
        meta: dict[str, Any] = {"normalised": prepared != text}
        if reference_audio is not None:
            meta["reference_audio"] = "ignored"
            meta["note"] = "MMS TTS has no voice cloning, so the reference audio was ignored."

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

        lang_code = self._language_code(language)
        self._prepare_language(lang_code)
        payload = f"{self._prefix} {prepared}".strip() if self._prefix else prepared
        meta["language_code"] = lang_code
        meta["language_token"] = self._lang_applied
        model_language = getattr(getattr(self._model, "config", None), "language", None)
        if model_language:
            meta["model_language"] = str(model_language)
            if str(model_language) != lang_code:
                meta["language_note"] = (
                    f"The loaded model speaks {model_language}, while {lang_code} was requested."
                )

        inputs = self._tokenizer(payload, return_tensors="pt")
        if int(inputs["input_ids"].shape[-1]) == 0:
            # The tokenizer dropped every character, so there is nothing to speak.
            meta["note"] = "The tokenizer produced no tokens for this text."
            return SynthesisResult(
                audio=np.zeros(0, dtype=np.float32),
                sample_rate=self._sample_rate,
                backend=self.name,
                model=self._model_id,
                voice=voice or self._model_id,
                elapsed=time.perf_counter() - started,
                meta=meta,
            )
        import torch  # imported late so the module stays light

        with torch.no_grad():
            outputs = self._model(**inputs)
        wave = outputs.waveform[0].detach().to("cpu").numpy().astype(np.float32)
        wave = _time_stretch_simple(wave, 1.0 / max(float(speed), 0.05))

        return SynthesisResult(
            audio=wave,
            sample_rate=self._sample_rate,
            backend=self.name,
            model=self._model_id,
            voice=voice or self._model_id,
            elapsed=time.perf_counter() - started,
            meta=meta,
        )

    # -- helpers --------------------------------------------------------
    def _language_code(self, language: str) -> str:
        code = _MMS_CODE.get((language or "").lower())
        if code:
            return code
        config = getattr(self._model, "config", None)
        model_lang = getattr(config, "language", None)
        if model_lang:
            return str(model_lang)
        return "eng"

    def _prepare_language(self, lang_code: str) -> None:
        """Make sure the tokenizer uses the language token of this model."""
        tokenizer = self._tokenizer
        self._prefix = ""
        setter = getattr(tokenizer, "set_lang", None)
        if callable(setter):
            try:
                setter(lang_code)
                self._lang_applied = f"set_lang:{lang_code}"
                return
            except Exception:
                pass
        # The transformers tokenizer keeps the language in its language attribute.
        for attribute in ("language", "lang"):
            if hasattr(tokenizer, attribute):
                try:
                    setattr(tokenizer, attribute, lang_code)
                    self._lang_applied = f"{attribute}:{lang_code}"
                    return
                except Exception:
                    continue
        token = self._language_token(lang_code)
        if token:
            self._prefix = token
            self._lang_applied = f"prefix:{token}"

    def _language_token(self, lang_code: str) -> str:
        """Find the language token id in the tokenizer vocabulary."""
        tokenizer = self._tokenizer
        unk = getattr(tokenizer, "unk_token_id", None)
        for candidate in (f"<{lang_code}>", f"<lang:{lang_code}>", lang_code):
            try:
                token_id = tokenizer.convert_tokens_to_ids(candidate)
            except Exception:
                continue
            if token_id is None:
                continue
            if unk is not None and token_id == unk:
                continue
            return candidate
        return ""

    def list_voices(self) -> list[str]:
        return list(_KNOWN_VOICES)

    def default_model_for(self, language: str) -> str:
        """Model id that matches a language, used when no model is given."""
        code = _MMS_CODE.get((language or "").lower())
        return _MODEL_FOR_CODE.get(code or "", _DEFAULT_MODEL)
