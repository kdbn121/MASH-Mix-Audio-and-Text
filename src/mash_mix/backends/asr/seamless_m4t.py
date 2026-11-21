# SPDX-License-Identifier: MIT | Copyright (c) 2026 MASH (Khaled Bhuiyan)
"""Meta SeamlessM4T speech to text backend.

SeamlessM4T is the multilingual model from Meta. It reads speech in about a
hundred languages, and Bengali is one of them, so it is a second strong
answer to the question of how to read a Bangla song well.

Two things about it shape how this backend is written.

* It is a translation model, so the language it is asked to write in is a
  parameter of the model, not only of the request. For plain recognition the
  target is set to the language that was asked for, which turns the model
  into a recogniser for that language.
* It gives the time of a sentence and not the time of a word, so the lyric
  lines are cut from the sentence times. A karaoke line that needs the time
  of a word inside it does not get one from this backend, and the local
  Whisper backends are the ones to reach for when that matters.

The model is large, so it needs room in memory. The backend reports itself
unavailable when transformers is not installed, and the caller falls back to
the next model in the ladder when the weights do not fit.
"""

from __future__ import annotations

import time
from typing import Any, Dict, List, Optional, Sequence, Tuple

import numpy as np

from ...errors import AudioDecodeError, BackendUnavailableError
from ...lang.dialect import detect_region
from ...lang.normalize import clean_asr_text, is_bangla_text, normalize_bangla
from ...types import Segment, Transcription
from ..base import BackendInfo, STTBackend
from ..registry import register_stt

try:  # pragma: no cover - optional dependency guard
    from transformers import (  # type: ignore
        AutoProcessor,
        SeamlessM4TForSpeechToText,
    )

    _HAS_TRANSFORMERS = True
except Exception:  # pragma: no cover - a broken install must not break the import
    AutoProcessor = None  # type: ignore
    SeamlessM4TForSpeechToText = None  # type: ignore

    _HAS_TRANSFORMERS = False

__all__ = ["SeamlessM4TSTTBackend"]

TARGET_RATE = 16000
DEFAULT_MODEL = "facebook/seamless-m4t-v2-large"

#: The language code SeamlessM4T uses for Bangla.
_BANGLA_TARGET = "ben"

#: The language codes for the tags the panel offers.
_TARGET_CODES: Dict[str, str] = {
    "bn-bd": _BANGLA_TARGET,
    "bn-in": _BANGLA_TARGET,
    "bn": _BANGLA_TARGET,
    "ben": _BANGLA_TARGET,
    "beng": _BANGLA_TARGET,
    "en": "eng",
    "eng": "eng",
    "hi": "hin",
    "hin": "hin",
    "ur": "urd",
    "urd": "urd",
    "ar": "arb",
    "arb": "arb",
    "es": "spa",
    "spa": "spa",
    "fr": "fra",
    "fra": "fra",
    "de": "deu",
    "deu": "deu",
    "it": "ita",
    "ita": "ita",
    "pt": "por",
    "por": "por",
    "ru": "rus",
    "rus": "rus",
    "zh": "cmn",
    "cmn": "cmn",
    "ja": "jpn",
    "jpn": "jpn",
    "ko": "kor",
    "kor": "kor",
    "tr": "tur",
    "tur": "tur",
    "vi": "vie",
    "vie": "vie",
    "id": "ind",
    "ind": "ind",
    "ms": "msa",
    "msa": "msa",
    "nl": "nld",
    "nld": "nld",
    "pl": "pol",
    "pol": "pol",
    "ta": "tam",
    "tam": "tam",
    "te": "tel",
    "tel": "tel",
    "mr": "mar",
    "mar": "mar",
    "gu": "guj",
    "guj": "guj",
    "pa": "pan",
    "pan": "pan",
    "th": "tha",
    "tha": "tha",
}

_BANGLA_BASES = {"bn", "beng", "ben", "bang", "bangla", "bangali", "bengali"}

_LANGUAGES = sorted({code for code in _TARGET_CODES.values()})

_PLATFORMS = ["windows", "linux", "macos"]


def _split_tag(tag: Optional[str]) -> Tuple[str, str]:
    """Split a language tag into a lowercase base and an uppercase region."""
    cleaned = (tag or "").strip().replace("_", "-")
    if not cleaned:
        return "", ""
    parts = [p for p in cleaned.split("-") if p]
    base = parts[0].lower()
    region = ""
    for part in parts[1:]:
        if len(part) == 2 and part.isalpha():
            region = part.upper()
    return base, region


def _target_code(language: Optional[str]) -> str:
    """The language code SeamlessM4T should write in.

    The model writes in the language it is told to write in, so a Bangla
    request becomes Bengali and an English request becomes English.
    """
    if not language:
        return "eng"
    base, _region = _split_tag(language)
    return _TARGET_CODES.get(base, _TARGET_CODES.get(language.lower(), "eng"))


def _clean(text: str) -> str:
    """Clean recognised text and apply the Bangla normalisation pass."""
    out = clean_asr_text(text or "")
    if out and is_bangla_text(out):
        out = normalize_bangla(out, digits="keep")
    return out.strip()


def _result_tag(requested: Optional[str], text: str) -> str:
    """Work out the language tag reported on the transcription."""
    if requested:
        base, region = _split_tag(requested)
        if base in _BANGLA_BASES:
            if region in ("BD", "IN"):
                return f"bn-{region}"
            if text and is_bangla_text(text):
                return detect_region(text, default="bn-BD").tag
            return "bn-BD"
        return requested
    if text and is_bangla_text(text):
        return detect_region(text, default="bn-BD").tag
    return "und"


def _resample(audio: np.ndarray, src_rate: int, dst_rate: int = TARGET_RATE) -> np.ndarray:
    """Resample mono float32 samples with linear interpolation."""
    data = np.asarray(audio, dtype=np.float32).reshape(-1)
    if src_rate == dst_rate or data.size == 0:
        return data.astype(np.float32, copy=False)
    new_length = max(1, int(round(data.size * float(dst_rate) / float(src_rate))))
    old_axis = np.arange(data.size, dtype=np.float64)
    new_axis = np.linspace(0.0, float(data.size - 1), num=new_length, dtype=np.float64)
    return np.interp(new_axis, old_axis, data).astype(np.float32)


def _prepare_audio(audio: Any, sample_rate: int) -> np.ndarray:
    """Validate the incoming block and return 16 kHz mono float32 samples."""
    if audio is None:
        raise AudioDecodeError("SeamlessM4T received no audio to transcribe.")
    arr = np.asarray(audio)
    if arr.ndim > 1:
        arr = arr.mean(axis=0) if arr.shape[0] <= 8 else arr.reshape(-1)
    arr = arr.astype(np.float32, copy=False).reshape(-1)
    if int(sample_rate) <= 0:
        raise AudioDecodeError("SeamlessM4T needs a positive sample rate.")
    if arr.size == 0:
        raise AudioDecodeError("SeamlessM4T cannot transcribe an empty audio block.")
    return _resample(arr, int(sample_rate))


@register_stt("seamless-m4t")
class SeamlessM4TSTTBackend(STTBackend):
    """Speech to text with the Meta SeamlessM4T model."""

    info = BackendInfo(
        name="seamless-m4t",
        kind="stt",
        summary="The Meta SeamlessM4T model, which reads Bengali as well as most languages.",
        package="transformers",
        extra="stt",
        languages=_LANGUAGES,
        platforms=_PLATFORMS,
        notes=(
            "Default model 'facebook/seamless-m4t-v2-large'. It gives the time "
            "of a sentence and not the time of a word, so the local Whisper "
            "backends are the better choice when a karaoke line needs the "
            "time of a word. The model is large and needs room in memory."
        ),
    )

    def __init__(self) -> None:
        super().__init__()
        self._model: Any = None
        self._processor: Any = None
        self._dtype: str = "float32"

    # -- lifecycle ------------------------------------------------------
    def is_available(self) -> bool:
        """True when transformers is installed."""
        try:
            return bool(
                _HAS_TRANSFORMERS
                and SeamlessM4TForSpeechToText is not None
                and AutoProcessor is not None
            )
        except Exception:  # pragma: no cover - defensive, must never raise
            return False

    def _pick_dtype(self) -> str:
        """Use float16 on a GPU and bfloat16 on the CPU."""
        try:
            import torch  # type: ignore

            if bool(torch.cuda.is_available()):
                return "float16"
            if hasattr(torch, "bfloat16"):
                return "bfloat16"
        except Exception:  # pragma: no cover - torch is optional here
            pass
        return "float32"

    def load(self, model: Optional[str] = None, **kwargs: Any) -> None:
        """Load the model and its processor. Safe to call twice."""
        if self._loaded:
            return
        if not self.is_available():
            raise BackendUnavailableError(
                "The 'seamless-m4t' backend needs the transformers package. "
                "Install it with: pip install 'mash-mix-audio-and-text[stt]'"
            )
        model_name = str(model or kwargs.get("model_id") or DEFAULT_MODEL)
        dtype_name = str(kwargs.get("torch_dtype") or self._pick_dtype())
        torch_dtype: Any = dtype_name
        try:
            import torch  # type: ignore

            torch_dtype = getattr(torch, dtype_name)
        except Exception:  # pragma: no cover - torch is optional here
            torch_dtype = dtype_name

        try:
            self._processor = AutoProcessor.from_pretrained(model_name)
            self._model = SeamlessM4TForSpeechToText.from_pretrained(
                model_name, torch_dtype=torch_dtype
            )
        except Exception as exc:
            raise BackendUnavailableError(
                f"Could not load SeamlessM4T model '{model_name}': {exc}"
            ) from exc
        self._dtype = dtype_name
        self._model_id = model_name
        self._loaded = True

    def unload(self) -> None:
        """Drop the loaded weights."""
        self._model = None
        self._processor = None
        self._loaded = False

    # -- inference ------------------------------------------------------
    def transcribe(
        self,
        audio: np.ndarray,
        sample_rate: int,
        *,
        language: Optional[str] = None,
        task: str = "transcribe",
        initial_prompt: Optional[str] = None,
        hotwords: Optional[Sequence[str]] = None,
        word_timestamps: bool = True,
        vad: bool = True,
        progress: bool = False,
        **kwargs: Any,
    ) -> Transcription:
        """Transcribe mono float32 ``audio`` sampled at ``sample_rate``."""
        if not self._loaded:
            self.load()
        samples = _prepare_audio(audio, sample_rate)
        duration = float(len(samples)) / float(TARGET_RATE)
        target = _target_code(language)

        started = time.perf_counter()
        try:
            features = self._processor(
                samples, sampling_rate=TARGET_RATE, return_tensors="pt"
            )
            generated = self._model.generate(
                features.input_features,
                tgt_lang=target,
                num_beams=kwargs.get("beam_size", 1),
                max_new_tokens=kwargs.get("max_new_tokens", 448),
            )
            decoded = self._processor.batch_decode(
                generated, skip_special_tokens=True, output_sequences=False
            )
        except Exception as exc:
            return self._empty(duration, language, exc)

        text = _clean(decoded[0] if decoded else "")
        segments: List[Segment] = []
        if text:
            # The model gives the time of a sentence and not the time of a
            # word, so the whole run is one segment.
            segments = [Segment(start=0.0, end=duration, text=text)]

        return Transcription(
            text=text,
            language=_result_tag(language, text),
            duration=duration,
            segments=segments,
            backend=self.name,
            model=self._model_id or DEFAULT_MODEL,
            elapsed=time.perf_counter() - started,
            meta={
                "dtype": self._dtype,
                "target_language": target,
                "task": task if task in ("transcribe", "translate") else "transcribe",
                "word_timestamps": False,
                "timestamp_level": "sentence",
            },
        )

    def _empty(self, duration: float, language: Optional[str], exc: Exception) -> Transcription:
        """Return an empty transcription instead of raising on a failure."""
        return Transcription(
            text="",
            language=_result_tag(language, ""),
            duration=duration,
            segments=[],
            backend=self.name,
            model=self._model_id or DEFAULT_MODEL,
            meta={"error": str(exc)},
        )
