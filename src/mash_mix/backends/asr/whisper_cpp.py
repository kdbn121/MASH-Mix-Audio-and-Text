# SPDX-License-Identifier: MIT | Copyright (c) 2026 MASH (Khaled Bhuiyan)
"""whisper.cpp speech to text backend.

The ``pywhispercpp`` package drives the whisper.cpp quantised models. It is the
fastest option on a phone and the only one that runs comfortably on a phone at
all, so it is the practical choice for Android on Termux and for iOS builds.

There is no default model. The caller must point the backend at a ggml file
with ``load(model_path='...')``.

Bangla is supported through the ggml Whisper models. A request for ``bn-BD`` or
``bn-IN`` is handed to whisper.cpp as ``bn`` and the regional tag is restored on
the returned :class:`~mash_mix.types.Transcription`.
"""

from __future__ import annotations

import os
import time
from typing import Any, Dict, List, Optional, Sequence, Tuple

import numpy as np

from ...errors import AudioDecodeError, BackendUnavailableError, ModelNotFoundError
from ...lang.dialect import detect_region
from ...lang.normalize import clean_asr_text, is_bangla_text, normalize_bangla
from ...types import Segment, Transcription, Word
from ..base import BackendInfo, STTBackend
from ..registry import register_stt

try:  # pragma: no cover - optional dependency guard
    from pywhispercpp.model import Model  # type: ignore

    _HAS_PYWHISPERCPP = True
except Exception:  # pragma: no cover - a broken install must not break the import
    Model = None  # type: ignore
    _HAS_PYWHISPERCPP = False

__all__ = ["WhisperCppSTTBackend"]

TARGET_RATE = 16000

_BANGLA_BASES = {"bn", "beng", "ben", "bang", "bangla", "bangali", "bengali"}

_LANGUAGES = [
    "bn",
    "bn-BD",
    "bn-IN",
    "en",
    "hi",
    "ur",
    "ar",
    "es",
    "fr",
    "de",
    "it",
    "pt",
    "ru",
    "zh",
    "ja",
    "ko",
    "tr",
    "vi",
    "id",
    "ms",
    "nl",
    "pl",
    "ta",
    "te",
    "mr",
    "gu",
    "pa",
    "fa",
    "sw",
    "th",
]

_PLATFORMS = ["windows", "linux", "macos", "android", "ios"]


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
        raise AudioDecodeError("whisper.cpp received no audio to transcribe.")
    arr = np.asarray(audio)
    if arr.ndim > 1:
        arr = arr.mean(axis=0) if arr.shape[0] <= 8 else arr.reshape(-1)
    arr = arr.astype(np.float32, copy=False).reshape(-1)
    if int(sample_rate) <= 0:
        raise AudioDecodeError("whisper.cpp needs a positive sample rate.")
    if arr.size == 0:
        raise AudioDecodeError("whisper.cpp cannot transcribe an empty audio block.")
    return _resample(arr, int(sample_rate))


def _clean(text: str) -> str:
    """Clean recognised text and apply the Bangla normalisation pass."""
    out = clean_asr_text(text or "")
    if out and is_bangla_text(out):
        out = normalize_bangla(out, digits="keep")
    return out.strip()


def _engine_language(language: Optional[str]) -> Optional[str]:
    """Map a requested tag onto a code whisper.cpp understands."""
    if not language:
        return None
    base, _region = _split_tag(language)
    if base in _BANGLA_BASES:
        return "bn"
    return language


def _result_tag(requested: Optional[str], detected: Optional[str], text: str) -> str:
    """Work out the language tag reported on the transcription."""
    for candidate in (requested, detected):
        if not candidate:
            continue
        base, region = _split_tag(candidate)
        if base not in _BANGLA_BASES:
            return candidate
        if region in ("BD", "IN"):
            return f"bn-{region}"
        if text and is_bangla_text(text):
            return detect_region(text, default="bn-BD").tag
        return "bn-BD"
    if text and is_bangla_text(text):
        return detect_region(text, default="bn-BD").tag
    return "und"


def _build_prompt(initial_prompt: Optional[str], hotwords: Optional[Sequence[str]]) -> Optional[str]:
    """Fold Bangla hotwords into the decoder prompt."""
    parts: List[str] = []
    if initial_prompt:
        parts.append(str(initial_prompt).strip())
    if hotwords:
        words = [str(w).strip() for w in hotwords if str(w).strip()]
        if words:
            parts.append(" ".join(words))
    if not parts:
        return None
    return " ".join(part for part in parts if part)


def _probability(value: Any) -> Optional[float]:
    try:
        if value is None:
            return None
        return float(value)
    except (TypeError, ValueError):
        return None


def _number(value: Any, default: float = 0.0) -> float:
    try:
        if value is None:
            return default
        return float(value)
    except (TypeError, ValueError):
        return default


def _finalise(segments: List[Segment]) -> List[Segment]:
    """Trim, drop empties, merge identical neighbours and enforce monotonic times."""
    out: List[Segment] = []
    for seg in segments:
        text = seg.text.strip()
        if not text:
            continue
        if out and out[-1].text == text:
            previous = out[-1]
            previous.end = max(previous.end, float(seg.end))
            previous.words.extend(seg.words)
            continue
        out.append(seg)
    cursor = 0.0
    for seg in out:
        start = max(float(seg.start), cursor)
        end = max(float(seg.end), start)
        seg.start = start
        seg.end = end
        cursor = end
    return out


@register_stt("whisper.cpp")
class WhisperCppSTTBackend(STTBackend):
    """Speech to text with whisper.cpp through the pywhispercpp bindings."""

    info = BackendInfo(
        name="whisper.cpp",
        kind="stt",
        summary="whisper.cpp quantised ggml models, the fastest option on a phone.",
        package="pywhispercpp",
        extra="stt",
        languages=_LANGUAGES,
        platforms=_PLATFORMS,
        notes=(
            "No default model. Call load(model_path='<path to a ggml model>'). "
            "Bangla needs a Whisper ggml model that covers bn, for example the "
            "multilingual ggml-large-v3 build."
        ),
    )

    def __init__(self) -> None:
        super().__init__()
        self._model: Any = None
        self._model_path: str = ""

    # -- lifecycle ------------------------------------------------------
    def is_available(self) -> bool:
        """True when ``pywhispercpp`` can be imported."""
        try:
            return bool(_HAS_PYWHISPERCPP and Model is not None)
        except Exception:  # pragma: no cover - defensive, must never raise
            return False

    def load(self, model: Optional[str] = None, *, model_path: Optional[str] = None, **kwargs: Any) -> None:
        """Load a ggml model. Safe to call twice."""
        if self._loaded:
            return
        if not self.is_available():
            raise BackendUnavailableError(
                "The 'whisper.cpp' backend needs the pywhispercpp package. "
                "Install it with: pip install 'mash-mix-audio-and-text[stt]'"
            )
        path = str(model_path or model or kwargs.get("model_path") or "").strip()
        if not path:
            raise ModelNotFoundError(
                "The 'whisper.cpp' backend has no default model. "
                "Call load(model_path='<path to a ggml model file>')."
            )
        if not os.path.isfile(path):
            raise ModelNotFoundError(
                f"The whisper.cpp model file '{path}' does not exist."
            )

        build_kwargs: Dict[str, Any] = {
            "n_threads": int(kwargs.get("n_threads") or 4),
            "language": str(kwargs.get("language") or "auto"),
            "print_realtime": False,
            "print_progress": False,
            "verbose": bool(kwargs.get("verbose", False)),
        }
        try:
            self._model = Model(path, **build_kwargs)
        except TypeError:
            try:
                self._model = Model(path)
            except Exception as exc:
                raise BackendUnavailableError(
                    f"Could not load the whisper.cpp model '{path}': {exc}"
                ) from exc
        except Exception as exc:
            raise BackendUnavailableError(
                f"Could not load the whisper.cpp model '{path}': {exc}"
            ) from exc
        self._model_path = path
        self._model_id = path
        self._loaded = True

    def unload(self) -> None:
        """Drop the loaded model."""
        self._model = None
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
        duration = float(len(np.asarray(audio).reshape(-1))) / float(sample_rate)

        hint = _engine_language(language)
        prompt = _build_prompt(initial_prompt, hotwords)

        call_kwargs: Dict[str, Any] = {}
        if hint:
            call_kwargs["language"] = hint
        if prompt:
            call_kwargs["prompt"] = prompt
        if kwargs.get("speed") is not None:
            call_kwargs["speed"] = kwargs["speed"]

        started = time.perf_counter()
        try:
            result = self._model.transcribe(samples, **call_kwargs)
        except TypeError:
            try:
                result = self._model.transcribe(samples)
            except Exception as exc:
                return self._empty(duration, language, exc)
        except Exception as exc:
            return self._empty(duration, language, exc)

        segments = self._segments_from_result(result, word_timestamps, duration)
        segments = _finalise(segments)
        joined = " ".join(seg.text for seg in segments)
        return Transcription(
            text=joined,
            language=_result_tag(language, None, joined),
            duration=duration,
            segments=segments,
            backend=self.name,
            model=self._model_id or "",
            elapsed=time.perf_counter() - started,
            meta={
                "model_path": self._model_path,
                "task": task,
                "word_timestamps": bool(word_timestamps),
                "vad": bool(vad),
            },
        )

    def _segments_from_result(self, result: Any, word_timestamps: bool, duration: float) -> List[Segment]:
        """Map whatever pywhispercpp returned onto segments."""
        raw_segments: Any = result
        if isinstance(result, dict):
            raw_segments = result.get("segments") or []
        elif result is not None and not isinstance(result, (list, tuple)):
            raw_segments = getattr(result, "segments", None) or []
        if raw_segments is None:
            raw_segments = []
        if isinstance(raw_segments, dict):
            raw_segments = [raw_segments]

        segments: List[Segment] = []
        for raw in raw_segments:
            if not isinstance(raw, dict):
                text = _clean(str(getattr(raw, "text", "") or ""))
                start = _number(getattr(raw, "start", 0.0), 0.0)
                end = _number(getattr(raw, "end", start), start)
                if text:
                    segments.append(Segment(start=start, end=end, text=text))
                continue
            text = _clean(str(raw.get("text") or ""))
            if not text:
                continue
            start = _number(raw.get("start"), 0.0)
            end = _number(raw.get("end"), start)
            words: List[Word] = []
            if word_timestamps:
                for raw_word in raw.get("words") or []:
                    if not isinstance(raw_word, dict):
                        continue
                    word_text = _clean(str(raw_word.get("word") or raw_word.get("text") or ""))
                    if not word_text:
                        continue
                    words.append(
                        Word(
                            start=_number(raw_word.get("start"), start),
                            end=_number(raw_word.get("end"), end),
                            text=word_text,
                            probability=_probability(raw_word.get("probability")),
                        )
                    )
            segments.append(Segment(start=start, end=end, text=text, words=words))
        return segments

    def _empty(self, duration: float, language: Optional[str], exc: Exception) -> Transcription:
        """Return an empty transcription instead of raising on a recognition failure."""
        return Transcription(
            text="",
            language=_result_tag(language, None, ""),
            duration=duration,
            segments=[],
            backend=self.name,
            model=self._model_id or "",
            meta={"error": str(exc)},
        )
