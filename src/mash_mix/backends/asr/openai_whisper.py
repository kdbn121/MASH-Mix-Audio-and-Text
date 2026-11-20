# SPDX-License-Identifier: MIT | Copyright (c) 2026 MASH (Khaled Bhuiyan)
"""OpenAI Whisper speech to text backend.

This is the reference Whisper implementation. It is slower on CPU than
``faster-whisper`` but it is the most faithful port of the original model and
the easiest one to install on Windows, Linux, macOS, Termux and iOS.

Bangla is a first class language here. A request for ``bn-BD`` or ``bn-IN`` is
handed to the model as ``bn`` and the regional tag is restored on the returned
:class:`~mash_mix.types.Transcription`.
"""

from __future__ import annotations

import time
from typing import Any, List, Optional, Sequence, Tuple

import numpy as np

from ...errors import AudioDecodeError, BackendUnavailableError
from ...lang.dialect import detect_region
from ...lang.normalize import clean_asr_text, is_bangla_text, normalize_bangla
from ...types import Segment, Transcription, Word
from ..base import BackendInfo, STTBackend
from ..registry import register_stt

try:  # pragma: no cover - optional dependency guard
    import whisper  # type: ignore

    _HAS_WHISPER = True
except Exception:  # pragma: no cover - a broken CUDA install must not break the import
    whisper = None  # type: ignore
    _HAS_WHISPER = False

__all__ = ["WhisperSTTBackend"]

TARGET_RATE = 16000
DEFAULT_MODEL = "small"

# Language codes that mean Bangla in one of the many registries.
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
        raise AudioDecodeError("Whisper received no audio to transcribe.")
    arr = np.asarray(audio)
    if arr.ndim > 1:
        arr = arr.mean(axis=0) if arr.shape[0] <= 8 else arr.reshape(-1)
    arr = arr.astype(np.float32, copy=False).reshape(-1)
    if int(sample_rate) <= 0:
        raise AudioDecodeError("Whisper needs a positive sample rate.")
    if arr.size == 0:
        raise AudioDecodeError("Whisper cannot transcribe an empty audio block.")
    return _resample(arr, int(sample_rate))


def _clean(text: str) -> str:
    """Clean recognised text and apply the Bangla normalisation pass."""
    out = clean_asr_text(text or "")
    if out and is_bangla_text(out):
        out = normalize_bangla(out, digits="keep")
    return out.strip()


def _engine_language(language: Optional[str]) -> Optional[str]:
    """Map a requested tag onto a code Whisper understands."""
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


@register_stt("whisper")
class WhisperSTTBackend(STTBackend):
    """Speech to text with the reference OpenAI Whisper package."""

    info = BackendInfo(
        name="whisper",
        kind="stt",
        summary="Reference OpenAI Whisper implementation with word timestamps.",
        package="openai-whisper",
        extra="stt",
        languages=_LANGUAGES,
        platforms=_PLATFORMS,
        notes=(
            "Default model 'small' runs on CPU. Use 'large-v3' for the best "
            "Bangla accuracy, or 'medium' as a middle ground. Needs a 16 kHz "
            "mono signal, so the audio is resampled internally."
        ),
    )

    def __init__(self) -> None:
        super().__init__()
        self._model: Any = None
        self._device: str = "cpu"

    # -- lifecycle ------------------------------------------------------
    def is_available(self) -> bool:
        """True when ``openai-whisper`` can be imported."""
        try:
            return bool(_HAS_WHISPER and whisper is not None)
        except Exception:  # pragma: no cover - defensive, must never raise
            return False

    def _pick_device(self) -> str:
        """Choose cuda, then Apple mps, then cpu."""
        try:
            import torch  # type: ignore

            if bool(torch.cuda.is_available()):
                return "cuda"
            mps = getattr(getattr(torch, "backends", None), "mps", None)
            if mps is not None and bool(mps.is_available()):
                return "mps"
        except Exception:  # pragma: no cover - torch is optional here
            pass
        return "cpu"

    def load(self, model: Optional[str] = None, **kwargs: Any) -> None:
        """Load the Whisper weights. Safe to call twice."""
        if self._loaded:
            return
        if not self.is_available():
            raise BackendUnavailableError(
                "The 'whisper' backend needs the openai-whisper package. "
                "Install it with: pip install 'mash-mix-audio-and-text[stt]'"
            )
        model_name = str(model or kwargs.get("model_id") or DEFAULT_MODEL)
        device = str(kwargs.get("device") or self._pick_device())
        try:
            self._model = whisper.load_model(model_name, device=device)
        except Exception as exc:
            raise BackendUnavailableError(
                f"Could not load Whisper model '{model_name}' on {device}: {exc}"
            ) from exc
        self._device = device
        self._model_id = model_name
        self._loaded = True

    def unload(self) -> None:
        """Drop the loaded weights."""
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

        engine_task = task if task in ("transcribe", "translate") else "transcribe"
        hint = _engine_language(language)
        if engine_task == "translate" and hint not in (None, "en"):
            # Whisper can only translate English into English.
            engine_task = "transcribe"
        prompt = _build_prompt(initial_prompt, hotwords)

        # Whisper inverts the meaning of verbose: the progress bar is drawn when
        # verbose is False, and the transcript is printed when verbose is True.
        # None is the only quiet setting, so it is used unless progress is asked
        # for, in which case the bar is welcome.
        verbose: Optional[bool] = False if progress else None
        started = time.perf_counter()
        try:
            result = self._model.transcribe(
                samples,
                language=hint,
                task=engine_task,
                word_timestamps=bool(word_timestamps),
                initial_prompt=prompt,
                verbose=verbose,
            )
        except Exception as exc:
            return self._empty(duration, language, exc)

        raw_segments = result.get("segments") or []
        segments: List[Segment] = []
        for raw in raw_segments:
            if not isinstance(raw, dict):
                continue
            text = _clean(str(raw.get("text") or ""))
            if not text:
                continue
            start = float(raw.get("start") or 0.0)
            end = float(raw.get("end") or 0.0)
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
                            start=float(raw_word.get("start") or start),
                            end=float(raw_word.get("end") or end),
                            text=word_text,
                            probability=_probability(raw_word.get("probability")),
                        )
                    )
            segments.append(Segment(start=start, end=end, text=text, words=words))

        segments = _finalise(segments)
        joined = " ".join(seg.text for seg in segments)
        detected = result.get("language")
        return Transcription(
            text=joined,
            language=_result_tag(language, detected, joined),
            duration=duration,
            segments=segments,
            backend=self.name,
            model=self._model_id or "",
            elapsed=time.perf_counter() - started,
            meta={
                "device": self._device,
                "task": engine_task,
                "detected_language": detected,
                "word_timestamps": bool(word_timestamps),
                "vad": bool(vad),
            },
        )

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
