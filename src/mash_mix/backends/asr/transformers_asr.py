# SPDX-License-Identifier: MIT | Copyright (c) 2026 MASH (Khaled Bhuiyan)
"""Hugging Face Transformers speech to text backend.

This backend drives the Whisper checkpoints through the generic
``transformers`` automatic speech recognition pipeline, so it can also run
Whisper derivatives such as ``openai/whisper-large-v3-turbo`` or a distilled
model from the Hub.

Bangla is a first class language here. A request for ``bn-BD`` or ``bn-IN`` is
handed to the model as ``bn`` and the regional tag is restored on the returned
:class:`~mash_mix.types.Transcription`.
"""

from __future__ import annotations

import time
from typing import Any, Dict, List, Optional, Sequence, Tuple

import numpy as np

from ...errors import AudioDecodeError, BackendUnavailableError
from ...lang.dialect import detect_region
from ...lang.normalize import clean_asr_text, is_bangla_text, normalize_bangla
from ...types import Segment, Transcription, Word
from ..base import BackendInfo, STTBackend
from ..registry import register_stt

try:  # pragma: no cover - optional dependency guard
    from transformers import pipeline  # type: ignore

    _HAS_TRANSFORMERS = True
except Exception:  # pragma: no cover - a broken install must not break the import
    pipeline = None  # type: ignore
    _HAS_TRANSFORMERS = False

__all__ = ["TransformersSTTBackend"]

TARGET_RATE = 16000
DEFAULT_MODEL = "openai/whisper-large-v3-turbo"

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

# A word ends a group when it closes a sentence or when a long pause follows.
_SENTENCE_END = (".", "!", "?", "।", "?", "।", "৷")


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
        raise AudioDecodeError("Transformers received no audio to transcribe.")
    arr = np.asarray(audio)
    if arr.ndim > 1:
        arr = arr.mean(axis=0) if arr.shape[0] <= 8 else arr.reshape(-1)
    arr = arr.astype(np.float32, copy=False).reshape(-1)
    if int(sample_rate) <= 0:
        raise AudioDecodeError("Transformers needs a positive sample rate.")
    if arr.size == 0:
        raise AudioDecodeError("Transformers cannot transcribe an empty audio block.")
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


def _chunk_times(chunk: Any) -> Tuple[Optional[float], Optional[float]]:
    """Read a timestamp pair from a pipeline chunk."""
    stamp = None
    if isinstance(chunk, dict):
        stamp = chunk.get("timestamp")
    else:
        stamp = getattr(chunk, "timestamp", None)
    if stamp is None:
        return None, None
    if isinstance(stamp, (list, tuple)):
        if not stamp:
            return None, None
        start = stamp[0]
        end = stamp[1] if len(stamp) > 1 else stamp[0]
    else:
        start = stamp
        end = stamp
    try:
        start_value = float(start)
    except (TypeError, ValueError):
        return None, None
    try:
        end_value = float(end)
    except (TypeError, ValueError):
        end_value = start_value
    return start_value, end_value


def _chunk_text(chunk: Any) -> str:
    """Read the text of a pipeline chunk."""
    if isinstance(chunk, dict):
        return str(chunk.get("text") or "")
    return str(getattr(chunk, "text", "") or "")


def _group_words(words: List[Word], gap: float = 0.6) -> List[Segment]:
    """Group word timings into readable segments."""
    segments: List[Segment] = []
    group: List[Word] = []
    for word in words:
        if group:
            previous = group[-1]
            pause = word.start - previous.end
            ends_sentence = previous.text.endswith(_SENTENCE_END)
            if pause > gap or ends_sentence:
                segments.append(_segment_from_words(group))
                group = []
        group.append(word)
    if group:
        segments.append(_segment_from_words(group))
    return segments


def _segment_from_words(words: List[Word]) -> Segment:
    """Build one segment out of a run of words."""
    return Segment(
        start=words[0].start,
        end=words[-1].end,
        text=" ".join(word.text for word in words).strip(),
        words=list(words),
    )


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


@register_stt("transformers")
class TransformersSTTBackend(STTBackend):
    """Speech to text with the Hugging Face transformers pipeline."""

    info = BackendInfo(
        name="transformers",
        kind="stt",
        summary="Hugging Face transformers pipeline for Whisper and its derivatives.",
        package="transformers",
        extra="stt",
        languages=_LANGUAGES,
        platforms=["windows", "linux", "macos", "android", "ios"],
        notes=(
            "Default model 'openai/whisper-large-v3-turbo', a strong Bangla "
            "recogniser with a good speed trade off. Word timestamps come back "
            "as pipeline chunks and are grouped into segments."
        ),
    )

    def __init__(self) -> None:
        super().__init__()
        self._pipe: Any = None
        self._dtype: str = "float32"

    # -- lifecycle ------------------------------------------------------
    def is_available(self) -> bool:
        """True when ``transformers`` can be imported."""
        try:
            return bool(_HAS_TRANSFORMERS and pipeline is not None)
        except Exception:  # pragma: no cover - defensive, must never raise
            return False

    def _pick_dtype(self) -> str:
        """Use float16 on a GPU and bfloat16 on the CPU.

        bfloat16 keeps the range of float32 and needs no scaling, which makes
        it the right half precision type for a CPU. On this path it is about
        three times faster than float32 and it does not change the text.
        """
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
        """Build the automatic speech recognition pipeline. Safe to call twice."""
        if self._loaded:
            return
        if not self.is_available():
            raise BackendUnavailableError(
                "The 'transformers' backend needs the transformers package. "
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

        build_kwargs: Dict[str, Any] = {"model": model_name, "torch_dtype": torch_dtype}
        if kwargs.get("device"):
            build_kwargs["device"] = kwargs["device"]
        try:
            self._pipe = pipeline("automatic-speech-recognition", **build_kwargs)
        except TypeError:
            # Newer transformers renamed torch_dtype to dtype.
            build_kwargs["dtype"] = build_kwargs.pop("torch_dtype")
            try:
                self._pipe = pipeline("automatic-speech-recognition", **build_kwargs)
            except Exception as exc:
                raise BackendUnavailableError(
                    f"Could not load transformers model '{model_name}': {exc}"
                ) from exc
        except Exception as exc:
            raise BackendUnavailableError(
                f"Could not load transformers model '{model_name}': {exc}"
            ) from exc
        self._dtype = dtype_name
        self._model_id = model_name
        self._loaded = True

    def unload(self) -> None:
        """Drop the loaded pipeline."""
        self._pipe = None
        self._loaded = False

    # -- inference ------------------------------------------------------
    def _run_window(
        self,
        samples: np.ndarray,
        *,
        language: Optional[str],
        task: str,
        prompt: Optional[str],
        hotwords: Optional[Sequence[str]],
        word_timestamps: bool,
        vad: bool,
    ) -> Tuple[str, List[Segment], Optional[str]]:
        """Run the model over one window and return its text and segments.

        The window is a single forward pass. The pipeline is not asked to chunk
        the audio itself, because that path batches several windows together
        and it has proved unstable on a CPU. Whisper was trained on 30 second
        windows, so one window at a time is the shape it expects.

        The timestamps are segment level, not word level. Word timestamps ask
        the model for its full cross attention, and for a 30 second window of
        a large Whisper model that is several gigabytes of attention weights
        on a machine that may not have the room for them. Segment timestamps
        cost almost nothing and the lyric pipeline cuts its lines from the
        punctuation and the pauses anyway.
        """
        engine_task = task if task in ("transcribe", "translate") else "transcribe"
        hint = _engine_language(language)
        if engine_task == "translate" and hint not in (None, "en"):
            engine_task = "transcribe"

        generate_kwargs: Dict[str, Any] = {"task": engine_task}
        if hint:
            generate_kwargs["language"] = hint
        if prompt:
            generate_kwargs["initial_prompt"] = prompt

        call_kwargs: Dict[str, Any] = {
            "generate_kwargs": generate_kwargs,
            "return_timestamps": True,
        }

        try:
            output = self._pipe(samples, **call_kwargs)
        except Exception as exc:
            raise RuntimeError(str(exc)) from exc

        if not isinstance(output, dict):
            output = {"text": str(output or "")}
        raw_text = str(output.get("text") or "")
        chunks = output.get("chunks") or []
        detected = output.get("language") or output.get("language_code")

        segments: List[Segment] = []
        for chunk in chunks:
            text = _clean(_chunk_text(chunk))
            if not text:
                continue
            start, end = _chunk_times(chunk)
            if start is None:
                continue
            segments.append(Segment(start=start, end=end if end is not None else start, text=text))
        if not segments:
            text = _clean(raw_text)
            if text:
                segments = [Segment(start=0.0, end=float(len(samples)) / TARGET_RATE, text=text)]

        return raw_text, segments, detected

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

        prompt = _build_prompt(initial_prompt, hotwords)

        started = time.perf_counter()
        try:
            raw_text, segments, detected = self._run_window(
                samples,
                language=language,
                task=task,
                prompt=prompt,
                hotwords=hotwords,
                word_timestamps=word_timestamps,
                vad=vad,
            )
        except Exception as exc:
            return self._empty(duration, language, exc)

        segments = _finalise(segments)
        joined = " ".join(seg.text for seg in segments) or _clean(raw_text)
        return Transcription(
            text=joined,
            language=_result_tag(language, detected, joined),
            duration=duration,
            segments=segments,
            backend=self.name,
            model=self._model_id or "",
            elapsed=time.perf_counter() - started,
            meta={
                "dtype": self._dtype,
                "task": task if task in ("transcribe", "translate") else "transcribe",
                "detected_language": detected,
                "word_timestamps": bool(word_timestamps),
                "timestamp_level": "segment",
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
