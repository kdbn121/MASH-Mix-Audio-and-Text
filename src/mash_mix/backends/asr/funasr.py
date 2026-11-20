# SPDX-License-Identifier: MIT | Copyright (c) 2026 MASH (Khaled Bhuiyan)
"""FunASR speech to text backend.

FunASR carries the SenseVoice and Paraformer model families. SenseVoiceSmall is
the default here because it is fast, it reports the detected language itself
and it handles Bangla well.

FunASR emits character level timestamps, so the backend converts them into word
timings by walking the recognised string. That keeps the subtitle exporter
usable with FunASR even though the raw output is character based.
"""

from __future__ import annotations

import re
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
    from funasr import AutoModel  # type: ignore

    _HAS_FUNASR = True
except Exception:  # pragma: no cover - a broken install must not break the import
    AutoModel = None  # type: ignore
    _HAS_FUNASR = False

__all__ = ["FunASRSTTBackend"]

TARGET_RATE = 16000
DEFAULT_MODEL = "FunAudioLLM/SenseVoiceSmall"

_BANGLA_BASES = {"bn", "beng", "ben", "bang", "bangla", "bangali", "bengali"}

# SenseVoice wraps its output in tags such as <|bn|><|NEUTRAL|><|woitn|>.
_TAG_RE = re.compile(r"<\|[^|]*\|>")

_LANGUAGES = [
    "bn",
    "bn-BD",
    "bn-IN",
    "zh",
    "yue",
    "en",
    "ja",
    "ko",
    "ru",
    "de",
    "fr",
    "es",
    "pt",
    "it",
    "hi",
    "th",
    "vi",
    "id",
    "ms",
    "ar",
    "tr",
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
        raise AudioDecodeError("FunASR received no audio to transcribe.")
    arr = np.asarray(audio)
    if arr.ndim > 1:
        arr = arr.mean(axis=0) if arr.shape[0] <= 8 else arr.reshape(-1)
    arr = arr.astype(np.float32, copy=False).reshape(-1)
    if int(sample_rate) <= 0:
        raise AudioDecodeError("FunASR needs a positive sample rate.")
    if arr.size == 0:
        raise AudioDecodeError("FunASR cannot transcribe an empty audio block.")
    return _resample(arr, int(sample_rate))


def _clean(text: str) -> str:
    """Clean recognised text and apply the Bangla normalisation pass."""
    out = clean_asr_text(text or "")
    if out and is_bangla_text(out):
        out = normalize_bangla(out, digits="keep")
    return out.strip()


def _engine_language(language: Optional[str]) -> Optional[str]:
    """Map a requested tag onto a code FunASR understands."""
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
    """Fold Bangla hotwords into a prompt. FunASR has no prompt channel, so this
    is only used to seed the cache and is otherwise ignored."""
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


def _strip_tags(text: str) -> Tuple[str, Optional[str]]:
    """Remove SenseVoice control tags and read the language tag out of them."""
    raw = text or ""
    language: Optional[str] = None
    for match in _TAG_RE.findall(raw):
        tag = match.strip("<|>").strip().lower()
        if tag and tag not in (
            "neutral",
            "speech",
            "woitn",
            "woitn2",
            "woitn3",
            "emotion",
            "event",
            "applause",
            "music",
            "noise",
            "it",
            "bgm",
            "laugh",
            "cough",
            "grain",
            "static",
        ):
            language = tag
    return _TAG_RE.sub("", raw), language


def _words_from_timestamps(text: str, timestamps: Any) -> List[Word]:
    """Turn FunASR character timestamps into word timings."""
    tokens = text.split()
    if not tokens:
        return []
    pairs: List[Tuple[float, float]] = []
    if timestamps:
        for item in timestamps:
            try:
                pairs.append((float(item[0]) / 1000.0, float(item[1]) / 1000.0))
            except (TypeError, ValueError, IndexError):
                continue
    if not pairs:
        return []

    words: List[Word] = []
    if len(pairs) == len(text):
        # Character aligned, so the exact span of every token is known.
        offsets: List[Tuple[int, int]] = []
        cursor = 0
        for token in tokens:
            index = text.find(token, cursor)
            if index < 0:
                index = cursor
            offsets.append((index, index + len(token)))
            cursor = index + len(token)
        for token, (first, last) in zip(tokens, offsets):
            span = pairs[first:last] if last > first else pairs[first : first + 1]
            if not span:
                continue
            words.append(
                Word(
                    start=min(pair[0] for pair in span),
                    end=max(pair[1] for pair in span),
                    text=token,
                    probability=None,
                )
            )
        return words

    # Fall back to spreading the available timestamps over the tokens.
    total = len(tokens)
    for index, token in enumerate(tokens):
        first = int(index * len(pairs) / float(total))
        last = max(first, int((index + 1) * len(pairs) / float(total)) - 1)
        span = pairs[first : last + 1]
        if not span:
            continue
        words.append(
            Word(
                start=min(pair[0] for pair in span),
                end=max(pair[1] for pair in span),
                text=token,
                probability=None,
            )
        )
    return words


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


@register_stt("funasr")
class FunASRSTTBackend(STTBackend):
    """Speech to text with FunASR (SenseVoice and Paraformer families)."""

    info = BackendInfo(
        name="funasr",
        kind="stt",
        summary="FunASR SenseVoice and Paraformer models, fast with Bangla support.",
        package="funasr",
        extra="stt",
        languages=_LANGUAGES,
        platforms=_PLATFORMS,
        notes=(
            "Default model 'FunAudioLLM/SenseVoiceSmall', which is quick on CPU "
            "and reports the detected language. Use 'paraformer-zh' for the "
            "Paraformer family. Timestamps are character level and are mapped "
            "onto word timings here."
        ),
    )

    def __init__(self) -> None:
        super().__init__()
        self._model: Any = None

    # -- lifecycle ------------------------------------------------------
    def is_available(self) -> bool:
        """True when ``funasr`` can be imported."""
        try:
            return bool(_HAS_FUNASR and AutoModel is not None)
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
        """Load the FunASR model. Safe to call twice."""
        if self._loaded:
            return
        if not self.is_available():
            raise BackendUnavailableError(
                "The 'funasr' backend needs the funasr package. "
                "Install it with: pip install 'mash-mix-audio-and-text[stt]'"
            )
        model_name = str(model or kwargs.get("model_id") or DEFAULT_MODEL)
        device = str(kwargs.get("device") or self._pick_device())
        try:
            self._model = AutoModel(model=model_name, device=device, disable_update=True)
        except Exception:
            # SenseVoice checkpoints need remote code to be trusted.
            try:
                self._model = AutoModel(
                    model=model_name, device=device, disable_update=True, trust_remote_code=True
                )
            except Exception as exc:
                raise BackendUnavailableError(
                    f"Could not load FunASR model '{model_name}': {exc}"
                ) from exc
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

        hint = _engine_language(language)
        prompt = _build_prompt(initial_prompt, hotwords)
        cache: Dict[str, Any] = {}
        if prompt:
            cache["prompt"] = prompt

        generate_kwargs: Dict[str, Any] = {"input": samples, "cache": cache}
        if hint:
            generate_kwargs["language"] = hint
        if kwargs.get("batch_size_s") is not None:
            generate_kwargs["batch_size_s"] = kwargs["batch_size_s"]
        if kwargs.get("merge_vad") is not None:
            generate_kwargs["merge_vad"] = kwargs["merge_vad"]

        started = time.perf_counter()
        try:
            result = self._model.generate(**generate_kwargs)
        except TypeError:
            # Older builds do not accept a language argument.
            generate_kwargs.pop("language", None)
            try:
                result = self._model.generate(**generate_kwargs)
            except Exception as exc:
                return self._empty(duration, language, exc)
        except Exception as exc:
            return self._empty(duration, language, exc)

        raw_text, detected = _strip_tags(self._first_text(result))
        timestamps = self._first_timestamps(result)
        text = _clean(raw_text)

        words: List[Word] = []
        if word_timestamps and text:
            words = _words_from_timestamps(text, timestamps)

        segments: List[Segment] = []
        if text:
            if words:
                segments = [
                    Segment(start=words[0].start, end=words[-1].end, text=text, words=words)
                ]
            else:
                segments = [Segment(start=0.0, end=duration, text=text)]
        segments = _finalise(segments)

        return Transcription(
            text=text,
            language=_result_tag(language, detected, text),
            duration=duration,
            segments=segments,
            backend=self.name,
            model=self._model_id or "",
            elapsed=time.perf_counter() - started,
            meta={
                "task": task,
                "detected_language": detected,
                "word_timestamps": bool(word_timestamps),
                "vad": bool(vad),
            },
        )

    def _first_text(self, result: Any) -> str:
        """Read the text out of the FunASR result list."""
        if result is None:
            return ""
        if isinstance(result, dict):
            return str(result.get("text") or "")
        if isinstance(result, (list, tuple)):
            for item in result:
                text = self._first_text(item)
                if text:
                    return text
            return ""
        return str(getattr(result, "text", "") or "")

    def _first_timestamps(self, result: Any) -> Any:
        """Read the timestamp pairs out of the FunASR result list."""
        if result is None:
            return None
        if isinstance(result, dict):
            return result.get("timestamp") or result.get("timestamps")
        if isinstance(result, (list, tuple)):
            for item in result:
                stamps = self._first_timestamps(item)
                if stamps:
                    return stamps
            return None
        return getattr(result, "timestamp", None)

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
