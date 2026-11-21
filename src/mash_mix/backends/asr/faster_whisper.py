# SPDX-License-Identifier: MIT | Copyright (c) 2026 MASH (Khaled Bhuiyan)
"""faster-whisper speech to text backend.

``faster-whisper`` is the recommended default engine for this package. It runs
the same Whisper weights through CTranslate2, so it is several times faster on
CPU and it supports int8 quantisation, which makes the ``small`` model usable
on a laptop and the ``large-v3`` model usable on a phone.

Bangla is a first class language here. A request for ``bn-BD`` or ``bn-IN`` is
handed to the model as ``bn`` and the regional tag is restored on the returned
:class:`~mash_mix.types.Transcription`.
"""

from __future__ import annotations

import glob
import inspect
import os
import shutil
import sys
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
    from faster_whisper import WhisperModel  # type: ignore

    _HAS_FASTER_WHISPER = True
except Exception:  # pragma: no cover - a broken install must not break the import
    WhisperModel = None  # type: ignore
    _HAS_FASTER_WHISPER = False

__all__ = ["FasterWhisperSTTBackend"]

TARGET_RATE = 16000
DEFAULT_MODEL = "small"

# The CUDA runtime libraries ctranslate2 links against. A machine can have an
# NVIDIA card without these, which is why they are checked before the GPU is
# chosen.
_CUDA_LIBRARIES = (
    "cublas64_12.dll",
    "cublas64_11.dll",
    "cudnn64_9.dll",
    "cudnn64_8.dll",
    "cudart64_12.dll",
    "cudart64_11.dll",
)

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
        raise AudioDecodeError("faster-whisper received no audio to transcribe.")
    arr = np.asarray(audio)
    if arr.ndim > 1:
        arr = arr.mean(axis=0) if arr.shape[0] <= 8 else arr.reshape(-1)
    arr = arr.astype(np.float32, copy=False).reshape(-1)
    if int(sample_rate) <= 0:
        raise AudioDecodeError("faster-whisper needs a positive sample rate.")
    if arr.size == 0:
        raise AudioDecodeError("faster-whisper cannot transcribe an empty audio block.")
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


@register_stt("faster-whisper")
class FasterWhisperSTTBackend(STTBackend):
    """Speech to text with faster-whisper (CTranslate2 Whisper)."""

    info = BackendInfo(
        name="faster-whisper",
        kind="stt",
        summary="CTranslate2 Whisper, fast on CPU with int8 quantisation.",
        package="faster-whisper",
        extra="stt",
        languages=_LANGUAGES,
        platforms=_PLATFORMS,
        notes=(
            "Recommended default engine. Default model 'small' with int8 on CPU. "
            "Use 'large-v3' for the best Bangla accuracy. VAD filtering is on by "
            "default and hotwords are supported by newer releases."
        ),
    )

    def __init__(self) -> None:
        super().__init__()
        self._model: Any = None
        self._compute_type: str = "int8"

    # -- lifecycle ------------------------------------------------------
    def is_available(self) -> bool:
        """True when ``faster-whisper`` can be imported."""
        try:
            return bool(_HAS_FASTER_WHISPER and WhisperModel is not None)
        except Exception:  # pragma: no cover - defensive, must never raise
            return False

    def _cuda_runtime_present(self) -> bool:
        """True when the CUDA runtime that ctranslate2 needs is actually installed.

        A machine can have an NVIDIA card and still lack the CUDA libraries
        that ctranslate2 links against. In that case the model loads fine and
        the failure only appears at inference time, as
        ``Library cublas64_12.dll is not found``. Checking the libraries up
        front turns a crash into a plain CPU run.
        """
        override = os.environ.get("MASH_MIX_DEVICE", "").strip().lower()
        if override in ("cpu", "cuda", "gpu", "auto"):
            return override in ("cuda", "gpu", "auto")
        try:
            import ctranslate2  # type: ignore

            if int(ctranslate2.get_cuda_device_count()) < 1:
                return False
        except Exception:  # pragma: no cover - ctranslate2 is optional here
            return False

        for name in _CUDA_LIBRARIES:
            if shutil.which(name):
                return True

        roots: List[str] = []
        try:
            import ctranslate2  # type: ignore

            roots.append(os.path.dirname(ctranslate2.__file__))
        except Exception:  # pragma: no cover - defensive
            pass
        roots.append(sys.prefix)
        roots.append(os.path.dirname(os.path.dirname(os.__file__)))

        patterns = (
            os.path.join("nvidia", "*", "bin"),
            os.path.join("nvidia", "*", "lib"),
            os.path.join("Lib", "site-packages", "nvidia", "*", "bin"),
            os.path.join("lib", "site-packages", "nvidia", "*", "lib"),
        )
        for root in roots:
            for pattern in patterns:
                for folder in glob.glob(os.path.join(root, pattern)):
                    try:
                        names = {item.lower() for item in os.listdir(folder)}
                    except OSError:
                        continue
                    if names & {item.lower() for item in _CUDA_LIBRARIES}:
                        return True
        return False

    def _pick_device(self) -> str:
        """Choose the device. The CPU is used unless the CUDA runtime is there."""
        override = os.environ.get("MASH_MIX_DEVICE", "").strip().lower()
        if override in ("cpu", "cuda", "gpu"):
            return "cpu" if override == "cpu" else "auto"
        return "auto" if self._cuda_runtime_present() else "cpu"

    def _pick_compute_type(self) -> str:
        """Prefer float16 on a working GPU and int8 on the CPU."""
        if self._cuda_runtime_present():
            try:
                import ctranslate2  # type: ignore

                if int(ctranslate2.get_cuda_device_count()) > 0:
                    return "float16"
            except Exception:  # pragma: no cover - ctranslate2 is optional here
                pass
            try:
                import torch  # type: ignore

                if bool(torch.cuda.is_available()):
                    return "float16"
            except Exception:  # pragma: no cover - torch is optional here
                pass
        return "int8"

    def load(self, model: Optional[str] = None, **kwargs: Any) -> None:
        """Load the faster-whisper model. Safe to call twice."""
        if self._loaded:
            return
        if not self.is_available():
            raise BackendUnavailableError(
                "The 'faster-whisper' backend needs the faster-whisper package. "
                "Install it with: pip install 'mash-mix-audio-and-text[stt]'"
            )
        model_name = str(model or kwargs.get("model_id") or DEFAULT_MODEL)
        compute_type = str(kwargs.get("compute_type") or self._pick_compute_type())
        device = str(kwargs.get("device") or self._pick_device())
        try:
            self._model = WhisperModel(
                model_name,
                device=device,
                compute_type=compute_type,
            )
        except Exception as exc:
            raise BackendUnavailableError(
                f"Could not load faster-whisper model '{model_name}' ({compute_type}): {exc}"
            ) from exc
        self._compute_type = compute_type
        self._device = device
        self._model_id = model_name
        self._loaded = True

    def unload(self) -> None:
        """Drop the loaded weights."""
        self._model = None
        self._loaded = False

    def _supports_hotwords(self) -> bool:
        """True when the installed faster-whisper can take a hotwords argument."""
        try:
            signature = inspect.signature(self._model.transcribe)
            if "hotwords" in signature.parameters:
                return True
            for parameter in signature.parameters.values():
                if parameter.kind is inspect.Parameter.VAR_KEYWORD:
                    return True
        except Exception:  # pragma: no cover - defensive
            pass
        return False

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
            engine_task = "transcribe"
        prompt = _build_prompt(initial_prompt, hotwords)

        call_kwargs: Dict[str, Any] = {
            "language": hint,
            "task": engine_task,
            "word_timestamps": bool(word_timestamps),
            "vad_filter": bool(vad),
            "initial_prompt": prompt,
        }
        if hotwords and self._supports_hotwords():
            call_kwargs["hotwords"] = " ".join(str(w).strip() for w in hotwords if str(w).strip())
        for key in ("beam_size", "condition_on_previous_text", "temperature"):
            if key in kwargs and kwargs[key] is not None:
                call_kwargs[key] = kwargs[key]

        started = time.perf_counter()
        try:
            raw_segments, info = self._model.transcribe(samples, **call_kwargs)
        except TypeError:
            # Older builds do not accept hotwords, so try again without them.
            if "hotwords" not in call_kwargs:
                return self._empty(duration, language, TypeError("transcribe call was rejected"))
            call_kwargs.pop("hotwords", None)
            try:
                raw_segments, info = self._model.transcribe(samples, **call_kwargs)
            except Exception as exc:
                return self._empty(duration, language, exc)
        except Exception as exc:
            if self._fall_back_to_cpu(exc):
                try:
                    raw_segments, info = self._model.transcribe(samples, **call_kwargs)
                except Exception as retry_exc:
                    return self._empty(duration, language, retry_exc)
            else:
                return self._empty(duration, language, exc)

        segments: List[Segment] = []
        try:
            for raw in raw_segments:
                text = _clean(str(getattr(raw, "text", "") or ""))
                if not text:
                    continue
                start = float(getattr(raw, "start", 0.0) or 0.0)
                end = float(getattr(raw, "end", 0.0) or 0.0)
                words: List[Word] = []
                if word_timestamps:
                    for raw_word in getattr(raw, "words", None) or []:
                        word_text = _clean(
                            str(getattr(raw_word, "word", None) or getattr(raw_word, "text", "") or "")
                        )
                        if not word_text:
                            continue
                        words.append(
                            Word(
                                start=float(getattr(raw_word, "start", start) or start),
                                end=float(getattr(raw_word, "end", end) or end),
                                text=word_text,
                                probability=_probability(getattr(raw_word, "probability", None)),
                            )
                        )
                segments.append(Segment(start=start, end=end, text=text, words=words))
        except Exception as exc:
            return self._empty(duration, language, exc)

        segments = _finalise(segments)
        joined = " ".join(seg.text for seg in segments)
        detected = getattr(info, "language", None)
        language_probability = _probability(getattr(info, "language_probability", None))
        return Transcription(
            text=joined,
            language=_result_tag(language, detected, joined),
            duration=duration,
            segments=segments,
            backend=self.name,
            model=self._model_id or "",
            elapsed=time.perf_counter() - started,
            meta={
                "compute_type": self._compute_type,
                "task": engine_task,
                "detected_language": detected,
                "language_probability": language_probability,
                "word_timestamps": bool(word_timestamps),
                "vad": bool(vad),
            },
        )

    def detect_language(
        self,
        audio: np.ndarray,
        sample_rate: int,
        *,
        max_seconds: float = 30.0,
    ) -> Optional[Tuple[str, float]]:
        """Work out the language of the audio with the model's own detector.

        The model reads the first seconds of the sound and scores every
        language it knows, which is much cheaper than reading the whole track
        and much better than guessing from the title.
        """
        if not self._loaded:
            self.load()
        samples = _prepare_audio(audio, sample_rate)
        limit = int(max(0.0, float(max_seconds)) * float(sample_rate))
        if limit:
            samples = samples[:limit]
        if samples.size == 0:
            return None
        try:
            code, probability, _scores = self._model.detect_language(audio=samples)
        except Exception:
            return None
        if not code:
            return None
        base = _split_tag(str(code))[0]
        return base, float(probability)

    def _fall_back_to_cpu(self, exc: Exception) -> bool:
        """Reload on the CPU when the GPU turns out to be unusable.

        The CUDA libraries can be missing on a machine that reports an NVIDIA
        card, and the failure only shows up at inference time. Rather than
        losing the transcription, the model is reloaded on the CPU.
        """
        message = str(exc).lower()
        cuda_markers = ("cuda", "cublas", "cudnn", "cudart", "gpu")
        if not any(marker in message for marker in cuda_markers):
            return False
        try:
            model_name = self._model_id or DEFAULT_MODEL
            self._model = WhisperModel(model_name, device="cpu", compute_type="int8")
            self._compute_type = "int8"
            self._device = "cpu"
            return True
        except Exception:  # pragma: no cover - defensive
            return False

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
