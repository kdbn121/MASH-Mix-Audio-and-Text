# SPDX-License-Identifier: MIT | Copyright (c) 2026 MASH (Khaled Bhuiyan)
"""sherpa-onnx speech to text backend.

sherpa-onnx runs ONNX models through onnxruntime, which makes it the lightest
option in this package. It needs no Python deep learning stack at all, so it
is the practical choice for Android on Termux and for iOS builds.

There is no default model. The caller must point the backend at a model
directory with ``load(model_dir=...)`` or at a transducer with the encoder,
decoder and joiner files.

Bangla models such as the sherpa-onnx Bangla zipformer are supported and the
regional tag ``bn-BD`` or ``bn-IN`` is preserved on the result.
"""

from __future__ import annotations

import inspect
import os
import re
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
    import sherpa_onnx  # type: ignore

    _HAS_SHERPA_ONNX = True
except Exception:  # pragma: no cover - a broken install must not break the import
    sherpa_onnx = None  # type: ignore
    _HAS_SHERPA_ONNX = False

__all__ = ["SherpaOnnxSTTBackend"]

TARGET_RATE = 16000

_BANGLA_BASES = {"bn", "beng", "ben", "bang", "bangla", "bangali", "bengali"}

_LANGUAGES = [
    "bn",
    "bn-BD",
    "bn-IN",
    "en",
    "hi",
    "ur",
    "zh",
    "ja",
    "ko",
    "de",
    "fr",
    "es",
    "ru",
    "it",
    "ar",
    "pt",
    "nl",
    "tr",
]

_PLATFORMS = ["windows", "linux", "macos", "android", "ios"]

_TOKEN_RE = re.compile(r"(\s+)")


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
        raise AudioDecodeError("sherpa-onnx received no audio to transcribe.")
    arr = np.asarray(audio)
    if arr.ndim > 1:
        arr = arr.mean(axis=0) if arr.shape[0] <= 8 else arr.reshape(-1)
    arr = arr.astype(np.float32, copy=False).reshape(-1)
    if int(sample_rate) <= 0:
        raise AudioDecodeError("sherpa-onnx needs a positive sample rate.")
    if arr.size == 0:
        raise AudioDecodeError("sherpa-onnx cannot transcribe an empty audio block.")
    return _resample(arr, int(sample_rate))


def _clean(text: str) -> str:
    """Clean recognised text and apply the Bangla normalisation pass."""
    out = clean_asr_text(text or "")
    if out and is_bangla_text(out):
        out = normalize_bangla(out, digits="keep")
    return out.strip()


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


def _probability(value: Any) -> Optional[float]:
    try:
        if value is None:
            return None
        return float(value)
    except (TypeError, ValueError):
        return None


def _words_from_tokens(tokens: Sequence[str], timestamps: Sequence[float]) -> List[Word]:
    """Group sherpa tokens into words using the token timestamps."""
    words: List[Word] = []
    buffer: List[str] = []
    start: Optional[float] = None
    end: Optional[float] = None
    for index, token in enumerate(tokens):
        stamp = None
        if index < len(timestamps):
            try:
                stamp = float(timestamps[index])
            except (TypeError, ValueError):
                stamp = None
        for piece in _TOKEN_RE.split(str(token)):
            if not piece:
                continue
            if piece.isspace():
                if buffer:
                    words.append(
                        Word(
                            start=start if start is not None else 0.0,
                            end=end if end is not None else (start if start is not None else 0.0),
                            text="".join(buffer).strip(),
                            probability=None,
                        )
                    )
                    buffer = []
                    start = None
                    end = None
                continue
            if stamp is not None:
                if start is None:
                    start = stamp
                end = stamp
            buffer.append(piece)
    if buffer:
        words.append(
            Word(
                start=start if start is not None else 0.0,
                end=end if end is not None else (start if start is not None else 0.0),
                text="".join(buffer).strip(),
                probability=None,
            )
        )
    return [word for word in words if word.text]


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


@register_stt("sherpa-onnx")
class SherpaOnnxSTTBackend(STTBackend):
    """Speech to text with sherpa-onnx and onnxruntime."""

    info = BackendInfo(
        name="sherpa-onnx",
        kind="stt",
        summary="sherpa-onnx ONNX models, the lightest runtime in this package.",
        package="sherpa-onnx",
        extra="stt",
        languages=_LANGUAGES,
        platforms=_PLATFORMS,
        notes=(
            "No default model. Load with load(model_dir='...') for a zipformer "
            "or sense-voice model, or pass encoder, decoder and joiner paths. "
            "Bangla zipformer models are supported."
        ),
    )

    def __init__(self) -> None:
        super().__init__()
        self._recognizer: Any = None
        self._model_dir: str = ""

    # -- lifecycle ------------------------------------------------------
    def is_available(self) -> bool:
        """True when ``sherpa-onnx`` can be imported."""
        try:
            return bool(_HAS_SHERPA_ONNX and sherpa_onnx is not None)
        except Exception:  # pragma: no cover - defensive, must never raise
            return False

    def _build_recognizer(self, model_dir: str, kwargs: Dict[str, Any]) -> Any:
        """Try the known sherpa-onnx factories until one of them accepts the files."""
        factories = ["from_pretrained", "from_transducer", "from_sense_voice", "from_paraformer"]
        last_error: Optional[Exception] = None
        for factory_name in factories:
            factory = getattr(sherpa_onnx.OfflineRecognizer, factory_name, None)
            if factory is None:
                continue
            call_kwargs = dict(kwargs)
            call_kwargs["model_dir"] = model_dir
            try:
                signature = inspect.signature(factory)
                accepted = {
                    name: value
                    for name, value in call_kwargs.items()
                    if name in signature.parameters
                }
            except (TypeError, ValueError):  # pragma: no cover - builtins without signature
                accepted = call_kwargs
            try:
                return factory(**accepted)
            except Exception as exc:
                last_error = exc
        raise BackendUnavailableError(
            f"sherpa-onnx could not build a recognizer from '{model_dir}': {last_error}"
        )

    def load(self, model: Optional[str] = None, *, model_dir: Optional[str] = None, **kwargs: Any) -> None:
        """Load an ONNX model. Safe to call twice."""
        if self._loaded:
            return
        if not self.is_available():
            raise BackendUnavailableError(
                "The 'sherpa-onnx' backend needs the sherpa-onnx package. "
                "Install it with: pip install 'mash-mix-audio-and-text[stt]'"
            )
        directory = str(model_dir or model or kwargs.get("model_dir") or "").strip()
        if not directory:
            raise ModelNotFoundError(
                "The 'sherpa-onnx' backend has no default model. "
                "Call load(model_dir='<path to the ONNX model directory>')."
            )
        if not os.path.isdir(directory):
            raise ModelNotFoundError(
                f"The sherpa-onnx model directory '{directory}' does not exist."
            )

        build_kwargs: Dict[str, Any] = {
            "tokens": kwargs.get("tokens") or "",
            "encoder": kwargs.get("encoder") or "",
            "decoder": kwargs.get("decoder") or "",
            "joiner": kwargs.get("joiner") or "",
            "num_threads": int(kwargs.get("num_threads") or 2),
            "sample_rate": int(kwargs.get("sample_rate") or TARGET_RATE),
            "provider": str(kwargs.get("provider") or "cpu"),
            "debug": bool(kwargs.get("debug", False)),
        }
        self._recognizer = self._build_recognizer(directory, build_kwargs)
        self._model_dir = directory
        self._model_id = directory
        self._loaded = True

    def unload(self) -> None:
        """Drop the loaded recognizer."""
        self._recognizer = None
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

        started = time.perf_counter()
        try:
            stream = self._recognizer.create_stream()
            stream.accept_waveform(samples, int(sample_rate))
            self._recognizer.decode_stream(stream)
            result = stream.result
        except Exception as exc:
            return self._empty(duration, language, exc)

        sentences = result if isinstance(result, (list, tuple)) else [result]
        segments: List[Segment] = []
        for sentence in sentences:
            text = _clean(str(getattr(sentence, "text", "") or ""))
            if not text:
                continue
            tokens = list(getattr(sentence, "tokens", None) or [])
            timestamps = list(getattr(sentence, "timestamps", None) or [])
            words = _words_from_tokens(tokens, timestamps) if word_timestamps else []
            if words:
                start = words[0].start
                end = words[-1].end
            elif timestamps:
                try:
                    start = float(timestamps[0])
                    end = float(timestamps[-1])
                except (TypeError, ValueError, IndexError):
                    start, end = 0.0, duration
            else:
                start, end = 0.0, duration
            segments.append(Segment(start=start, end=end, text=text, words=words))

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
                "model_dir": self._model_dir,
                "task": task,
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
