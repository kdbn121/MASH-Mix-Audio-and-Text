# SPDX-License-Identifier: MIT | Copyright (c) 2026 MASH (Khaled Bhuiyan)
"""WhisperX speech to text backend.

WhisperX does three things that a plain recogniser does not. It aligns the
words of a transcript to the sound, so the time of every word is right rather
than roughly right. It can tell who is speaking, so a song with two voices and
a conversation with four of them both come back with the name of the speaker on
every line. And it runs the same Whisper weights through CTranslate2, so it is
quick on a machine with no graphics card.

The speaker labels are the reason this backend is here. Every other backend in
this package can hear a voice but cannot say whose voice it was, so the
``speaker`` field of a segment was always empty. This one fills it.

The speaker part needs a model from Hugging Face, so it needs a token in the
``HF_TOKEN`` environment variable. Without a token the words are still aligned
and the speakers are simply left empty. Nothing is sent anywhere: the model is
downloaded once and then it runs on this machine.
"""

from __future__ import annotations

import os
import tempfile
import time
import wave
from typing import Any, Dict, List, Optional, Sequence, Tuple

import numpy as np

from ...errors import AudioDecodeError, BackendUnavailableError
from ...lang.dialect import detect_region
from ...lang.normalize import clean_asr_text, is_bangla_text, normalize_bangla
from ...types import Segment, Transcription, Word
from ..base import BackendInfo, STTBackend
from ..registry import register_stt

__all__ = ["WhisperXBackend"]

TARGET_RATE = 16000
DEFAULT_MODEL = "small"

#: The environment variable that holds the token for the speaker model.
TOKEN_ENV = "HF_TOKEN"

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
    "nl",
    "tr",
    "pl",
    "id",
    "ms",
    "zh",
    "ja",
    "ko",
    "th",
    "vi",
    "fa",
    "he",
    "ta",
    "te",
    "mr",
    "gu",
    "pa",
    "sw",
    "el",
    "uk",
    "cs",
    "hu",
    "ka",
    "lv",
]

_PLATFORMS = ["windows", "linux", "macos"]


def _has_whisperx() -> bool:
    """True when the package can be imported."""
    try:
        import whisperx  # noqa: F401

        return True
    except Exception:  # pragma: no cover - optional dependency guard
        return False


def _token() -> Optional[str]:
    """The token for the speaker model, or nothing."""
    for name in (TOKEN_ENV, "HUGGING_FACE_HUB_TOKEN", "MASH_MIX_HF_TOKEN"):
        value = os.environ.get(name, "").strip()
        if value:
            return value
    return None


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


def _align_language(language: Optional[str]) -> Optional[str]:
    """The language code the aligner understands."""
    if not language:
        return None
    base, _region = _split_tag(language)
    if not base or base == "auto":
        return None
    return base


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


def _write_wav(samples: np.ndarray, sample_rate: int) -> str:
    """Write the audio to a wav file and return where it was written."""
    data = np.asarray(samples, dtype=np.float32).reshape(-1)
    clipped = np.clip(data, -1.0, 1.0)
    pcm = (clipped * 32767.0).astype("<i2")
    handle = tempfile.NamedTemporaryFile(prefix="mashmix-wx-", suffix=".wav", delete=False)
    path = handle.name
    handle.close()
    with wave.open(path, "wb") as out:
        out.setnchannels(1)
        out.setsampwidth(2)
        out.setframerate(int(sample_rate))
        out.writeframes(pcm.tobytes())
    return path


def _device_and_type() -> Tuple[str, str]:
    """The device and the number format to run the model in."""
    override = os.environ.get("MASH_MIX_DEVICE", "").strip().lower()
    device = "cpu"
    compute = "int8"
    if override in ("cpu", "cuda", "gpu"):
        device = "cpu" if override == "cpu" else "cuda"
    else:
        try:
            import torch  # type: ignore

            if bool(torch.cuda.is_available()):
                device = "cuda"
                compute = "float16"
        except Exception:  # pragma: no cover - torch is optional here
            pass
    return device, compute


def _to_segments(raw: List[Dict[str, Any]]) -> List[Segment]:
    """Turn the segments of WhisperX into the segments of this package."""
    out: List[Segment] = []
    for item in raw or []:
        if not isinstance(item, dict):
            continue
        text = _clean(str(item.get("text") or ""))
        if not text:
            continue
        try:
            start = float(item.get("start") or 0.0)
        except (TypeError, ValueError):
            start = 0.0
        try:
            end = float(item.get("end") or start)
        except (TypeError, ValueError):
            end = start
        words: List[Word] = []
        for word in item.get("words") or []:
            if not isinstance(word, dict):
                continue
            word_text = _clean(str(word.get("word") or word.get("text") or ""))
            if not word_text:
                continue
            try:
                word_start = float(word.get("start") or start)
            except (TypeError, ValueError):
                word_start = start
            try:
                word_end = float(word.get("end") or word_start)
            except (TypeError, ValueError):
                word_end = word_start
            speaker = word.get("speaker")
            words.append(
                Word(
                    start=word_start,
                    end=word_end,
                    text=word_text,
                    probability=_probability(word.get("score")),
                )
            )
        speaker = item.get("speaker")
        out.append(
            Segment(
                start=start,
                end=max(end, start),
                text=text,
                words=words,
                speaker=str(speaker) if speaker is not None else None,
            )
        )
    return out


def _probability(value: Any) -> Optional[float]:
    """The certainty of a word, when there is one."""
    if value is None:
        return None
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def _finalise(segments: List[Segment]) -> List[Segment]:
    """Trim, drop empties and enforce monotonic times."""
    out: List[Segment] = []
    for seg in segments:
        if not seg.text.strip():
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


@register_stt("whisperx")
class WhisperXBackend(STTBackend):
    """Speech to text with WhisperX: aligned words and speaker labels."""

    info = BackendInfo(
        name="whisperx",
        kind="stt",
        summary="Whisper with the words aligned to the sound and the speakers named.",
        package="whisperx",
        extra="align",
        languages=_LANGUAGES,
        platforms=_PLATFORMS,
        notes=(
            "Needs the whisperx package. The speaker labels need a model from "
            "Hugging Face, so put a token in the HF_TOKEN environment variable; "
            "without it the words are still aligned and the speakers are left "
            "empty. Everything runs on this machine."
        ),
    )

    def __init__(self) -> None:
        super().__init__()
        self._model: Any = None
        self._device = "cpu"
        self._compute_type = "int8"

    # -- lifecycle ------------------------------------------------------
    def is_available(self) -> bool:
        """True when ``whisperx`` can be imported."""
        try:
            return _has_whisperx()
        except Exception:  # pragma: no cover - defensive, must never raise
            return False

    def load(self, model: Optional[str] = None, **kwargs: Any) -> None:
        """Load the WhisperX model. Safe to call twice."""
        if self._loaded:
            return
        if not self.is_available():
            raise BackendUnavailableError(
                "The 'whisperx' backend needs the whisperx package. Install it "
                "with: pip install 'mash-mix-audio-and-text[align]'"
            )
        self._model_id = str(model or kwargs.get("model_id") or DEFAULT_MODEL)
        self._device, self._compute_type = _device_and_type()
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
        data = np.asarray(audio, dtype=np.float32).reshape(-1)
        if data.size == 0:
            raise AudioDecodeError("WhisperX received no audio to transcribe.")
        duration = float(data.size) / float(sample_rate or TARGET_RATE)

        try:
            import whisperx  # type: ignore
        except Exception as exc:
            return self._empty(duration, language, exc)

        path = ""
        started = time.perf_counter()
        try:
            path = _write_wav(data, int(sample_rate or TARGET_RATE))
            if self._model is None:
                self._model = whisperx.load_model(
                    self._model_id or DEFAULT_MODEL,
                    self._device,
                    compute_type=self._compute_type,
                )
            spoken = whisperx.load_audio(path)
            result = self._model.transcribe(spoken, language=_align_language(language))

            segments = _to_segments(result.get("segments") or [])
            aligned = False
            code = _align_language(language) or str(result.get("language") or "")
            if code and segments:
                try:
                    align_model, metadata = whisperx.load_align_model(
                        language_code=code,
                        device=self._device,
                    )
                    aligned_segments = whisperx.align(
                        result.get("segments") or [],
                        align_model,
                        metadata,
                        spoken,
                        self._device,
                    )
                    better = _to_segments(aligned_segments or [])
                    if better:
                        segments = better
                        aligned = True
                except Exception:
                    # The words are already timed, so a missing aligner is not
                    # a failure. The times are just less exact.
                    aligned = False

            speakers = False
            if bool(kwargs.get("diarize", True)) and _token():
                try:
                    pipeline = whisperx.DiarizationPipeline(
                        use_auth_token=_token(),
                        device=self._device,
                    )
                    diarized = pipeline(spoken)
                    with_speakers = whisperx.assign_word_speakers(
                        diarized,
                        {"segments": result.get("segments") or []},
                    )
                    better = _to_segments(with_speakers.get("segments") or [])
                    if better:
                        segments = better
                        speakers = True
                except Exception:
                    speakers = False

            segments = _finalise(segments)
            joined = " ".join(seg.text for seg in segments)
            detected = str(result.get("language") or "") or None
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
                    "compute_type": self._compute_type,
                    "detected_language": detected,
                    "word_timestamps": bool(word_timestamps),
                    "aligned": aligned,
                    "speakers": speakers,
                },
            )
        except Exception as exc:
            return self._empty(duration, language, exc)
        finally:
            if path:
                try:
                    os.unlink(path)
                except OSError:  # pragma: no cover - defensive
                    pass

    def _empty(self, duration: float, language: Optional[str], exc: Exception) -> Transcription:
        """Return an empty transcription instead of raising on a failure."""
        return Transcription(
            text="",
            language=_result_tag(language, None, ""),
            duration=duration,
            segments=[],
            backend=self.name,
            model=self._model_id or "",
            meta={"error": str(exc)},
        )
