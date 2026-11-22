# SPDX-License-Identifier: MIT | Copyright (c) 2026 MASH (Khaled Bhuiyan)
"""ElevenLabs Scribe speech to text backend.

Scribe is the hosted recogniser from ElevenLabs. It is the strongest
recogniser that can be pointed at a Bangla song without a download and
without the memory a large local model needs, so it is worth having next to
the local backends.

Two things are true about it and both matter:

* It needs an account. The key is read from the environment, never from a
  file inside the project, and the backend reports itself unavailable when
  there is no key. Nothing is ever sent to ElevenLabs without one.
* It sends the audio away from the machine. Every other backend here runs
  locally, so this one is opt in and says so plainly. The panel in the
  extension never reaches it on its own, it only asks the local engine,
  and the engine only calls ElevenLabs when the caller named this backend
  and a key is present.

The response carries word level timings, so the lyrics come back with the
same shape as the local backends and the karaoke timing keeps working.
"""

from __future__ import annotations

import io
import json
import os
import time
import urllib.error
import urllib.request
import wave
from typing import Any, Dict, List, Optional, Sequence, Tuple

import numpy as np

from ...errors import AudioDecodeError, BackendUnavailableError
from ...lang.dialect import detect_region
from ...lang.normalize import clean_asr_text, is_bangla_text, normalize_bangla
from ...types import Segment, Transcription, Word
from ..base import BackendInfo, STTBackend
from ..registry import register_stt

__all__ = ["ElevenLabsSTTBackend"]

TARGET_RATE = 16000

#: The endpoint that takes the audio.
ENDPOINT = "https://api.elevenlabs.io/v1/speech-to-text"

#: The model to ask for. Scribe v3 is the most accurate of the Scribe models.
DEFAULT_MODEL = "scribe_v3"

#: The environment variable that holds the key.
API_KEY_ENV = "ELEVENLABS_API_KEY"

#: The language codes ElevenLabs knows for Bangla.
_BANGLA_CODES = ("ben", "bn", "beng")

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
    "ne",
    "si",
    "th",
    "km",
    "lo",
    "my",
]

_PLATFORMS = ["windows", "linux", "macos"]


def _api_key() -> Optional[str]:
    """The key from the environment, or nothing."""
    for name in (API_KEY_ENV, "MASH_MIX_ELEVENLABS_API_KEY", "XI_API_KEY"):
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


def _engine_language(language: Optional[str]) -> Optional[str]:
    """Map a requested tag onto a code ElevenLabs understands."""
    if not language:
        return None
    base, _region = _split_tag(language)
    if base in _BANGLA_BASES:
        return "ben"
    if base == "zh":
        return "zho"
    if base == "he":
        return "heb"
    if base == "ja":
        return "jpn"
    if base == "ko":
        return "kor"
    if base == "hi":
        return "hin"
    if base == "ur":
        return "urd"
    if base == "ta":
        return "tam"
    if base == "te":
        return "tel"
    if base == "mr":
        return "mar"
    if base == "gu":
        return "guj"
    if base == "pa":
        return "pan"
    if base == "ne":
        return "nep"
    if base == "si":
        return "sin"
    if base == "th":
        return "tha"
    if base == "km":
        return "khm"
    if base == "lo":
        return "lao"
    if base == "my":
        return "mya"
    if base == "ms":
        return "msa"
    return language


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


def _wav_bytes(samples: np.ndarray, sample_rate: int) -> bytes:
    """Wrap mono float32 samples in a wav container."""
    data = np.asarray(samples, dtype=np.float32).reshape(-1)
    clipped = np.clip(data, -1.0, 1.0)
    pcm = (clipped * 32767.0).astype("<i2")
    buffer = io.BytesIO()
    with wave.open(buffer, "wb") as handle:
        handle.setnchannels(1)
        handle.setsampwidth(2)
        handle.setframerate(int(sample_rate))
        handle.writeframes(pcm.tobytes())
    return buffer.getvalue()


def _multipart(
    fields: Sequence[Tuple[str, str]],
    filename: str,
    content_type: str,
    payload: bytes,
) -> Tuple[bytes, str]:
    """Build a multipart body with one file and some plain fields."""
    boundary = "mashmixboundary7f3a9c1e"
    parts: List[bytes] = []
    for name, value in fields:
        parts.append(f"--{boundary}\r\n".encode())
        parts.append(f'Content-Disposition: form-data; name="{name}"\r\n\r\n'.encode())
        parts.append(f"{value}\r\n".encode())
    parts.append(f"--{boundary}\r\n".encode())
    parts.append(
        f'Content-Disposition: form-data; name="file"; filename="{filename}"\r\n'.encode()
    )
    parts.append(f"Content-Type: {content_type}\r\n\r\n".encode())
    parts.append(payload)
    parts.append(f"\r\n--{boundary}--\r\n".encode())
    return b"".join(parts), f"multipart/form-data; boundary={boundary}"


def _parse_words(payload: Dict[str, Any]) -> List[Word]:
    """Read the word timings out of the response."""
    words: List[Word] = []
    for item in payload.get("words") or []:
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
        logprob = item.get("logprob")
        probability: Optional[float] = None
        if isinstance(logprob, (int, float)):
            try:
                probability = float(np.exp(float(logprob)))
            except (OverflowError, ValueError):  # pragma: no cover - defensive
                probability = None
        words.append(Word(start=start, end=end, text=text, probability=probability))
    return words


def _group_words(words: List[Word], gap: float = 0.6) -> List[Segment]:
    """Group word timings into readable segments."""
    segments: List[Segment] = []
    group: List[Word] = []
    for word in words:
        if group:
            previous = group[-1]
            if word.start - previous.end > gap:
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
    """Trim, drop empties and enforce monotonic times."""
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


@register_stt("elevenlabs")
class ElevenLabsSTTBackend(STTBackend):
    """Speech to text with the hosted ElevenLabs Scribe model."""

    info = BackendInfo(
        name="elevenlabs",
        kind="stt",
        summary="Hosted ElevenLabs Scribe, the strongest hosted Bangla recogniser.",
        package="none, the standard library is enough",
        extra="",
        languages=_LANGUAGES,
        platforms=_PLATFORMS,
        notes=(
            "Needs an ElevenLabs account and the key in the ELEVENLABS_API_KEY "
            "environment variable. The audio is sent to ElevenLabs, so this "
            "backend is opt in. Word level timings come back with the result."
        ),
        missing="Put the key in the ELEVENLABS_API_KEY environment variable.",
    )

    def __init__(self) -> None:
        super().__init__()
        self._model_id = DEFAULT_MODEL

    # -- lifecycle ------------------------------------------------------
    def is_available(self) -> bool:
        """True when a key is present.

        The backend needs no package, so the key is the only thing that can
        be missing. Without one the backend stays out of the way.
        """
        try:
            return bool(_api_key())
        except Exception:  # pragma: no cover - defensive, must never raise
            return False

    def load(self, model: Optional[str] = None, **kwargs: Any) -> None:
        """Check the key and note the model to ask for."""
        if self._loaded:
            return
        if not self.is_available():
            raise BackendUnavailableError(
                "The 'elevenlabs' backend needs an ElevenLabs key. Put it in the "
                "ELEVENLABS_API_KEY environment variable and start the engine again."
            )
        self._model_id = str(model or DEFAULT_MODEL)
        self._loaded = True

    def unload(self) -> None:
        """Nothing is held in memory, so there is nothing to drop."""
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
        key = _api_key()
        if not key:
            return self._empty(
                float(len(audio)) / float(sample_rate or 1),
                language,
                RuntimeError("No ElevenLabs key is present."),
            )

        data = np.asarray(audio, dtype=np.float32).reshape(-1)
        if data.size == 0:
            raise AudioDecodeError("ElevenLabs received no audio to transcribe.")
        duration = float(data.size) / float(sample_rate or TARGET_RATE)

        fields: List[Tuple[str, str]] = [
            ("model_id", self._model_id),
            ("timestamps_granularity", "word" if word_timestamps else "none"),
            ("tag_audio_events", "false"),
        ]
        code = _engine_language(language)
        if code:
            fields.append(("language_code", code))
        if hotwords:
            words = [str(w).strip() for w in hotwords if str(w).strip()][:1000]
            if words:
                fields.append(("keyterms", " ".join(words)))

        body, content_type = _multipart(
            fields,
            "audio.wav",
            "audio/wav",
            _wav_bytes(data, int(sample_rate or TARGET_RATE)),
        )

        request = urllib.request.Request(ENDPOINT, data=body, method="POST")
        request.add_header("xi-api-key", key)
        request.add_header("Content-Type", content_type)
        request.add_header("Accept", "application/json")

        started = time.perf_counter()
        try:
            with urllib.request.urlopen(request, timeout=300) as response:
                raw = response.read().decode("utf-8")
            payload = json.loads(raw)
        except urllib.error.HTTPError as exc:
            detail = ""
            try:
                detail = exc.read().decode("utf-8")[:400]
            except Exception:  # pragma: no cover - defensive
                detail = ""
            return self._empty(duration, language, RuntimeError(f"ElevenLabs said {exc.code}. {detail}"))
        except Exception as exc:
            return self._empty(duration, language, exc)

        text = _clean(str(payload.get("text") or ""))
        words = _parse_words(payload)
        segments = _finalise(_group_words(words)) if words else []
        if not segments and text:
            segments = [Segment(start=0.0, end=duration, text=text)]
        detected = str(payload.get("language_code") or "") or None

        return Transcription(
            text=text or " ".join(seg.text for seg in segments),
            language=_result_tag(language, detected, text),
            duration=duration,
            segments=segments,
            backend=self.name,
            model=self._model_id,
            elapsed=time.perf_counter() - started,
            meta={
                "cloud": True,
                "provider": "elevenlabs",
                "detected_language": detected,
                "language_probability": payload.get("language_probability"),
                "word_timestamps": bool(word_timestamps),
            },
        )

    def _empty(self, duration: float, language: Optional[str], exc: Exception) -> Transcription:
        """Return an empty transcription instead of raising on a failure."""
        return Transcription(
            text="",
            language=_result_tag(language, None, ""),
            duration=duration,
            segments=[],
            backend=self.name,
            model=self._model_id,
            meta={"error": str(exc), "cloud": True, "provider": "elevenlabs"},
        )
