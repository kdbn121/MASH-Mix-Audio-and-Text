# SPDX-License-Identifier: MIT | Copyright (c) 2026 MASH (Khaled Bhuiyan)
"""Azure AI Speech Fast Transcription backend.

This is the hosted recogniser of Microsoft Azure. It is worth having beside the
local backends because it hears many languages well without a download and
without the memory a large model needs, and because it can tell who is talking
without a Hugging Face token.

Two things are true about it and both matter:

* It needs an account. The key is read from the environment, never from a file
  inside the project, and the backend reports itself unavailable when there is
  no key. Nothing is ever sent to Azure without one.
* It sends the audio away from the machine. Every other backend here runs
  locally, so this one is opt in and says so plainly. The panel in the
  extension never reaches it on its own, it only asks the local engine, and the
  engine only calls Azure when the caller named this backend and a key is
  present.

The response carries word level timings and the speaker of every phrase, so the
lyrics come back with the same shape as the local backends and the karaoke
timing keeps working.
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

__all__ = ["AzureSTTBackend"]

TARGET_RATE = 16000

#: The version of the Fast Transcription route that is called.
API_VERSION = "2024-11-15"

#: The environment variable that holds the key.
API_KEY_ENV = "AZURE_SPEECH_KEY"

#: The environment variable that holds the region.
REGION_ENV = "AZURE_SPEECH_REGION"

#: The environment variable that holds a whole endpoint, when there is one.
ENDPOINT_ENV = "AZURE_SPEECH_ENDPOINT"

#: The locales that are offered when the caller does not name a language.
#:
#: Bangla is first, because this package is built for Bangla, and English
#: comes next because it is the most common language in a song.
DEFAULT_LOCALES = [
    "bn-IN",
    "bn-BD",
    "en-US",
    "hi-IN",
    "ur-PK",
    "es-ES",
    "fr-FR",
    "de-DE",
    "it-IT",
    "pt-BR",
    "ru-RU",
    "ar-SA",
    "zh-CN",
    "ja-JP",
    "ko-KR",
    "tr-TR",
    "id-ID",
    "ms-MY",
    "nl-NL",
    "pl-PL",
    "th-TH",
    "vi-VN",
]

#: The shortest code to a locale, so a caller can pass "bn" or "en".
_ISO_TO_LOCALE = {
    "bn": "bn-IN",
    "bn-in": "bn-IN",
    "bn-bd": "bn-BD",
    "en": "en-US",
    "hi": "hi-IN",
    "ur": "ur-PK",
    "ar": "ar-SA",
    "es": "es-ES",
    "fr": "fr-FR",
    "de": "de-DE",
    "it": "it-IT",
    "pt": "pt-BR",
    "ru": "ru-RU",
    "nl": "nl-NL",
    "tr": "tr-TR",
    "pl": "pl-PL",
    "id": "id-ID",
    "ms": "ms-MY",
    "zh": "zh-CN",
    "ja": "ja-JP",
    "ko": "ko-KR",
    "th": "th-TH",
    "vi": "vi-VN",
    "fa": "fa-IR",
    "he": "he-IL",
    "ta": "ta-IN",
    "te": "te-IN",
    "mr": "mr-IN",
    "gu": "gu-IN",
    "pa": "pa-IN",
    "sw": "sw-KE",
    "el": "el-GR",
    "uk": "uk-UA",
    "cs": "cs-CZ",
    "hu": "hu-HU",
    "ka": "ka-GE",
    "lv": "lv-LV",
}

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


def _api_key() -> Optional[str]:
    """The key from the environment, or nothing."""
    for name in (API_KEY_ENV, "MASH_MIX_AZURE_SPEECH_KEY"):
        value = os.environ.get(name, "").strip()
        if value:
            return value
    return None


def _region() -> str:
    """The region from the environment, or nothing."""
    for name in (REGION_ENV, "MASH_MIX_AZURE_SPEECH_REGION"):
        value = os.environ.get(name, "").strip()
        if value:
            return value
    return ""


def _endpoint() -> str:
    """The address the request goes to."""
    override = os.environ.get(ENDPOINT_ENV, "").strip()
    if override:
        return override.rstrip("/")
    region = _region()
    if not region:
        return ""
    return f"https://{region}.api.cognitive.microsoft.com"


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


def _locale_for(language: Optional[str]) -> Optional[str]:
    """The locale to ask Azure for, from the tag the caller gave."""
    if not language:
        return None
    base, region = _split_tag(language)
    if not base or base == "auto":
        return None
    if region:
        return f"{base}-{region}"
    return _ISO_TO_LOCALE.get(base, f"{base}-US")


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
    boundary = "mashmixazure9d4b7e2"
    parts: List[bytes] = []
    for name, value in fields:
        parts.append(f"--{boundary}\r\n".encode())
        parts.append(f'Content-Disposition: form-data; name="{name}"\r\n\r\n'.encode())
        parts.append(f"{value}\r\n".encode())
    parts.append(f"--{boundary}\r\n".encode())
    parts.append(
        f'Content-Disposition: form-data; name="audio"; filename="{filename}"\r\n'.encode()
    )
    parts.append(f"Content-Type: {content_type}\r\n\r\n".encode())
    parts.append(payload)
    parts.append(f"\r\n--{boundary}--\r\n".encode())
    return b"".join(parts), f"multipart/form-data; boundary={boundary}"


def _parse_words(phrase: Dict[str, Any]) -> List[Word]:
    """The word timings of one phrase."""
    words: List[Word] = []
    for item in phrase.get("words") or []:
        if not isinstance(item, dict):
            continue
        text = _clean(str(item.get("text") or ""))
        if not text:
            continue
        try:
            start = float(item.get("offsetMilliseconds") or 0) / 1000.0
        except (TypeError, ValueError):
            start = 0.0
        try:
            end = float(item.get("durationMilliseconds") or 0) / 1000.0
        except (TypeError, ValueError):
            end = start
        words.append(Word(start=start, end=end + start, text=text))
    return words


def _finalise(segments: List[Segment]) -> List[Segment]:
    """Trim, drop empties and enforce monotonic times."""
    out: List[Segment] = []
    for seg in segments:
        text = seg.text.strip()
        if not text:
            continue
        if out and out[-1].text == text and out[-1].speaker == seg.speaker:
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


@register_stt("azure-stt")
class AzureSTTBackend(STTBackend):
    """Speech to text with the hosted Azure AI Speech Fast Transcription route."""

    info = BackendInfo(
        name="azure-stt",
        kind="stt",
        summary="Hosted Azure AI Speech, word timings and speaker labels without a token.",
        package="none, the standard library is enough",
        extra="",
        languages=_LANGUAGES,
        platforms=_PLATFORMS,
        notes=(
            "Needs an Azure Speech resource and the key in the AZURE_SPEECH_KEY "
            "environment variable with the region in AZURE_SPEECH_REGION. The "
            "audio is sent to Azure, so this backend is opt in. Word level "
            "timings and the speaker of every phrase come back with the result."
        ),
        missing=(
            "Put the key in the AZURE_SPEECH_KEY environment variable and the "
            "region in AZURE_SPEECH_REGION."
        ),
    )

    def __init__(self) -> None:
        super().__init__()
        self._model_id = "azure-fast-transcription"

    # -- lifecycle ------------------------------------------------------
    def is_available(self) -> bool:
        """True when a key and a place to send the request are both present."""
        try:
            return bool(_api_key()) and bool(_endpoint())
        except Exception:  # pragma: no cover - defensive, must never raise
            return False

    def load(self, model: Optional[str] = None, **kwargs: Any) -> None:
        """Check the key and note the model to ask for."""
        if self._loaded:
            return
        if not self.is_available():
            raise BackendUnavailableError(
                "The 'azure-stt' backend needs an Azure Speech key. Put it in the "
                "AZURE_SPEECH_KEY environment variable, the region in "
                "AZURE_SPEECH_REGION, and start the engine again."
            )
        self._model_id = str(model or "azure-fast-transcription")
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
        endpoint = _endpoint()
        if not key or not endpoint:
            return self._empty(
                float(len(audio)) / float(sample_rate or 1),
                language,
                RuntimeError("No Azure Speech key or region is present."),
            )

        data = np.asarray(audio, dtype=np.float32).reshape(-1)
        if data.size == 0:
            raise AudioDecodeError("Azure Speech received no audio to transcribe.")
        duration = float(data.size) / float(sample_rate or TARGET_RATE)

        locale = _locale_for(language)
        locales = [locale] if locale else list(DEFAULT_LOCALES)

        definition: Dict[str, Any] = {
            "locales": locales,
            "profanityFilterMode": str(kwargs.get("profanity_filter", "Masked")),
        }
        if kwargs.get("diarize"):
            definition["diarization"] = {
                "maxSpeakers": int(kwargs.get("max_speakers", 4) or 4),
                "enabled": True,
            }

        body, content_type = _multipart(
            [("definition", json.dumps(definition))],
            "audio.wav",
            "audio/wav",
            _wav_bytes(data, int(sample_rate or TARGET_RATE)),
        )

        url = f"{endpoint}/speechtotext/transcriptions:transcribe?api-version={API_VERSION}"
        request = urllib.request.Request(url, data=body, method="POST")
        request.add_header("Ocp-Apim-Subscription-Key", key)
        request.add_header("Content-Type", content_type)
        request.add_header("Accept", "application/json")

        started = time.perf_counter()
        try:
            with urllib.request.urlopen(request, timeout=600) as response:
                raw = response.read().decode("utf-8")
            payload = json.loads(raw)
        except urllib.error.HTTPError as exc:
            detail = ""
            try:
                detail = exc.read().decode("utf-8")[:400]
            except Exception:  # pragma: no cover - defensive
                detail = ""
            return self._empty(
                duration,
                language,
                RuntimeError(f"Azure Speech said {exc.code}. {detail}"),
            )
        except Exception as exc:
            return self._empty(duration, language, exc)

        segments: List[Segment] = []
        detected: Optional[str] = None
        for phrase in payload.get("phrases") or []:
            if not isinstance(phrase, dict):
                continue
            text = _clean(str(phrase.get("text") or ""))
            if not text:
                continue
            if detected is None and phrase.get("locale"):
                detected = str(phrase.get("locale"))
            try:
                start = float(phrase.get("offsetMilliseconds") or 0) / 1000.0
            except (TypeError, ValueError):
                start = 0.0
            try:
                length = float(phrase.get("durationMilliseconds") or 0) / 1000.0
            except (TypeError, ValueError):
                length = 0.0
            speaker = phrase.get("speaker")
            segments.append(
                Segment(
                    start=start,
                    end=start + max(0.0, length),
                    text=text,
                    words=_parse_words(phrase) if word_timestamps else [],
                    speaker=str(speaker) if speaker is not None else None,
                )
            )

        segments = _finalise(segments)
        joined = " ".join(seg.text for seg in segments)
        if not joined:
            try:
                reported = float(payload.get("durationMilliseconds") or 0) / 1000.0
            except (TypeError, ValueError):
                reported = duration
            return Transcription(
                text="",
                language=_result_tag(language, detected, ""),
                duration=reported or duration,
                segments=[],
                backend=self.name,
                model=self._model_id,
                elapsed=time.perf_counter() - started,
                meta={
                    "cloud": True,
                    "provider": "azure",
                    "detected_language": detected,
                    "word_timestamps": bool(word_timestamps),
                    "diarization": bool(kwargs.get("diarize")),
                },
            )

        return Transcription(
            text=joined,
            language=_result_tag(language, detected, joined),
            duration=duration,
            segments=segments,
            backend=self.name,
            model=self._model_id,
            elapsed=time.perf_counter() - started,
            meta={
                "cloud": True,
                "provider": "azure",
                "detected_language": detected,
                "locales": locales,
                "word_timestamps": bool(word_timestamps),
                "diarization": bool(kwargs.get("diarize")),
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
            meta={"error": str(exc), "cloud": True, "provider": "azure"},
        )
