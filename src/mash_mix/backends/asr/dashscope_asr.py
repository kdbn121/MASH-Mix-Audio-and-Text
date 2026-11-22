# SPDX-License-Identifier: MIT | Copyright (c) 2026 MASH (Khaled Bhuiyan)
"""Alibaba DashScope Qwen3 ASR backend.

Qwen3 ASR hears many languages well and gives the time of every word, which is
what a karaoke subtitle needs. It is a hosted service, so it is opt in.

One thing about it is different from the other hosted backends here. DashScope
fetches the audio itself, from a place on the internet it can reach. A file on
this machine cannot be named to it, so this backend needs a public address for
the audio. The caller passes one as ``audio_url`` and the backend says so
plainly when it is missing, rather than pretending to work.

The work is done in three steps: the job is submitted, the job is polled until
it is finished, and the transcript that the job points at is fetched and read.
"""

from __future__ import annotations

import json
import os
import time
import urllib.error
import urllib.request
from typing import Any, Dict, List, Optional, Sequence, Tuple

import numpy as np

from ...errors import BackendUnavailableError
from ...lang.dialect import detect_region
from ...lang.normalize import clean_asr_text, is_bangla_text, normalize_bangla
from ...types import Segment, Transcription, Word
from ..base import BackendInfo, STTBackend
from ..registry import register_stt

__all__ = ["DashScopeASRBackend"]

#: The address that starts the job.
SUBMIT_URL = "https://dashscope.aliyuncs.com/api/v1/services/audio/asr/transcription"

#: The address that asks how the job is going.
POLL_URL = "https://dashscope.aliyuncs.com/api/v1/tasks/{task_id}"

#: The model to ask for. This is the route that gives the time of every word.
DEFAULT_MODEL = "qwen3-asr-flash-filetrans"

#: The environment variable that holds the key.
API_KEY_ENV = "DASHSCOPE_API_KEY"

#: How often the job is asked about, and how long the wait is allowed to be.
POLL_SECONDS = 5.0
TIMEOUT_SECONDS = 600.0

#: The language hints that are offered when the caller does not name a language.
DEFAULT_HINTS = ["bn", "en", "hi", "ur", "zh", "ja"]

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
    for name in (API_KEY_ENV, "MASH_MIX_DASHSCOPE_API_KEY"):
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


def _request_json(
    url: str,
    key: str,
    *,
    payload: Optional[Dict[str, Any]] = None,
    headers: Optional[Dict[str, str]] = None,
    timeout: float = 60.0,
) -> Dict[str, Any]:
    """Send a request to DashScope and read the answer as JSON."""
    head = {"Authorization": f"Bearer {key}", "Accept": "application/json"}
    body: Optional[bytes] = None
    method = "GET"
    if payload is not None:
        body = json.dumps(payload).encode("utf-8")
        method = "POST"
        head["Content-Type"] = "application/json"
    if headers:
        head.update(headers)
    request = urllib.request.Request(url, data=body, method=method, headers=head)
    with urllib.request.urlopen(request, timeout=timeout) as response:
        raw = response.read().decode("utf-8")
    return json.loads(raw)


def _parse_words(transcription: Dict[str, Any]) -> List[Word]:
    """Every word of the transcript, with its time in seconds."""
    words: List[Word] = []
    for transcript in transcription.get("transcripts") or []:
        if not isinstance(transcript, dict):
            continue
        for sentence in transcript.get("sentences") or []:
            if not isinstance(sentence, dict):
                continue
            for word in sentence.get("words") or []:
                if not isinstance(word, dict):
                    continue
                text = _clean(str(word.get("text") or ""))
                if not text:
                    continue
                try:
                    start = float(word.get("begin_time") or 0) / 1000.0
                except (TypeError, ValueError):
                    start = 0.0
                try:
                    end = float(word.get("end_time") or 0) / 1000.0
                except (TypeError, ValueError):
                    end = start
                words.append(Word(start=start, end=max(end, start), text=text))
    return words


def _finalise(segments: List[Segment]) -> List[Segment]:
    """Trim, drop empties and enforce monotonic times."""
    out: List[Segment] = []
    for seg in segments:
        text = seg.text.strip()
        if not text:
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


@register_stt("dashscope-asr")
class DashScopeASRBackend(STTBackend):
    """Speech to text with the hosted Qwen3 ASR model of Alibaba DashScope."""

    info = BackendInfo(
        name="dashscope-asr",
        kind="stt",
        summary="Hosted Qwen3 ASR with the time of every word.",
        package="none, the standard library is enough",
        extra="",
        languages=_LANGUAGES,
        platforms=_PLATFORMS,
        notes=(
            "Needs an Alibaba Cloud key in the DASHSCOPE_API_KEY environment "
            "variable. The audio is sent to DashScope, so this backend is opt "
            "in. DashScope fetches the audio itself, so the caller has to give "
            "a public address for it as audio_url."
        ),
        missing="Put the key in the DASHSCOPE_API_KEY environment variable.",
    )

    def __init__(self) -> None:
        super().__init__()
        self._model_id = DEFAULT_MODEL

    # -- lifecycle ------------------------------------------------------
    def is_available(self) -> bool:
        """True when a key is present."""
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
                "The 'dashscope-asr' backend needs an Alibaba Cloud key. Put it in "
                "the DASHSCOPE_API_KEY environment variable and start the engine "
                "again."
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
        """Transcribe the audio that ``audio_url`` points at.

        The samples that are handed in are not sent. DashScope fetches the file
        itself, so the address of the file is what this backend needs.
        """
        if not self._loaded:
            self.load()
        key = _api_key()
        if not key:
            return self._empty(
                float(len(audio)) / float(sample_rate or 1),
                language,
                RuntimeError("No DashScope key is present."),
            )

        audio_url = str(kwargs.get("audio_url") or "").strip()
        if not audio_url:
            return self._empty(
                float(len(audio)) / float(sample_rate or 1),
                language,
                RuntimeError(
                    "The 'dashscope-asr' backend needs a public address for the "
                    "audio. Pass audio_url with the address of the file."
                ),
            )
        if not audio_url.startswith(("http://", "https://")):
            return self._empty(
                float(len(audio)) / float(sample_rate or 1),
                language,
                RuntimeError(
                    "The address of the audio has to be one the internet can "
                    "reach, so it must start with http:// or https://."
                ),
            )
        if audio_url.startswith("http://"):
            audio_url = "https://" + audio_url[len("http://") :]

        data = np.asarray(audio, dtype=np.float32).reshape(-1)
        duration = float(data.size) / float(sample_rate or 1)

        hints = kwargs.get("language_hints")
        if not hints:
            base, _region = _split_tag(language)
            hints = [base] if base and base != "auto" else list(DEFAULT_HINTS)
        hints = [str(item).strip().lower() for item in hints if str(item).strip()]

        payload: Dict[str, Any] = {
            "model": self._model_id,
            "input": {"file_url": audio_url},
            "parameters": {
                "enable_words": bool(word_timestamps),
                "language_hints": hints,
            },
        }

        started = time.perf_counter()
        try:
            submitted = _request_json(
                SUBMIT_URL,
                key,
                payload=payload,
                headers={"X-DashScope-Async": "enable"},
            )
            task_id = str((submitted.get("output") or {}).get("task_id") or "")
            if not task_id:
                raise RuntimeError(
                    "DashScope took the job but did not give a task id back."
                )
            transcript = self._wait_for_task(task_id, key)
        except urllib.error.HTTPError as exc:
            detail = ""
            try:
                detail = exc.read().decode("utf-8")[:400]
            except Exception:  # pragma: no cover - defensive
                detail = ""
            return self._empty(
                duration,
                language,
                RuntimeError(f"DashScope said {exc.code}. {detail}"),
            )
        except Exception as exc:
            return self._empty(duration, language, exc)

        words = _parse_words(transcript) if word_timestamps else []
        segments: List[Segment] = []
        if words:
            group: List[Word] = []
            for word in words:
                if group and word.start - group[-1].end > 0.8:
                    segments.append(_segment_from_words(group))
                    group = []
                group.append(word)
            if group:
                segments.append(_segment_from_words(group))
        else:
            text = _clean(str((transcript.get("transcripts") or [{}])[0].get("text") or ""))
            if text:
                segments = [Segment(start=0.0, end=duration, text=text)]

        segments = _finalise(segments)
        joined = " ".join(seg.text for seg in segments)
        detected = (transcript.get("transcripts") or [{}])[0].get("language")

        return Transcription(
            text=joined,
            language=_result_tag(language, str(detected) if detected else None, joined),
            duration=duration,
            segments=segments,
            backend=self.name,
            model=self._model_id,
            elapsed=time.perf_counter() - started,
            meta={
                "cloud": True,
                "provider": "dashscope",
                "detected_language": str(detected) if detected else None,
                "language_hints": hints,
                "word_timestamps": bool(word_timestamps),
                "audio_url": audio_url,
            },
        )

    def _wait_for_task(self, task_id: str, key: str) -> Dict[str, Any]:
        """Wait for the job and fetch the transcript it points at."""
        deadline = time.time() + TIMEOUT_SECONDS
        url = POLL_URL.format(task_id=task_id)
        while time.time() < deadline:
            time.sleep(POLL_SECONDS)
            answer = _request_json(url, key)
            output = answer.get("output") or {}
            status = str(output.get("task_status") or "").upper()
            if status == "SUCCEEDED":
                result = output.get("result") or {}
                transcription_url = str(result.get("transcription_url") or "")
                if not transcription_url:
                    raise RuntimeError(
                        "The DashScope job finished but did not point at a transcript."
                    )
                return _request_json(transcription_url, key)
            if status in ("FAILED", "CANCELED", "UNKNOWN"):
                raise RuntimeError(
                    f"The DashScope job did not finish: {output.get('message') or status}"
                )
        raise TimeoutError(
            f"The DashScope job did not finish within {int(TIMEOUT_SECONDS)} seconds."
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
            meta={"error": str(exc), "cloud": True, "provider": "dashscope"},
        )


def _segment_from_words(words: List[Word]) -> Segment:
    """Build one segment out of a run of words."""
    return Segment(
        start=words[0].start,
        end=words[-1].end,
        text=" ".join(word.text for word in words).strip(),
        words=list(words),
    )
