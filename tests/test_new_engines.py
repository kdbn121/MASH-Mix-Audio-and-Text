# SPDX-License-Identifier: MIT | Copyright (c) 2026 MASH (Khaled Bhuiyan)
"""The three speech engines that came in with the merge: Azure, DashScope and WhisperX."""

from __future__ import annotations

import json
import urllib.request
from typing import Any, Dict, List

import numpy as np
import pytest

from mash_mix.backends.asr import azure_stt, dashscope_asr, whisperx_backend
from mash_mix.backends.registry import stt_names


class _Answer:
    """A stand in for the answer of a web call."""

    def __init__(self, payload: Dict[str, Any]) -> None:
        self._payload = payload
        self._body = json.dumps(payload).encode("utf-8")

    def read(self) -> bytes:
        return self._body

    def __enter__(self) -> _Answer:
        return self

    def __exit__(self, *args: object) -> bool:
        return False


def _tone(seconds: float = 1.0) -> np.ndarray:
    """A short block of sound, so the backends have something to work on."""
    count = int(16000 * seconds)
    time = np.arange(count, dtype=np.float32) / 16000.0
    return (0.1 * np.sin(2 * np.pi * 220 * time)).astype(np.float32)


# -- the registry ---------------------------------------------------------


def test_the_new_backends_are_registered():
    names = stt_names()
    assert "azure-stt" in names
    assert "dashscope-asr" in names
    assert "whisperx" in names


def test_the_new_backends_import_without_their_packages():
    assert azure_stt.AzureSTTBackend().info.name == "azure-stt"
    assert dashscope_asr.DashScopeASRBackend().info.name == "dashscope-asr"
    assert whisperx_backend.WhisperXBackend().info.name == "whisperx"


# -- Azure ----------------------------------------------------------------


AZADE_ANSWER = {
    "durationMilliseconds": 4000,
    "phrases": [
        {
            "offsetMilliseconds": 0,
            "durationMilliseconds": 1500,
            "text": "ওরে নীল দরিয়া",
            "locale": "bn-IN",
            "confidence": 0.92,
            "speaker": "SPEAKER_00",
            "words": [
                {"text": "ওরে", "offsetMilliseconds": 0, "durationMilliseconds": 500},
                {"text": "নীল", "offsetMilliseconds": 500, "durationMilliseconds": 500},
                {"text": "দরিয়া", "offsetMilliseconds": 1000, "durationMilliseconds": 500},
            ],
        },
        {
            "offsetMilliseconds": 1500,
            "durationMilliseconds": 1200,
            "text": "বন্দী হইয়া মনোয়া পাখি হায়রে",
            "locale": "bn-IN",
            "confidence": 0.88,
            "speaker": "SPEAKER_01",
        },
    ],
}


def test_the_azure_backend_is_away_without_a_key(monkeypatch):
    monkeypatch.delenv("AZURE_SPEECH_KEY", raising=False)
    monkeypatch.delenv("MASH_MIX_AZURE_SPEECH_KEY", raising=False)
    monkeypatch.setenv("AZURE_SPEECH_REGION", "eastus")
    assert azure_stt.AzureSTTBackend().is_available() is False


def test_the_azure_backend_is_away_without_a_region(monkeypatch):
    monkeypatch.setenv("AZURE_SPEECH_KEY", "a-key")
    monkeypatch.delenv("AZURE_SPEECH_REGION", raising=False)
    monkeypatch.delenv("MASH_MIX_AZURE_SPEECH_REGION", raising=False)
    monkeypatch.delenv("AZURE_SPEECH_ENDPOINT", raising=False)
    assert azure_stt.AzureSTTBackend().is_available() is False


def test_the_azure_backend_is_here_with_a_key_and_a_region(monkeypatch):
    monkeypatch.setenv("AZURE_SPEECH_KEY", "a-key")
    monkeypatch.setenv("AZURE_SPEECH_REGION", "eastus")
    assert azure_stt.AzureSTTBackend().is_available() is True


def test_the_azure_backend_reads_the_phrases_and_the_words(monkeypatch):
    monkeypatch.setenv("AZURE_SPEECH_KEY", "a-key")
    monkeypatch.setenv("AZURE_SPEECH_REGION", "eastus")

    def fake_urlopen(request, timeout=None):
        return _Answer(AZADE_ANSWER)

    monkeypatch.setattr(urllib.request, "urlopen", fake_urlopen)
    backend = azure_stt.AzureSTTBackend()
    backend.load()
    result = backend.transcribe(_tone(2.0), 16000, language="bn")
    assert len(result.segments) == 2
    assert result.segments[0].text == "ওরে নীল দরিয়া"
    assert result.segments[0].speaker == "SPEAKER_00"
    assert len(result.segments[0].words) == 3
    assert result.segments[0].words[0].start == 0.0
    assert result.segments[0].words[2].end == 1.5
    assert result.segments[1].speaker == "SPEAKER_01"
    assert result.language.startswith("bn")
    assert result.meta["cloud"] is True
    assert result.meta["provider"] == "azure"


def test_the_azure_backend_reports_a_failure_without_raising(monkeypatch):
    monkeypatch.setenv("AZURE_SPEECH_KEY", "a-key")
    monkeypatch.setenv("AZURE_SPEECH_REGION", "eastus")

    def fake_urlopen(request, timeout=None):
        raise urllib.error.HTTPError("url", 401, "no", {}, None)  # type: ignore[arg-type]

    monkeypatch.setattr(urllib.request, "urlopen", fake_urlopen)
    backend = azure_stt.AzureSTTBackend()
    backend.load()
    result = backend.transcribe(_tone(1.0), 16000)
    assert result.text == ""
    assert "error" in result.meta


# -- DashScope ------------------------------------------------------------


def test_the_dashscope_backend_is_away_without_a_key(monkeypatch):
    monkeypatch.delenv("DASHSCOPE_API_KEY", raising=False)
    monkeypatch.delenv("MASH_MIX_DASHSCOPE_API_KEY", raising=False)
    assert dashscope_asr.DashScopeASRBackend().is_available() is False


def test_the_dashscope_backend_needs_a_public_address(monkeypatch):
    monkeypatch.setenv("DASHSCOPE_API_KEY", "a-key")
    backend = dashscope_asr.DashScopeASRBackend()
    backend.load()
    result = backend.transcribe(_tone(1.0), 16000)
    assert result.text == ""
    assert "audio_url" in result.meta["error"]


def test_the_dashscope_backend_refuses_a_local_path(monkeypatch):
    monkeypatch.setenv("DASHSCOPE_API_KEY", "a-key")
    backend = dashscope_asr.DashScopeASRBackend()
    backend.load()
    result = backend.transcribe(_tone(1.0), 16000, audio_url="C:/music/song.wav")
    assert result.text == ""
    assert "http" in result.meta["error"]


def test_the_dashscope_backend_reads_the_words(monkeypatch):
    monkeypatch.setenv("DASHSCOPE_API_KEY", "a-key")
    monkeypatch.setattr(dashscope_asr, "POLL_SECONDS", 0.0)

    transcript = {
        "transcripts": [
            {
                "language": "bn",
                "sentences": [
                    {
                        "words": [
                            {"text": "ওরে", "begin_time": 0, "end_time": 500},
                            {"text": "নীল", "begin_time": 500, "end_time": 1000},
                        ]
                    }
                ],
            }
        ]
    }
    calls: List[str] = []

    def fake_urlopen(request, timeout=None):
        calls.append(request.full_url)
        if "tasks" in request.full_url:
            return _Answer(
                {
                    "output": {
                        "task_status": "SUCCEEDED",
                        "result": {"transcription_url": "https://example.invalid/out.json"},
                    }
                }
            )
        if request.method == "POST":
            return _Answer({"output": {"task_id": "task-1"}})
        return _Answer(transcript)

    monkeypatch.setattr(urllib.request, "urlopen", fake_urlopen)
    backend = dashscope_asr.DashScopeASRBackend()
    backend.load()
    result = backend.transcribe(
        _tone(2.0),
        16000,
        language="bn",
        audio_url="https://example.invalid/song.wav",
    )
    assert result.text == "ওরে নীল"
    assert len(result.segments) == 1
    assert len(result.segments[0].words) == 2
    assert result.segments[0].words[1].start == 0.5
    assert result.segments[0].words[1].end == 1.0
    assert result.meta["provider"] == "dashscope"
    assert "bn" in result.meta["language_hints"]
    assert any("transcription" in call for call in calls)


# -- WhisperX -------------------------------------------------------------


def test_the_whisperx_backend_is_away_without_the_package(monkeypatch):
    monkeypatch.setattr(whisperx_backend, "_has_whisperx", lambda: False)
    assert whisperx_backend.WhisperXBackend().is_available() is False


def test_the_whisperx_backend_is_here_with_the_package(monkeypatch):
    monkeypatch.setattr(whisperx_backend, "_has_whisperx", lambda: True)
    assert whisperx_backend.WhisperXBackend().is_available() is True


def test_the_whisperx_backend_fills_the_speaker():
    raw = [
        {
            "start": 0.0,
            "end": 1.0,
            "text": "ওরে নীল",
            "words": [
                {"word": "ওরে", "start": 0.0, "end": 0.4, "score": 0.9},
                {"word": "নীল", "start": 0.4, "end": 1.0, "score": 0.8},
            ],
        },
        {
            "start": 1.0,
            "end": 2.0,
            "text": "দরিয়া",
            "speaker": "SPEAKER_01",
            "words": [{"word": "দরিয়া", "start": 1.0, "end": 2.0}],
        },
    ]
    segments = whisperx_backend._to_segments(raw)
    assert len(segments) == 2
    assert segments[0].words[0].probability == 0.9
    assert segments[1].speaker == "SPEAKER_01"


def test_the_whisperx_backend_reports_a_failure_without_raising(monkeypatch):
    monkeypatch.setattr(whisperx_backend, "_has_whisperx", lambda: True)
    backend = whisperx_backend.WhisperXBackend()
    backend.load()
    result = backend.transcribe(_tone(1.0), 16000)
    assert result.text == ""
    assert "error" in result.meta


def test_the_whisperx_backend_honours_the_device_choice(monkeypatch):
    monkeypatch.setenv("MASH_MIX_DEVICE", "cpu")
    device, compute = whisperx_backend._device_and_type()
    assert device == "cpu"
    assert compute == "int8"


@pytest.mark.parametrize("name", ["azure-stt", "dashscope-asr", "whisperx"])
def test_every_new_backend_says_when_it_is_not_ready(name):
    from mash_mix.backends.registry import get_stt
    from mash_mix.errors import BackendUnavailableError

    with pytest.raises(BackendUnavailableError):
        get_stt(name)
