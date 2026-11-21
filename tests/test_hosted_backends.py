# SPDX-License-Identifier: MIT | Copyright (c) 2026 MASH (Khaled Bhuiyan)
"""Tests for the two hosted speech to text backends.

The ElevenLabs Scribe backend and the Meta SeamlessM4T backend both live
outside the machine, so the tests here are about the parts that can be
checked without a key and without the weights: the request that is built,
the answer that is read back, and the rule that a backend without a key
keeps out of the way.

No test here reaches the network.
"""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

import numpy as np  # noqa: E402

from mash_mix.backends.asr import elevenlabs_stt, seamless_m4t  # noqa: E402


def test_the_elevenlabs_backend_stays_away_without_a_key(monkeypatch):
    """No key means no backend, so nothing is ever sent away."""
    monkeypatch.delenv("ELEVENLABS_API_KEY", raising=False)
    monkeypatch.delenv("MASH_MIX_ELEVENLABS_API_KEY", raising=False)
    monkeypatch.delenv("XI_API_KEY", raising=False)
    assert elevenlabs_stt._api_key() is None
    assert not elevenlabs_stt.ElevenLabsSTTBackend().is_available()


def test_the_elevenlabs_backend_finds_its_key(monkeypatch):
    """The key comes from the environment."""
    monkeypatch.setenv("ELEVENLABS_API_KEY", "test-key")
    assert elevenlabs_stt._api_key() == "test-key"
    assert elevenlabs_stt.ElevenLabsSTTBackend().is_available()


def test_the_elevenlabs_backend_loads_without_a_key_it_must_refuse(monkeypatch):
    """A load without a key is refused with a clear message."""
    monkeypatch.delenv("ELEVENLABS_API_KEY", raising=False)
    monkeypatch.delenv("MASH_MIX_ELEVENLABS_API_KEY", raising=False)
    monkeypatch.delenv("XI_API_KEY", raising=False)
    backend = elevenlabs_stt.ElevenLabsSTTBackend()
    try:
        backend.load()
    except Exception as exc:
        assert "ELEVENLABS_API_KEY" in str(exc)
        return
    raise AssertionError("the load should have been refused")


def test_the_elevenlabs_backend_maps_bangla_to_its_code():
    """The Bangla tags become the code ElevenLabs knows."""
    assert elevenlabs_stt._engine_language("bn-BD") == "ben"
    assert elevenlabs_stt._engine_language("bn-IN") == "ben"
    assert elevenlabs_stt._engine_language("bn") == "ben"
    assert elevenlabs_stt._engine_language("en") == "en"


def test_the_elevenlabs_backend_builds_a_multipart_body():
    """The body carries the file and the fields."""
    body, content_type = elevenlabs_stt._multipart(
        [("model_id", "scribe_v3")],
        "audio.wav",
        "audio/wav",
        b"RIFFDATA",
    )
    assert content_type.startswith("multipart/form-data; boundary=")
    assert b'name="model_id"' in body
    assert b"scribe_v3" in body
    assert b'name="file"; filename="audio.wav"' in body
    assert b"RIFFDATA" in body


def test_the_elevenlabs_backend_reads_the_word_times():
    """The answer comes back as words with their times."""
    payload = {
        "text": "dol dol doluni",
        "language_code": "ben",
        "words": [
            {"text": "dol", "start": 0.0, "end": 0.4, "logprob": -0.1},
            {"text": "dol", "start": 0.4, "end": 0.9, "logprob": -0.2},
            {"text": "doluni", "start": 0.9, "end": 1.6, "logprob": -0.3},
        ],
    }
    words = elevenlabs_stt._parse_words(payload)
    assert len(words) == 3
    assert words[0].start == 0.0
    assert words[-1].end == 1.6
    assert words[0].probability is not None
    assert 0.0 < words[0].probability <= 1.0


def test_the_elevenlabs_backend_groups_words_into_segments():
    """A pause between words starts a new segment."""
    words = elevenlabs_stt._parse_words(
        {
            "words": [
                {"text": "dol", "start": 0.0, "end": 0.4},
                {"text": "chiruni", "start": 3.0, "end": 3.6},
            ]
        }
    )
    segments = elevenlabs_stt._group_words(words)
    assert len(segments) == 2
    assert segments[0].text == "dol"


def test_the_elevenlabs_backend_makes_a_wav():
    """The samples go out as a wav container."""
    payload = elevenlabs_stt._wav_bytes(np.zeros(1600, dtype=np.float32), 16000)
    assert payload[:4] == b"RIFF"
    assert b"WAVE" in payload[:16]


def test_the_seamless_backend_maps_bangla_to_its_code():
    """Bengali is the target the model writes in."""
    assert seamless_m4t._target_code("bn-BD") == "ben"
    assert seamless_m4t._target_code("bn-IN") == "ben"
    assert seamless_m4t._target_code("en-US") == "eng"
    assert seamless_m4t._target_code("hi-IN") == "hin"


def test_the_seamless_backend_reports_the_right_tag():
    """The Bangla region comes back on the result."""
    assert seamless_m4t._result_tag("bn-BD", "") == "bn-BD"
    assert seamless_m4t._result_tag("bn-IN", "") == "bn-IN"
    assert seamless_m4t._result_tag("en-US", "") == "en-US"


def test_the_seamless_backend_needs_transformers():
    """Without transformers the backend keeps out of the way."""
    backend = seamless_m4t.SeamlessM4TSTTBackend()
    assert isinstance(backend.is_available(), bool)
    if not seamless_m4t._HAS_TRANSFORMERS:
        assert not backend.is_available()
