# SPDX-License-Identifier: MIT | Copyright (c) 2026 MASH (Khaled Bhuiyan)
"""Tests for the model ladder and the fallback that walks it.

The ladder is the reason a Bangla song comes back with real lyrics now. The
small model used to be the only choice, and it wrote the words wrong. These
tests pin down the order of the ladder, the rule that a named model is the
only candidate, and the rule that a model which cannot be loaded does not end
the job.
"""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

import numpy as np  # noqa: E402

from mash_mix.model_plan import (  # noqa: E402
    DEFAULT_QUALITY,
    QUALITY_LEVELS,
    Candidate,
    plan_for,
    quality_label,
)
from mash_mix.tasks.lyrics import _recognise, _takes_long_audio  # noqa: E402
from mash_mix.types import AudioBuffer, Segment, Transcription, Word  # noqa: E402


class FakeEngine:
    """A recogniser that stands in for a real one."""

    def __init__(self, name: str = "faster-whisper", text: str = "ranga mathar chiruni") -> None:
        self.name = name
        self.text = text
        self.calls = []

    def transcribe(self, audio, sample_rate, **kwargs):
        self.calls.append((len(audio), dict(kwargs)))
        return Transcription(
            text=self.text,
            language="bn-BD",
            duration=len(audio) / float(sample_rate),
            segments=[
                Segment(
                    start=0.0,
                    end=1.0,
                    text=self.text,
                    words=[Word(start=0.0, end=1.0, text=self.text)],
                )
            ],
            backend=self.name,
            model="fake",
        )


class BrokenEngine(FakeEngine):
    """A recogniser that always fails."""

    def transcribe(self, audio, sample_rate, **kwargs):
        raise RuntimeError("out of memory")


class SilentEngine(FakeEngine):
    """A recogniser that loads but hears nothing."""

    def transcribe(self, audio, sample_rate, **kwargs):
        return Transcription(
            text="",
            language="bn-BD",
            duration=len(audio) / float(sample_rate),
            segments=[],
            backend=self.name,
            model="silent",
        )


def test_the_bangla_ladder_starts_with_the_large_model():
    """The large model read the test song better than the fine tune did."""
    candidates = plan_for("bn-BD")
    assert candidates[0].backend == "faster-whisper"
    assert candidates[0].model == "large-v3"


def test_the_bangla_fine_tune_stays_in_the_ladder():
    """The Bangla fine tune is the second rung, because it is Bangla only."""
    candidates = plan_for("bn-BD")
    models = [candidate.model for candidate in candidates]
    assert "mozilla-ai/whisper-large-v3-turbo-bn" in models
    assert models.index("mozilla-ai/whisper-large-v3-turbo-bn") == 1


def test_the_bangla_ladder_ends_with_the_small_model():
    """The small model stays as the floor of the ladder."""
    candidates = plan_for("bn-BD")
    models = [candidate.model for candidate in candidates]
    assert "small" in models
    assert models.index("small") > 0
    assert models[-1] == "tiny"


def test_the_ladder_never_starts_at_the_small_model():
    """The old default was the small model, and it was the wrong choice."""
    for language in ("bn-BD", "bn-IN", "bn", "en-US", "hi-IN"):
        candidates = plan_for(language)
        assert candidates[0].model != "small", language
        assert candidates[0].model != "tiny", language


def test_a_named_model_is_the_only_candidate():
    """When the caller names a model, that model is the one."""
    candidates = plan_for("bn-BD", model="large-v3")
    assert len(candidates) == 1
    assert candidates[0].model == "large-v3"


def test_a_named_backend_is_used_for_every_candidate():
    """When the caller names a backend, every candidate runs on it."""
    candidates = plan_for("bn-BD", backend="whisper")
    assert candidates
    for candidate in candidates:
        assert candidate.backend == "whisper"


def test_the_fast_level_starts_at_the_small_model():
    """The fast level is the quick one, and it says so."""
    candidates = plan_for("bn-BD", quality="fast")
    assert candidates[0].model == "small"


def test_the_best_level_still_falls_back():
    """The best level is a ladder too, so a failure is survivable."""
    candidates = plan_for("bn-BD", quality="best")
    assert len(candidates) > 1
    assert candidates[0].model == "large-v3"


def test_an_unknown_quality_falls_back_to_the_default():
    """A nonsense level must not raise."""
    candidates = plan_for("bn-BD", quality="nonsense")
    assert candidates == plan_for("bn-BD", quality=DEFAULT_QUALITY)


def test_the_transformers_ladder_starts_with_the_bangla_fine_tune():
    """On the transformers backend the fine tune comes first."""
    candidates = plan_for("bn-BD", backend="transformers")
    assert candidates[0].model == "mozilla-ai/whisper-large-v3-turbo-bn"


def test_the_hosted_backends_are_never_reached_on_their_own():
    """A hosted backend sends the audio away, so it has to be named."""
    for language in ("bn-BD", "en-US"):
        for candidate in plan_for(language):
            assert candidate.backend not in ("elevenlabs", "seamless-m4t"), language


def test_a_named_hosted_backend_gets_its_own_model():
    """The hosted backends have their own models."""
    eleven = plan_for("bn-BD", backend="elevenlabs")
    assert len(eleven) == 1
    assert eleven[0].backend == "elevenlabs"
    assert eleven[0].model == "scribe_v3"

    meta = plan_for("bn-BD", backend="seamless-m4t")
    assert len(meta) == 1
    assert meta[0].backend == "seamless-m4t"
    assert "seamless" in meta[0].model


def test_the_quality_levels_are_the_three_that_are_documented():
    assert QUALITY_LEVELS == ("auto", "best", "fast")
    assert quality_label("best") != quality_label("fast")


def test_a_candidate_compares_by_backend_and_model():
    assert Candidate("faster-whisper", "small") == Candidate("faster-whisper", "small")
    assert Candidate("faster-whisper", "small") != Candidate("faster-whisper", "tiny")


def test_a_whole_track_goes_to_a_model_that_can_take_one():
    """The Whisper family reads a long file in one call."""
    assert _takes_long_audio(FakeEngine("faster-whisper"))
    assert _takes_long_audio(FakeEngine("whisper"))
    assert not _takes_long_audio(FakeEngine("transformers"))


def test_a_whole_track_is_read_in_one_call():
    """One call for the whole track, so the song keeps its thread."""
    engine = FakeEngine()
    buffer = AudioBuffer(np.zeros(16000 * 45, dtype=np.float32), 16000, 1)
    segments, chunks = _recognise(
        engine,
        buffer,
        language="bn-BD",
        initial_prompt=None,
        hotwords=None,
        engine_vad=False,
        progress=False,
        chunk_seconds=30.0,
        overlap_seconds=2.0,
        kwargs={},
    )
    assert len(engine.calls) == 1
    assert chunks == 1
    assert segments


def test_a_model_that_cannot_read_the_track_falls_back_to_chunks():
    """When the whole call fails, the chunked path is used."""
    class HalfBroken(FakeEngine):
        def transcribe(self, audio, sample_rate, **kwargs):
            if len(audio) > 16000 * 40:
                raise RuntimeError("too long")
            return super().transcribe(audio, sample_rate, **kwargs)

    engine = HalfBroken()
    buffer = AudioBuffer(np.zeros(16000 * 45, dtype=np.float32), 16000, 1)
    segments, chunks = _recognise(
        engine,
        buffer,
        language="bn-BD",
        initial_prompt=None,
        hotwords=None,
        engine_vad=False,
        progress=False,
        chunk_seconds=30.0,
        overlap_seconds=2.0,
        kwargs={},
    )
    assert chunks > 1
    assert segments


def test_a_broken_model_raises_so_the_ladder_can_move_on():
    """The failure has to reach the ladder, not be swallowed here."""
    engine = BrokenEngine()
    buffer = AudioBuffer(np.zeros(16000 * 5, dtype=np.float32), 16000, 1)
    try:
        _recognise(
            engine,
            buffer,
            language="bn-BD",
            initial_prompt=None,
            hotwords=None,
            engine_vad=False,
            progress=False,
            chunk_seconds=30.0,
            overlap_seconds=2.0,
            kwargs={},
        )
    except RuntimeError:
        return
    raise AssertionError("the failure should have reached the ladder")


def test_the_ladder_walks_past_a_model_that_cannot_be_loaded(monkeypatch):
    """The real failure is a model that will not load, so the ladder must move.

    This is the case that bit the engine on a machine that was short of
    memory: the strong model could not be loaded and the request has to come
    back with the lyrics read by the next model rather than an error.
    """
    import mash_mix.tasks.lyrics as lyrics_module

    loaded: list = []

    def fake_get_stt(backend, *, model=None):
        if model == "large-v3":
            # This is the failure the machine really gave.
            raise RuntimeError("Could not load faster-whisper model 'large-v3': mkl_malloc")
        loaded.append((backend, model))
        return FakeEngine(backend, "ranga mathar chiruni")

    monkeypatch.setattr(lyrics_module, "pick_backend", lambda kind, preferred="auto": "faster-whisper")
    monkeypatch.setattr("mash_mix.backends.registry.get_stt", fake_get_stt)

    result = lyrics_module.extract_lyrics(
        AudioBuffer(np.zeros(16000 * 5, dtype=np.float32), 16000, 1),
        language="bn-BD",
        quality="auto",
    )
    assert result.lines
    # The ladder moved past the model that would not load.
    assert result.meta["model_used"] != "large-v3"
    assert loaded
    assert all(model != "large-v3" for _backend, model in loaded)
    assert result.meta["model_fallbacks"]


def test_the_ladder_reports_when_it_used_the_strong_model(monkeypatch):
    """The result says which model did the work."""
    import mash_mix.tasks.lyrics as lyrics_module

    def fake_get_stt(backend, *, model=None):
        return FakeEngine(backend, "ranga mathar chiruni")

    monkeypatch.setattr(lyrics_module, "pick_backend", lambda kind, preferred="auto": "faster-whisper")
    monkeypatch.setattr("mash_mix.backends.registry.get_stt", fake_get_stt)

    result = lyrics_module.extract_lyrics(
        AudioBuffer(np.zeros(16000 * 5, dtype=np.float32), 16000, 1),
        language="bn-BD",
        quality="auto",
    )
    assert result.meta["model_used"] == "large-v3"
    assert result.meta["model_label"]
    assert not result.meta["model_fallbacks"]


def test_a_model_that_hears_nothing_is_not_the_last_word(monkeypatch):
    """A model that hears nothing lets the ladder continue."""
    import mash_mix.tasks.lyrics as lyrics_module

    def fake_get_stt(backend, *, model=None):
        if model == "large-v3":
            return SilentEngine()
        return FakeEngine(backend, "ranga mathar chiruni")

    monkeypatch.setattr(lyrics_module, "pick_backend", lambda kind, preferred="auto": "faster-whisper")
    monkeypatch.setattr("mash_mix.backends.registry.get_stt", fake_get_stt)

    result = lyrics_module.extract_lyrics(
        AudioBuffer(np.zeros(16000 * 5, dtype=np.float32), 16000, 1),
        language="bn-BD",
        quality="auto",
    )
    assert result.meta["model_used"] != "large-v3"
    assert result.lines


def test_a_backend_that_rejects_a_keyword_still_works():
    """A backend without every keyword still does the work."""
    class Narrow(FakeEngine):
        def transcribe(self, audio, sample_rate, language=None, task=None):
            return super().transcribe(audio, sample_rate, language=language, task=task)

    engine = Narrow()
    buffer = AudioBuffer(np.zeros(16000 * 5, dtype=np.float32), 16000, 1)
    segments, _chunks = _recognise(
        engine,
        buffer,
        language="bn-BD",
        initial_prompt=None,
        hotwords=None,
        engine_vad=False,
        progress=False,
        chunk_seconds=30.0,
        overlap_seconds=2.0,
        kwargs={},
    )
    assert segments
