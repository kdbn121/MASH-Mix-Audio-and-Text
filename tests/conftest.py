# SPDX-License-Identifier: MIT | Copyright (c) 2026 MASH (Khaled Bhuiyan)
"""Shared fixtures for the test suite.

The suite runs without the package installed and without any model weights,
so the ``src`` directory is put on ``sys.path`` here. Every fixture is
synthetic: no file is downloaded, no network is touched and no model is loaded.
"""

from __future__ import annotations

import sys
from pathlib import Path
from typing import List, NamedTuple

import numpy as np
import pytest

REPO_ROOT = Path(__file__).resolve().parents[1]
SRC_DIR = REPO_ROOT / "src"

if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from mash_mix.types import Segment, Transcription, Word  # noqa: E402


class Waveform(NamedTuple):
    """A synthetic waveform with its sample rate."""

    samples: np.ndarray
    sample_rate: int


@pytest.fixture
def bangla_waveform() -> Waveform:
    """A short synthetic signal that stands in for a Bangla recording.

    It is a 440 Hz tone with a raised cosine amplitude envelope, so it has a
    clear beginning and a clear end, which is what the chunking and the
    decoding tests need. The array is float32 mono at 22050 Hz.
    """
    sample_rate = 22050
    duration = 1.5
    count = int(sample_rate * duration)
    time = np.arange(count, dtype=np.float64) / float(sample_rate)

    envelope = np.sin(np.pi * time / duration) ** 2
    carrier = 0.5 * np.sin(2.0 * np.pi * 440.0 * time)
    samples = (envelope * carrier).astype(np.float32)
    return Waveform(samples=samples, sample_rate=sample_rate)


@pytest.fixture
def fake_transcription() -> Transcription:
    """A transcription with three segments and word level timings."""
    segments: List[Segment] = [
        Segment(
            start=0.0,
            end=2.5,
            text="প্রথম লাইন",
            words=[
                Word(start=0.0, end=0.6, text="প্রথম", probability=0.98),
                Word(start=0.6, end=2.5, text="লাইন", probability=0.95),
            ],
            language="bn-BD",
        ),
        Segment(
            start=2.5,
            end=5.125,
            text="দ্বিতীয় লাইন",
            words=[
                Word(start=2.5, end=3.4, text="দ্বিতীয়", probability=0.97),
                Word(start=3.4, end=5.125, text="লাইন", probability=0.94),
            ],
            language="bn-BD",
        ),
        Segment(
            start=5.125,
            end=8.0,
            text="তৃতীয় লাইন",
            words=[
                Word(start=5.125, end=6.0, text="তৃতীয়", probability=0.96),
                Word(start=6.0, end=8.0, text="লাইন", probability=0.93),
            ],
            language="bn-BD",
        ),
    ]
    return Transcription(
        text="\n".join(segment.text for segment in segments),
        language="bn-BD",
        duration=8.0,
        segments=segments,
        backend="fake",
        model="fake-small",
        elapsed=0.5,
        meta={"sample_rate": 16000},
    )
