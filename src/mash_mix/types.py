# SPDX-License-Identifier: MIT | Copyright (c) 2026 MASH (Khaled Bhuiyan)
"""Plain data carriers shared by every task, backend and exporter.

Every dataclass here is JSON friendly: ``to_dict`` returns only builtin types so
results can travel over HTTP or land in a report without extra plumbing.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
from typing import Any, Dict, List, Optional

import numpy as np

__all__ = [
    "Word",
    "Segment",
    "Transcription",
    "LyricLine",
    "LyricsResult",
    "SynthesisResult",
    "SeparationResult",
    "AnalysisResult",
    "AudioBuffer",
]


@dataclass
class AudioBuffer:
    """A mono or multi channel float32 audio block plus its sample rate.

    ``samples`` is shaped ``(channels, frames)`` when ``channels > 1`` and
    ``(frames,)`` when mono. Keeping the sample rate next to the samples means
    no backend ever has to guess the tempo of an array it was handed.
    """

    samples: np.ndarray
    sample_rate: int
    channels: int = 1

    def __post_init__(self) -> None:
        arr = np.asarray(self.samples)
        if arr.ndim == 1:
            arr = arr[None, :]
            self.channels = 1
        elif arr.ndim == 2:
            self.channels = arr.shape[0]
        else:
            raise ValueError("AudioBuffer expects 1-D or 2-D samples")
        self.samples = arr.astype(np.float32, copy=False)

    @property
    def frames(self) -> int:
        return int(self.samples.shape[1])

    @property
    def duration(self) -> float:
        if self.sample_rate <= 0:
            return 0.0
        return self.frames / float(self.sample_rate)

    def mono(self) -> np.ndarray:
        """Return a single channel float32 array."""
        if self.samples.shape[0] == 1:
            return self.samples[0].astype(np.float32, copy=False)
        return self.samples.mean(axis=0).astype(np.float32, copy=False)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "sample_rate": int(self.sample_rate),
            "channels": int(self.channels),
            "frames": self.frames,
            "duration": round(self.duration, 6),
        }


@dataclass
class Word:
    """A single recognised word with its time span."""

    start: float
    end: float
    text: str
    probability: Optional[float] = None

    def to_dict(self) -> Dict[str, Any]:
        return {
            "start": round(float(self.start), 3),
            "end": round(float(self.end), 3),
            "text": self.text,
            "probability": None if self.probability is None else round(float(self.probability), 4),
        }


@dataclass
class Segment:
    """A contiguous block of speech. ``words`` may be empty when the backend has no word timings."""

    start: float
    end: float
    text: str
    words: List[Word] = field(default_factory=list)
    speaker: Optional[str] = None
    language: Optional[str] = None
    probability: Optional[float] = None

    def to_dict(self) -> Dict[str, Any]:
        return {
            "start": round(float(self.start), 3),
            "end": round(float(self.end), 3),
            "text": self.text,
            "speaker": self.speaker,
            "language": self.language,
            "probability": None if self.probability is None else round(float(self.probability), 4),
            "words": [w.to_dict() for w in self.words],
        }


@dataclass
class Transcription:
    """Full speech to text result."""

    text: str
    language: str
    duration: float
    segments: List[Segment] = field(default_factory=list)
    backend: str = ""
    model: str = ""
    elapsed: float = 0.0
    meta: Dict[str, Any] = field(default_factory=dict)

    @property
    def words(self) -> List[Word]:
        out: List[Word] = []
        for seg in self.segments:
            out.extend(seg.words)
        return out

    def plain_text(self) -> str:
        return "\n".join(s.text.strip() for s in self.segments if s.text.strip())

    def to_dict(self) -> Dict[str, Any]:
        return {
            "text": self.text,
            "language": self.language,
            "duration": round(float(self.duration), 3),
            "backend": self.backend,
            "model": self.model,
            "elapsed": round(float(self.elapsed), 3),
            "segments": [s.to_dict() for s in self.segments],
            "meta": self.meta,
        }


@dataclass
class LyricLine:
    """One lyric line with its time span. Used for SRT, LRC and VTT export.

    ``words`` holds the time of every word when the backend gave one. It is
    empty for a line that came from a page subtitle or from a picture, because
    those have no word timings to give.
    """

    start: float
    end: float
    text: str
    speaker: Optional[str] = None
    probability: Optional[float] = None
    words: List[Word] = field(default_factory=list)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "start": round(float(self.start), 3),
            "end": round(float(self.end), 3),
            "text": self.text,
            "speaker": self.speaker,
            "probability": None if self.probability is None else round(float(self.probability), 4),
            "words": [w.to_dict() for w in self.words],
        }


@dataclass
class LyricsResult:
    """Lyrics extracted from a song or spoken track."""

    lines: List[LyricLine] = field(default_factory=list)
    language: str = "bn-BD"
    duration: float = 0.0
    title: Optional[str] = None
    artist: Optional[str] = None
    backend: str = ""
    model: str = ""
    elapsed: float = 0.0
    meta: Dict[str, Any] = field(default_factory=dict)

    @property
    def text(self) -> str:
        return "\n".join(line.text.strip() for line in self.lines if line.text.strip())

    def to_dict(self) -> Dict[str, Any]:
        return {
            "text": self.text,
            "language": self.language,
            "duration": round(float(self.duration), 3),
            "title": self.title,
            "artist": self.artist,
            "backend": self.backend,
            "model": self.model,
            "elapsed": round(float(self.elapsed), 3),
            "lines": [line.to_dict() for line in self.lines],
            "meta": self.meta,
        }


@dataclass
class SynthesisResult:
    """Generated speech."""

    audio: np.ndarray
    sample_rate: int
    backend: str = ""
    model: str = ""
    voice: Optional[str] = None
    elapsed: float = 0.0
    meta: Dict[str, Any] = field(default_factory=dict)

    @property
    def duration(self) -> float:
        if self.sample_rate <= 0:
            return 0.0
        return len(self.audio) / float(self.sample_rate)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "sample_rate": int(self.sample_rate),
            "duration": round(self.duration, 3),
            "backend": self.backend,
            "model": self.model,
            "voice": self.voice,
            "elapsed": round(float(self.elapsed), 3),
            "meta": self.meta,
        }


@dataclass
class SeparationResult:
    """Separated stems. ``stems`` maps a stem name to its waveform."""

    stems: Dict[str, np.ndarray] = field(default_factory=dict)
    sample_rate: int = 44100
    backend: str = ""
    model: str = ""
    elapsed: float = 0.0
    meta: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "stems": {k: {"frames": int(len(v))} for k, v in self.stems.items()},
            "sample_rate": int(self.sample_rate),
            "backend": self.backend,
            "model": self.model,
            "elapsed": round(float(self.elapsed), 3),
            "meta": self.meta,
        }


@dataclass
class AnalysisResult:
    """Musical and acoustic measurements of a track."""

    duration: float = 0.0
    sample_rate: int = 0
    channels: int = 1
    bpm: Optional[float] = None
    key: Optional[str] = None
    mode: Optional[str] = None
    loudness_lufs: Optional[float] = None
    peak_dbfs: Optional[float] = None
    rms_dbfs: Optional[float] = None
    spectral_centroid: Optional[float] = None
    sections: List[Dict[str, Any]] = field(default_factory=list)
    beats: List[float] = field(default_factory=list)
    clipped_samples: int = 0
    dc_offset_percent: Optional[float] = None
    head_silence_seconds: Optional[float] = None
    crest_db: Optional[float] = None
    backend: str = ""
    meta: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        out = asdict(self)
        out["duration"] = round(float(self.duration), 3)
        out["beats"] = [round(float(b), 3) for b in self.beats]
        return out
