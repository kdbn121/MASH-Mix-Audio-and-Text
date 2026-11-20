# SPDX-License-Identifier: MIT | Copyright (c) 2026 MASH (Khaled Bhuiyan)
"""One facade over every task.

``MASHMix`` keeps a single configuration for the whole engine, so a caller
loads the audio once, keeps the backend selection consistent, and can move
from analysis to lyrics to synthesis without repeating options.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, Optional, Sequence

from .types import (
    AnalysisResult,
    AudioBuffer,
    LyricsResult,
    SeparationResult,
    SynthesisResult,
    Transcription,
)

__all__ = ["MASHMix", "PipelineConfig"]


@dataclass
class PipelineConfig:
    """Shared settings for a run."""

    language: str = "bn-BD"
    stt_backend: str = "auto"
    tts_backend: str = "auto"
    sts_backend: str = "auto"
    model: Optional[str] = None
    sample_rate: int = 16000
    extra: Dict[str, Any] = field(default_factory=dict)


class MASHMix:
    """The engine, configured once.

    Example
    -------
    >>> engine = MASHMix(language="bn-BD")
    >>> lyrics = engine.lyrics("song.mp3")            # doctest: +SKIP
    >>> engine.write(lyrics, "out", formats=["srt", "txt"])   # doctest: +SKIP
    """

    def __init__(self, config: Optional[PipelineConfig] = None, **kwargs: Any) -> None:
        if config is None:
            config = PipelineConfig(**kwargs)
        else:
            for key, value in kwargs.items():
                if hasattr(config, key):
                    setattr(config, key, value)
        self.config = config

    # -- helpers --------------------------------------------------------
    @property
    def language(self) -> str:
        return self.config.language

    def _buffer(self, audio: Any, sample_rate: Optional[int] = None) -> AudioBuffer:
        from .tasks._common import to_buffer

        return to_buffer(audio, sample_rate=sample_rate or self.config.sample_rate)

    # -- tasks ----------------------------------------------------------
    def analyze(self, audio: Any, **kwargs: Any) -> AnalysisResult:
        """Measure tempo, key, loudness and structure."""
        from .tasks.analyze import analyze

        return analyze(audio, **kwargs)

    def separate(self, audio: Any, **kwargs: Any) -> SeparationResult:
        """Split a track into stems."""
        from .tasks.analyze import separate

        return separate(audio, **kwargs)

    def transcribe(self, audio: Any, **kwargs: Any) -> Transcription:
        """Recognise speech."""
        from .tasks.transcribe import transcribe

        kwargs.setdefault("backend", self.config.stt_backend)
        kwargs.setdefault("model", self.config.model)
        kwargs.setdefault("language", self.config.language)
        return transcribe(audio, **kwargs)

    def lyrics(self, audio: Any, **kwargs: Any) -> LyricsResult:
        """Extract lyrics from a song or spoken track."""
        from .tasks.lyrics import extract_lyrics

        kwargs.setdefault("backend", self.config.stt_backend)
        kwargs.setdefault("model", self.config.model)
        kwargs.setdefault("language", self.config.language)
        return extract_lyrics(audio, **kwargs)

    def speak(self, text: str, **kwargs: Any) -> SynthesisResult:
        """Synthesise speech."""
        from .tasks.synthesize import synthesize

        kwargs.setdefault("backend", self.config.tts_backend)
        kwargs.setdefault("model", self.config.model)
        kwargs.setdefault("language", self.config.language)
        return synthesize(text, **kwargs)

    def sts(self, audio: Any, **kwargs: Any) -> SynthesisResult:
        """Speech to speech."""
        from .tasks.speech_to_speech import speak

        kwargs.setdefault("backend", self.config.sts_backend)
        kwargs.setdefault("stt_backend", self.config.stt_backend)
        kwargs.setdefault("tts_backend", self.config.tts_backend)
        return speak(audio, **kwargs)

    # -- output ---------------------------------------------------------
    def write(
        self,
        result: Any,
        output_dir: str,
        *,
        stem: Optional[str] = None,
        formats: Sequence[str] = ("srt", "txt"),
        title: Optional[str] = None,
    ) -> Dict[str, str]:
        """Write a result to disk in the requested formats."""
        from .export.plaintext import write_text
        from .export.subtitles import write_subtitles

        folder = Path(output_dir)
        folder.mkdir(parents=True, exist_ok=True)
        base = stem or getattr(result, "title", None) or "output"
        written: Dict[str, str] = {}

        for fmt in formats:
            lowered = fmt.lower()
            if lowered == "srt":
                path = folder / f"{base}.srt"
                write_subtitles(result.lines, str(path), fmt="srt", title=title)
            elif lowered == "txt":
                path = folder / f"{base}.txt"
                write_text(str(path), result.text + "\n")
            elif lowered == "vtt":
                path = folder / f"{base}.vtt"
                write_subtitles(result.lines, str(path), fmt="vtt", title=title)
            elif lowered == "lrc":
                path = folder / f"{base}.lrc"
                write_subtitles(result.lines, str(path), fmt="lrc")
            elif lowered == "json":
                import json

                path = folder / f"{base}.json"
                write_text(str(path), json.dumps(result.to_dict(), ensure_ascii=False, indent=2))
            else:
                continue
            written[lowered] = str(path)

        return written
