# SPDX-License-Identifier: MIT | Copyright (c) 2026 MASH (Khaled Bhuiyan)
"""Backend contracts.

A backend is a thin adapter around one inference stack. It must be importable
without the stack installed, report its own availability, and expose a stable
call signature so the task layer never needs to know which engine ran.
"""

from __future__ import annotations

import abc
from typing import Any, Dict, List, Optional, Sequence, Tuple

import numpy as np

from ..types import AudioBuffer, SynthesisResult, Transcription

__all__ = ["STTBackend", "TTSBackend", "STSBackend", "BackendInfo"]


class BackendInfo:
    """Static description of a backend, used by ``mash-mix doctor`` and the server."""

    def __init__(
        self,
        name: str,
        kind: str,
        summary: str,
        package: Optional[str] = None,
        extra: Optional[str] = None,
        languages: Sequence[str] = (),
        platforms: Sequence[str] = (),
        notes: str = "",
    ) -> None:
        self.name = name
        self.kind = kind
        self.summary = summary
        self.package = package
        self.extra = extra
        self.languages = list(languages)
        self.platforms = list(platforms)
        self.notes = notes

    def to_dict(self) -> Dict[str, Any]:
        return {
            "name": self.name,
            "kind": self.kind,
            "summary": self.summary,
            "package": self.package,
            "extra": self.extra,
            "languages": self.languages,
            "platforms": self.platforms,
            "notes": self.notes,
        }


class STTBackend(abc.ABC):
    """Speech to text.

    Implementations receive mono float32 samples and return a
    :class:`~mash_mix.types.Transcription`. Word timestamps are optional but
    strongly preferred, because the lyric pipeline needs them for subtitles.
    """

    name: str = "stt"
    info: BackendInfo

    def __init__(self) -> None:
        self._loaded = False
        self._model_id: Optional[str] = None

    # -- lifecycle ------------------------------------------------------
    @abc.abstractmethod
    def is_available(self) -> bool:
        """True when the runtime dependencies for this backend are importable."""

    @abc.abstractmethod
    def load(self, model: Optional[str] = None) -> None:
        """Load weights. Safe to call more than once."""

    def unload(self) -> None:
        """Release weights. Optional for a backend to override."""
        self._loaded = False
        self._model_id = None

    @property
    def loaded(self) -> bool:
        return self._loaded

    @property
    def model_id(self) -> Optional[str]:
        return self._model_id

    # -- inference ------------------------------------------------------
    @abc.abstractmethod
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

    def detect_language(
        self,
        audio: np.ndarray,
        sample_rate: int,
        *,
        max_seconds: float = 30.0,
    ) -> Optional[Tuple[str, float]]:
        """Work out the language of the audio, with how sure the backend is.

        Returns the language code and a confidence, or ``None`` when the
        backend cannot tell. A backend that can tell does it in one short pass
        over the first seconds of the sound, which is far cheaper than reading
        the whole track.
        """
        return None

    def supported_languages(self) -> List[str]:
        return list(self.info.languages)

    def __repr__(self) -> str:  # pragma: no cover - debug helper
        return f"<{type(self).__name__} name={self.name} loaded={self._loaded}>"


class TTSBackend(abc.ABC):
    """Text to speech."""

    name: str = "tts"
    info: BackendInfo

    def __init__(self) -> None:
        self._loaded = False
        self._model_id: Optional[str] = None

    @abc.abstractmethod
    def is_available(self) -> bool:
        """True when the runtime dependencies for this backend are importable."""

    @abc.abstractmethod
    def load(self, model: Optional[str] = None) -> None:
        """Load weights. Safe to call more than once."""

    def unload(self) -> None:
        """Release weights. Optional for a backend to override."""
        self._loaded = False
        self._model_id = None

    @property
    def loaded(self) -> bool:
        return self._loaded

    @property
    def model_id(self) -> Optional[str]:
        return self._model_id

    @abc.abstractmethod
    def synthesize(
        self,
        text: str,
        *,
        language: str = "bn-BD",
        voice: Optional[str] = None,
        speed: float = 1.0,
        reference_audio: Optional[AudioBuffer] = None,
        **kwargs: Any,
    ) -> SynthesisResult:
        """Turn ``text`` into speech."""

    def list_voices(self) -> List[str]:
        """Voice identifiers this backend can use. Empty when unknown."""
        return []

    def supported_languages(self) -> List[str]:
        return list(self.info.languages)

    def __repr__(self) -> str:  # pragma: no cover - debug helper
        return f"<{type(self).__name__} name={self.name} loaded={self._loaded}>"


class STSBackend(abc.ABC):
    """Speech to speech.

    The default implementation in this package is a pipeline of STT plus TTS,
    which keeps every language benefit of the text layer. A backend may also be
    a native duplex model.
    """

    name: str = "sts"
    info: BackendInfo

    def __init__(self) -> None:
        self._loaded = False

    @abc.abstractmethod
    def is_available(self) -> bool:
        """True when the runtime dependencies for this backend are importable."""

    @abc.abstractmethod
    def speak(
        self,
        audio: AudioBuffer,
        *,
        target_language: str = "bn-BD",
        target_voice: Optional[str] = None,
        reference_audio: Optional[AudioBuffer] = None,
        **kwargs: Any,
    ) -> SynthesisResult:
        """Convert speech in ``audio`` into speech in ``target_language``."""

    def __repr__(self) -> str:  # pragma: no cover - debug helper
        return f"<{type(self).__name__} name={self.name} loaded={self._loaded}>"
