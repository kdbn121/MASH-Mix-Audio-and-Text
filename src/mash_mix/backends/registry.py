# SPDX-License-Identifier: MIT | Copyright (c) 2026 MASH (Khaled Bhuiyan)
"""Backend registry.

Backends register themselves at import time. The registry keeps imports lazy so
a machine with only one inference stack installed still starts cleanly.
"""

from __future__ import annotations

import importlib
from typing import Any, Dict, List, Optional, Type

from ..errors import BackendUnavailableError
from .base import BackendInfo, STSBackend, STTBackend, TTSBackend

__all__ = [
    "register_stt",
    "register_tts",
    "register_sts",
    "get_stt",
    "get_tts",
    "get_sts",
    "stt_names",
    "tts_names",
    "sts_names",
    "backend_infos",
    "discover",
]

_STT: Dict[str, Type[STTBackend]] = {}
_TTS: Dict[str, Type[TTSBackend]] = {}
_STS: Dict[str, Type[STSBackend]] = {}

# Modules that register backends. Each one must import cleanly even when its
# inference stack is missing, so the module level import guard is required.
_DISCOVERY_MODULES = (
    "mash_mix.backends.asr.openai_whisper",
    "mash_mix.backends.asr.faster_whisper",
    "mash_mix.backends.asr.transformers_asr",
    "mash_mix.backends.asr.elevenlabs_stt",
    "mash_mix.backends.asr.azure_stt",
    "mash_mix.backends.asr.dashscope_asr",
    "mash_mix.backends.asr.whisperx_backend",
    "mash_mix.backends.asr.seamless_m4t",
    "mash_mix.backends.asr.funasr",
    "mash_mix.backends.asr.sherpa_onnx",
    "mash_mix.backends.asr.whisper_cpp",
    "mash_mix.backends.tts.mms_tts",
    "mash_mix.backends.tts.piper",
    "mash_mix.backends.tts.vits_local",
    "mash_mix.backends.tts.indic_tts",
    "mash_mix.backends.tts.parler_tts",
    "mash_mix.backends.tts.chatterbox",
    "mash_mix.backends.tts.xtts",
    "mash_mix.backends.tts.edge_tts",
    "mash_mix.backends.sts.pipeline_sts",
)


def register_stt(name: str) -> Any:
    """Class decorator that registers a speech to text backend."""

    def deco(cls: Type[STTBackend]) -> Type[STTBackend]:
        cls.name = name
        _STT[name] = cls
        return cls

    return deco


def register_tts(name: str) -> Any:
    """Class decorator that registers a text to speech backend."""

    def deco(cls: Type[TTSBackend]) -> Type[TTSBackend]:
        cls.name = name
        _TTS[name] = cls
        return cls

    return deco


def register_sts(name: str) -> Any:
    """Class decorator that registers a speech to speech backend."""

    def deco(cls: Type[STSBackend]) -> Type[STSBackend]:
        cls.name = name
        _STS[name] = cls
        return cls

    return deco


def discover() -> None:
    """Import every backend module once, ignoring modules that fail to import."""
    for module in _DISCOVERY_MODULES:
        try:
            importlib.import_module(module)
        except Exception:  # pragma: no cover - a broken optional backend must not break the CLI
            continue


def _instantiate(kind: str, name: str) -> Any:
    table = {"stt": _STT, "tts": _TTS, "sts": _STS}[kind]
    if name not in table:
        discover()
    if name not in table:
        raise BackendUnavailableError(
            f"Unknown {kind} backend '{name}'. Known backends: {', '.join(sorted(table))}"
        )
    return table[name]()


def get_stt(name: str, *, model: Optional[str] = None) -> STTBackend:
    """Return a loaded speech to text backend."""
    backend = _instantiate("stt", name)
    if not backend.is_available():
        info: BackendInfo = backend.info
        hint = f"pip install 'mash-mix-audio-and-text[{info.extra}]'" if info.extra else ""
        hint = hint or (info.missing or "")
        raise BackendUnavailableError(
            f"Speech to text backend '{name}' is not available on this machine. {hint}".strip()
        )
    backend.load(model)
    return backend


def get_tts(name: str, *, model: Optional[str] = None) -> TTSBackend:
    """Return a loaded text to speech backend."""
    backend = _instantiate("tts", name)
    if not backend.is_available():
        info = backend.info
        hint = f"pip install 'mash-mix-audio-and-text[{info.extra}]'" if info.extra else ""
        raise BackendUnavailableError(
            f"Text to speech backend '{name}' is not available on this machine. {hint}".strip()
        )
    backend.load(model)
    return backend


def get_sts(name: str, *, model: Optional[str] = None) -> STSBackend:
    """Return a loaded speech to speech backend."""
    backend = _instantiate("sts", name)
    if not backend.is_available():
        raise BackendUnavailableError(
            f"Speech to speech backend '{name}' is not available on this machine."
        )
    return backend


def stt_names() -> List[str]:
    discover()
    return sorted(_STT)


def tts_names() -> List[str]:
    discover()
    return sorted(_TTS)


def sts_names() -> List[str]:
    discover()
    return sorted(_STS)


def backend_infos() -> List[Dict[str, Any]]:
    """Describe every known backend for the doctor command and the server."""
    discover()
    out: List[Dict[str, Any]] = []
    for kind, table in (("stt", _STT), ("tts", _TTS), ("sts", _STS)):
        for name in sorted(table):
            cls = table[name]
            info: BackendInfo = getattr(cls, "info", None) or BackendInfo(name, kind, "")
            try:
                available = bool(cls().is_available())
            except Exception:
                available = False
            data = info.to_dict()
            data["available"] = available
            out.append(data)
    return out
