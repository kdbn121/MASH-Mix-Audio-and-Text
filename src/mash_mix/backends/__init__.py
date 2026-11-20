# SPDX-License-Identifier: MIT | Copyright (c) 2026 MASH (Khaled Bhuiyan)
"""Backend registry and base contracts."""

from __future__ import annotations

from .base import BackendInfo, STSBackend, STTBackend, TTSBackend
from .registry import (
    backend_infos,
    discover,
    get_sts,
    get_stt,
    get_tts,
    register_sts,
    register_stt,
    register_tts,
    sts_names,
    stt_names,
    tts_names,
)

__all__ = [
    "BackendInfo",
    "STTBackend",
    "STSBackend",
    "TTSBackend",
    "backend_infos",
    "discover",
    "get_sts",
    "get_stt",
    "get_tts",
    "register_sts",
    "register_stt",
    "register_tts",
    "stt_names",
    "sts_names",
    "tts_names",
]
