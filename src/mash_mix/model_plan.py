# SPDX-License-Identifier: MIT | Copyright (c) 2026 MASH (Khaled Bhuiyan)
"""Choosing the speech to text model for a piece of audio.

The small model that used to be the default is quick but it writes Bangla
songs badly. A lyric read out of the small model comes back as a rough
guess, with the words mangled and often in romanised Bangla even when the
caller asked for the native script.

The strong models read the same song properly, so the default is now the
strong one and the small model is the last resort rather than the first
choice. This module holds the ladder that decides which models are tried in
which order, and it holds one rule above all others:

**A model that cannot be loaded must not end the job.**

A strong model needs room in memory and a download that may not be
available on a machine that is offline. So every request walks a ladder of
candidates and moves to the next one when a candidate fails. The caller
always gets lyrics, and the result says which model actually did the work.
"""

from __future__ import annotations

from typing import Dict, List, Optional, Sequence, Tuple

__all__ = [
    "Candidate",
    "QUALITY_LEVELS",
    "DEFAULT_QUALITY",
    "plan_for",
    "quality_label",
]

#: The quality levels a caller may ask for.
QUALITY_LEVELS = ("auto", "best", "fast")

#: The level used when the caller does not say.
DEFAULT_QUALITY = "auto"

#: The Bangla fine tune of the turbo model. It was trained on Bengali, so it
#: writes Bengali in the native script instead of romanised Bangla.
_BANGLA_TUNED = ("transformers", "mozilla-ai/whisper-large-v3-turbo-bn")

#: The general large model, run through CTranslate2. It reads Bangla well and
#: it needs much less memory than the fine tune does.
_LARGE = ("faster-whisper", "large-v3")

#: The small model. It is quick and it fits on any machine, so it is the
#: floor of every ladder.
_SMALL = ("faster-whisper", "small")

#: The smallest model. It is the last resort of the last resort.
_TINY = ("faster-whisper", "tiny")

#: The ladder for Bangla, from the best model to the smallest one.
#:
#: The large model comes first because it read a real Bangla song better in
#: the test: it caught "নুটন নুটন খোপাতি তুলে হেরে দোপাতি" where the fine
#: tune produced "ন্টন নোটন খপাতি তুলে রেদপাতি", and it is also the faster
#: of the two because the weights run through CTranslate2 in int8. The fine
#: tune is Bangla only and stays as the second rung, because it is the
#: stronger choice on some Bangla songs.
_BANGLA_LADDER: Tuple[Tuple[str, str], ...] = (_LARGE, _BANGLA_TUNED, _SMALL, _TINY)

#: The ladder for every other language.
_GENERAL_LADDER: Tuple[Tuple[str, str], ...] = (_LARGE, _SMALL, _TINY)

#: The ladder for the fast level. It starts at the small model on purpose.
_FAST_LADDER: Tuple[Tuple[str, str], ...] = (_SMALL, _TINY)

#: The hosted ladder. These backends send the audio away from the machine,
#: so they are never reached on their own. The caller has to name one.
_HOSTED_LADDER: Tuple[Tuple[str, str], ...] = (
    ("elevenlabs", "scribe_v3"),
    ("seamless-m4t", "facebook/seamless-m4t-v2-large"),
)

#: The models the transformers pipeline can drive. It takes a checkpoint
#: from the Hub, not the short CTranslate2 names.
_TRANSFORMERS_MODELS: Tuple[str, ...] = (
    "mozilla-ai/whisper-large-v3-turbo-bn",
    "openai/whisper-large-v3",
    "openai/whisper-large-v3-turbo",
)

#: The models the faster-whisper backend can drive.
_FASTER_MODELS: Tuple[str, ...] = ("large-v3", "small", "tiny")

#: The models of each backend the caller may name.
_BY_BACKEND: Dict[str, Tuple[str, ...]] = {
    "faster-whisper": _FASTER_MODELS,
    "whisper": _FASTER_MODELS,
    "transformers": _TRANSFORMERS_MODELS,
}


class Candidate:
    """One model the engine may try.

    A candidate names the backend that drives the model and the model
    itself. The backend and the model belong together, because a CTranslate2
    checkpoint cannot be loaded by the transformers pipeline and the other
    way round.
    """

    __slots__ = ("backend", "model", "label")

    def __init__(self, backend: str, model: str, label: str = "") -> None:
        self.backend = backend
        self.model = model
        self.label = label or model

    def __repr__(self) -> str:  # pragma: no cover - for a human reading a log
        return f"Candidate({self.backend!r}, {self.model!r})"

    def __eq__(self, other: object) -> bool:
        if not isinstance(other, Candidate):
            return NotImplemented
        return self.backend == other.backend and self.model == other.model

    def __hash__(self) -> int:
        return hash((self.backend, self.model))


def _is_bangla(language: Optional[str]) -> bool:
    """True when the language tag asks for Bangla."""
    if not language:
        return False
    from .lang.registry import is_bangla

    try:
        return bool(is_bangla(language))
    except Exception:  # pragma: no cover - defensive, an odd tag is not Bangla
        return False


def _describe(backend: str, model: str) -> str:
    """A short name for a model, for the report and the panel."""
    if model == "mozilla-ai/whisper-large-v3-turbo-bn":
        return "Whisper turbo, Bangla fine tune"
    if model == "large-v3":
        return "Whisper large v3"
    if model == "small":
        return "Whisper small"
    if model == "tiny":
        return "Whisper tiny"
    if model == "scribe_v3":
        return "ElevenLabs Scribe"
    if model == "facebook/seamless-m4t-v2-large":
        return "Meta SeamlessM4T"
    return model


def _build(pairs: Sequence[Tuple[str, str]]) -> List[Candidate]:
    return [Candidate(backend, model, _describe(backend, model)) for backend, model in pairs]


def plan_for(
    language: Optional[str] = None,
    *,
    quality: str = DEFAULT_QUALITY,
    model: Optional[str] = None,
    backend: Optional[str] = None,
) -> List[Candidate]:
    """The candidates to try, in order.

    Parameters
    ----------
    language:
        The language tag of the audio. A Bangla tag gets the Bangla fine
        tune at the top of the ladder.
    quality:
        ``auto`` picks the ladder that fits the language, ``best`` forces
        the strongest model of that ladder and ``fast`` starts at the small
        model.
    model:
        A model the caller named. When it is given it is the only candidate,
        because the caller has made the choice.
    backend:
        A backend the caller named. It is used for every candidate, because
        the caller has asked for that engine.
    """
    level = (quality or DEFAULT_QUALITY).strip().lower()
    if level not in QUALITY_LEVELS:
        level = DEFAULT_QUALITY

    if model:
        chosen_backend = backend or "faster-whisper"
        return [Candidate(chosen_backend, model, _describe(chosen_backend, model))]

    if backend and backend != "auto":
        hosted = {name: hosted_model for name, hosted_model in _HOSTED_LADDER}
        if backend in hosted:
            # A hosted backend has its own model. The local model names do
            # not mean anything to it.
            hosted_model = hosted[backend]
            return [Candidate(backend, hosted_model, _describe(backend, hosted_model))]
        ladder = _BY_BACKEND.get(backend)
        if ladder:
            return [
                Candidate(backend, model_name, _describe(backend, model_name))
                for model_name in ladder
            ]

    if level == "fast":
        pairs = _FAST_LADDER
    elif _is_bangla(language):
        pairs = _BANGLA_LADDER
    else:
        pairs = _GENERAL_LADDER

    candidates = _build(pairs)
    if backend and backend != "auto":
        # The caller asked for a backend that has no ladder of its own, so
        # every candidate runs on it.
        for candidate in candidates:
            candidate.backend = backend
    return candidates


def quality_label(quality: str) -> str:
    """A short name for a quality level."""
    level = (quality or DEFAULT_QUALITY).strip().lower()
    if level == "best":
        return "the strongest model"
    if level == "fast":
        return "the quick model"
    return "the model that fits the language"
