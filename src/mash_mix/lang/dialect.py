# SPDX-License-Identifier: MIT | Copyright (c) 2026 MASH (Khaled Bhuiyan)
"""Regional detection for Bangla: Bangladesh or India, and the district dialect.

Bangla is one language with two standard registers in daily use. The written
forms are close enough that a general model handles both, but the vocabulary is
not identical, and a transcript that claims to be Bengali while carrying
Bangladeshi markers is more useful to the caller when the region is named.

The detector is a weighted marker vote. It is explainable on purpose: every
decision comes back with the markers that produced it, so a wrong call can be
seen and corrected rather than guessed at.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple

from .lexicon import BD_MARKERS, DIALECT_MARKERS, IN_MARKERS
from .normalize import bangla_ratio, is_bangla_text
from .registry import BANGLA, BANGLA_BD, BANGLA_IN, LanguageSpec, resolve

__all__ = ["DialectReport", "detect_region", "detect_dialect", "detect_language"]


@dataclass
class DialectReport:
    """The outcome of a regional detection pass."""

    tag: str
    language: str
    confidence: float
    bd_score: float
    in_score: float
    dialect: Optional[str] = None
    dialect_confidence: float = 0.0
    evidence: List[str] = field(default_factory=list)
    script: str = "unknown"

    def to_dict(self) -> Dict[str, object]:
        return {
            "tag": self.tag,
            "language": self.language,
            "confidence": round(self.confidence, 4),
            "bd_score": round(self.bd_score, 3),
            "in_score": round(self.in_score, 3),
            "dialect": self.dialect,
            "dialect_confidence": round(self.dialect_confidence, 4),
            "evidence": self.evidence,
            "script": self.script,
        }


def _count_markers(text: str, markers: Dict[str, float]) -> Tuple[float, List[str]]:
    """Weighted marker vote with the matched evidence."""
    score = 0.0
    evidence: List[str] = []
    for marker, weight in markers.items():
        count = text.count(marker)
        if count:
            score += weight * min(count, 5)
            evidence.append(f"{marker} x{count}")
    return score, evidence


def detect_region(text: str, *, default: str = "bn-BD") -> DialectReport:
    """Detect whether a Bangla text is Bangladeshi or Indian, and how sure we are."""
    if not text or not text.strip():
        return DialectReport(
            tag=resolve(default).tag,
            language=default,
            confidence=0.0,
            bd_score=0.0,
            in_score=0.0,
            script="unknown",
        )

    ratio = bangla_ratio(text)
    script = "bangla" if ratio >= 0.3 else ("latin" if ratio < 0.1 else "mixed")

    bd_score, bd_evidence = _count_markers(text, BD_MARKERS)
    in_score, in_evidence = _count_markers(text, IN_MARKERS)

    total = bd_score + in_score
    if total <= 0:
        # No regional markers at all. Keep the caller's default and say so.
        return DialectReport(
            tag=resolve(default).tag,
            language=default,
            confidence=0.0,
            bd_score=bd_score,
            in_score=in_score,
            evidence=[],
            script=script,
        )

    if bd_score > in_score:
        tag = "bn-BD"
        confidence = bd_score / total
        evidence = [f"BD:{item}" for item in bd_evidence]
    else:
        tag = "bn-IN"
        confidence = in_score / total
        evidence = [f"IN:{item}" for item in in_evidence]

    # Confidence also needs the amount of text to be believable.
    length_factor = min(1.0, len(text) / 200.0)
    confidence = round(confidence * (0.6 + 0.4 * length_factor), 4)

    dialect, dialect_confidence, dialect_evidence = detect_dialect(text)

    return DialectReport(
        tag=tag,
        language=tag,
        confidence=confidence,
        bd_score=bd_score,
        in_score=in_score,
        dialect=dialect,
        dialect_confidence=dialect_confidence,
        evidence=evidence + [f"dialect:{item}" for item in dialect_evidence],
        script=script,
    )


def detect_dialect(text: str) -> Tuple[Optional[str], float, List[str]]:
    """Detect the regional dialect cluster inside Bangladesh."""
    if not text:
        return None, 0.0, []

    scores: Dict[str, float] = {}
    evidence: List[str] = []
    for cluster, markers in DIALECT_MARKERS.items():
        score = 0.0
        cluster_evidence: List[str] = []
        for marker, weight in markers.items():
            count = text.count(marker)
            if count:
                score += weight * min(count, 4)
                cluster_evidence.append(f"{marker} x{count}")
        if score > 0:
            scores[cluster] = score
            evidence.extend(f"{cluster}:{item}" for item in cluster_evidence)

    if not scores:
        return None, 0.0, []

    best = max(scores.items(), key=lambda kv: kv[1])
    total = sum(scores.values())
    confidence = best[1] / total if total else 0.0
    return best[0], round(confidence, 4), evidence


def detect_language(text: str, *, default: str = "bn-BD") -> LanguageSpec:
    """Detect the language of a text and return its full specification."""
    if not text or not text.strip():
        return resolve(default)

    ratio = bangla_ratio(text)
    if ratio >= 0.3:
        return resolve(detect_region(text, default=default).tag)

    # Not Bangla script. If it looks like romanised Bangla, convert and retry.
    from .translit import has_banglish

    if has_banglish(text):
        from .translit import banglish_to_bangla

        converted = banglish_to_bangla(text)
        if is_bangla_text(converted):
            return resolve(detect_region(converted, default=default).tag)

    # Fall back to a script check for the other Indic scripts.
    for ch in text:
        if "\u0900" <= ch <= "\u097F":
            from .registry import _INDEX  # local import keeps the public API clean

            return _INDEX.get("hi", BANGLA)
    return BANGLA_BD if resolve(default).region == "BD" else BANGLA_IN
