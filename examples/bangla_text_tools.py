# SPDX-License-Identifier: MIT | Copyright (c) 2026 MASH (Khaled Bhuiyan)
"""Prepare Bangla text for a speech engine and report what changed.

Shows the pieces of the Bangla layer: normalisation, number reading, symbol
expansion and romanised Bangla conversion.

Usage
-----
    python examples/bangla_text_tools.py "100% off, ami tomay bhalobashi"
"""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from mash_mix.lang import (  # noqa: E402
    bangla_to_latin,
    banglish_to_bangla,
    detect_region,
    expand_symbols,
    normalize_bangla,
    normalize_for_tts,
    number_to_bangla_words,
)


def show(label: str, value: str) -> None:
    print(f"{label:<26} {value}")


def main() -> int:
    raw = " ".join(sys.argv[1:]) or "100% off, ami tomay bhalobashi"

    show("original", raw)
    show("normalised", normalize_bangla(raw, digits="native"))
    show("for the voice engine", normalize_for_tts(raw, language="bn-BD"))
    show("symbols spelled out", expand_symbols(raw))
    show("as Bangla script", banglish_to_bangla(raw))
    show("as Latin", bangla_to_latin(raw))
    print()
    for value in (0, 11, 42, 100, 1234, 12345678):
        show(f"number {value}", number_to_bangla_words(value))
    print()
    report = detect_region(banglish_to_bangla(raw))
    show("detected region", f"{report.tag} (confidence {report.confidence})")
    show("evidence", ", ".join(report.evidence[:6]) or "no regional markers")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
