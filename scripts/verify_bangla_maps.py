# SPDX-License-Identifier: MIT | Copyright (c) 2026 MASH (Khaled Bhuiyan)
"""Self check for the Bangla script tables.

The Bangla and Devanagari letters look alike, so a mapping table can lose a
nukta or pick up the wrong sign without anyone noticing. This script checks
the tables in ``mash_mix.lang.script`` and ``mash_mix.lang.normalize`` and
prints the result.

Usage
-----
    python scripts/verify_bangla_maps.py
"""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from mash_mix.lang.normalize import (  # noqa: E402
    _NUKTA_COMPOSE_MAP,
    _NUKTA_MAP,
    canonical_nukta,
)
from mash_mix.lang.script import (  # noqa: E402
    _BANGLA_TO_DEVANAGARI,
    _DEVANAGARI_TO_BANGLA,
    bangla_to_devanagari,
    devanagari_to_bangla,
)


def show(text: str) -> str:
    return " ".join(f"U+{ord(ch):04X}" for ch in text)


def main() -> int:
    problems = []

    print("Nukta letters, the precomposed form and the base plus NUKTA:")
    for precomposed, decomposed in _NUKTA_MAP.items():
        print(f"  {show(precomposed)}  ->  {show(decomposed)}")
        if len(decomposed) != 2:
            problems.append(f"the nukta map entry {show(precomposed)} is not a pair")
        if canonical_nukta(precomposed) != decomposed:
            problems.append(f"canonical_nukta does not split {show(precomposed)}")
        if _NUKTA_COMPOSE_MAP.get(decomposed) != precomposed:
            problems.append(f"the compose map does not rebuild {show(precomposed)}")

    print()
    print("Script repair, the two directions:")
    for source, target in list(_DEVANAGARI_TO_BANGLA.items())[:8]:
        print(f"  {show(source)}  ->  {show(target)}")

    sample_deva = "".join(sorted(_DEVANAGARI_TO_BANGLA)[:40])
    sample_bangla = devanagari_to_bangla(sample_deva)
    if bangla_to_devanagari(sample_bangla) != sample_deva:
        problems.append("the two directions do not round trip")

    print()
    print(f"Devanagari to Bangla entries: {len(_DEVANAGARI_TO_BANGLA)}")
    print(f"Bangla to Devanagari entries: {len(_BANGLA_TO_DEVANAGARI)}")

    if problems:
        print()
        print("PROBLEMS")
        for item in problems:
            print(f"  - {item}")
        return 1

    print()
    print("all checks passed")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
