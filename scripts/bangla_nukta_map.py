# SPDX-License-Identifier: MIT | Copyright (c) 2026 MASH (Khaled Bhuiyan)
"""Show the Bangla nukta map with explicit code points.

Editors and chat clients normalise Bengali characters silently, so the map in
``mash_mix/lang/normalize.py`` is written from escapes. This script prints the
map so the pairs can be checked by eye.

Usage
-----
    python scripts/bangla_nukta_map.py
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


def show(text: str) -> str:
    return " ".join(f"U+{ord(ch):04X}" for ch in text)


def main() -> int:
    print("Decompose, the precomposed letter becomes the base plus NUKTA:")
    for precomposed, decomposed in _NUKTA_MAP.items():
        print(f"  {show(precomposed)}  ->  {show(decomposed)}")

    print()
    print("Compose, the other way:")
    for decomposed, precomposed in _NUKTA_COMPOSE_MAP.items():
        print(f"  {show(decomposed)}  ->  {show(precomposed)}")

    print()
    sample = "ড় ঢ় য়"
    print(f"sample      {sample}   {show(sample)}")
    print(f"decomposed  {canonical_nukta(sample)}   {show(canonical_nukta(sample))}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
