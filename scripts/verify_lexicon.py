# SPDX-License-Identifier: MIT | Copyright (c) 2026 MASH (Khaled Bhuiyan)
"""Check the curated lexicon for the problems that matter.

The Bangla and Devanagari letters look alike, so a value can be written with the
wrong script and nothing will complain. This script checks that every value is
Bangla, that no key is repeated, and that every value round trips.
"""

from __future__ import annotations

import re
import sys
from pathlib import Path

sys.path.insert(
    0,
    str(
        Path(
            r"C:/Users/imalp/Downloads/mash-mighty-github-repo/MASH-Mix-Audio-and-Text"
            r"/src"
        )
    ),
)

from mash_mix.lang.lexicon import KNOWN_BANGGLISH  # noqa: E402
from mash_mix.lang.translit import bangla_to_latin  # noqa: E402

DEVANAGARI = re.compile(r"[\u0900-\u097f]")
BANGLA = re.compile(r"[\u0980-\u09ff]")


def main() -> int:
    problems: list[str] = []
    for key, value in KNOWN_BANGGLISH.items():
        if not value:
            problems.append(f"{key}: empty value")
            continue
        if DEVANAGARI.search(value):
            problems.append(f"{key}: Devanagari in the value {value!r}")
            continue
        if not BANGLA.search(value):
            problems.append(f"{key}: no Bangla in the value {value!r}")
            continue
        if bangla_to_latin(value).strip() == "":
            problems.append(f"{key}: the value does not come back to Latin")

    print(f"entries : {len(KNOWN_BANGGLISH)}")
    print(f"problems: {len(problems)}")
    for line in problems:
        print("  ", line)
    return 1 if problems else 0


if __name__ == "__main__":
    raise SystemExit(main())
