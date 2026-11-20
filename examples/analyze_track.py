# SPDX-License-Identifier: MIT | Copyright (c) 2026 MASH (Khaled Bhuiyan)
"""Measure a track: tempo, key, loudness and structure.

Usage
-----
    python examples/analyze_track.py "song.mp3"
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from mash_mix.tasks.analyze import analyze  # noqa: E402


def main() -> int:
    if len(sys.argv) < 2:
        print("Give me an audio file.", file=sys.stderr)
        return 1

    result = analyze(sys.argv[1])
    report = result.to_dict()
    print(json.dumps(report, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
