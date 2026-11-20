# SPDX-License-Identifier: MIT | Copyright (c) 2026 MASH (Khaled Bhuiyan)
"""Extract lyrics from a Bangla song and write them as SRT and TXT.

Usage
-----
    python examples/lyrics_from_song.py "my song.mp3"
    python examples/lyrics_from_song.py "my song.mp3" --lang bn-IN --separate
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from mash_mix.tasks.lyrics import extract_lyrics_to_files  # noqa: E402


def main() -> int:
    parser = argparse.ArgumentParser(description="Extract lyrics from a song.")
    parser.add_argument("audio", help="The song file. Any format, any length.")
    parser.add_argument("--lang", default="bn-BD", help="bn, bn-BD or bn-IN. Default: bn-BD")
    parser.add_argument("--out", default="lyrics-out", help="Output folder.")
    parser.add_argument(
        "--separate",
        action="store_true",
        help="Separate the vocals before recognition. Better on a dense mix.",
    )
    parser.add_argument("--title", default=None)
    parser.add_argument("--artist", default=None)
    args = parser.parse_args()

    written = extract_lyrics_to_files(
        args.audio,
        args.out,
        stem=args.title or Path(args.audio).stem,
        formats=["srt", "txt", "vtt", "lrc"],
        language=args.lang,
        title=args.title,
        artist=args.artist,
        separate_vocals=args.separate,
    )
    result = written.pop("result")

    print(f"language   {result.language}")
    print(f"lines      {len(result.lines)}")
    print(f"duration   {result.duration:.1f} s")
    print(f"elapsed    {result.elapsed:.1f} s")
    print()
    for fmt, path in written.items():
        print(f"{fmt:<5} {path}")
    print()
    print(result.text)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
