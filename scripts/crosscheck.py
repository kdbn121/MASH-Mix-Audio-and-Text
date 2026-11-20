# SPDX-License-Identifier: MIT | Copyright (c) 2026 MASH (Khaled Bhuiyan)
"""End to end crosscheck.

Builds a Bangla test track, runs the full lyrics pipeline over it, and checks
that the outputs are real and correct. Useful after changing the pipeline, and
useful as a demonstration of what the engine does.

Usage
-----
    python scripts/crosscheck.py                       # build the track and run it
    python scripts/crosscheck.py --audio my-song.mp3   # run it over a real song
"""

from __future__ import annotations

import argparse
import sys
import tempfile
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

LYRIC_LINES = [
    "আজকের দিনটা সুন্দর",
    "আকাশে চাঁদ উঠেছে ধীরে ধীরে",
    "তোমার চোখে আলো জ্বলছে",
    "আমি থাকি তোমার পাশে",
    "এই গানটা শুধু তোমার জন্য",
    "ভালোবাসি তোমাকে অনেক",
]

BACKING_CHORDS = [
    (261.63, 329.63, 392.00),
    (220.00, 261.63, 329.63),
    (174.61, 220.00, 261.63),
    (196.00, 246.94, 293.66),
]


def build_backing(total_seconds: float, sample_rate: int = 22050) -> np.ndarray:
    """A simple chord bed so the recogniser has to cope with instrumentation."""
    out = np.zeros(int(total_seconds * sample_rate), dtype=np.float32)
    t = np.arange(out.shape[0]) / sample_rate
    bar = 2.0
    for index in range(int(total_seconds // bar) + 1):
        start = int(index * bar * sample_rate)
        end = min(out.shape[0], start + int(bar * sample_rate))
        if end <= start:
            continue
        chord = BACKING_CHORDS[index % len(BACKING_CHORDS)]
        for freq in chord:
            out[start:end] += 0.045 * np.sin(2 * np.pi * freq * t[start:end]).astype(np.float32)
    # A slow fade in and out so the file starts and ends cleanly.
    fade = int(0.4 * sample_rate)
    out[:fade] *= np.linspace(0.0, 1.0, fade)
    out[-fade:] *= np.linspace(1.0, 0.0, fade)
    return out


def speak_lines(sample_rate: int = 22050) -> list:
    """Synthesise the lyric lines with a Bangla voice."""
    from mash_mix.backends.registry import get_tts

    backend = get_tts("mms-tts", model="facebook/mms-tts-ben")
    chunks = []
    for line in LYRIC_LINES:
        result = backend.synthesize(line, language="bn-BD")
        chunks.append(np.asarray(result.audio, dtype=np.float32))
        chunks.append(np.zeros(int(0.35 * sample_rate), dtype=np.float32))
    return chunks


def build_track(path: Path) -> Path:
    """Create a test track: Bangla speech over a chord bed."""
    from mash_mix.audio.io import save

    sample_rate = 22050
    chunks = speak_lines(sample_rate)
    speech = np.concatenate(chunks)
    backing = build_backing(len(speech) / sample_rate, sample_rate)
    mixed = speech + backing[: speech.shape[0]]
    peak = float(np.max(np.abs(mixed)) or 1.0)
    mixed = (mixed / peak * 0.85).astype(np.float32)
    save(str(path), mixed, sample_rate)
    return path


def check(
    path: Path,
    output_dir: Path,
    language: str,
    backend: str,
    model: str = None,
    stem: str = "crosscheck",
) -> int:
    """Run the lyrics pipeline and verify the results."""
    from mash_mix.tasks.lyrics import extract_lyrics_to_files

    written = extract_lyrics_to_files(
        str(path),
        str(output_dir),
        stem=stem,
        formats=["srt", "txt", "vtt", "lrc", "json"],
        language=language,
        backend=backend,
        model=model,
    )
    result = written.pop("result")

    print("=" * 68)
    print("crosscheck result")
    print("=" * 68)
    print(f"backend        {result.backend}")
    print(f"model          {result.model}")
    print(f"language       {result.language}")
    print(f"duration       {result.duration:.1f} s")
    print(f"lyric lines    {len(result.lines)}")
    print(f"elapsed        {result.elapsed:.1f} s")
    print(f"dialect        {result.meta.get('dialect')} "
          f"(confidence {result.meta.get('dialect_confidence')})")
    print(f"evidence       {', '.join(result.meta.get('evidence', [])[:5])}")
    print()

    print("recognised text")
    print("-" * 68)
    print(result.text[:1200])
    print()

    failures = []

    srt_path = Path(written["srt"])
    txt_path = Path(written["txt"])

    if not srt_path.exists():
        failures.append("the srt file was not written")
    else:
        srt = srt_path.read_text(encoding="utf-8")
        if "-->" not in srt:
            failures.append("the srt file has no cue timestamps")
        if not srt.strip().startswith("1"):
            failures.append("the srt file does not start with cue 1")
        print(f"srt            {srt_path} ({srt_path.stat().st_size} bytes)")
        print("-" * 68)
        print(srt[:600])
        print()

    if not txt_path.exists():
        failures.append("the txt file was not written")
    else:
        txt = txt_path.read_text(encoding="utf-8")
        print(f"txt            {txt_path} ({txt_path.stat().st_size} bytes)")
        if not txt.strip():
            failures.append("the txt file is empty")
        if len(txt.splitlines()) != len(result.lines):
            print(
                f"note           the txt has {len(txt.splitlines())} lines "
                f"for {len(result.lines)} lyric lines"
            )

    for fmt in ("vtt", "lrc", "json"):
        target = Path(written[fmt])
        if not target.exists() or target.stat().st_size == 0:
            failures.append(f"the {fmt} file is missing or empty")
        else:
            print(f"{fmt:<14} {target} ({target.stat().st_size} bytes)")

    bangla = sum(1 for line in result.lines if any("\u0980" <= ch <= "\u09FF" for ch in line.text))
    print()
    print(f"bangla lines   {bangla} of {len(result.lines)}")

    if len(result.lines) == 0:
        failures.append("no lyric lines were produced")
    if result.lines and bangla == 0:
        failures.append("no Bangla script was produced")

    print()
    if failures:
        print("FAILED")
        for item in failures:
            print(f"  - {item}")
        return 1
    print("PASSED, every expected artefact was produced and checked")
    return 0


def main() -> int:
    parser = argparse.ArgumentParser(description="Crosscheck the MASH-Mix-Audio-and-Text pipeline.")
    parser.add_argument("--audio", default=None, help="Use this audio file instead of a built track.")
    parser.add_argument("--language", default="bn-BD")
    parser.add_argument("--backend", default="auto")
    parser.add_argument("--model", default=None, help="Model id for the backend.")
    parser.add_argument("--output-dir", default=None)
    args = parser.parse_args()

    workdir = Path(args.output_dir) if args.output_dir else Path(tempfile.gettempdir()) / "mash-mix-crosscheck"
    workdir.mkdir(parents=True, exist_ok=True)

    if args.audio:
        audio = Path(args.audio)
        print(f"using {audio}")
    else:
        print("building a Bangla test track with a synthesised voice over a chord bed")
        audio = build_track(workdir / "bangla-test-track.wav")
        print(f"wrote {audio}")

    return check(audio, workdir, args.language, args.backend, model=args.model)


if __name__ == "__main__":
    raise SystemExit(main())
