# SPDX-License-Identifier: MIT | Copyright (c) 2026 MASH (Khaled Bhuiyan)
"""Score extracted lyrics against a ground truth lyric file.

A song lyric engine is easy to build and hard to trust. This script puts a
number on the result. It runs the lyric pipeline over an audio file, then
compares the output with the real lyric text and reports:

* word error rate over the whole text, the standard measure for speech
* line recall, the share of true lines that the engine found with a close match
* the worst lines, so a failure is visible and not hidden inside an average

Usage
-----
    python scripts/lyrics_eval.py song.mp3 truth.txt
    python scripts/lyrics_eval.py song.mp3 truth.txt --backend faster-whisper \
        --model large-v3 --separate-vocals --out-dir out
"""

from __future__ import annotations

import argparse
import json
import re
import sys
import unicodedata
from pathlib import Path
from typing import Dict, List, Sequence, Tuple

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from mash_mix.tasks.lyrics import extract_lyrics  # noqa: E402

PUNCT = re.compile(r"[\u0964\u0965,.;:!?\"'()\[\]\u2013\u2014-]+")
WHITESPACE = re.compile(r"\s+")


def normalise(text: str) -> str:
    """Lower case, strip punctuation, collapse spaces, NFC."""
    text = unicodedata.normalize("NFC", text)
    text = PUNCT.sub(" ", text)
    return WHITESPACE.sub(" ", text).strip().lower()


def tokens(text: str) -> List[str]:
    return normalise(text).split()


def levenshtein(a: Sequence[str], b: Sequence[str]) -> int:
    """Edit distance over token lists."""
    if not a:
        return len(b)
    if not b:
        return len(a)
    previous = list(range(len(b) + 1))
    for i, item_a in enumerate(a, start=1):
        current = [i]
        for j, item_b in enumerate(b, start=1):
            cost = 0 if item_a == item_b else 1
            current.append(min(previous[j] + 1, current[j - 1] + 1, previous[j - 1] + cost))
        previous = current
    return previous[-1]


def word_error_rate(hypothesis: str, reference: str) -> float:
    """Word error rate of the hypothesis against the reference."""
    ref = tokens(reference)
    hyp = tokens(hypothesis)
    if not ref:
        return 0.0 if not hyp else 1.0
    return levenshtein(hyp, ref) / len(ref)


def similarity(a: str, b: str) -> float:
    """Character level similarity between two lines, from 0 to 1."""
    sa = normalise(a).replace(" ", "")
    sb = normalise(b).replace(" ", "")
    if not sa or not sb:
        return 1.0 if sa == sb else 0.0
    return 1.0 - (levenshtein(sa, sb) / max(len(sa), len(sb)))


def phrase_recall(
    truth: Sequence[str],
    hypothesis: Sequence[str],
    *,
    threshold: float = 0.7,
) -> Tuple[List[Dict[str, object]], float]:
    """Check whether every true line appears anywhere in the transcript.

    Line by line alignment is the wrong measure for a song. The same lyric can
    be transcribed as one long line or as four short ones, and that choice is
    made by the segmenter, not by the recogniser. This measure asks the
    content question instead: is this true line present somewhere in what was
    transcribed, wherever the line breaks happen to fall?

    A true line is counted as found when the closest window of transcript
    words scores at least ``threshold`` on character similarity.
    """
    hyp_tokens: List[str] = []
    hyp_owner: List[int] = []
    for index, line in enumerate(hypothesis):
        for token in tokens(line):
            hyp_tokens.append(token)
            hyp_owner.append(index)

    rows: List[Dict[str, object]] = []
    hits = 0
    for index, true_line in enumerate(truth):
        want = tokens(true_line)
        best_score = 0.0
        best_text = ""
        if want and hyp_tokens:
            width = len(want)
            for start in range(0, max(1, len(hyp_tokens) - width + 1)):
                window = hyp_tokens[start : start + width]
                score = similarity(" ".join(window), true_line)
                if score > best_score:
                    best_score = score
                    owner = hyp_owner[start]
                    best_text = hypothesis[owner] if owner < len(hypothesis) else ""
                if start + width >= len(hyp_tokens):
                    break
        ok = best_score >= threshold
        if ok:
            hits += 1
        rows.append(
            {
                "index": index,
                "truth": true_line,
                "best": best_text,
                "score": round(best_score, 3),
                "match": ok,
            }
        )
    recall = hits / len(truth) if truth else 0.0
    return rows, recall


def read_truth(path: Path) -> List[str]:
    """Read the truth file, keeping non empty lines and dropping the numbering."""
    out: List[str] = []
    for raw in path.read_text(encoding="utf-8").splitlines():
        line = raw.strip()
        if not line:
            continue
        line = re.sub(r"^\d+[.):]\s*", "", line)
        if line:
            out.append(line)
    return out


def match_lines(truth: Sequence[str], hypothesis: Sequence[str]) -> Tuple[List[Dict[str, object]], float]:
    """Match every true line with its best hypothesis line."""
    rows: List[Dict[str, object]] = []
    hits = 0
    for index, true_line in enumerate(truth):
        best_text = ""
        best_score = 0.0
        for candidate in hypothesis:
            score = similarity(true_line, candidate)
            if score > best_score:
                best_score = score
                best_text = candidate
        ok = best_score >= 0.7
        if ok:
            hits += 1
        rows.append(
            {
                "index": index,
                "truth": true_line,
                "best": best_text,
                "score": round(best_score, 3),
                "match": ok,
            }
        )
    recall = hits / len(truth) if truth else 0.0
    return rows, recall


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Score extracted lyrics against ground truth.")
    parser.add_argument("audio", help="Song file")
    parser.add_argument("truth", help="Ground truth lyric file")
    parser.add_argument("--backend", default="faster-whisper")
    parser.add_argument("--model", default="small")
    parser.add_argument("--language", default="bn-BD")
    parser.add_argument("--script", default="", choices=["", "bangla", "banglish"])
    parser.add_argument("--separate-vocals", action="store_true")
    parser.add_argument("--chunk-seconds", type=float, default=30.0)
    parser.add_argument("--out-dir", default="")
    args = parser.parse_args(argv)

    audio = Path(args.audio)
    truth_path = Path(args.truth)
    if not audio.exists():
        print(f"Audio file not found: {audio}")
        return 2
    if not truth_path.exists():
        print(f"Truth file not found: {truth_path}")
        return 2

    truth = read_truth(truth_path)

    kwargs: Dict[str, object] = {
        "backend": args.backend,
        "model": args.model,
        "language": args.language or None,
        "separate_vocals": bool(args.separate_vocals),
        "chunk_seconds": float(args.chunk_seconds),
    }
    if args.script:
        kwargs["script"] = args.script

    result = extract_lyrics(str(audio), **kwargs)  # type: ignore[arg-type]

    hypothesis = [line.text for line in result.lines]
    joined_hyp = "\n".join(hypothesis)
    joined_truth = "\n".join(truth)

    wer = word_error_rate(joined_hyp, joined_truth)
    rows, recall = match_lines(truth, hypothesis)
    phrase_rows, phrase = phrase_recall(truth, hypothesis)

    print("=" * 66)
    print(f"audio   : {audio.name}")
    print(f"model   : {args.backend} / {result.model or args.model}")
    print(f"language: {result.language}   duration: {result.duration:.1f} s")
    print(f"elapsed : {result.elapsed:.1f} s")
    print(f"lines   : {len(result.lines)} produced, {len(truth)} in the truth file")
    print(f"WER     : {wer * 100:.1f} %")
    print(f"lines   : {recall * 100:.1f} % of true lines matched one to one at 0.70 or better")
    print(f"content : {phrase * 100:.1f} % of true lines found somewhere in the transcript")
    print("=" * 66)
    for row in phrase_rows:
        mark = "OK  " if row["match"] else "MISS"
        print(f"{mark} {row['score']:.2f}  truth: {row['truth']}")
        if not row["match"]:
            print(f"          got : {row['best']}")
    print("=" * 66)

    if args.out_dir:
        out = Path(args.out_dir)
        out.mkdir(parents=True, exist_ok=True)
        (out / "hypothesis.txt").write_text(joined_hyp + "\n", encoding="utf-8")
        (out / "truth.txt").write_text(joined_truth + "\n", encoding="utf-8")
        report = {
            "audio": str(audio),
            "backend": args.backend,
            "model": result.model or args.model,
            "language": result.language,
            "duration": result.duration,
            "elapsed": result.elapsed,
            "wer": wer,
            "line_recall": recall,
            "content_recall": phrase,
            "lines_produced": len(result.lines),
            "lines_in_truth": len(truth),
            "meta": result.meta,
            "lines": phrase_rows,
        }
        (out / "report.json").write_text(
            json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8"
        )
        print(f"report written to {out / 'report.json'}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
