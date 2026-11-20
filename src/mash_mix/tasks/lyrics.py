# SPDX-License-Identifier: MIT | Copyright (c) 2026 MASH (Khaled Bhuiyan)
"""Song and speech to lyric extraction.

This is the headline task: drop in a Bangla song of any length and any format,
get back lyric lines with timings, ready to export as SRT or plain text.

Singing voice is hard for any speech model, so the pipeline does the work that
makes the difference:

* optional vocal separation, so the model hears the voice and not the mix
* voice activity detection, so instrumental passages are skipped instead of
  being hallucinated into lyrics
* repetition control, because a model that loops on a sung phrase will
  otherwise repeat it for the rest of the track
* Bangla cleaning and regional detection on the final text
"""

from __future__ import annotations

from typing import Any, List, Optional, Sequence

from ..types import AudioBuffer, LyricLine, LyricsResult, Segment
from ._common import Timer, pick_backend, to_buffer

__all__ = ["extract_lyrics", "extract_lyrics_to_files"]


def _drop_repeated_lines(lines: List[LyricLine], *, max_repeats: int = 2) -> List[LyricLine]:
    """Remove consecutive identical lyric lines produced by a decoding loop."""
    out: List[LyricLine] = []
    run_text: Optional[str] = None
    run_count = 0
    for line in lines:
        key = " ".join(line.text.split()).strip().lower()
        if key and key == run_text:
            run_count += 1
            if run_count > max_repeats:
                continue
        else:
            run_text = key
            run_count = 1
        out.append(line)
    return out


def _merge_short_lines(
    lines: List[LyricLine],
    *,
    min_duration: float = 1.2,
    max_gap: float = 0.8,
) -> List[LyricLine]:
    """Join very short fragments with the next line when they belong together."""
    merged: List[LyricLine] = []
    for line in lines:
        if merged:
            previous = merged[-1]
            duration = previous.end - previous.start
            gap = line.start - previous.end
            if duration < min_duration and gap <= max_gap:
                previous.text = f"{previous.text} {line.text}".strip()
                previous.end = max(previous.end, line.end)
                continue
        merged.append(line)
    return merged


def _shift(segments: Sequence[Segment], offset: float) -> List[Segment]:
    out: List[Segment] = []
    for seg in segments:
        out.append(
            Segment(
                start=seg.start + offset,
                end=seg.end + offset,
                text=seg.text,
                words=[
                    type(w)(
                        start=w.start + offset,
                        end=w.end + offset,
                        text=w.text,
                        probability=w.probability,
                    )
                    for w in seg.words
                ],
                speaker=seg.speaker,
                language=seg.language,
                probability=seg.probability,
            )
        )
    return out


def extract_lyrics(
    audio: Any,
    *,
    backend: str = "auto",
    model: Optional[str] = None,
    language: Optional[str] = "bn-BD",
    title: Optional[str] = None,
    artist: Optional[str] = None,
    separate_vocals: bool = False,
    separation_model: str = "htdemucs",
    chunk_seconds: float = 30.0,
    overlap_seconds: float = 2.0,
    hotwords: Optional[Sequence[str]] = None,
    initial_prompt: Optional[str] = None,
    max_repeats: int = 2,
    merge_short: bool = True,
    engine_vad: bool = False,
    sample_rate: int = 16000,
    progress: bool = False,
    **kwargs: Any,
) -> LyricsResult:
    """Extract lyrics from a song or spoken track.

    Parameters
    ----------
    audio:
        Path, file object or AudioBuffer. Any format and any length.
    language:
        Language hint. ``bn-BD`` for Bangladeshi Bangla, ``bn-IN`` for Indian
        Bengali, ``bn`` for automatic regional choice. Pass ``None`` to let
        the model decide.
    separate_vocals:
        Pull the vocal stem out of the mix before recognition. Much better on
        dense mixes, slower because it runs a separation model.
    engine_vad:
        Let the recogniser run its own voice activity detection. Off by
        default, because a speech VAD trained on conversation cuts through
        sustained sung phrases and leaves a choir with fragments. The silence
        aware chunking in this module already skips the quiet parts.
    """
    from ..audio.chunk import chunk_audio
    from ..backends.registry import get_stt
    from ..lang.dialect import detect_region
    from ..lang.normalize import clean_asr_text, is_bangla_text, normalize_bangla
    from ..lang.registry import is_bangla
    from ..lang.script import needs_script_repair, repair_to_bangla

    buffer = to_buffer(audio, sample_rate=sample_rate)

    if separate_vocals:
        from ..audio.separate import separate

        try:
            separated = separate(buffer.mono(), buffer.sample_rate, model=separation_model)
        except Exception:
            separated = None
        if separated and "vocals" in separated.stems:
            buffer = AudioBuffer(separated.stems["vocals"], separated.sample_rate, 1)
            buffer = to_buffer(buffer, sample_rate=sample_rate)

    name = pick_backend("stt", backend)

    with Timer() as timer:
        engine = get_stt(name, model=model)

        chunks = chunk_audio(
            buffer.mono(),
            buffer.sample_rate,
            chunk_seconds=chunk_seconds,
            overlap_seconds=overlap_seconds,
        )

        segments: List[Segment] = []
        for chunk in chunks:
            samples = chunk["samples"]
            if samples is None or len(samples) == 0:
                continue
            offset = chunk["start"]
            try:
                partial = engine.transcribe(
                    samples,
                    buffer.sample_rate,
                    language=language,
                    task="transcribe",
                    initial_prompt=initial_prompt,
                    hotwords=list(hotwords) if hotwords else None,
                    word_timestamps=True,
                    vad=engine_vad,
                    progress=progress,
                    condition_on_previous_text=False,
                    **kwargs,
                )
            except TypeError:
                partial = engine.transcribe(
                    samples,
                    buffer.sample_rate,
                    language=language,
                    task="transcribe",
                    initial_prompt=initial_prompt,
                    hotwords=list(hotwords) if hotwords else None,
                    word_timestamps=True,
                    vad=engine_vad,
                    progress=progress,
                    **kwargs,
                )
            segments.extend(_shift(partial.segments, offset))

    lines: List[LyricLine] = []
    bangla_requested = is_bangla(language) if language else False
    repaired = 0
    for seg in segments:
        text = clean_asr_text(seg.text)
        if not text:
            continue
        if bangla_requested and needs_script_repair(text, target="bangla"):
            # The model answered in Devanagari. Bring it back to Bangla.
            text = repair_to_bangla(text)
            repaired += 1
        if is_bangla_text(text):
            text = normalize_bangla(text, digits="keep")
        lines.append(
            LyricLine(
                start=float(seg.start),
                end=float(seg.end),
                text=text,
                speaker=seg.speaker,
                probability=seg.probability,
            )
        )

    lines = _drop_repeated_lines(lines, max_repeats=max_repeats)
    if merge_short:
        lines = _merge_short_lines(lines)

    full_text = "\n".join(line.text for line in lines)
    detected = detect_region(full_text, default=language or "bn-BD")

    return LyricsResult(
        lines=lines,
        language=detected.tag if language is None else language,
        duration=buffer.duration,
        title=title,
        artist=artist,
        backend=name,
        model=model or (engine.model_id or ""),
        elapsed=timer.elapsed,
        meta={
            "chunks": len(chunks),
            "dialect_confidence": detected.confidence,
            "dialect": detected.dialect,
            "evidence": detected.evidence,
            "bd_score": detected.bd_score,
            "in_score": detected.in_score,
            "script": detected.script,
            "script_repaired_lines": repaired,
            "engine_vad": engine_vad,
        },
    )


def extract_lyrics_to_files(
    audio: Any,
    output_dir: str,
    *,
    stem: Optional[str] = None,
    formats: Sequence[str] = ("srt", "txt"),
    **kwargs: Any,
) -> dict:
    """Extract lyrics and write the requested files.

    Returns a dict with the written paths keyed by format, plus the result.
    """
    from pathlib import Path

    from ..export.plaintext import write_text
    from ..export.subtitles import write_subtitles

    result = extract_lyrics(audio, **kwargs)

    folder = Path(output_dir)
    folder.mkdir(parents=True, exist_ok=True)
    base = stem or (result.title or "lyrics")

    written: dict = {"result": result}
    for fmt in formats:
        lowered = fmt.lower()
        if lowered == "srt":
            path = folder / f"{base}.srt"
            write_subtitles(result.lines, str(path), fmt="srt", title=result.title)
        elif lowered == "txt":
            path = folder / f"{base}.txt"
            write_text(str(path), result.text + "\n")
        elif lowered in ("vtt", "webvtt"):
            path = folder / f"{base}.vtt"
            write_subtitles(result.lines, str(path), fmt="vtt", title=result.title)
        elif lowered == "lrc":
            path = folder / f"{base}.lrc"
            write_subtitles(result.lines, str(path), fmt="lrc")
        elif lowered == "json":
            path = folder / f"{base}.json"
            write_text(str(path), result.to_dict() and __import__("json").dumps(
                result.to_dict(), ensure_ascii=False, indent=2
            ))
        else:
            continue
        written[lowered] = str(path)

    return written
