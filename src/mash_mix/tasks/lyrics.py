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
from ._common import Timer, pick_backend, to_buffer, to_stereo_buffer

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


def _collapse_word_repeats(text: str, max_repeat: int = 2) -> str:
    """Collapse a word that the recogniser repeated over and over."""
    words = text.split()
    if not words:
        return ""
    out: List[str] = []
    previous: Optional[str] = None
    count = 0
    for word in words:
        key = word.lower()
        if key == previous:
            count += 1
            if count > max_repeat:
                continue
        else:
            previous = key
            count = 1
        out.append(word)
    return " ".join(out)


def _is_repetition_loop(text: str, threshold: int = 6) -> bool:
    """True when a line is one word or one phrase repeated over and over.

    A speech model that loses the plot on sung audio falls into this loop and
    writes the same syllable until it runs out of tokens. The line carries no
    lyric, so it is dropped.
    """
    words = text.split()
    if len(words) < threshold:
        return False
    distinct = len({word.lower() for word in words})
    return distinct <= max(2, len(words) // 4)


def _split_on_pauses(
    segments: Sequence[Segment],
    *,
    gap: float = 0.35,
    max_chars: int = 55,
) -> List[Segment]:
    """Break long recognised segments into lyric lines at the pauses.

    A recogniser returns one segment per utterance, and on a song that is one
    long run of words for a whole verse. Karaoke timing needs the lines, so
    the segment is cut where the singer breathes, and where a line would grow
    past a comfortable length.
    """
    out: List[Segment] = []
    for seg in segments:
        words = list(seg.words or [])
        if not words:
            out.append(seg)
            continue
        group: List[Any] = []
        length = 0
        for word in words:
            if group:
                previous = group[-1]
                pause = float(word.start) - float(previous.end)
                if pause > gap or length + len(word.text) > max_chars:
                    out.append(_segment_from_words(group))
                    group = []
                    length = 0
            group.append(word)
            length += len(word.text) + 1
        if group:
            out.append(_segment_from_words(group))
    return out


def _segment_from_words(words: Sequence[Any]) -> Segment:
    """Build one segment out of a run of words."""
    return Segment(
        start=float(words[0].start),
        end=float(words[-1].end),
        text=" ".join(word.text for word in words).strip(),
        words=list(words),
    )


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
    script: str = "bangla",
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
    script:
        The writing system of the returned lines. ``bangla`` keeps the native
        script and writes out any romanised Bangla the recogniser produced.
        ``banglish`` returns the same content in Latin letters.
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
            # The separator needs the stereo mix at its own sample rate. The
            # mono 16 kHz buffer below has already thrown that away, so the
            # file is opened again for the separation step.
            stereo = to_stereo_buffer(audio)
            separated = separate(
                stereo.samples,
                stereo.sample_rate,
                model=separation_model,
            )
        except Exception:
            separated = None
        if separated and "vocals" in separated.stems:
            buffer = to_buffer(
                AudioBuffer(separated.stems["vocals"], separated.sample_rate, 1),
                sample_rate=sample_rate,
            )

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
    dropped_loops = 0
    collapsed = 0
    for seg in _split_on_pauses(segments):
        text = clean_asr_text(seg.text)
        if not text:
            continue
        if _is_repetition_loop(text):
            # The recogniser fell into a loop on this passage. It carries no
            # lyric, so the line goes.
            dropped_loops += 1
            continue
        tightened = _collapse_word_repeats(text)
        if tightened != text:
            collapsed += 1
            text = tightened
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

    # One decision for the whole document: is this song written in romanised
    # Bangla, and which script should the lines come back in.
    from ..lang.translit import apply_script, resolve_script_target

    target_script = resolve_script_target(full_text, script)
    if target_script:
        for line in lines:
            line.text = apply_script(line.text, target_script)
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
            "output_script": target_script or "as recognised",
            "sample_rate": buffer.sample_rate,
            "channels": buffer.channels,
            "dropped_loops": dropped_loops,
            "collapsed_repeats": collapsed,
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
