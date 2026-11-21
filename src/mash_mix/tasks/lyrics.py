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

from typing import Any, List, Optional, Sequence, Tuple

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
    max_chars: int = 45,
) -> List[LyricLine]:
    """Join very short fragments with the next line when they belong together."""
    merged: List[LyricLine] = []
    for line in lines:
        if merged:
            previous = merged[-1]
            duration = previous.end - previous.start
            gap = line.start - previous.end
            combined = len(previous.text) + 1 + len(line.text)
            if duration < min_duration and gap <= max_gap and combined <= max_chars:
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


def _is_character_loop(text: str, min_repeats: int = 3) -> bool:
    """True when a run of characters repeats itself over and over.

    Some failures are not word shaped. A model can emit a syllable that is
    spelled with several letters, for example ``ma-th-ya-lam-ya-lam``, and
    the word level filter never sees a repeated word. This looks for a short
    block of characters that repeats back to back.

    A short block, three characters or fewer, is a syllable and three repeats
    of it are already a loop. A longer block has to cover at least half of the
    text, so a line that merely mentions the same sound twice is not thrown
    away.
    """
    compact = "".join(text.split())
    if len(compact) < 12:
        return False
    needed = max(12, int(len(compact) * 0.5))
    for size in range(1, 25):
        if len(compact) < size * min_repeats:
            break
        for start in range(0, len(compact) - size * min_repeats + 1):
            block = compact[start : start + size]
            repeats = 1
            cursor = start + size
            while compact.startswith(block, cursor):
                repeats += 1
                cursor += size
            if repeats < min_repeats:
                continue
            if size <= 3 or cursor - start >= needed:
                return True
    return False


def _has_replacement_char(text: str) -> bool:
    """True when the text carries the unicode replacement character.

    A broken decode leaves U+FFFD in the string. No lyric contains it, so the
    line is treated as noise.
    """
    return "\ufffd" in text


#: Punctuation that closes a phrase inside one recognised segment.
_SENTENCE_BREAK = ("\u0964", "\u0965", "?", "!", ".", "\u2026")
#: Punctuation that marks a smaller pause inside a phrase.
_CLAUSE_BREAK = (",", ";", ":", "\u0963")
#: A clause break only splits a line once this much text has accumulated.
_CLAUSE_MIN_CHARS = 18


def _ends_with(text: str, marks: Sequence[str]) -> bool:
    """True when the text ends with one of the given punctuation marks."""
    stripped = text.rstrip()
    if not stripped:
        return False
    return stripped[-1] in marks


def _split_text_on_punctuation(text: str, max_chars: int = 55) -> List[Tuple[str, int, int]]:
    """Cut a run of text into pieces at the punctuation it carries.

    Returns a list of ``(piece, start_offset, end_offset)`` where the offsets
    are character positions in the original text, so the caller can give each
    piece a share of the time the whole run took.

    This is the path used when the recogniser gives segment timings only. A
    sung line carries almost no pause between words, so the punctuation the
    model wrote is the better place to cut.
    """
    pieces: List[Tuple[str, int, int]] = []
    start = 0
    length = 0
    for index, char in enumerate(text):
        if _ends_with(text[index : index + 1], _SENTENCE_BREAK):
            piece = text[start : index + 1].strip()
            if piece:
                pieces.append((piece, start, index + 1))
            start = index + 1
            length = 0
            continue
        if _ends_with(text[index : index + 1], _CLAUSE_BREAK) and length >= _CLAUSE_MIN_CHARS:
            piece = text[start : index + 1].strip()
            if piece:
                pieces.append((piece, start, index + 1))
            start = index + 1
            length = 0
            continue
        length += 1
        if length >= max_chars and char == " ":
            piece = text[start : index + 1].strip()
            if piece:
                pieces.append((piece, start, index + 1))
            start = index + 1
            length = 0
    tail = text[start:].strip()
    if tail:
        pieces.append((tail, start, len(text)))
    return pieces


def _split_segment_without_words(seg: Segment, max_chars: int = 55) -> List[Segment]:
    """Cut a segment that carries no word timings into timed lyric lines."""
    pieces = _split_text_on_punctuation(seg.text, max_chars=max_chars)
    if len(pieces) <= 1:
        return [seg]
    total = max(1, len(seg.text))
    span = max(0.0, float(seg.end) - float(seg.start))
    out: List[Segment] = []
    for piece, begin, end in pieces:
        start = float(seg.start) + span * (begin / total)
        stop = float(seg.start) + span * (end / total)
        if stop <= start:
            stop = start + 0.01
        out.append(Segment(start=start, end=stop, text=piece))
    return out


def _split_on_pauses(
    segments: Sequence[Segment],
    *,
    gap: float = 0.35,
    max_chars: int = 55,
) -> List[Segment]:
    """Break long recognised segments into lyric lines.

    A recogniser returns one segment per utterance, and on a song that is one
    long run of words for a whole verse. Karaoke timing needs the lines, so
    the segment is cut where the singer breathes, where the recogniser wrote a
    full stop, and where a line would grow past a comfortable length.

    A sung line rarely carries a long pause between words, so the punctuation
    written by the model is the more useful signal here. A clause break only
    splits a line once enough text has gathered, which keeps short clauses
    attached to the phrase they belong to.
    """
    out: List[Segment] = []
    for seg in segments:
        words = list(seg.words or [])
        if not words:
            out.extend(_split_segment_without_words(seg, max_chars=max_chars))
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
            if _ends_with(word.text, _SENTENCE_BREAK):
                out.append(_segment_from_words(group))
                group = []
                length = 0
            elif _ends_with(word.text, _CLAUSE_BREAK) and length >= _CLAUSE_MIN_CHARS:
                out.append(_segment_from_words(group))
                group = []
                length = 0
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


def _recognise(
    engine: Any,
    buffer: AudioBuffer,
    *,
    language: Optional[str],
    initial_prompt: Optional[str],
    hotwords: Optional[Sequence[str]],
    engine_vad: bool,
    progress: bool,
    chunk_seconds: float,
    overlap_seconds: float,
    kwargs: dict,
) -> Tuple[List[Segment], int]:
    """Read a buffer with one recogniser and return its segments.

    The audio is handed to the model in one piece when the model can take a
    long file. A Whisper model reads a whole track on its own and keeps the
    thread of the song across the whole track, while the chunked path asks
    the model the same question again and again with no memory of what came
    before, and a song is exactly where that shows. The chunked path is kept
    for the models that cannot take a long file.
    """
    samples = buffer.mono()
    if not len(samples):
        return [], 0

    call_kwargs: dict = {
        "language": language,
        "task": "transcribe",
        "initial_prompt": initial_prompt,
        "hotwords": list(hotwords) if hotwords else None,
        "word_timestamps": True,
        "vad": engine_vad,
        "progress": progress,
        "condition_on_previous_text": False,
    }
    call_kwargs.update(kwargs)

    def run(block: Any) -> Any:
        try:
            return engine.transcribe(block, buffer.sample_rate, **call_kwargs)
        except TypeError:
            # A backend that does not take every keyword still does the work.
            # The optional keywords are dropped one at a time, from the one
            # that matters least to the one that matters most, until the
            # call is accepted.
            optional = [
                "condition_on_previous_text",
                "progress",
                "vad",
                "word_timestamps",
                "hotwords",
                "initial_prompt",
            ]
            accepted = dict(call_kwargs)
            for name in optional:
                if name not in accepted:
                    continue
                trimmed = dict(accepted)
                trimmed.pop(name, None)
                try:
                    return engine.transcribe(block, buffer.sample_rate, **trimmed)
                except TypeError:
                    accepted = trimmed
                    continue
            raise

    if _takes_long_audio(engine):
        try:
            partial = run(samples)
        except Exception:
            partial = None
        if partial is not None and partial.segments:
            return list(partial.segments), 1

    from ..audio.chunk import chunk_audio

    chunks = chunk_audio(
        samples,
        buffer.sample_rate,
        chunk_seconds=chunk_seconds,
        overlap_seconds=overlap_seconds,
    )
    segments: List[Segment] = []
    for chunk in chunks:
        block = chunk["samples"]
        if block is None or len(block) == 0:
            continue
        offset = chunk["start"]
        partial = run(block)
        segments.extend(_shift(partial.segments, offset))
    return segments, len(chunks)


def _takes_long_audio(engine: Any) -> bool:
    """True when the recogniser can read a whole track in one call.

    The Whisper family handles a long file itself, with its own windowing
    and its own voice activity detection, and it keeps the context of the
    song as it goes. A model that answers one window at a time cannot.
    """
    name = str(getattr(engine, "name", "") or "").lower()
    if not name:
        return False
    if name in ("faster-whisper", "whisper", "openai-whisper"):
        return True
    if name == "transformers":
        # The transformers pipeline batches the windows itself, which has
        # proved unstable on a CPU, so the chunked path is used for it.
        return False
    return False


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
    quality: str = "auto",
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
    quality:
        Which model to reach for. ``auto`` picks the model that fits the
        language, ``best`` forces the strongest one and ``fast`` starts at
        the quick one. When a model cannot be loaded the next one in the
        ladder is tried, so a request always comes back with lyrics.
    """
    from ..backends.registry import get_stt
    from ..lang.dialect import detect_region
    from ..lang.normalize import clean_asr_text, is_bangla_text, normalize_bangla
    from ..lang.registry import is_bangla
    from ..lang.script import needs_script_repair, repair_to_bangla
    from ..model_plan import plan_for

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

    # The ladder of models to try. The first one that loads and reads the
    # audio does the work, and the result says which one it was.
    candidates = plan_for(language, quality=quality, model=model, backend=backend)
    chosen = None
    segments: List[Segment] = []
    failures: List[str] = []
    chunk_count = 0

    with Timer() as timer:
        for candidate in candidates:
            try:
                engine = get_stt(candidate.backend, model=candidate.model)
            except Exception as exc:
                # The model is not installed, not downloaded or does not fit in
                # memory. The next one in the ladder is tried.
                failures.append(f"{candidate.model}: {exc}")
                continue

            try:
                read_segments, read_chunks = _recognise(
                    engine,
                    buffer,
                    language=language,
                    initial_prompt=initial_prompt,
                    hotwords=hotwords,
                    engine_vad=engine_vad,
                    progress=progress,
                    chunk_seconds=chunk_seconds,
                    overlap_seconds=overlap_seconds,
                    kwargs=kwargs,
                )
            except Exception as exc:
                failures.append(f"{candidate.model}: {exc}")
                continue

            if not read_segments and candidates.index(candidate) + 1 < len(candidates):
                # The model loaded but heard nothing. A stronger model may do
                # better, so the ladder continues.
                failures.append(f"{candidate.model}: heard nothing")
                continue

            chosen = candidate
            segments = read_segments
            chunk_count = read_chunks
            name = candidate.backend
            break

    if chosen is None:
        detail = "; ".join(failures) if failures else "no model was available"
        raise RuntimeError(f"The lyrics could not be read. {detail}")

    lines: List[LyricLine] = []
    bangla_requested = is_bangla(language) if language else False
    repaired = 0
    dropped_loops = 0
    dropped_noise = 0
    collapsed = 0
    for seg in _split_on_pauses(segments):
        text = clean_asr_text(seg.text)
        if not text:
            continue
        if _has_replacement_char(text):
            # A broken decode leaves a replacement character in the string.
            # No lyric contains one, so the line is noise.
            dropped_noise += 1
            continue
        if _is_repetition_loop(text) or _is_character_loop(text):
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
        model=(chosen.model if chosen else model) or "",
        elapsed=timer.elapsed,
        meta={
            "chunks": chunk_count,
            "model_used": (chosen.model if chosen else "") or "",
            "model_label": (chosen.label if chosen else "") or "",
            "quality": quality,
            "model_fallbacks": failures,
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
            "dropped_noise": dropped_noise,
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
