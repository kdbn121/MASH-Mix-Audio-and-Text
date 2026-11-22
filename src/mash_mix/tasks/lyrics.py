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

import time
from collections import OrderedDict
from typing import Any, Dict, List, Optional, Sequence, Tuple

from ..types import AudioBuffer, LyricLine, LyricsResult, Segment
from ._common import Timer, pick_backend, to_buffer, to_stereo_buffer

__all__ = ["extract_lyrics", "extract_lyrics_to_files", "extract_lyrics_from_url"]

# The words of a karaoke are read off the picture of the video with a model,
# and that takes minutes. A link that has been read is kept, so that the same
# link asked for again is answered at once instead of being read again.
_KARAOKE_READS: OrderedDict[str, Tuple[List[LyricLine], Dict[str, Any]]] = OrderedDict()
_KARAOKE_READS_MAX = 4


def _copy_lines(lines: Sequence[LyricLine]) -> List[LyricLine]:
    """A copy of a list of lines, so that a kept one is never changed."""
    return [
        LyricLine(
            start=line.start,
            end=line.end,
            text=line.text,
            speaker=line.speaker,
            probability=line.probability,
            words=list(line.words),
        )
        for line in lines
    ]


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
) -> Tuple[List[Segment], int, Dict[str, Any]]:
    """Read a buffer with one recogniser and return its segments.

    The audio is handed to the model in one piece when the model can take a
    long file. A Whisper model reads a whole track on its own and keeps the
    thread of the song across the whole track, while the chunked path asks
    the model the same question again and again with no memory of what came
    before, and a song is exactly where that shows. The chunked path is kept
    for the models that cannot take a long file.

    The third value is what the model said about the language of the audio.
    A model that was given no language hint works it out on its own, and that
    answer is what the caller needs when the language was left open.
    """
    samples = buffer.mono()
    if not len(samples):
        return [], 0, {}

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

    def heard(partial: Any) -> Dict[str, Any]:
        """What the model said about the language of the audio."""
        meta = getattr(partial, "meta", None)
        if not isinstance(meta, dict):
            return {}
        code = meta.get("detected_language")
        if not code:
            return {}
        return {
            "detected_language": str(code),
            "language_probability": meta.get("language_probability"),
        }

    if _takes_long_audio(engine):
        try:
            partial = run(samples)
        except Exception:
            partial = None
        if partial is not None and partial.segments:
            return list(partial.segments), 1, heard(partial)

    from ..audio.chunk import chunk_audio

    chunks = chunk_audio(
        samples,
        buffer.sample_rate,
        chunk_seconds=chunk_seconds,
        overlap_seconds=overlap_seconds,
    )
    segments: List[Segment] = []
    detected: Dict[str, Any] = {}
    for chunk in chunks:
        block = chunk["samples"]
        if block is None or len(block) == 0:
            continue
        offset = chunk["start"]
        partial = run(block)
        segments.extend(_shift(partial.segments, offset))
        if not detected:
            detected = heard(partial)
    return segments, len(chunks), detected


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


def _detect_language(engine: Any, buffer: AudioBuffer) -> Tuple[str, Optional[float]]:
    """Ask a loaded recogniser what language the audio is in.

    A backend that cannot tell says so by returning nothing, and the caller
    falls back on what the recognition itself reported.
    """
    detector = getattr(engine, "detect_language", None)
    if not callable(detector):
        return "", None
    try:
        answer = detector(buffer.mono(), buffer.sample_rate)
    except Exception:
        return "", None
    if not answer:
        return "", None
    try:
        code, probability = answer
    except (TypeError, ValueError):
        return "", None
    return str(code or ""), probability


def _resolve_result_language(
    full_text: str,
    effective: Optional[str],
    language_source: str,
    detected_language: str = "",
) -> Tuple[str, Any]:
    """The language tag of a result, and the regional reading of its text.

    A text that came back in the Bengali script is Bangla, whatever the
    detector said about it. A detector that guesses wrong on a sung track
    must not cost the caller the Bangla repair and the region.

    A Bangla text carries a region, Bangladesh or India, and the text decides
    which one. A text in any other language carries the language itself and
    nothing more.
    """
    from ..lang.dialect import detect_region
    from ..lang.normalize import is_bangla_text
    from ..lang.registry import is_bangla

    bangla_requested = is_bangla(effective) if effective else False
    if not bangla_requested and is_bangla_text(full_text):
        bangla_requested = True

    asked_for = language_source == "requested"
    if bangla_requested and not asked_for:
        # The text itself says Bangla, so the region comes from the text.
        detected = detect_region(full_text, default="bn-BD")
        result_language = detected.tag
    elif effective and is_bangla(effective):
        detected = detect_region(full_text, default=effective)
        result_language = detected.tag
    else:
        detected = detect_region(full_text, default="bn-BD")
        result_language = effective or detected_language or "und"
    return result_language, detected


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
    language_hint: Optional[str] = None,
    **kwargs: Any,
) -> LyricsResult:
    """Extract lyrics from a song or spoken track.

    Parameters
    ----------
    audio:
        Path, file object or AudioBuffer. Any format and any length.
    language:
        Language hint. ``bn-BD`` for Bangladeshi Bangla, ``bn-IN`` for Indian
        Bengali, ``bn`` for automatic regional choice. ``auto`` or ``None``
        finds the language of the audio on its own.
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
        ``banglish`` returns the same content in Latin letters. A language
        that is not Bangla keeps the script the recogniser used, because
        there is nothing to convert it to.
    quality:
        Which model to reach for. ``auto`` picks the model that fits the
        language, ``best`` forces the strongest one and ``fast`` starts at
        the quick one. When a model cannot be loaded the next one in the
        ladder is tried, so a request always comes back with lyrics.
    language_hint:
        What is already known about the language, used when the language was
        left open. The page a link points at may name its own language, and
        that answer is better than a guess.
    """
    from ..backends.registry import get_stt
    from ..lang.catalogue import model_language, resolve_entry
    from ..lang.corrections import apply_corrections
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

    # The language of the request. An open request means the engine finds the
    # language of the audio on its own.
    requested = (language or "").strip()
    auto_language = requested.lower() in ("", "auto")
    effective: Optional[str] = None if auto_language else requested
    language_source = "requested"
    if auto_language and language_hint:
        entry = resolve_entry(language_hint)
        if entry:
            effective = entry.iso639_1
            language_source = "hint"

    # The ladder of models to try. The first one that loads and reads the
    # audio does the work, and the result says which one it was.
    candidates = plan_for(effective, quality=quality, model=model, backend=backend)
    chosen = None
    segments: List[Segment] = []
    failures: List[str] = []
    chunk_count = 0
    detected_language = ""
    language_probability: Optional[float] = None
    index = 0

    with Timer() as timer:
        while index < len(candidates):
            candidate = candidates[index]
            index += 1
            try:
                engine = get_stt(candidate.backend, model=candidate.model)
            except Exception as exc:
                # The model is not installed, not downloaded or does not fit in
                # memory. The next one in the ladder is tried.
                failures.append(f"{candidate.model}: {exc}")
                continue

            if auto_language and not effective:
                # The model is loaded already, so it can be asked what it
                # hears. One short pass over the first seconds is enough.
                code, probability = _detect_language(engine, buffer)
                if code:
                    effective = code
                    language_source = "model"
                    language_probability = probability
                    # The ladder is built for a language, and now there is one.
                    # The model that is loaded stays at the front, because it
                    # is the one that has to do the work.
                    replanned = plan_for(
                        effective, quality=quality, model=model, backend=backend
                    )
                    rest = [item for item in replanned if item != candidate]
                    candidates = [candidate] + rest

            try:
                read_segments, read_chunks, heard = _recognise(
                    engine,
                    buffer,
                    # A model is taught one code per language, so a regional
                    # tag such as pt-BR has to reach it as pt. The region the
                    # caller asked for is not lost, because it is kept in the
                    # report and in the fallback chain.
                    language=(model_language(effective) or effective) if effective else None,
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

            if not read_segments and index < len(candidates):
                # The model loaded but heard nothing. A stronger model may do
                # better, so the ladder continues.
                failures.append(f"{candidate.model}: heard nothing")
                continue

            chosen = candidate
            segments = read_segments
            chunk_count = read_chunks
            name = candidate.backend
            if heard:
                detected_language = str(heard.get("detected_language") or "")
                if auto_language and not effective and detected_language:
                    effective = detected_language
                    language_source = "model"
                    if language_probability is None:
                        # The model worked the language out on its own here,
                        # so the number means something. A model that was
                        # told the language reports a certainty of one, and
                        # that is not a measurement of anything.
                        language_probability = heard.get("language_probability")
            break

    if chosen is None:
        detail = "; ".join(failures) if failures else "no model was available"
        raise RuntimeError(f"The lyrics could not be read. {detail}")

    lines: List[LyricLine] = []
    # The language of the audio decides what the text needs. A language that
    # is not Bangla keeps the script the recogniser used.
    bangla_requested = is_bangla(effective) if effective else False
    repaired = 0
    dropped_loops = 0
    dropped_noise = 0
    collapsed = 0
    corrected = 0
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
        changed = tightened != text
        if changed:
            collapsed += 1
            text = tightened
        if bangla_requested and needs_script_repair(text, target="bangla"):
            # The model answered in Devanagari. Bring it back to Bangla.
            text = repair_to_bangla(text)
            repaired += 1
            changed = True
        if is_bangla_text(text):
            text = normalize_bangla(text, digits="keep")
        # The time of every word is kept only when the text still matches the
        # words. A line that was repaired or collapsed no longer lines up with
        # the timings it came with, so a highlight built on them would lie.
        words = list(seg.words) if not changed else []
        if words:
            from ..lang.corrections import correct_words

            fixed = correct_words(words)
            if [w.text for w in fixed] != [w.text for w in words]:
                corrected += 1
            words = fixed
        else:
            fixed_text = apply_corrections(text)
            if fixed_text != text:
                corrected += 1
                text = fixed_text
        lines.append(
            LyricLine(
                start=float(seg.start),
                end=float(seg.end),
                text=text,
                speaker=seg.speaker,
                probability=seg.probability,
                words=words,
            )
        )

    lines = _drop_repeated_lines(lines, max_repeats=max_repeats)
    if merge_short:
        lines = _merge_short_lines(lines)

    full_text = "\n".join(line.text for line in lines)

    # One decision for the whole document: is this song written in romanised
    # Bangla, and which script should the lines come back in.
    from ..lang.translit import apply_script, resolve_script_target

    target_script = ""
    if bangla_requested or is_bangla_text(full_text):
        target_script = resolve_script_target(full_text, script)
        if target_script:
            for line in lines:
                line.text = apply_script(line.text, target_script)
            full_text = "\n".join(line.text for line in lines)

    result_language, detected = _resolve_result_language(
        full_text,
        effective,
        language_source,
        detected_language,
    )

    return LyricsResult(
        lines=lines,
        language=result_language,
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
            "language_requested": requested or "auto",
            "language_effective": effective or "",
            "language_source": language_source,
            "language_detected": detected_language,
            "language_probability": language_probability,
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
            "corrected_words": corrected,
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
        elif lowered == "karaoke":
            # The karaoke file is an SRT with the word that is being sung
            # marked inside the line, so it carries the .srt name.
            path = folder / f"{base}.karaoke.srt"
            write_subtitles(result.lines, str(path), fmt="karaoke", title=result.title)
        elif lowered == "json":
            path = folder / f"{base}.json"
            write_text(str(path), result.to_dict() and __import__("json").dumps(
                result.to_dict(), ensure_ascii=False, indent=2
            ))
        else:
            continue
        written[lowered] = str(path)

    return written


def _result_from_subtitle_lines(
    lines: List[LyricLine],
    *,
    duration: float,
    title: Optional[str],
    artist: Optional[str],
    language: str,
    language_source: str,
    script: str,
    language_requested: str,
) -> LyricsResult:
    """Build a result out of the subtitles a page carried.

    The words a human wrote are left as they were written. The only change
    made is the writing system the caller asked for, and the region of a
    Bangla text, which is read out of the text itself.
    """
    from ..lang.translit import apply_script, resolve_script_target

    full_text = "\n".join(line.text for line in lines)
    target_script = resolve_script_target(full_text, script)
    if target_script:
        for line in lines:
            line.text = apply_script(line.text, target_script)
        full_text = "\n".join(line.text for line in lines)

    result_language, detected = _resolve_result_language(full_text, language, language_source)

    return LyricsResult(
        lines=lines,
        language=result_language,
        duration=duration,
        title=title,
        artist=artist,
        backend="site-subtitles",
        model="",
        elapsed=0.0,
        meta={
            "chunks": 0,
            "model_used": "",
            "model_label": "",
            "quality": "subtitles",
            "model_fallbacks": [],
            "language_requested": language_requested or "auto",
            "language_effective": language or "",
            "language_source": language_source,
            "language_detected": language or "",
            "language_probability": None,
            "dialect_confidence": detected.confidence,
            "dialect": detected.dialect,
            "evidence": detected.evidence,
            "bd_score": detected.bd_score,
            "in_score": detected.in_score,
            "script": detected.script,
            "script_repaired_lines": 0,
            "engine_vad": False,
            "output_script": target_script or "as recognised",
            "sample_rate": 0,
            "channels": 0,
            "dropped_loops": 0,
            "dropped_noise": 0,
            "collapsed_repeats": 0,
            "lyrics_source": "site subtitles",
        },
    )


def _read_karaoke(
    url: str,
    work_dir: str,
    language: str,
    steps: List[Dict[str, Any]],
    *,
    interval: float = 1.0,
) -> Tuple[List[LyricLine], Dict[str, Any]]:
    """Read the words of a karaoke off the picture of the video.

    A karaoke has no subtitles and often nobody singing either, so the words
    of the song are on the picture and nowhere else. The reading begins with
    the survey, which is the cheap look: a few frames are read, and a video
    with no text that moves is not a karaoke, so the run carries on to the
    model rather than spending the rest of the work on a video that has
    nothing to read.
    """
    from ..karaoke import KaraokeError, have_ocr, read_burned_in_lyrics
    from ..sources import SourceError, fetch_video

    if not have_ocr():
        steps.append(
            {
                "step": "karaoke",
                "ok": False,
                "reason": "the reader of the picture is not installed",
            }
        )
        return [], {}

    # A link that has been read before is answered from what was kept, so the
    # video is not fetched and the picture is not read a second time.
    key = f"{url}|{language}|{interval}"
    kept = _KARAOKE_READS.get(key)
    if kept is not None:
        _KARAOKE_READS.move_to_end(key)
        lines, facts = kept
        steps.append(
            {
                "step": "karaoke",
                "ok": True,
                "lines": len(lines),
                "swept": facts.get("swept", False),
                "elapsed": facts.get("elapsed", 0.0),
                "band": facts.get("band", []),
                "rows": facts.get("rows", 0),
                "kept": True,
            }
        )
        return _copy_lines(lines), dict(facts)

    try:
        video_path = fetch_video(url, work_dir)
    except SourceError as exc:
        steps.append({"step": "karaoke", "ok": False, "reason": str(exc)})
        return [], {}

    try:
        result = read_burned_in_lyrics(video_path, language=language, interval=interval)
    except KaraokeError as exc:
        steps.append({"step": "karaoke", "ok": False, "reason": str(exc)})
        return [], {}

    steps.append(
        {
            "step": "karaoke",
            "ok": True,
            "lines": len(result.lines),
            "swept": result.swept,
            "elapsed": result.elapsed,
            "band": list(result.band),
            "rows": result.rows,
        }
    )
    facts = {
        "rows": result.rows,
        "swept": result.swept,
        "band": list(result.band),
        "elapsed": result.elapsed,
        "frames_read": result.meta.get("frames_read", 0),
        "seconds_opening": result.meta.get("seconds_opening", 0.0),
        "seconds_survey": result.meta.get("seconds_survey", 0.0),
        "seconds_reading": result.meta.get("seconds_reading", 0.0),
        "seconds_refining": result.meta.get("seconds_refining", 0.0),
    }
    _KARAOKE_READS[key] = (list(result.lines), dict(facts))
    while len(_KARAOKE_READS) > _KARAOKE_READS_MAX:
        _KARAOKE_READS.popitem(last=False)
    return list(result.lines), facts


def extract_lyrics_from_url(
    url: str,
    work_dir: str,
    *,
    language: Optional[str] = "auto",
    karaoke: str = "auto",
    **kwargs: Any,
) -> Dict[str, Any]:
    """Fetch the audio behind a link and read the lyrics out of it.

    Four things can answer the request, and the best one that is there wins:

    * the subtitles the page itself carries. They are the truth about the
      words and the timings, and using them costs no model, no memory and no
      waiting.
    * the words burned into the picture of a karaoke video, when the page has
      no subtitles and the words are on the screen rather than in the sound.
    * the audio, read by a speech model, when the page has no subtitles.
    * the page's own answer about the language, used as a hint for the model
      when the caller did not name one.

    The audio is fetched either way, so the panel can play the song it wrote
    the lyrics for. Everything stays on this machine.

    Returns the result and the facts about the source.
    """
    from ..export.subtitles import parse_subtitles
    from ..sources import (
        SourceError,
        fetch_audio,
        fetch_metadata,
        fetch_subtitles,
        rank_subtitle_tracks,
        read_subtitle_file,
    )

    started = time.perf_counter()
    steps: List[Dict[str, Any]] = []

    metadata = fetch_metadata(url)
    steps.append(
        {
            "step": "metadata",
            "ok": True,
            "title": metadata.title,
            "uploader": metadata.uploader,
            "language_source": metadata.language_source,
            "subtitle_tracks": len(metadata.subtitles),
        }
    )

    requested = (language or "").strip()
    auto_language = requested.lower() in ("", "auto")

    # The language to look for. A caller who named one wins, and the page's
    # own answer is used when the caller did not.
    prefer = metadata.language if auto_language else requested

    # The subtitles the page offers, if any. The best track is tried first and
    # the next one is tried when a site refuses the first, because a busy site
    # does refuse a track now and then.
    ranked = rank_subtitle_tracks(metadata.subtitles, prefer)
    subtitle_lines: List[LyricLine] = []
    subtitle_info: Dict[str, Any] = {}

    if not ranked:
        steps.append({"step": "subtitles", "ok": False, "reason": "the page offers no subtitles"})
    else:
        track = None
        try:
            track = fetch_subtitles(
                url,
                work_dir,
                tracks=ranked,
                prefer_language=prefer,
            )
        except SourceError as exc:
            steps.append({"step": "subtitles", "ok": False, "reason": str(exc)})
        if track is None:
            if not any(step["step"] == "subtitles" for step in steps):
                steps.append(
                    {
                        "step": "subtitles",
                        "ok": False,
                        "reason": "no track could be read from the page",
                    }
                )
        else:
            text = read_subtitle_file(track.path, language=track.language)
            subtitle_lines = parse_subtitles(text)
            if subtitle_lines:
                subtitle_info = track.to_dict()
                subtitle_info["lines"] = len(subtitle_lines)
                steps.append(
                    {
                        "step": "subtitles",
                        "ok": True,
                        "language": track.language,
                        "automatic": track.automatic,
                        "lines": len(subtitle_lines),
                    }
                )
            else:
                subtitle_lines = []
                steps.append(
                    {"step": "subtitles", "ok": False, "reason": "the track held no cues"}
                )

    # The audio is fetched either way, because the panel plays it.
    source = fetch_audio(url, work_dir)
    steps.append(
        {
            "step": "audio",
            "ok": True,
            "duration": round(float(source.duration), 3),
            "extractor": source.extractor,
        }
    )

    title = kwargs.pop("title", None) or source.title
    artist = kwargs.pop("artist", None) or source.uploader

    # A karaoke keeps the words of the song in the picture. It is looked at
    # when the page has no subtitles of its own, because a page that carries
    # its subtitles is the better answer and is cheaper.
    karaoke_mode = str(karaoke or "auto").strip().lower()
    karaoke_lines: List[LyricLine] = []
    karaoke_info: Dict[str, Any] = {}
    if not subtitle_lines and karaoke_mode != "off":
        karaoke_lines, karaoke_info = _read_karaoke(url, work_dir, prefer or "bn", steps)

    if subtitle_lines:
        # The page named the language of its own subtitles, so the language is
        # known rather than guessed.
        effective = requested if not auto_language else (subtitle_info.get("language") or "")
        language_source = "requested" if not auto_language else "declared"
        result = _result_from_subtitle_lines(
            subtitle_lines,
            duration=float(source.duration or 0.0),
            title=title,
            artist=artist,
            language=effective,
            language_source=language_source,
            script=str(kwargs.pop("script", "bangla") or "bangla"),
            language_requested=requested or "auto",
        )
        lyrics_source = "site subtitles"
    elif karaoke_lines:
        # The words were read off the picture, so the script they are in is
        # the script of the picture.
        effective = requested if not auto_language else (prefer or "")
        result = _result_from_subtitle_lines(
            karaoke_lines,
            duration=float(source.duration or 0.0),
            title=title,
            artist=artist,
            language=effective,
            language_source="requested" if not auto_language else "read",
            script=str(kwargs.pop("script", "bangla") or "bangla"),
            language_requested=requested or "auto",
        )
        result.backend = "video-ocr"
        result.meta["lyrics_source"] = "video ocr"
        result.meta["karaoke"] = karaoke_info
        lyrics_source = "video ocr"
    else:
        hint = source.language
        result = extract_lyrics(
            source.audio_path,
            language=language,
            language_hint=hint or None,
            title=title,
            artist=artist,
            **kwargs,
        )
        lyrics_source = "speech model"

    result.meta["source"] = source.to_dict()
    result.meta["source_language"] = source.language
    result.meta["source_language_source"] = source.language_source
    result.meta["lyrics_source"] = lyrics_source
    result.meta["subtitle_track"] = subtitle_info
    result.meta["pipeline"] = {
        "url": url,
        "steps": steps,
        "lyrics_source": lyrics_source,
        "subtitle_track": subtitle_info,
        "karaoke": karaoke_info,
        "elapsed": round(time.perf_counter() - started, 3),
    }
    if source.title and not result.title:
        result.title = source.title
    if source.uploader and not result.artist:
        result.artist = source.uploader
    return {"result": result, "source": source}
