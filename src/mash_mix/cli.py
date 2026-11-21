# SPDX-License-Identifier: MIT | Copyright (c) 2026 MASH (Khaled Bhuiyan)
"""Command line interface for MASH-Mix-Audio-and-Text."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Optional, Sequence

from . import __version__

__all__ = ["main", "build_parser"]


def _add_common_language(parser: argparse.ArgumentParser, default: str = "bn-BD") -> None:
    parser.add_argument(
        "--lang",
        "--language",
        dest="language",
        default=default,
        help=(
            "Language tag. Bangla tags: bn, bn-BD, bn-IN. Any ISO code works, "
            "for example hi or ar. 'auto' finds the language of the audio. "
            "Default: %(default)s"
        ),
    )


def _add_script_option(parser: argparse.ArgumentParser, default: str = "bangla") -> None:
    parser.add_argument(
        "--script",
        dest="script",
        default=default,
        choices=["bangla", "banglish"],
        help=(
            "Writing system of the text. bangla keeps the native script, "
            "banglish writes it in Latin letters. Default: %(default)s"
        ),
    )


def build_parser() -> argparse.ArgumentParser:
    """Build the full argument parser."""
    parser = argparse.ArgumentParser(
        prog="mash-mix",
        description=(
            "Cross platform speech and text engine. Analysis, extraction, song to lyric, "
            "text to speech, speech to text and speech to speech, with Bangla first."
        ),
    )
    parser.add_argument("--version", action="version", version=f"mash-mix {__version__}")
    sub = parser.add_subparsers(dest="command", required=True)

    # -- lyrics ---------------------------------------------------------
    lyrics = sub.add_parser("lyrics", help="Extract lyrics from a song or spoken track.")
    lyrics.add_argument(
        "audio",
        help="Audio or video file, or a link to one. Any format, any length.",
    )
    lyrics.add_argument("-o", "--output-dir", default="lyrics-out", help="Where to write files.")
    lyrics.add_argument("--stem", default=None, help="Base name for the written files.")
    lyrics.add_argument(
        "--format",
        dest="formats",
        action="append",
        choices=["srt", "txt", "vtt", "lrc", "json"],
        help="Output format. Repeat for more than one. Default: srt and txt.",
    )
    _add_common_language(lyrics, default="auto")
    _add_script_option(lyrics)
    lyrics.add_argument("--backend", default="auto", help="Speech to text backend. Default: auto.")
    lyrics.add_argument("--model", default=None, help="Model id for the backend.")
    lyrics.add_argument(
        "--quality",
        default="auto",
        choices=["auto", "best", "fast"],
        help="Which model to reach for. Default: auto.",
    )
    lyrics.add_argument("--title", default=None, help="Title written into the file header.")
    lyrics.add_argument("--artist", default=None, help="Artist written into the file header.")
    lyrics.add_argument(
        "--separate-vocals",
        action="store_true",
        help="Pull the vocal stem out of the mix before recognition.",
    )
    lyrics.add_argument(
        "--separation-model",
        default="htdemucs",
        help="Separation model. Use mid_side for a dependency free vocal removal.",
    )
    lyrics.add_argument("--chunk-seconds", type=float, default=30.0)
    lyrics.add_argument("--overlap-seconds", type=float, default=2.0)
    lyrics.add_argument("--hotword", action="append", dest="hotwords", default=None)
    lyrics.add_argument("--prompt", default=None, help="Initial prompt to guide the recogniser.")
    lyrics.add_argument("--max-repeats", type=int, default=2)
    lyrics.add_argument("--print", dest="print_text", action="store_true", help="Print the lyrics.")

    # -- transcribe -----------------------------------------------------
    tr = sub.add_parser("transcribe", help="Transcribe speech to text.")
    tr.add_argument("audio", help="Audio or video file.")
    tr.add_argument("-o", "--output", default=None, help="Write the transcript to this file.")
    tr.add_argument(
        "--format",
        dest="output_format",
        default="txt",
        choices=["txt", "json", "srt", "vtt"],
        help="Output format. Default: txt",
    )
    _add_common_language(tr, default="")
    _add_script_option(tr)
    tr.add_argument("--backend", default="auto")
    tr.add_argument("--model", default=None)
    tr.add_argument("--task", default="transcribe", choices=["transcribe", "translate"])
    tr.add_argument("--hotword", action="append", dest="hotwords", default=None)
    tr.add_argument("--prompt", default=None)
    tr.add_argument("--timestamps", action="store_true", help="Prefix each line with a time.")

    # -- speak ----------------------------------------------------------
    speak = sub.add_parser("speak", help="Synthesize speech from text.")
    speak.add_argument("text", help="The text to speak.")
    speak.add_argument("-o", "--output", default="speech.wav", help="Output audio file.")
    _add_common_language(speak)
    _add_script_option(speak)
    speak.add_argument("--backend", default="auto")
    speak.add_argument("--model", default=None)
    speak.add_argument("--voice", default=None)
    speak.add_argument("--speed", type=float, default=1.0)
    speak.add_argument("--reference-audio", default=None, help="Reference clip for voice cloning.")
    speak.add_argument("--no-normalize", action="store_true", help="Skip Bangla text preparation.")

    # -- sts ------------------------------------------------------------
    sts = sub.add_parser("sts", help="Speech to speech. Speak a recording in another voice or language.")
    sts.add_argument("audio")
    sts.add_argument("-o", "--output", default="speech-sts.wav")
    sts.add_argument("--target-lang", default="bn-BD")
    sts.add_argument("--target-voice", default=None)
    sts.add_argument("--reference-audio", default=None)
    sts.add_argument("--stt-backend", default="auto")
    sts.add_argument("--tts-backend", default="auto")
    _add_script_option(sts)

    # -- analyze --------------------------------------------------------
    an = sub.add_parser("analyze", help="Measure tempo, key, loudness and sections.")
    an.add_argument("audio")
    an.add_argument("-o", "--output", default=None, help="Write the report as JSON.")
    an.add_argument("--json", action="store_true", help="Print the report as JSON.")

    # -- separate -------------------------------------------------------
    sep = sub.add_parser("separate", help="Split a track into stems.")
    sep.add_argument("audio")
    sep.add_argument("-o", "--output-dir", default="stems-out")
    sep.add_argument("--model", default="htdemucs", help="Use mid_side for a dependency free split.")

    # -- text tools -----------------------------------------------------
    text = sub.add_parser("text", help="Bangla text tools.")
    text_sub = text.add_subparsers(dest="text_command", required=True)

    norm = text_sub.add_parser("normalize", help="Normalise Bangla text.")
    norm.add_argument("text", nargs="?", default=None, help="Text. Reads stdin when omitted.")
    norm.add_argument("--file", default=None)
    norm.add_argument("--digits", default="keep", choices=["keep", "native", "ascii"])

    num = text_sub.add_parser("numbers", help="Read a number in Bangla words.")
    num.add_argument("value")

    trl = text_sub.add_parser("translit", help="Convert romanised Bangla to Bangla script.")
    trl.add_argument("text", nargs="?", default=None)
    trl.add_argument("--file", default=None)
    trl.add_argument("--reverse", action="store_true", help="Bangla script to Latin.")

    det = text_sub.add_parser("detect", help="Detect the region and dialect of a Bangla text.")
    det.add_argument("text", nargs="?", default=None)
    det.add_argument("--file", default=None)

    # -- info -----------------------------------------------------------
    sub.add_parser("languages", help="List the supported language tags.")
    sub.add_parser("backends", help="List the backends and what is installed.")
    models = sub.add_parser("models", help="List the model catalogue with licences.")
    models.add_argument("--task", default=None, choices=["stt", "tts", "sts", "separation"])
    models.add_argument("--bengali", action="store_true", help="Only the Bangla capable models.")
    models.add_argument("--json", action="store_true")

    doctor = sub.add_parser("doctor", help="Check this machine for everything the engine needs.")
    doctor.add_argument("--json", action="store_true")

    # -- serve ----------------------------------------------------------
    serve = sub.add_parser("serve", help="Start the web interface and the HTTP API.")
    serve.add_argument("--host", default="127.0.0.1")
    serve.add_argument("--port", type=int, default=8000)
    serve.add_argument("--reload", action="store_true")

    return parser


def _read_text(args: argparse.Namespace) -> str:
    if getattr(args, "file", None):
        return Path(args.file).read_text(encoding="utf-8")
    value = getattr(args, "text", None)
    if value:
        return value
    if not sys.stdin.isatty():
        return sys.stdin.read()
    return ""


def _cmd_lyrics(args: argparse.Namespace) -> int:
    from .sources import is_url
    from .tasks.lyrics import extract_lyrics_from_url, extract_lyrics_to_files

    formats = args.formats or ["srt", "txt"]

    if is_url(args.audio):
        # The audio behind a link is fetched into the output folder and then
        # read exactly like a file that was dropped in.
        from pathlib import Path

        from .export.plaintext import write_text
        from .export.subtitles import write_subtitles

        folder = Path(args.output_dir)
        folder.mkdir(parents=True, exist_ok=True)
        outcome = extract_lyrics_from_url(
            args.audio,
            str(folder),
            language=args.language or "auto",
            backend=args.backend,
            model=args.model,
            title=args.title,
            artist=args.artist,
            separate_vocals=args.separate_vocals,
            separation_model=args.separation_model,
            chunk_seconds=args.chunk_seconds,
            overlap_seconds=args.overlap_seconds,
            hotwords=args.hotwords,
            initial_prompt=args.prompt,
            max_repeats=args.max_repeats,
            script=args.script,
            quality=args.quality,
        )
        result = outcome["result"]
        source = outcome["source"]
        base = args.stem or result.title or "lyrics"
        base = "".join(char for char in base if char not in '<>:"/\\|?*').strip() or "lyrics"
        written: dict = {}
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
            else:
                continue
            written[lowered] = str(path)
        if args.print_text:
            print(result.text)
        for fmt, path in written.items():
            print(f"{fmt}: {path}")
        print(
            f"language={result.language} lines={len(result.lines)} "
            f"duration={result.duration:.1f}s elapsed={result.elapsed:.1f}s "
            f"model={result.meta.get('model_used') or 'unknown'} "
            f"source={source.title or source.url}",
            file=sys.stderr,
        )
        return 0

    written = extract_lyrics_to_files(
        args.audio,
        args.output_dir,
        stem=args.stem,
        formats=formats,
        language=args.language or None,
        backend=args.backend,
        model=args.model,
        title=args.title,
        artist=args.artist,
        separate_vocals=args.separate_vocals,
        separation_model=args.separation_model,
        chunk_seconds=args.chunk_seconds,
        overlap_seconds=args.overlap_seconds,
        hotwords=args.hotwords,
        initial_prompt=args.prompt,
        max_repeats=args.max_repeats,
        script=args.script,
        quality=args.quality,
    )
    result = written.pop("result")
    if args.print_text:
        print(result.text)
    for fmt, path in written.items():
        print(f"{fmt}: {path}")
    print(
        f"language={result.language} lines={len(result.lines)} "
        f"duration={result.duration:.1f}s elapsed={result.elapsed:.1f}s "
        f"model={result.meta.get('model_used') or 'unknown'}",
        file=sys.stderr,
    )
    return 0


def _cmd_transcribe(args: argparse.Namespace) -> int:
    from .export.plaintext import transcript_to_text, write_text
    from .export.subtitles import segments_to_lines, write_subtitles
    from .tasks.transcribe import transcribe

    result = transcribe(
        args.audio,
        backend=args.backend,
        model=args.model,
        language=args.language or None,
        task=args.task,
        hotwords=args.hotwords,
        initial_prompt=args.prompt,
        script=args.script,
    )

    if args.output_format == "json":
        content = json.dumps(result.to_dict(), ensure_ascii=False, indent=2)
    elif args.output_format in ("srt", "vtt"):
        content = ""
        path = args.output
        if path is None:
            path = f"transcript.{args.output_format}"
        write_subtitles(
            segments_to_lines(result.segments),
            path,
            fmt=args.output_format,
            title=None,
        )
        print(f"{args.output_format}: {path}")
        return 0
    else:
        content = transcript_to_text(result, timestamps=args.timestamps)

    if args.output:
        write_text(args.output, content)
        print(f"txt: {args.output}")
    else:
        print(content)
    return 0


def _cmd_speak(args: argparse.Namespace) -> int:
    from .tasks.synthesize import synthesize_to_file

    written = synthesize_to_file(
        args.text,
        args.output,
        backend=args.backend,
        model=args.model,
        language=args.language,
        voice=args.voice,
        speed=args.speed,
        reference_audio=args.reference_audio,
        normalize=not args.no_normalize,
        script=args.script,
    )
    print(f"audio: {written['audio']}")
    return 0


def _cmd_sts(args: argparse.Namespace) -> int:
    from .audio.io import save
    from .tasks.speech_to_speech import speak as _speak

    result = _speak(
        args.audio,
        target_language=args.target_lang,
        target_voice=args.target_voice,
        reference_audio=args.reference_audio,
        stt_backend=args.stt_backend,
        tts_backend=args.tts_backend,
        script=args.script,
    )
    save(args.output, result.audio, result.sample_rate)
    print(f"audio: {args.output}")
    return 0


def _cmd_analyze(args: argparse.Namespace) -> int:
    from .tasks.analyze import analyze

    result = analyze(args.audio)
    payload = result.to_dict()
    if args.output:
        Path(args.output).parent.mkdir(parents=True, exist_ok=True)
        Path(args.output).write_text(
            json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8"
        )
        print(f"json: {args.output}")
    if args.json or not args.output:
        print(json.dumps(payload, ensure_ascii=False, indent=2))
    return 0


def _cmd_separate(args: argparse.Namespace) -> int:
    from .tasks.analyze import separate

    result = separate(args.audio, model=args.model, output_dir=args.output_dir)
    for name, wave in result.stems.items():
        print(f"stem {name}: {len(wave)} frames")
    print(f"output dir: {args.output_dir}")
    return 0


def _cmd_text(args: argparse.Namespace) -> int:
    from .lang.dialect import detect_region
    from .lang.normalize import normalize_bangla
    from .lang.numerals import number_to_bangla_words
    from .lang.translit import bangla_to_latin, banglish_to_bangla

    command = args.text_command
    if command == "numbers":
        print(number_to_bangla_words(args.value))
        return 0

    content = _read_text(args)
    if command == "normalize":
        print(normalize_bangla(content, digits=args.digits))
        return 0
    if command == "translit":
        if args.reverse:
            print(bangla_to_latin(content))
        else:
            print(banglish_to_bangla(content))
        return 0
    if command == "detect":
        report = detect_region(content)
        print(json.dumps(report.to_dict(), ensure_ascii=False, indent=2))
        return 0
    return 1


def _cmd_languages(_: argparse.Namespace) -> int:
    from .lang.registry import list_languages

    print(json.dumps(list_languages(), ensure_ascii=False, indent=2))
    return 0


def _cmd_backends(_: argparse.Namespace) -> int:
    from .backends.registry import backend_infos

    for info in backend_infos():
        mark = "ok " if info.get("available") else "no "
        print(f"[{mark}] {info['kind']:3} {info['name']:<16} {info['summary']}")
    return 0


def _cmd_models(args: argparse.Namespace) -> int:
    from .models import MODEL_CATALOG, bengali_models, models_for

    if args.bengali:
        entries = bengali_models(args.task)
    elif args.task:
        entries = models_for(args.task)
    else:
        entries = list(MODEL_CATALOG)

    if args.json:
        print(json.dumps(entries, ensure_ascii=False, indent=2))
        return 0

    for entry in entries:
        print(f"{entry['task']:<11} {entry['backend']:<13} {entry['id']}")
        print(f"            {entry['name']}")
        print(f"            languages: {entry['languages']}")
        print(f"            bangla:    {entry['bengali']}")
        print(f"            licence:   {entry['license']}")
        print(f"            platforms: {entry['platforms']}")
        print(f"            {entry['url']}")
        print()
    return 0


def _cmd_doctor(args: argparse.Namespace) -> int:
    from .doctor import run_doctor

    report = run_doctor()
    if args.json:
        print(json.dumps(report, ensure_ascii=False, indent=2))
        return 0 if report.get("ok") else 1
    print(report["text"])
    return 0 if report.get("ok") else 1


def _cmd_serve(args: argparse.Namespace) -> int:
    try:
        import uvicorn
    except ImportError:
        print(
            "The server needs uvicorn and fastapi. Install them with:\n"
            "  pip install 'mash-mix-audio-and-text[server]'",
            file=sys.stderr,
        )
        return 1

    uvicorn.run(
        "mash_mix.server.app:app",
        host=args.host,
        port=args.port,
        reload=args.reload,
    )
    return 0


_COMMANDS = {
    "lyrics": _cmd_lyrics,
    "transcribe": _cmd_transcribe,
    "speak": _cmd_speak,
    "sts": _cmd_sts,
    "analyze": _cmd_analyze,
    "separate": _cmd_separate,
    "text": _cmd_text,
    "languages": _cmd_languages,
    "backends": _cmd_backends,
    "models": _cmd_models,
    "doctor": _cmd_doctor,
    "serve": _cmd_serve,
}


def main(argv: Optional[Sequence[str]] = None) -> int:
    """Entry point for the ``mash-mix`` command."""
    parser = build_parser()
    args = parser.parse_args(list(argv) if argv is not None else None)
    handler = _COMMANDS.get(args.command)
    if handler is None:
        parser.print_help()
        return 1
    try:
        return handler(args)
    except KeyboardInterrupt:
        print("\nInterrupted.", file=sys.stderr)
        return 130
    except Exception as exc:  # pragma: no cover - top level guard
        print(f"error: {exc}", file=sys.stderr)
        return 1


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
