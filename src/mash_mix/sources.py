# SPDX-License-Identifier: MIT | Copyright (c) 2026 MASH (Khaled Bhuiyan)
"""Fetching the audio behind a link.

A link is a way of naming a piece of audio. This module takes that name and
fetches the audio itself, so the rest of the engine never has to know where
the sound came from.

The fetching is done with ``yt-dlp``, which is the tool that reads the pages
of the public sites, and it is an optional dependency: the engine works
without it, and only a request for a link needs it.

Two facts come back with the audio:

* what the page says about the language, when the page says anything at all,
* what the script of the title and the description says about the language,
  when the page says nothing.

Those two are reported apart on purpose. A page that names its language is
telling the truth about itself. A script read out of a title is a guess, and a
guess is marked as a guess.
"""

from __future__ import annotations

import json
import shutil
import subprocess
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Tuple

from .lang.catalogue import detect_language_from_text, resolve_entry

__all__ = [
    "SourceInfo",
    "SourceError",
    "have_yt_dlp",
    "fetch_audio",
    "fetch_metadata",
    "is_url",
    "read_subtitle_file",
]

#: The hosts the fetcher is meant for. Anything else is still allowed, because
#: the tool reads whatever page the caller names, but these are the ones that
#: are known to work.
KNOWN_SITES = (
    "youtube.com",
    "youtu.be",
    "dailymotion.com",
    "dai.ly",
    "soundcloud.com",
    "vimeo.com",
    "bandcamp.com",
    "facebook.com",
    "instagram.com",
    "twitter.com",
    "x.com",
    "tiktok.com",
    "reddit.com",
    "bilibili.com",
    "nicovideo.jp",
    "vk.com",
    "rumble.com",
    "odysee.com",
)


class SourceError(RuntimeError):
    """The audio behind a link could not be fetched."""


@dataclass
class SourceInfo:
    """What is known about the audio behind a link."""

    url: str
    title: str = ""
    uploader: str = ""
    duration: float = 0.0
    extractor: str = ""
    webpage_url: str = ""
    audio_path: str = ""
    #: The language the page itself named, as an ISO code.
    declared_language: str = ""
    #: The language guessed from the script of the title and the description.
    guessed_language: str = ""
    #: The script the title and the description are written in.
    script: str = ""
    #: How the language was worked out: declared, script, model or none.
    language_source: str = "none"
    #: The subtitle tracks the page offers.
    subtitles: List[Dict[str, Any]] = field(default_factory=list)
    meta: Dict[str, Any] = field(default_factory=dict)

    @property
    def language(self) -> str:
        """The best language this source can offer."""
        return self.declared_language or self.guessed_language or ""

    def to_dict(self) -> Dict[str, Any]:
        return {
            "url": self.url,
            "title": self.title,
            "uploader": self.uploader,
            "duration": round(float(self.duration), 3),
            "extractor": self.extractor,
            "webpage_url": self.webpage_url,
            "audio_path": self.audio_path,
            "declared_language": self.declared_language,
            "guessed_language": self.guessed_language,
            "script": self.script,
            "language_source": self.language_source,
            "language": self.language,
            "subtitles": self.subtitles,
            "meta": self.meta,
        }


def have_yt_dlp() -> bool:
    """True when the fetcher can run on this machine."""
    if shutil.which("yt-dlp"):
        return True
    try:
        import yt_dlp  # noqa: F401
    except Exception:
        return False
    return True


def is_url(text: str) -> bool:
    """True when the text looks like a web address."""
    cleaned = (text or "").strip()
    if not cleaned:
        return False
    lowered = cleaned.lower()
    return lowered.startswith("http://") or lowered.startswith("https://")


def _yt_dlp_command(args: List[str]) -> List[str]:
    """The command that runs the fetcher, the module or the binary."""
    if shutil.which("yt-dlp"):
        return ["yt-dlp", *args]
    return ["python", "-m", "yt_dlp", *args]


def _run(args: List[str], timeout: int = 600) -> str:
    """Run the fetcher and return its output."""
    command = _yt_dlp_command(args)
    try:
        proc = subprocess.run(
            command,
            capture_output=True,
            text=True,
            encoding="utf-8",
            errors="replace",
            timeout=timeout,
        )
    except FileNotFoundError as exc:
        raise SourceError(
            "The fetcher yt-dlp is not installed. Install it with: "
            "pip install 'mash-mix-audio-and-text[sources]'"
        ) from exc
    except subprocess.TimeoutExpired as exc:
        raise SourceError("The link took too long to answer and the fetch was stopped.") from exc
    if proc.returncode != 0:
        message = (proc.stderr or proc.stdout or "").strip()
        raise SourceError(message or f"The fetch failed with code {proc.returncode}.")
    return proc.stdout


def _metadata_from_json(payload: str) -> Dict[str, Any]:
    """Read the metadata the fetcher printed."""
    try:
        data = json.loads(payload)
    except ValueError as exc:
        raise SourceError("The fetcher did not return readable metadata.") from exc
    if isinstance(data, list):
        data = data[0] if data else {}
    if not isinstance(data, dict):
        raise SourceError("The fetcher did not return readable metadata.")
    return data


def _declared_language(info: Dict[str, Any]) -> str:
    """The language the page named, when it named one."""
    for key in ("language", "language_preference", "audio_language"):
        value = info.get(key)
        if not value or not isinstance(value, str):
            continue
        entry = resolve_entry(value)
        if entry:
            return entry.iso639_1
    return ""


def _script_language(info: Dict[str, Any]) -> Tuple[str, str]:
    """The language guessed from the script of the title and the description."""
    pieces: List[str] = []
    for key in ("title", "description", "tags"):
        value = info.get(key)
        if isinstance(value, str):
            pieces.append(value)
        elif isinstance(value, (list, tuple)):
            pieces.append(" ".join(str(item) for item in value))
    text = "\n".join(pieces)
    if not text.strip():
        return "", ""
    iso, _share, script = detect_language_from_text(text)
    if not iso or iso == "en":
        # The Latin script says nothing, so the guess is left empty.
        return "", script
    return iso, script


def _subtitle_list(info: Dict[str, Any]) -> List[Dict[str, Any]]:
    """The subtitle tracks the page offers, as a plain list."""
    out: List[Dict[str, Any]] = []
    for key in ("subtitles", "automatic_captions"):
        tracks = info.get(key) or {}
        if not isinstance(tracks, dict):
            continue
        for lang, formats in tracks.items():
            entry = resolve_entry(lang)
            code = entry.iso639_1 if entry else str(lang)
            names = []
            if isinstance(formats, list):
                for item in formats:
                    if isinstance(item, dict):
                        names.append(str(item.get("ext") or ""))
            out.append(
                {
                    "language": code,
                    "raw_language": str(lang),
                    "automatic": key == "automatic_captions",
                    "formats": names,
                }
            )
    return out


def fetch_metadata(url: str, *, timeout: int = 120) -> SourceInfo:
    """Read what a page says about itself, without fetching the sound."""
    if not is_url(url):
        raise SourceError(f"'{url}' is not a web address.")
    payload = _run(
        [
            "--dump-single-json",
            "--skip-download",
            "--no-warnings",
            "--no-playlist",
            url,
        ],
        timeout=timeout,
    )
    info = _metadata_from_json(payload)
    return _build_info(url, info)


def _build_info(url: str, info: Dict[str, Any]) -> SourceInfo:
    """Turn the fetcher's metadata into a :class:`SourceInfo`."""
    declared = _declared_language(info)
    guessed, script = _script_language(info)
    if declared:
        source = "declared"
    elif guessed:
        source = "script"
    else:
        source = "none"
    try:
        duration = float(info.get("duration") or 0.0)
    except (TypeError, ValueError):
        duration = 0.0
    return SourceInfo(
        url=url,
        title=str(info.get("title") or "").strip(),
        uploader=str(info.get("uploader") or info.get("channel") or "").strip(),
        duration=duration,
        extractor=str(info.get("extractor_key") or info.get("extractor") or "").strip(),
        webpage_url=str(info.get("webpage_url") or "").strip(),
        declared_language=declared,
        guessed_language=guessed,
        script=script,
        language_source=source,
        subtitles=_subtitle_list(info),
        meta={
            "id": str(info.get("id") or ""),
            "ext": str(info.get("ext") or ""),
            "thumbnail": str(info.get("thumbnail") or ""),
            "is_live": bool(info.get("is_live") or False),
            "view_count": info.get("view_count"),
        },
    )


def fetch_audio(url: str, target_dir: str, *, timeout: int = 900) -> SourceInfo:
    """Fetch the audio behind a link into a folder.

    The sound is written as a wav file, because that is what the rest of the
    engine reads, and the fetcher asks for the audio track on its own so that
    no picture is downloaded.
    """
    if not is_url(url):
        raise SourceError(f"'{url}' is not a web address.")

    folder = Path(target_dir)
    folder.mkdir(parents=True, exist_ok=True)

    template = str(folder / "source.%(ext)s")
    payload = _run(
        [
            "--dump-single-json",
            # The dump flag makes the fetcher simulate, and a simulation
            # writes no file. This one says: do the work and then print it.
            "--no-simulate",
            "--no-warnings",
            "--no-playlist",
            "--no-progress",
            "--extract-audio",
            "--audio-format",
            "wav",
            "--force-overwrites",
            "--output",
            template,
            url,
        ],
        timeout=timeout,
    )
    info = _metadata_from_json(payload)
    source = _build_info(url, info)

    # The fetcher names the file it wrote in the metadata. The name is taken
    # from there rather than guessed, because the extension can change.
    audio_path = ""
    for item in info.get("requested_downloads") or []:
        if isinstance(item, dict):
            candidate = str(item.get("filepath") or "")
            if candidate and Path(candidate).exists():
                audio_path = candidate
                break
    if not audio_path:
        for candidate in sorted(folder.glob("source.*")):
            if candidate.suffix.lower() == ".wav":
                audio_path = str(candidate)
                break
    if not audio_path:
        raise SourceError("The audio was fetched but no file was written.")
    source.audio_path = audio_path
    return source


def read_subtitle_file(path: str) -> str:
    """Read a subtitle file as text, whatever encoding it was written in.

    A subtitle that a site offers may be in UTF-8, and a subtitle that came
    from an older page may be in the code page of its language. Both are read,
    and a broken byte is replaced rather than raising.
    """
    target = Path(path)
    raw = target.read_bytes()
    try:
        return raw.decode("utf-8")
    except UnicodeDecodeError:
        pass
    for encoding in ("utf-8-sig", "cp1252", "latin-1"):
        try:
            return raw.decode(encoding)
        except UnicodeDecodeError:
            continue
    return raw.decode("utf-8", errors="replace")
