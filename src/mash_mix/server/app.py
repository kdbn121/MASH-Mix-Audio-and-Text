# SPDX-License-Identifier: MIT | Copyright (c) 2026 MASH (Khaled Bhuiyan)
"""HTTP API and web interface for MASH-Mix-Audio-and-Text.

The server gives the same engine a face: upload a song, get lyric lines back,
download them as SRT or plain text. It also exposes OpenAI style speech and
transcription routes so existing tools can point at it.
"""

from __future__ import annotations

import shutil
import tempfile
import uuid
from pathlib import Path
from typing import Any, Dict, List, Optional

from fastapi import FastAPI, File, Form, HTTPException, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse, HTMLResponse, JSONResponse, Response
from fastapi.staticfiles import StaticFiles

from .. import __version__
from ..export.plaintext import write_text
from ..export.subtitles import write_subtitles

STATIC_DIR = Path(__file__).parent / "static"

app = FastAPI(
    title="MASH-Mix-Audio-and-Text",
    version=__version__,
    description=(
        "Cross platform speech and text engine. Song to lyric, speech to text, "
        "text to speech and speech to speech, with Bangla first."
    ),
)

# The browser extension on the other side of the loopback calls this server,
# and a browser applies its own rules to a call that comes from another
# origin. Only the local origins are let in, so nothing off this machine
# can reach the engine through the browser.
app.add_middleware(
    CORSMiddleware,
    allow_origin_regex=r"^https?://(localhost|127\.0\.0\.1)(:\d+)?$",
    allow_credentials=False,
    allow_methods=["GET", "POST", "OPTIONS"],
    allow_headers=["*"],
)

# Finished jobs live here until the server restarts.
_JOBS: Dict[str, Dict[str, Any]] = {}
_JOB_ROOT = Path(tempfile.gettempdir()) / "mash-mix-jobs"


def _job_dir(job_id: str) -> Path:
    folder = _JOB_ROOT / job_id
    folder.mkdir(parents=True, exist_ok=True)
    return folder


def _save_upload(upload: UploadFile, folder: Path) -> Path:
    """Stream an upload to disk. No size limit, the file is written in chunks."""
    suffix = Path(upload.filename or "upload.bin").suffix
    target = folder / f"input{suffix}"
    with open(target, "wb") as handle:
        shutil.copyfileobj(upload.file, handle, length=1024 * 1024)
    return target


def _source_details(path: Path, original_name: str) -> Dict[str, Any]:
    """Facts about the uploaded file, so the interface can show what arrived."""
    details: Dict[str, Any] = {
        "name": original_name or path.name,
        "size_bytes": None,
        "duration": None,
        "sample_rate": None,
        "channels": None,
        "bit_rate": None,
        "format_name": None,
    }
    try:
        from ..audio.io import probe

        facts = probe(path)
    except Exception:
        facts = {}
    for key in ("size_bytes", "duration", "sample_rate", "channels", "bit_rate", "format_name"):
        value = facts.get(key)
        if value is not None:
            details[key] = value
    if details["size_bytes"] is None:
        try:
            details["size_bytes"] = int(path.stat().st_size)
        except OSError:
            details["size_bytes"] = None
    return details


@app.get("/", response_class=HTMLResponse)
def index() -> Any:
    """The web interface."""
    page = STATIC_DIR / "index.html"
    if not page.exists():
        return HTMLResponse("<h1>MASH-Mix-Audio-and-Text</h1><p>Web interface file missing.</p>")
    return HTMLResponse(page.read_text(encoding="utf-8"))


@app.get("/api/health")
def health() -> Dict[str, Any]:
    """Liveness plus a quick view of what is installed."""
    from ..backends.registry import backend_infos

    backends = backend_infos()
    return {
        "status": "ok",
        "version": __version__,
        "stt_backends": [b["name"] for b in backends if b["kind"] == "stt" and b.get("available")],
        "tts_backends": [b["name"] for b in backends if b["kind"] == "tts" and b.get("available")],
    }


def _analysis_details(path: Path) -> Dict[str, Any]:
    """Measure the uploaded file: levels, headroom, pitch and tempo.

    Every measurement is optional. A file that cannot be decoded still returns
    the facts the probe found, so the interface always has something to show.
    """
    from ..audio.analyze import analyze as run_analyze
    from ..audio.io import load as load_audio

    details: Dict[str, Any] = {
        "peak_dbfs": None,
        "rms_dbfs": None,
        "loudness_lufs": None,
        "clipped_samples": 0,
        "dc_offset_percent": None,
        "head_silence_seconds": None,
        "crest_db": None,
        "key": None,
        "mode": None,
        "bpm": None,
        "backend": "",
    }
    try:
        buffer = load_audio(str(path), sr=None, mono=False)
    except Exception:
        return details
    try:
        # The channels are kept apart, because the loudness of a stereo file is
        # measured per channel and not on a downmix.
        result = run_analyze(buffer.samples, buffer.sample_rate)
    except Exception:
        return details

    details.update(
        {
            "peak_dbfs": result.peak_dbfs,
            "rms_dbfs": result.rms_dbfs,
            "loudness_lufs": result.loudness_lufs,
            "clipped_samples": int(result.clipped_samples),
            "dc_offset_percent": result.dc_offset_percent,
            "head_silence_seconds": result.head_silence_seconds,
            "crest_db": result.crest_db,
            "key": result.key,
            "mode": result.mode,
            "bpm": result.bpm,
            "backend": result.backend,
        }
    )
    return details


@app.post("/api/analyze")
async def analyze_upload(file: UploadFile = File(...)) -> Dict[str, Any]:
    """Measure an uploaded file. Returns the levels, the headroom, the pitch and the tempo."""
    from ..audio.io import probe

    job_id = uuid.uuid4().hex[:12]
    folder = _job_dir(job_id)
    source = _save_upload(file, folder)
    try:
        facts: Dict[str, Any] = {}
        try:
            facts = probe(source)
        except Exception:
            facts = {}
        payload = _analysis_details(source)
    finally:
        shutil.rmtree(folder, ignore_errors=True)

    payload["source"] = {
        "name": file.filename or source.name,
        "size_bytes": facts.get("size_bytes"),
        "duration": facts.get("duration"),
        "sample_rate": facts.get("sample_rate"),
        "channels": facts.get("channels"),
        "bit_rate": facts.get("bit_rate"),
        "format_name": facts.get("format_name"),
    }
    return payload


@app.get("/api/languages")
def languages() -> List[Dict[str, Any]]:
    """Every registered language tag with its standard identifiers."""
    from ..lang.registry import list_languages

    return list_languages()


@app.get("/api/models")
def models() -> List[Dict[str, Any]]:
    """The known model catalogue."""
    from ..models import MODEL_CATALOG

    return MODEL_CATALOG


@app.post("/api/lyrics")
async def lyrics(
    file: UploadFile = File(...),
    language: str = Form("bn-BD"),
    script: str = Form("bangla"),
    title: Optional[str] = Form(None),
    artist: Optional[str] = Form(None),
    separate_vocals: bool = Form(False),
    backend: str = Form("auto"),
    model: Optional[str] = Form(None),
    quality: str = Form("auto"),
) -> Dict[str, Any]:
    """Extract lyrics from an uploaded song. Returns the lines and download links."""
    from ..tasks.lyrics import extract_lyrics

    job_id = uuid.uuid4().hex[:12]
    folder = _job_dir(job_id)
    source = _save_upload(file, folder)
    source_info = _source_details(source, file.filename or source.name)

    try:
        result = extract_lyrics(
            str(source),
            backend=backend,
            model=model,
            language=language or None,
            title=title,
            artist=artist,
            separate_vocals=bool(separate_vocals),
            script=script or "bangla",
            quality=quality or "auto",
        )
    except Exception as exc:
        shutil.rmtree(folder, ignore_errors=True)
        raise HTTPException(status_code=500, detail=str(exc)) from exc

    stem = (title or Path(file.filename or "lyrics").stem) or "lyrics"
    srt_path = folder / f"{stem}.srt"
    txt_path = folder / f"{stem}.txt"
    vtt_path = folder / f"{stem}.vtt"
    lrc_path = folder / f"{stem}.lrc"

    write_subtitles(result.lines, str(srt_path), fmt="srt", title=title)
    write_text(str(txt_path), result.text + "\n")
    write_subtitles(result.lines, str(vtt_path), fmt="vtt", title=title)
    write_subtitles(result.lines, str(lrc_path), fmt="lrc")

    _JOBS[job_id] = {
        "result": result,
        "files": {
            "srt": str(srt_path),
            "txt": str(txt_path),
            "vtt": str(vtt_path),
            "lrc": str(lrc_path),
        },
    }

    return {
        "job_id": job_id,
        "language": result.language,
        "script": result.meta.get("output_script", script or "bangla"),
        "duration": result.duration,
        "title": result.title,
        "artist": result.artist,
        "source": source_info,
        "text": result.text,
        "lines": [line.to_dict() for line in result.lines],
        "meta": result.meta,
        "downloads": {
            "srt": f"/api/lyrics/{job_id}/download/srt",
            "txt": f"/api/lyrics/{job_id}/download/txt",
            "vtt": f"/api/lyrics/{job_id}/download/vtt",
            "lrc": f"/api/lyrics/{job_id}/download/lrc",
        },
    }


@app.get("/api/lyrics/{job_id}/download/{fmt}")
def download(job_id: str, fmt: str) -> Any:
    """Download a finished lyric file."""
    job = _JOBS.get(job_id)
    if not job:
        raise HTTPException(status_code=404, detail="Unknown job id")
    path = job["files"].get(fmt.lower())
    if not path or not Path(path).exists():
        raise HTTPException(status_code=404, detail=f"Format '{fmt}' was not produced")
    media = {
        "srt": "application/x-subrip",
        "txt": "text/plain",
        "vtt": "text/vtt",
        "lrc": "text/plain",
    }.get(fmt.lower(), "application/octet-stream")
    return FileResponse(path, media_type=media, filename=Path(path).name)


@app.post("/api/transcribe")
async def transcribe(
    file: UploadFile = File(...),
    language: str = Form(""),
    script: str = Form("bangla"),
    backend: str = Form("auto"),
    model: Optional[str] = Form(None),
) -> Dict[str, Any]:
    """Transcribe an uploaded recording."""
    from ..tasks.transcribe import transcribe as run_transcribe

    job_id = uuid.uuid4().hex[:12]
    folder = _job_dir(job_id)
    source = _save_upload(file, folder)
    try:
        result = run_transcribe(
            str(source),
            backend=backend,
            model=model,
            language=language or None,
            script=script or "bangla",
        )
    except Exception as exc:
        shutil.rmtree(folder, ignore_errors=True)
        raise HTTPException(status_code=500, detail=str(exc)) from exc
    payload = result.to_dict()
    payload["source"] = _source_details(source, file.filename or source.name)
    return payload


@app.post("/api/speech")
def speech(payload: Dict[str, Any]) -> Response:
    """OpenAI style text to speech. Returns audio bytes."""
    from ..tasks.synthesize import synthesize

    text = payload.get("input") or payload.get("text") or ""
    if not text:
        raise HTTPException(status_code=400, detail="Field 'input' is required")
    language = payload.get("language") or "bn-BD"
    try:
        result = synthesize(
            text,
            backend=payload.get("backend", "auto"),
            model=payload.get("model"),
            language=language,
            voice=payload.get("voice"),
            speed=float(payload.get("speed", 1.0)),
            script=payload.get("script") or "bangla",
        )
    except Exception as exc:
        raise HTTPException(status_code=500, detail=str(exc)) from exc

    import io

    import soundfile as sf

    buffer = io.BytesIO()
    sf.write(buffer, result.audio, result.sample_rate, format="WAV")
    return Response(
        content=buffer.getvalue(),
        media_type="audio/wav",
        headers={"Content-Disposition": 'inline; filename="speech.wav"'},
    )


@app.post("/api/audio/transcriptions")
async def audio_transcriptions(
    file: UploadFile = File(...),
    model: str = Form("auto"),
    language: str = Form(""),
    response_format: str = Form("json"),
) -> Any:
    """OpenAI style speech to text."""
    from ..tasks.transcribe import transcribe as run_transcribe

    job_id = uuid.uuid4().hex[:12]
    folder = _job_dir(job_id)
    source = _save_upload(file, folder)
    try:
        result = run_transcribe(
            str(source),
            backend=model if model in ("auto", "whisper", "faster-whisper") else "auto",
            language=language or None,
        )
    except Exception as exc:
        shutil.rmtree(folder, ignore_errors=True)
        raise HTTPException(status_code=500, detail=str(exc)) from exc

    if response_format == "text":
        return Response(content=result.text, media_type="text/plain")
    if response_format == "vtt":
        from ..export.subtitles import segments_to_lines

        path = folder / "out.vtt"
        write_subtitles(segments_to_lines(result.segments), str(path), fmt="vtt")
        return FileResponse(path, media_type="text/vtt")
    if response_format == "srt":
        from ..export.subtitles import segments_to_lines

        path = folder / "out.srt"
        write_subtitles(segments_to_lines(result.segments), str(path), fmt="srt")
        return FileResponse(path, media_type="application/x-subrip")
    return JSONResponse(content=result.to_dict())


if STATIC_DIR.exists():
    app.mount("/static", StaticFiles(directory=str(STATIC_DIR)), name="static")
