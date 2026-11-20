# SPDX-License-Identifier: MIT | Copyright (c) 2026 MASH (Khaled Bhuiyan)
"""Audio decoding, encoding and format probing.

The module never assumes a particular container. Every request is served by a
chain of decoders:

1. ``soundfile`` (libsndfile) for wav, flac, ogg and anything else the host
   libsndfile build understands.
2. The ``ffmpeg`` binary on PATH, decoding to raw ``f32le`` PCM on stdout.
   This is the important route because it is the only one that handles mp3,
   m4a, aac, wma, opus, mp4, mkv and webm on every platform, and because it has
   no duration or file size limit.
3. ``torchaudio`` when it happens to be installed.

Only numpy is a hard requirement. Every other dependency is optional and the
module still imports when none of them are present.
"""

from __future__ import annotations

import json
import os
import re
import shutil
import subprocess
import tempfile
import wave
from collections.abc import Iterator
from pathlib import Path
from typing import Any

import numpy as np

from ..errors import AudioDecodeError
from ..types import AudioBuffer

__all__ = [
    "load",
    "save",
    "probe",
    "is_supported_format",
    "SUPPORTED_SUFFIXES",
    "HAVE_SOUNDFILE",
    "HAVE_TORCHAUDIO",
    "ffmpeg_path",
    "ffprobe_path",
]

try:  # pragma: no cover - depends on the host environment
    import soundfile as _soundfile

    HAVE_SOUNDFILE = True
except Exception:  # pragma: no cover - optional dependency
    _soundfile = None  # type: ignore[assignment]
    HAVE_SOUNDFILE = False

try:  # pragma: no cover - depends on the host environment
    import torchaudio as _torchaudio

    HAVE_TORCHAUDIO = True
except Exception:  # pragma: no cover - optional dependency
    _torchaudio = None  # type: ignore[assignment]
    HAVE_TORCHAUDIO = False


#: Container suffixes the layer is expected to open on every platform.
SUPPORTED_SUFFIXES = frozenset(
    {
        ".wav",
        ".wave",
        ".mp3",
        ".flac",
        ".ogg",
        ".oga",
        ".opus",
        ".m4a",
        ".mp4",
        ".aac",
        ".wma",
        ".mkv",
        ".webm",
        ".aiff",
        ".aif",
        ".caf",
        ".amr",
        ".3gp",
        ".mka",
        ".spx",
        ".au",
    }
)

_STREAM_RE = re.compile(
    r"Stream #\d+:\d+(?::\d+)?: Audio: [^,]+, (\d+) Hz, ([A-Za-z0-9.]+)"
)
_CHANNEL_NAMES = {
    "mono": 1,
    "stereo": 2,
    "quad": 4,
    "quad(side)": 4,
    "3.0": 3,
    "4.0": 4,
    "5.0": 5,
    "5.1": 6,
    "6.1": 7,
    "7.1": 8,
}


# ---------------------------------------------------------------------------
# external tool discovery
# ---------------------------------------------------------------------------


def ffmpeg_path() -> str | None:
    """Return the path to the ``ffmpeg`` binary or ``None`` when not on PATH."""
    return shutil.which("ffmpeg")


def ffprobe_path() -> str | None:
    """Return the path to the ``ffprobe`` binary or ``None`` when not on PATH."""
    return shutil.which("ffprobe")


def _decode_channels(token: str) -> int:
    """Translate an ffmpeg channel layout name such as ``5.1`` into a count."""
    key = token.strip().lower()
    if key in _CHANNEL_NAMES:
        return _CHANNEL_NAMES[key]
    try:
        if "." in key:
            whole = float(key)
            return int(round(whole + 1))
        return int(round(float(key)))
    except (TypeError, ValueError):
        return 2


# ---------------------------------------------------------------------------
# source helpers
# ---------------------------------------------------------------------------


def _looks_like_path(value: Any) -> bool:
    return isinstance(value, (str, os.PathLike))


def _source_and_handle(path_or_file: Any) -> tuple[str, Any | None]:
    """Return ``(ffmpeg_source, file_object)`` for the requested input."""
    if _looks_like_path(path_or_file):
        path = Path(os.fspath(path_or_file))
        if not path.exists():
            raise AudioDecodeError(f"Audio file not found: {path}")
        return str(path), None
    if hasattr(path_or_file, "read"):
        return "pipe:0", path_or_file
    raise AudioDecodeError(
        "Expected a path, a pathlib.Path or a binary file object, "
        f"got {type(path_or_file).__name__}"
    )


def _probe_native(path: str) -> dict[str, Any] | None:
    """Return native stream facts using ffprobe, or ``None`` when unavailable."""
    probe_bin = ffprobe_path()
    if not probe_bin:
        return None
    cmd = [
        probe_bin,
        "-v",
        "error",
        "-print_format",
        "json",
        "-show_format",
        "-show_streams",
        path,
    ]
    try:
        proc = subprocess.run(cmd, capture_output=True)
    except OSError:
        return None
    if proc.returncode != 0:
        return None
    try:
        payload = json.loads(proc.stdout.decode("utf-8", "replace"))
    except (ValueError, UnicodeDecodeError):
        return None
    streams = [s for s in payload.get("streams", []) if s.get("codec_type") == "audio"]
    if not streams:
        return None
    stream = streams[0]
    fmt = payload.get("format", {})
    return {
        "sample_rate": int(stream.get("sample_rate") or 0) or None,
        "channels": int(stream.get("channels") or 0) or None,
        "duration": float(stream.get("duration") or 0.0) or None,
        "bit_rate": int(fmt.get("bit_rate") or 0) or None,
        "format_name": str(fmt.get("format_name") or "") or None,
        "size_bytes": int(fmt.get("size") or 0) or None,
    }


def _parse_stream_info(stderr: str) -> tuple[int | None, int | None]:
    """Recover the output sample rate and channel count from ffmpeg logs."""
    matches = _STREAM_RE.findall(stderr or "")
    if not matches:
        return None, None
    rate, layout = matches[-1]
    try:
        return int(rate), _decode_channels(layout)
    except (TypeError, ValueError):
        return None, None


def _finalize(
    samples: np.ndarray,
    sample_rate: int,
    mono: bool,
    target_sr: int | None,
) -> AudioBuffer:
    """Normalise an array to the layout the caller asked for."""
    data = np.asarray(samples, dtype=np.float32)
    if data.ndim == 1:
        data = data[None, :]
    elif data.ndim != 2:
        raise AudioDecodeError(f"Unexpected audio shape {data.shape}")
    if mono and data.shape[0] > 1:
        data = data.mean(axis=0, keepdims=True)
    elif not mono and data.shape[0] == 1 and sample_rate:
        data = np.repeat(data, 2, axis=0)
    if target_sr and int(target_sr) != int(sample_rate):
        from . import resample as _resample

        data = _resample.resample(data, int(sample_rate), int(target_sr))
        sample_rate = int(target_sr)
    return AudioBuffer(samples=data, sample_rate=int(sample_rate), channels=data.shape[0])


# ---------------------------------------------------------------------------
# decoders
# ---------------------------------------------------------------------------


def _load_soundfile(
    source: str,
    file_obj: Any | None,
    sr: int | None,
    mono: bool,
    offset: float,
    duration: float | None,
) -> AudioBuffer:
    """Decode with soundfile. Raises on anything the libsndfile build refuses."""
    handle = file_obj if file_obj is not None else source
    info = _soundfile.info(handle)
    native_sr = int(info.samplerate)
    start = int(max(0.0, float(offset)) * native_sr)
    frames = -1
    if duration is not None:
        frames = int(max(0.0, float(duration)) * native_sr)
    data, native_sr = _soundfile.read(
        handle, dtype="float32", always_2d=True, start=start, frames=frames
    )
    if data.size == 0 and frames != 0:
        data = np.zeros((0, 1), dtype=np.float32)
    samples = np.asarray(data, dtype=np.float32).T
    return _finalize(samples, native_sr, mono, sr)


def _ffmpeg_command(
    source: str,
    is_pipe: bool,
    out_sr: int | None,
    out_channels: int | None,
    offset: float,
    duration: float | None,
    loglevel: str = "info",
) -> list[str]:
    """Build the ffmpeg argv used for raw PCM decoding."""
    ffmpeg = ffmpeg_path()
    cmd = [ffmpeg or "ffmpeg", "-nostdin", "-hide_banner", "-loglevel", loglevel]
    offset = max(0.0, float(offset or 0.0))
    if is_pipe:
        # A pipe cannot be seeked, so the offset is applied after the input.
        cmd += ["-i", source]
        if offset > 0:
            cmd += ["-ss", f"{offset:.6f}"]
    else:
        if offset > 0:
            cmd += ["-ss", f"{offset:.6f}"]
        cmd += ["-i", source]
    if duration is not None and float(duration) > 0:
        cmd += ["-t", f"{float(duration):.6f}"]
    if out_channels:
        cmd += ["-ac", str(int(out_channels))]
    if out_sr:
        cmd += ["-ar", str(int(out_sr))]
    cmd += ["-f", "f32le", "-acodec", "pcm_f32le", "-"]
    return cmd


def _load_ffmpeg(
    source: str,
    file_obj: Any | None,
    sr: int | None,
    mono: bool,
    offset: float,
    duration: float | None,
) -> AudioBuffer:
    """Decode with the ffmpeg binary into raw float32 PCM."""
    if not ffmpeg_path():
        raise RuntimeError("the ffmpeg binary was not found on PATH")
    is_pipe = source == "pipe:0"
    native = None
    if not is_pipe:
        native = _probe_native(source)
    out_sr = int(sr) if sr else (native or {}).get("sample_rate")
    out_channels = 1 if mono else None
    cmd = _ffmpeg_command(
        source, is_pipe, out_sr, out_channels, offset, duration, loglevel="info"
    )
    stdin_data = None
    stdin_handle: Any | None = None
    if is_pipe:
        position = None
        try:
            position = file_obj.tell()
            file_obj.seek(0)
        except Exception:
            position = None
        if hasattr(file_obj, "fileno"):
            stdin_handle = file_obj
        else:
            stdin_data = file_obj.read()
        if position is not None and stdin_data is not None:
            file_obj.seek(position)
    try:
        proc = subprocess.run(
            cmd,
            input=stdin_data,
            stdin=None if stdin_handle is None else stdin_handle,
            capture_output=True,
        )
    except OSError as exc:
        raise RuntimeError(f"could not start ffmpeg: {exc}") from exc
    stderr = proc.stderr.decode("utf-8", "replace")
    if proc.returncode != 0:
        raise RuntimeError(stderr.strip() or f"ffmpeg exited with {proc.returncode}")
    parsed_sr, parsed_ch = _parse_stream_info(stderr)
    native_sr = int(out_sr or parsed_sr or 44100)
    channels = int(out_channels or parsed_ch or 1)
    raw = np.frombuffer(proc.stdout, dtype="<f4")
    if channels > 1:
        usable = (raw.size // channels) * channels
        raw = raw[:usable]
        samples = raw.reshape(-1, channels).T
    else:
        samples = raw
    return _finalize(samples, native_sr, mono, sr)


def _load_torchaudio(
    source: str,
    file_obj: Any | None,
    sr: int | None,
    mono: bool,
    offset: float,
    duration: float | None,
) -> AudioBuffer:
    """Decode with torchaudio as the last resort."""
    handle = file_obj if file_obj is not None else source
    waveform, native_sr = _torchaudio.load(handle)
    samples = waveform.detach().cpu().numpy().astype(np.float32)
    offset_samples = int(max(0.0, float(offset)) * int(native_sr))
    if offset_samples:
        samples = samples[:, offset_samples:]
    if duration is not None:
        limit = int(max(0.0, float(duration)) * int(native_sr))
        samples = samples[:, :limit]
    return _finalize(samples, int(native_sr), mono, sr)


# ---------------------------------------------------------------------------
# public loading API
# ---------------------------------------------------------------------------


def load(
    path_or_file: Any,
    *,
    sr: int | None = None,
    mono: bool = True,
    offset: float = 0.0,
    duration: float | None = None,
    stream: bool = False,
) -> AudioBuffer | Iterator[AudioBuffer]:
    """Decode an audio file into an :class:`AudioBuffer`.

    Parameters
    ----------
    path_or_file:
        A filesystem path, a :class:`pathlib.Path` or a binary file object.
    sr:
        Resample the result to this rate. ``None`` keeps the native rate.
    mono:
        Downmix to a single channel.
    offset:
        Skip this many seconds before decoding starts.
    duration:
        Only decode this many seconds after the offset.
    stream:
        Return an iterator of one second :class:`AudioBuffer` blocks instead of
        a single buffer. Nothing is decoded until the iterator is consumed.

    Raises
    ------
    AudioDecodeError
        When no decoder could read the input. The message lists every method
        that was tried together with the reason it failed.
    """
    source, file_obj = _source_and_handle(path_or_file)
    if stream:
        return _load_stream(source, file_obj, sr, mono, offset)
    failures: list[str] = []
    if HAVE_SOUNDFILE:
        try:
            return _load_soundfile(source, file_obj, sr, mono, offset, duration)
        except Exception as exc:  # noqa: BLE001 - reported in the final error
            failures.append(f"soundfile: {exc}")
    try:
        return _load_ffmpeg(source, file_obj, sr, mono, offset, duration)
    except Exception as exc:  # noqa: BLE001 - reported in the final error
        failures.append(f"ffmpeg: {exc}")
    if HAVE_TORCHAUDIO:
        try:
            return _load_torchaudio(source, file_obj, sr, mono, offset, duration)
        except Exception as exc:  # noqa: BLE001 - reported in the final error
            failures.append(f"torchaudio: {exc}")
    tried = " | ".join(failures) if failures else "no decoder was available"
    raise AudioDecodeError(
        f"Could not decode audio from {source!r}. Methods tried: {tried}"
    )


def _load_stream(
    source: str,
    file_obj: Any | None,
    sr: int | None,
    mono: bool,
    offset: float,
) -> Iterator[AudioBuffer]:
    """Yield decoded blocks of about one second without holding the whole file."""
    native: dict[str, Any] | None = None
    if source != "pipe:0":
        native = _probe_native(source)
    if sr is None and native is None:
        raise AudioDecodeError(
            "Streaming needs the native sample rate, which could not be probed. "
            "Install ffprobe or pass sr explicitly."
        )
    out_sr = int(sr) if sr else int((native or {}).get("sample_rate") or 44100)
    out_channels = 1 if mono else int((native or {}).get("channels") or 1)
    if not ffmpeg_path():
        raise AudioDecodeError(
            "Streaming decode requires the ffmpeg binary on PATH"
        )
    cmd = _ffmpeg_command(
        source,
        source == "pipe:0",
        out_sr,
        out_channels,
        offset,
        None,
        loglevel="error",
    )
    stdin_handle = None
    if source == "pipe:0" and hasattr(file_obj, "fileno"):
        try:
            file_obj.seek(0)
        except Exception:
            pass
        stdin_handle = file_obj
    with tempfile.TemporaryFile() as err_file:
        proc = subprocess.Popen(
            cmd,
            stdin=stdin_handle if stdin_handle is not None else subprocess.DEVNULL,
            stdout=subprocess.PIPE,
            stderr=err_file,
        )
        completed = False
        try:
            block = max(1, int(out_sr))
            channels = max(1, out_channels)
            while True:
                raw = proc.stdout.read(block * channels * 4)
                if not raw:
                    completed = True
                    break
                values = np.frombuffer(raw, dtype="<f4")
                usable = (values.size // channels) * channels
                values = values[:usable]
                if values.size == 0:
                    continue
                if channels > 1:
                    samples = values.reshape(-1, channels).T
                else:
                    samples = values
                yield AudioBuffer(samples=samples, sample_rate=out_sr, channels=channels)
        finally:
            try:
                proc.stdout.close()
            except Exception:
                pass
            code = proc.wait()
            if completed and code not in (0, None):
                err_file.seek(0)
                message = err_file.read().decode("utf-8", "replace").strip()
                raise AudioDecodeError(
                    f"ffmpeg stream failed with code {code}: {message}"
                )


# ---------------------------------------------------------------------------
# public saving API
# ---------------------------------------------------------------------------

_CODEC_BY_FORMAT = {
    "wav": "pcm_s16le",
    "wave": "pcm_s16le",
    "mp3": "libmp3lame",
    "flac": "flac",
    "ogg": "libvorbis",
    "oga": "libvorbis",
    "opus": "libopus",
    "m4a": "aac",
    "mp4": "aac",
    "aac": "aac",
    "wav ": "pcm_s16le",
}

_CODEC_BY_SUBTYPE = {
    "pcm_s16le": "pcm_s16le",
    "pcm_16": "pcm_s16le",
    "pcm_s24le": "pcm_s24le",
    "pcm_24": "pcm_s24le",
    "pcm_s32le": "pcm_s32le",
    "pcm_32": "pcm_s32le",
    "pcm_f32le": "pcm_f32le",
    "float": "pcm_f32le",
    "pcm_32f": "pcm_f32le",
    "ulaw": "pcm_mulaw",
    "alaw": "pcm_alaw",
}


def _ffmpeg_write(
    target: Path,
    data: np.ndarray,
    sr: int,
    fmt: str,
    subtype: str | None,
) -> None:
    """Write PCM float32 through ffmpeg, encoding into the requested format."""
    ffmpeg = ffmpeg_path()
    if not ffmpeg:
        raise RuntimeError("the ffmpeg binary was not found on PATH")
    channels = int(data.shape[0])
    codec = None
    if subtype:
        codec = _CODEC_BY_SUBTYPE.get(str(subtype).strip().lower())
    if codec is None:
        codec = _CODEC_BY_FORMAT.get(fmt.lower())
    cmd = [
        ffmpeg,
        "-nostdin",
        "-hide_banner",
        "-loglevel",
        "error",
        "-y",
        "-f",
        "f32le",
        "-acodec",
        "pcm_f32le",
        "-ar",
        str(int(sr)),
        "-ac",
        str(channels),
        "-i",
        "pipe:0",
    ]
    if codec:
        cmd += ["-c:a", codec]
    if fmt.lower() in ("mp3", "mp2"):
        cmd += ["-b:a", "192k"]
    elif fmt.lower() in ("opus",):
        cmd += ["-b:a", "160k"]
    elif fmt.lower() in ("ogg", "oga"):
        cmd += ["-q:a", "5"]
    cmd += [str(target)]
    payload = np.ascontiguousarray(data, dtype="<f4").tobytes()
    proc = subprocess.run(cmd, input=payload, capture_output=True)
    if proc.returncode != 0:
        message = proc.stderr.decode("utf-8", "replace").strip()
        raise RuntimeError(message or f"ffmpeg exited with {proc.returncode}")


def _wave_write(target: Path, data: np.ndarray, sr: int, subtype: str | None) -> None:
    """Write a WAV file with the standard library only."""
    channels = int(data.shape[0])
    if channels > 2:
        raise RuntimeError("the wave module can only write mono or stereo files")
    requested = (subtype or "PCM_16").strip().upper()
    if requested in ("FLOAT", "PCM_32F"):
        raise RuntimeError("float wav files need the ffmpeg encoder")
    width = 4 if requested in ("PCM_32", "PCM_24") else 2
    clipped = np.clip(data, -1.0, 1.0)
    if width == 4:
        ints = (clipped * 2147483647.0).astype("<i4")
    else:
        ints = (clipped * 32767.0).astype("<i2")
    with wave.open(str(target), "wb") as handle:
        handle.setnchannels(channels)
        handle.setsampwidth(width)
        handle.setframerate(int(sr))
        handle.writeframes(ints.T.tobytes())


def save(
    path: str | os.PathLike[str],
    audio: np.ndarray,
    sr: int,
    *,
    format: str | None = None,
    subtype: str | None = None,
) -> str:
    """Write audio to disk and return the written path.

    The container is taken from ``format`` or from the file suffix. Anything
    ffmpeg can encode is supported, including wav, mp3, flac, ogg, opus and
    m4a. WAV also has a fallback writer built on the standard library so a
    plain wav file can be produced without any external tool.
    """
    data = np.asarray(audio, dtype=np.float32)
    if data.ndim == 1:
        data = data[None, :]
    elif data.ndim != 2:
        raise ValueError(f"Expected 1-D or 2-D audio, got shape {data.shape}")
    if int(sr) <= 0:
        raise ValueError("Sample rate must be a positive number")
    target = Path(os.fspath(path))
    if str(target.parent) not in ("", "."):
        target.parent.mkdir(parents=True, exist_ok=True)
    fmt = (format or target.suffix.lstrip(".") or "wav").strip().lower().lstrip(".")
    failures: list[str] = []
    if ffmpeg_path():
        try:
            _ffmpeg_write(target, data, int(sr), fmt, subtype)
            return str(target)
        except Exception as exc:  # noqa: BLE001 - reported in the final error
            failures.append(f"ffmpeg: {exc}")
    if fmt in ("wav", "wave"):
        try:
            _wave_write(target, data, int(sr), subtype)
            return str(target)
        except Exception as exc:  # noqa: BLE001 - reported in the final error
            failures.append(f"wave: {exc}")
    tried = " | ".join(failures) if failures else "no encoder was available"
    raise AudioDecodeError(f"Could not write audio to {target}. Methods tried: {tried}")


# ---------------------------------------------------------------------------
# probing
# ---------------------------------------------------------------------------


def probe(path: str | os.PathLike[str]) -> dict[str, Any]:
    """Return container facts for a file.

    The mapping always carries the keys ``duration``, ``sample_rate``,
    ``channels``, ``bit_rate``, ``format_name`` and ``size_bytes``. A value is
    ``None`` when it could not be determined.
    """
    target = Path(os.fspath(path))
    info: dict[str, Any] = {
        "duration": None,
        "sample_rate": None,
        "channels": None,
        "bit_rate": None,
        "format_name": None,
        "size_bytes": None,
    }
    try:
        info["size_bytes"] = int(target.stat().st_size)
    except OSError:
        pass
    native = _probe_native(str(target))
    if native:
        info["sample_rate"] = native.get("sample_rate")
        info["channels"] = native.get("channels")
        info["duration"] = native.get("duration")
        info["bit_rate"] = native.get("bit_rate")
        info["format_name"] = native.get("format_name")
        if info["duration"] is None:
            info["duration"] = None
        return info
    if HAVE_SOUNDFILE:
        try:
            sf_info = _soundfile.info(str(target))
            info["sample_rate"] = int(sf_info.samplerate)
            info["channels"] = int(sf_info.channels)
            info["duration"] = float(sf_info.duration)
            info["format_name"] = str(sf_info.format or "") or None
            return info
        except Exception:  # noqa: BLE001 - fall through to the empty result
            pass
    return info


def is_supported_format(suffix: str) -> bool:
    """Return ``True`` when the suffix names a container the layer can open."""
    if not suffix:
        return False
    text = str(suffix).strip().lower()
    if not text.startswith("."):
        text = "." + text
    return text in SUPPORTED_SUFFIXES
