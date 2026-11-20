# SPDX-License-Identifier: MIT | Copyright (c) 2026 MASH (Khaled Bhuiyan)
"""Microsoft Edge neural voices through the edge-tts package.

This is the online backend. It needs no weights on disk and it has the best
Bangla voices of anything in this package, with two of them:
``bn-BD-NabanitaNeural`` for Bangladeshi Bangla and ``bn-IN-SwaraNeural`` for
Indian Bengali.

Because the service is online, the voice list is fetched when the machine has
a connection and falls back to a short static list when it does not.
"""

from __future__ import annotations

import asyncio
import io
import time
from concurrent.futures import ThreadPoolExecutor
from typing import Any, Optional

import numpy as np

from ...errors import DependencyMissingError, MashMixError
from ...lang.normalize import normalize_for_tts
from ...types import AudioBuffer, SynthesisResult
from ..base import BackendInfo, TTSBackend
from ..registry import register_tts

try:  # pragma: no cover - the import guard keeps this module importable
    import edge_tts

    _HAS_EDGE_TTS = True
except ImportError:  # pragma: no cover
    edge_tts = None  # type: ignore[assignment]
    _HAS_EDGE_TTS = False


_DEFAULT_VOICE = "en-US-AriaNeural"
_DEFAULT_SAMPLE_RATE = 24000

# The Bangla voices, plus a few common ones for other languages.
_STATIC_VOICES = [
    "bn-BD-NabanitaNeural",
    "bn-IN-SwaraNeural",
    "en-US-AriaNeural",
    "en-US-GuyNeural",
    "en-GB-SoniaNeural",
    "en-IN-NeerjaNeural",
    "hi-IN-SwaraNeural",
    "ta-IN-VasanthiNeural",
    "te-IN-MrityunjayiNeural",
    "bn-IN-AaravNeural",
]

# Preferred voice per language tag.
_VOICE_FOR_LANGUAGE = {
    "bn": "bn-BD-NabanitaNeural",
    "bn-bd": "bn-BD-NabanitaNeural",
    "bn-in": "bn-IN-SwaraNeural",
}


def _time_stretch_simple(wave: np.ndarray, factor: float) -> np.ndarray:
    """Change the duration of ``wave`` by ``factor`` and keep the pitch.

    A factor below 1 makes the speech shorter, so it plays faster, and the
    pitch stays where it was. The work is an overlap add pass: frames are cut
    from the input at a steady spacing, each frame is crossfaded with linear
    interpolation ramps, and the frames are laid down at the new spacing. The
    samples inside a frame are untouched, which is what holds the pitch in
    place, and a short search aligns the frames so the overlaps do not click.
    """
    if wave is None or wave.size == 0:
        return np.asarray(wave, dtype=np.float32)
    factor = float(factor)
    if factor <= 0:
        raise ValueError("time stretch factor must be positive")
    source = wave.astype(np.float64, copy=False)
    if abs(factor - 1.0) < 1e-6:
        return source.astype(np.float32)
    n_in = source.size
    n_out = max(1, int(round(n_in * factor)))
    frame = 1024
    hop = frame // 2
    if n_in < 2 * frame:
        # Too short for an overlap add pass, so fall back to a plain resample.
        old_x = np.arange(n_in, dtype=np.float64)
        new_x = np.linspace(0.0, n_in - 1, n_out, dtype=np.float64)
        return np.interp(new_x, old_x, source).astype(np.float32)

    out_hop = max(1, int(round(hop * factor)))
    search = max(1, hop // 2)
    overlap = max(1, frame - out_hop)
    ramp = np.linspace(0.0, 1.0, hop, endpoint=False)
    window = np.concatenate([ramp, ramp[::-1]])
    out = np.zeros(n_out + frame, dtype=np.float64)
    weight = np.zeros_like(out)

    previous = None
    template = None
    out_pos = 0
    while out_pos < n_out:
        if previous is None:
            start = 0
        else:
            ideal = previous + hop
            low = max(0, ideal - search)
            high = min(n_in - frame, ideal + search)
            start = max(0, min(ideal, n_in - frame))
            if high > low and template is not None and template.size == overlap:
                region = source[low : high + overlap]
                if region.size >= overlap:
                    corr = np.correlate(region, template, mode="valid")
                    start = low + int(np.argmax(corr))
        segment = source[start : start + frame]
        if segment.size < frame:
            break
        out[out_pos : out_pos + frame] += segment * window
        weight[out_pos : out_pos + frame] += window
        template = source[start + out_hop : start + frame]
        previous = start
        out_pos += out_hop

    result = out[:n_out]
    norm = np.where(weight[:n_out] > 1e-6, weight[:n_out], 1.0)
    return (result / norm).astype(np.float32)


def async_to_sync(coro: Any) -> Any:
    """Run an awaitable from synchronous code.

    The desktop tools of this package are synchronous, while edge-tts is
    asynchronous, so the coroutine is driven from a fresh event loop. When the
    caller already sits inside a running loop the work moves to a worker
    thread so the loop is never nested.
    """
    try:
        asyncio.get_running_loop()
    except RuntimeError:
        return asyncio.run(coro)
    with ThreadPoolExecutor(max_workers=1) as pool:
        return pool.submit(asyncio.run, coro).result()


async def _collect_audio(communicate: Any) -> bytes:
    """Gather the mp3 chunks of an edge-tts stream."""
    chunks: list[bytes] = []
    async for item in communicate.stream():
        if item.get("type") == "audio":
            chunks.append(item["data"])
    return b"".join(chunks)


async def _fetch_voices() -> list[str]:
    """Ask the service for the current voice list."""
    voices = await edge_tts.list_voices()
    names = []
    for voice in voices:
        short = voice.get("ShortName") if isinstance(voice, dict) else None
        if short:
            names.append(str(short))
    return sorted(names)


def _decode_audio(data: bytes) -> tuple[np.ndarray, int]:
    """Decode an mp3 payload into mono float32 samples."""
    problems: list[str] = []
    try:
        import soundfile as sf

        samples, rate = sf.read(io.BytesIO(data), dtype="float32", always_2d=False)
        return np.asarray(samples, dtype=np.float32).reshape(-1), int(rate)
    except Exception as exc:
        problems.append(f"soundfile: {exc}")
    try:
        import torchaudio

        tensor, rate = torchaudio.load(io.BytesIO(data))
        return tensor.squeeze(0).numpy().astype(np.float32), int(rate)
    except Exception as exc:
        problems.append(f"torchaudio: {exc}")
    try:
        import miniaudio

        decoded = miniaudio.decode(data)
        return np.asarray(decoded.samples, dtype=np.float32), int(decoded.sample_rate)
    except Exception as exc:
        problems.append(f"miniaudio: {exc}")
    raise DependencyMissingError(
        "soundfile",
        "tts",
        "decoding the edge-tts mp3 stream (tried soundfile, torchaudio and miniaudio: "
        + "; ".join(problems)
        + ")",
    )


@register_tts("edge-tts")
class EdgeTTSBackend(TTSBackend):
    """Microsoft Edge neural voices, the online Bangla option."""

    info = BackendInfo(
        name="edge-tts",
        kind="tts",
        summary="Microsoft Edge neural voices online, with the best Bangla voices available.",
        package="edge-tts",
        extra="tts",
        languages=["bn", "bn-BD", "bn-IN", "en", "en-US", "en-GB", "en-IN", "hi", "ta", "te"],
        platforms=["windows", "linux", "macos", "android-termux", "ios"],
        notes=(
            "Bangla is covered by bn-BD-NabanitaNeural for Bangladeshi Bangla and "
            "bn-IN-SwaraNeural for Indian Bengali. This backend is online, it needs no weights and "
            "it returns mp3, so a decoder such as soundfile must be present."
        ),
    )

    def __init__(self) -> None:
        super().__init__()
        self._voices: list[str] = []
        self._sample_rate = _DEFAULT_SAMPLE_RATE

    # -- lifecycle ------------------------------------------------------
    def is_available(self) -> bool:
        return _HAS_EDGE_TTS

    def load(self, model: Optional[str] = None) -> None:
        if self._loaded:
            return
        if not _HAS_EDGE_TTS:
            raise DependencyMissingError("edge-tts", "tts", "Edge text to speech")
        self._model_id = model or "edge-tts-online"
        self._loaded = True

    def unload(self) -> None:
        self._voices = []
        self._loaded = False

    # -- inference ------------------------------------------------------
    def synthesize(
        self,
        text: str,
        *,
        language: str = "bn-BD",
        voice: Optional[str] = None,
        speed: float = 1.0,
        reference_audio: Optional[AudioBuffer] = None,
        **kwargs: Any,
    ) -> SynthesisResult:
        started = time.perf_counter()
        if not self._loaded:
            self.load(kwargs.get("model"))
        prepared = normalize_for_tts(text, language=language, expand_numbers=False)
        meta: dict[str, Any] = {"normalised": prepared != text}
        if reference_audio is not None:
            meta["reference_audio"] = "ignored"
            meta["note"] = "Edge voices are fixed, so the reference audio was ignored."

        if not prepared.strip():
            return SynthesisResult(
                audio=np.zeros(0, dtype=np.float32),
                sample_rate=self._sample_rate,
                backend=self.name,
                model=self._model_id,
                voice=voice,
                elapsed=time.perf_counter() - started,
                meta=meta,
            )

        voice_id = voice or _VOICE_FOR_LANGUAGE.get((language or "").lower(), _DEFAULT_VOICE)
        meta["voice"] = voice_id
        rate = f"{(float(speed) - 1.0) * 100.0:+.0f}%" if abs(float(speed) - 1.0) > 1e-6 else "+0%"

        payload, used_rate = self._fetch(prepared, voice_id, rate)
        if not used_rate:
            meta["rate"] = "fallback"
        samples, sample_rate = _decode_audio(payload)
        if not used_rate:
            samples = _time_stretch_simple(samples, 1.0 / max(float(speed), 0.05))

        return SynthesisResult(
            audio=samples,
            sample_rate=sample_rate,
            backend=self.name,
            model=self._model_id,
            voice=voice_id,
            elapsed=time.perf_counter() - started,
            meta=meta,
        )

    # -- helpers --------------------------------------------------------
    def _fetch(self, text: str, voice: str, rate: str) -> tuple[bytes, bool]:
        """Ask the service for the audio, with a plain rate as a fallback."""
        try:
            communicate = edge_tts.Communicate(text, voice, rate=rate)
            payload = async_to_sync(_collect_audio(communicate))
            if payload:
                return payload, True
        except Exception:
            # The rate argument was refused, so the plain call below is used
            # and the tempo is applied to the audio afterwards.
            pass
        try:
            communicate = edge_tts.Communicate(text, voice)
            payload = async_to_sync(_collect_audio(communicate))
        except Exception as exc:
            raise MashMixError(f"The edge-tts service could not be reached: {exc}") from exc
        return payload, False

    def list_voices(self) -> list[str]:
        """Voices from the service, with a static list when offline."""
        if self._voices:
            return list(self._voices)
        if _HAS_EDGE_TTS:
            try:
                self._voices = async_to_sync(_fetch_voices())
            except Exception:
                self._voices = []
        if not self._voices:
            self._voices = list(_STATIC_VOICES)
        return list(self._voices)
