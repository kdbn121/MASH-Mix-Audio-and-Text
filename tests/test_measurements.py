# SPDX-License-Identifier: MIT | Copyright (c) 2026 MASH (Khaled Bhuiyan)
"""The measurements of an uploaded file.

These tests cover the numbers the web interface shows in the file panel: peak,
loudness, clipped samples, dc offset, head silence, crest, the key of the song
and its tempo. They also cover the ``/api/analyze`` route that returns them.

Every signal here is synthetic, so the expected answer is known in advance.
Nothing is downloaded, no model is loaded and the network is never touched.
"""

from __future__ import annotations

import asyncio
import io
import math

import numpy as np
import pytest


def _analyze_module():
    """The analysis module, or a skip when it cannot be imported."""
    return pytest.importorskip("mash_mix.audio.analyze")


def _server_module():
    """The server module, or a skip when FastAPI is not installed."""
    return pytest.importorskip("mash_mix.server.app")


def _tone(freq: float, seconds: float, rate: int, amplitude: float = 0.5) -> np.ndarray:
    """A pure sine tone."""
    time = np.arange(int(rate * seconds), dtype=np.float64) / float(rate)
    return amplitude * np.sin(2.0 * math.pi * freq * time)


def _triad(root: float, minor: bool, seconds: float, rate: int) -> np.ndarray:
    """A root triad with a few harmonics, so the chroma profile has something to read.

    ``minor`` picks the minor third over the major third. The chord is repeated
    so that the profile is stable.
    """
    third = root * (6.0 / 5.0) if minor else root * (5.0 / 4.0)
    fifth = root * (3.0 / 2.0)
    time = np.arange(int(rate * seconds), dtype=np.float64) / float(rate)
    signal = np.zeros_like(time)
    for freq, weight in ((root, 1.0), (third, 0.8), (fifth, 0.7)):
        for harmonic in (1, 2, 3):
            signal += weight / harmonic * np.sin(2.0 * math.pi * freq * harmonic * time)
    peak = float(np.max(np.abs(signal))) or 1.0
    return (signal / peak * 0.6).astype(np.float32)


def _clicks(bpm: float, seconds: float, rate: int) -> np.ndarray:
    """A click track at the given tempo, so the beat tracker has a known answer."""
    signal = np.zeros(int(rate * seconds), dtype=np.float64)
    period = 60.0 / bpm
    time = np.arange(signal.size, dtype=np.float64) / float(rate)
    for start in np.arange(0.0, seconds, period):
        index = int(start * rate)
        length = int(0.02 * rate)
        signal[index:index + length] += 0.9 * np.exp(-np.arange(length) / (0.004 * rate))
    return (signal * np.cos(2.0 * math.pi * 180.0 * time)).astype(np.float32)


# ---------------------------------------------------------------------------
# the individual measurements
# ---------------------------------------------------------------------------


def test_clipped_samples_counts_the_samples_at_full_scale():
    """Only the samples that reached the ceiling count as clipped."""
    module = _analyze_module()
    signal = np.array([0.0, 0.5, -0.999, 1.0, -1.0, 0.998], dtype=np.float32)
    assert module._clipped_samples(signal) == 3


def test_clipped_samples_of_a_quiet_signal_is_zero():
    """A normal recording does not clip."""
    module = _analyze_module()
    signal = (np.linspace(-0.8, 0.8, 512)).astype(np.float32)
    assert module._clipped_samples(signal) == 0


def test_clipped_samples_of_an_empty_signal_is_zero():
    """An empty buffer has nothing to clip."""
    module = _analyze_module()
    assert module._clipped_samples(np.array([], dtype=np.float32)) == 0


def test_dc_offset_is_the_mean_of_the_signal_as_a_percentage():
    """A signal sitting above zero has a dc offset."""
    module = _analyze_module()
    signal = np.full(1000, 0.02, dtype=np.float32)
    assert module._dc_offset(signal) == pytest.approx(2.0, abs=0.001)


def test_dc_offset_of_a_centred_signal_is_zero():
    """A signal that swings both ways has no dc offset."""
    module = _analyze_module()
    signal = np.array([0.5, -0.5] * 500, dtype=np.float32)
    assert module._dc_offset(signal) == pytest.approx(0.0, abs=1e-6)


def test_head_silence_measures_the_quiet_start_of_the_file():
    """Half a second of silence comes back as half a second."""
    module = _analyze_module()
    rate = 8000
    silence = np.zeros(rate // 2, dtype=np.float32)
    sound = _tone(440.0, 1.0, rate, amplitude=0.5)
    assert module._head_silence(np.concatenate([silence, sound]), rate) == pytest.approx(0.5, abs=0.01)


def test_head_silence_of_a_file_that_starts_sounding_is_zero():
    """A file with no leading silence has nothing to report."""
    module = _analyze_module()
    rate = 8000
    sound = _tone(440.0, 1.0, rate, amplitude=0.5)
    assert module._head_silence(sound, rate) == pytest.approx(0.0, abs=0.01)


def test_head_silence_of_a_fully_silent_file_is_its_length():
    """A file with no sound at all is silent for its whole length."""
    module = _analyze_module()
    rate = 8000
    silence = np.zeros(rate, dtype=np.float32)
    assert module._head_silence(silence, rate) == pytest.approx(1.0, abs=0.01)


def test_crest_is_the_gap_between_the_peak_and_the_average_level():
    """The crest factor is the peak level minus the rms level."""
    module = _analyze_module()
    assert module._crest_db(-0.5, -14.0) == pytest.approx(13.5, abs=0.001)


def test_crest_is_none_without_a_level():
    """Without both levels there is no crest."""
    module = _analyze_module()
    assert module._crest_db(None, -14.0) is None
    assert module._crest_db(-0.5, None) is None


# ---------------------------------------------------------------------------
# the whole analysis
# ---------------------------------------------------------------------------


def test_analyze_fills_every_new_measurement():
    """The result carries the new fields and the meta block repeats them."""
    module = _analyze_module()
    rate = 22050
    signal = _tone(440.0, 1.0, rate, amplitude=0.5)
    result = module.analyze(signal, rate, backend="numpy")

    assert result.clipped_samples == 0
    assert result.dc_offset_percent is not None
    assert result.head_silence_seconds is not None
    assert result.crest_db is not None
    assert result.crest_db == pytest.approx(result.peak_dbfs - result.rms_dbfs, abs=0.001)
    assert result.meta["clipped_samples"] == 0
    assert result.meta["dc_offset_percent"] == pytest.approx(result.dc_offset_percent)
    assert result.meta["head_silence_seconds"] == pytest.approx(result.head_silence_seconds)
    assert result.meta["crest_db"] == pytest.approx(result.crest_db)


def test_analyze_reports_a_clipped_signal():
    """A signal at full scale is counted as clipped."""
    module = _analyze_module()
    rate = 8000
    signal = _tone(440.0, 0.5, rate, amplitude=1.0)
    result = module.analyze(signal, rate, backend="numpy")
    assert result.clipped_samples > 0


def test_analyze_of_an_empty_signal_is_safe():
    """An empty buffer gives a result without raising."""
    module = _analyze_module()
    result = module.analyze(np.array([], dtype=np.float32), 22050, backend="numpy")
    assert result.duration == 0.0
    assert result.clipped_samples == 0


def test_analyze_finds_the_key_of_a_d_minor_triad():
    """A D minor triad is reported as D minor."""
    module = _analyze_module()
    rate = 22050
    signal = _triad(146.83, minor=True, seconds=4.0, rate=rate)
    result = module.analyze(signal, rate, backend="numpy")
    assert result.key == "D"
    assert result.mode == "minor"


def test_analyze_finds_the_key_of_a_c_major_triad():
    """A C major triad is reported as C major."""
    module = _analyze_module()
    rate = 22050
    signal = _triad(130.81, minor=False, seconds=4.0, rate=rate)
    result = module.analyze(signal, rate, backend="numpy")
    assert result.key == "C"
    assert result.mode == "major"


@pytest.mark.parametrize("rate", [22050, 44100, 48000])
def test_the_key_does_not_depend_on_the_sample_rate(rate):
    """The same chord is the same key at every sample rate.

    The peak of a tone is worked out from the bins around it, so the answer
    does not move when the tone lands between two bins in a different way.
    Before that, a C major triad read as E minor at 48 kHz.
    """
    module = _analyze_module()
    signal = _triad(130.81, minor=False, seconds=4.0, rate=rate)
    result = module.analyze(signal, rate, backend="numpy")
    assert result.key == "C"
    assert result.mode == "major"


@pytest.mark.parametrize("rate", [22050, 44100, 48000])
def test_the_key_of_a_minor_triad_does_not_depend_on_the_sample_rate(rate):
    """The same minor chord is the same key at every sample rate."""
    module = _analyze_module()
    signal = _triad(146.83, minor=True, seconds=4.0, rate=rate)
    result = module.analyze(signal, rate, backend="numpy")
    assert result.key == "D"
    assert result.mode == "minor"


def test_the_chroma_of_a_tone_sits_on_the_pitch_class_of_that_tone():
    """One tone gives one pitch class, whatever the rate is."""
    module = _analyze_module()
    for rate in (22050, 44100, 48000):
        signal = _tone(440.0, 4.0, rate, amplitude=0.5)
        chroma = module._chroma(signal.astype(np.float32), rate)
        assert chroma is not None
        # A is the ninth semitone from C, so it sits at index nine.
        assert int(np.argmax(chroma)) == 9


def test_analyze_finds_the_tempo_of_a_click_track():
    """A click track at 120 beats a minute is read back near 120."""
    module = _analyze_module()
    rate = 22050
    signal = _clicks(120.0, 12.0, rate)
    result = module.analyze(signal, rate, backend="numpy")
    assert result.bpm is not None
    assert 110.0 <= result.bpm <= 130.0


def test_analyze_reports_a_tempo_and_a_key_for_a_real_shape_of_signal():
    """A signal with a beat and a chord gives both numbers."""
    module = _analyze_module()
    rate = 22050
    signal = _clicks(110.0, 12.0, rate) + _triad(146.83, minor=True, seconds=12.0, rate=rate)
    result = module.analyze(signal, rate, backend="numpy")
    assert result.bpm is not None and result.bpm > 0
    assert result.key == "D"


# ---------------------------------------------------------------------------
# the http route
# ---------------------------------------------------------------------------


def _upload(name: str, payload: bytes):
    """A FastAPI upload that carries bytes in memory."""
    from fastapi import UploadFile

    return UploadFile(filename=name, file=io.BytesIO(payload))


def _wav_bytes(samples: np.ndarray, rate: int) -> bytes:
    """Encode a mono signal as a wav file in memory."""
    soundfile = pytest.importorskip("soundfile")
    buffer = io.BytesIO()
    soundfile.write(buffer, samples, rate, format="WAV")
    return buffer.getvalue()


def test_the_analyze_route_returns_the_measurements_and_the_source_facts():
    """The route gives the panel every number it needs."""
    server = _server_module()
    rate = 22050
    signal = _tone(440.0, 1.0, rate, amplitude=0.5)
    upload = _upload("tone.wav", _wav_bytes(signal, rate))

    payload = asyncio.run(server.analyze_upload(upload))

    for key in (
        "peak_dbfs",
        "rms_dbfs",
        "loudness_lufs",
        "clipped_samples",
        "dc_offset_percent",
        "head_silence_seconds",
        "crest_db",
        "key",
        "mode",
        "bpm",
    ):
        assert key in payload

    source = payload["source"]
    assert source["name"] == "tone.wav"
    assert source["size_bytes"] > 0
    assert source["duration"] == pytest.approx(1.0, abs=0.05)
    assert source["sample_rate"] == rate
    assert source["channels"] == 1


def test_the_analyze_route_measures_a_stereo_file():
    """A stereo file keeps both channels and the loudness is per channel."""
    server = _server_module()
    rate = 22050
    left = _tone(440.0, 1.0, rate, amplitude=0.5)
    right = _tone(660.0, 1.0, rate, amplitude=0.3)
    stereo = np.stack([left, right], axis=1)
    upload = _upload("stereo.wav", _wav_bytes(stereo, rate))

    payload = asyncio.run(server.analyze_upload(upload))

    assert payload["source"]["channels"] == 2
    assert payload["peak_dbfs"] is not None
    assert payload["clipped_samples"] == 0


def test_the_analyze_route_cleans_up_its_temporary_folder():
    """The uploaded copy is removed once the measurement is done."""
    server = _server_module()
    rate = 22050
    signal = _tone(440.0, 0.5, rate, amplitude=0.5)
    upload = _upload("tone.wav", _wav_bytes(signal, rate))

    before = set(server._JOB_ROOT.glob("*")) if server._JOB_ROOT.exists() else set()
    asyncio.run(server.analyze_upload(upload))
    after = set(server._JOB_ROOT.glob("*")) if server._JOB_ROOT.exists() else set()

    assert after == before


def test_the_analyze_route_survives_a_file_it_cannot_decode():
    """A broken file still returns a payload instead of raising."""
    server = _server_module()
    upload = _upload("broken.wav", b"not a wav file at all")

    payload = asyncio.run(server.analyze_upload(upload))

    assert payload["source"]["name"] == "broken.wav"
    assert payload["source"]["size_bytes"] > 0


def test_the_task_measures_a_stereo_file_on_its_own_channels():
    """The task keeps the channels, so the peak is the loudest channel of the file."""
    tasks = pytest.importorskip("mash_mix.tasks.analyze")
    types = pytest.importorskip("mash_mix.types")
    rate = 22050
    left = _tone(440.0, 1.0, rate, amplitude=0.5)
    right = _tone(660.0, 1.0, rate, amplitude=0.2)
    stereo = np.stack([left, right], axis=0)
    buffer = types.AudioBuffer(samples=stereo, sample_rate=rate)

    result = tasks.analyze(buffer)

    assert result.channels == 2
    expected = 20.0 * math.log10(0.5)
    assert result.peak_dbfs == pytest.approx(expected, abs=0.01)


def test_the_web_page_carries_the_measurement_panel():
    server = _server_module()
    page = (server.STATIC_DIR / "index.html").read_text(encoding="utf-8")

    assert "fcMeasure" in page
    assert "fcPitch" in page
    assert "Own pitch and speed" in page
    assert "/api/analyze" in page
    for label in ("Peak", "Loudness", "Clipped", "DC offset", "Head silence", "Crest"):
        assert label in page


def test_the_web_page_is_responsive():
    """The page adapts to a phone, a tablet and a desktop."""
    server = _server_module()
    page = (server.STATIC_DIR / "index.html").read_text(encoding="utf-8")

    assert "viewport" in page
    assert "@media (max-width: 900px)" in page
    assert "@media (max-width: 560px)" in page
    assert "@media (pointer: coarse)" in page
    assert "prefers-reduced-motion" in page
    assert "clamp(" in page
