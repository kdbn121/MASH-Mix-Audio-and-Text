# SPDX-License-Identifier: MIT | Copyright (c) 2026 MASH (Khaled Bhuiyan)
"""The audio input and output layer.

The audio package is written with optional dependencies, so every test here
imports it inside the test body and skips when it is not importable. Nothing
in this file downloads a model or touches the network.
"""

from __future__ import annotations

import shutil

import numpy as np
import pytest


def _audio_io():
    """The audio io module, or a skip when it cannot be imported."""
    return pytest.importorskip("mash_mix.audio.io")


def _has_decoder() -> bool:
    """True when this machine can decode an audio file."""
    try:
        import soundfile  # noqa: F401

        return True
    except Exception:
        return shutil.which("ffmpeg") is not None


def _has_ffmpeg() -> bool:
    return shutil.which("ffmpeg") is not None


def test_the_audio_package_is_importable():
    """The module imports cleanly even when nothing optional is installed."""
    module = _audio_io()
    assert hasattr(module, "load")
    assert hasattr(module, "save")
    assert hasattr(module, "probe")


def test_save_and_load_a_wav_file(tmp_path, bangla_waveform):
    """Write a synthetic waveform to disk and read it back."""
    io = _audio_io()
    if not _has_decoder():
        pytest.skip("no audio decoder on this machine, install soundfile or ffmpeg")

    target = tmp_path / "tone.wav"
    written = io.save(str(target), bangla_waveform.samples, bangla_waveform.sample_rate)
    assert written == str(target)
    assert target.exists()
    assert target.stat().st_size > 0

    buffer = io.load(str(target))
    assert buffer.sample_rate == bangla_waveform.sample_rate
    assert buffer.channels == 1
    assert abs(buffer.duration - 1.5) < 0.05

    original = bangla_waveform.samples
    loaded = buffer.mono()
    assert loaded.shape == original.shape
    # The file is 16 bit, so the samples come back very close but not exact.
    assert np.allclose(loaded, original, atol=1e-3)


def test_load_can_resample(tmp_path, bangla_waveform):
    io = _audio_io()
    if not _has_decoder():
        pytest.skip("no audio decoder on this machine, install soundfile or ffmpeg")

    target = tmp_path / "tone.wav"
    io.save(str(target), bangla_waveform.samples, bangla_waveform.sample_rate)

    buffer = io.load(str(target), sr=16000)
    assert buffer.sample_rate == 16000
    assert abs(buffer.duration - 1.5) < 0.1


def test_probe_reports_the_container_facts(tmp_path, bangla_waveform):
    io = _audio_io()
    if not _has_decoder():
        pytest.skip("no audio decoder on this machine, install soundfile or ffmpeg")

    target = tmp_path / "tone.wav"
    io.save(str(target), bangla_waveform.samples, bangla_waveform.sample_rate)

    info = io.probe(str(target))
    for key in ("duration", "sample_rate", "channels", "bit_rate", "format_name", "size_bytes"):
        assert key in info
    assert info["size_bytes"] > 0


def test_load_raises_for_a_missing_file(tmp_path):
    io = _audio_io()
    from mash_mix.errors import AudioDecodeError

    with pytest.raises(AudioDecodeError):
        io.load(str(tmp_path / "does-not-exist.wav"))


def test_resample_changes_the_length():
    resample = pytest.importorskip("mash_mix.audio.resample")
    samples = np.linspace(-1.0, 1.0, 22050, dtype=np.float32)
    out = resample.resample(samples, 22050, 16000)
    assert len(out) == pytest.approx(16000, abs=2)
    assert out.dtype == np.float32


def test_chunk_audio_returns_plain_dictionaries():
    chunk = pytest.importorskip("mash_mix.audio.chunk")
    samples = np.zeros(22050 * 2, dtype=np.float32)
    chunks = chunk.chunk_audio(samples, 22050, chunk_seconds=1.0, overlap_seconds=0.0)
    assert len(chunks) >= 2
    for item in chunks:
        for key in ("index", "start", "end", "samples", "start_sample", "end_sample"):
            assert key in item
    assert chunks[0]["start"] == 0.0
    assert chunks[0]["index"] == 0


@pytest.mark.slow
def test_ffmpeg_encodes_and_decodes_an_mp3(tmp_path, bangla_waveform):
    """The ffmpeg path handles a compressed container."""
    io = _audio_io()
    if not _has_ffmpeg():
        pytest.skip("ffmpeg is not on PATH")

    target = tmp_path / "tone.mp3"
    io.save(str(target), bangla_waveform.samples, bangla_waveform.sample_rate, format="mp3")
    assert target.exists()
    assert target.stat().st_size > 0

    buffer = io.load(str(target))
    assert buffer.sample_rate > 0
    assert abs(buffer.duration - 1.5) < 0.25
