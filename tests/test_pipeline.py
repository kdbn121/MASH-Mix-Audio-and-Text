# SPDX-License-Identifier: MIT | Copyright (c) 2026 MASH (Khaled Bhuiyan)
"""The pipeline facade, the backend registry and the task guards."""

from __future__ import annotations

import importlib
import importlib.util

import pytest

from mash_mix.export.subtitles import to_srt
from mash_mix.pipeline import MASHMix, PipelineConfig
from mash_mix.types import LyricLine, LyricsResult

LINES = [
    LyricLine(start=0.0, end=2.5, text="প্রথম লাইন"),
    LyricLine(start=2.5, end=5.125, text="দ্বিতীয় লাইন"),
]


def test_write_writes_the_files_and_the_srt_content_is_correct(tmp_path):
    result = LyricsResult(
        lines=LINES,
        language="bn-BD",
        duration=5.125,
        title="Kobita tuku",
        artist="MASH",
    )

    engine = MASHMix(language="bn-BD")
    written = engine.write(result, str(tmp_path), formats=["srt", "txt"])

    assert set(written) == {"srt", "txt"}
    srt_path = tmp_path / "Kobita tuku.srt"
    txt_path = tmp_path / "Kobita tuku.txt"
    assert srt_path.exists()
    assert txt_path.exists()
    assert written["srt"] == str(srt_path)

    content = srt_path.read_text(encoding="utf-8")
    assert content == to_srt(LINES)
    assert "00:00:00,000 --> 00:00:02,500" in content
    assert "প্রথম লাইন" in content

    assert txt_path.read_text(encoding="utf-8") == "প্রথম লাইন\nদ্বিতীয় লাইন\n"


def test_write_supports_the_other_formats(tmp_path):
    result = LyricsResult(lines=LINES, language="bn-BD", title="out")
    engine = MASHMix()
    written = engine.write(result, str(tmp_path), formats=["vtt", "lrc", "json"])

    assert set(written) == {"vtt", "lrc", "json"}
    assert (tmp_path / "out.vtt").read_text(encoding="utf-8").startswith("WEBVTT")
    assert (tmp_path / "out.lrc").read_text(encoding="utf-8").startswith("[00:00.00]")
    assert "bn-BD" in (tmp_path / "out.json").read_text(encoding="utf-8")


def test_write_uses_the_stem_when_one_is_given(tmp_path):
    result = LyricsResult(lines=LINES, language="bn-BD", title="ignored")
    engine = MASHMix()
    written = engine.write(result, str(tmp_path), stem="custom", formats=["srt"])
    assert written["srt"] == str(tmp_path / "custom.srt")


def test_the_pipeline_configuration():
    engine = MASHMix(language="bn-IN", stt_backend="whisper", tts_backend="piper")
    assert engine.language == "bn-IN"
    assert engine.config.stt_backend == "whisper"
    assert engine.config.tts_backend == "piper"

    config = PipelineConfig(language="bn-BD")
    assert MASHMix(config).language == "bn-BD"


def test_every_backend_module_imports():
    """Each backend module imports cleanly, even without its runtime."""
    from mash_mix.backends.registry import _DISCOVERY_MODULES

    for module_name in _DISCOVERY_MODULES:
        if importlib.util.find_spec(module_name) is None:
            continue
        module = importlib.import_module(module_name)
        assert module is not None


def test_backend_infos_have_a_name_a_kind_and_an_availability():
    from mash_mix.backends.registry import backend_infos

    infos = backend_infos()
    assert isinstance(infos, list)
    assert infos
    for info in infos:
        assert info["name"]
        assert info["kind"] in ("stt", "tts", "sts")
        assert isinstance(info["available"], bool)
        assert "summary" in info
        assert "package" in info


def test_the_registry_lists_the_known_backends():
    from mash_mix.backends.registry import stt_names, tts_names

    stt = stt_names()
    assert "faster-whisper" in stt
    assert "whisper" in stt
    assert isinstance(tts_names(), list)


def test_synthesize_refuses_an_empty_text_without_a_tts_backend():
    """The guard runs before any backend is looked up."""
    from mash_mix.tasks.synthesize import synthesize

    with pytest.raises(ValueError):
        synthesize("")
    with pytest.raises(ValueError):
        synthesize("   ")


def test_transcribe_is_importable_from_the_package():
    import mash_mix

    assert callable(mash_mix.available_backends)
    assert mash_mix.available_backends()
