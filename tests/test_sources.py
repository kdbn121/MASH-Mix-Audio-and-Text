# SPDX-License-Identifier: MIT | Copyright (c) 2026 MASH (Khaled Bhuiyan)
"""Tests for fetching a link and for working the language out on its own.

Nothing here touches the network and nothing here loads a model. The fetcher
is stood in for, so the tests say what the engine does with what a page says.
"""

from __future__ import annotations

import asyncio
import json
import sys
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from mash_mix.lang.catalogue import (  # noqa: E402
    LANGUAGES,
    ansi_codepage,
    detect_language_from_text,
    detect_script,
    iso_for,
    resolve_entry,
)
from mash_mix.sources import (  # noqa: E402
    SourceError,
    SourceInfo,
    fetch_metadata,
    is_url,
    read_subtitle_file,
)
from mash_mix.types import AudioBuffer, LyricLine, LyricsResult, Segment, Word  # noqa: E402

# The letters are written as code points on purpose. The Bengali and the
# Devanagari letters look alike, so a literal can be changed into its twin.
BENGALI_TITLE = "\u09ad\u09ac\u09c7\u09b0 \u098f\u0987"
HINDI_TITLE = "\u092d\u094b\u092d\u0930 \u090f\u0987"
ARABIC_TITLE = "\u0627\u0644\u0623\u063a\u0646\u064a\u0629"
RUSSIAN_TITLE = "\u041f\u0435\u0441\u043d\u044f"


# -- the language catalogue ------------------------------------------------


def test_a_language_is_found_by_name_by_alias_and_by_code():
    assert iso_for("Hindi") == "hi"
    assert iso_for("hindi") == "hi"
    assert iso_for("hi") == "hi"
    assert iso_for("BN-bd") == "bn"


def test_auto_means_the_language_is_not_known_yet():
    assert resolve_entry("auto") is None
    assert resolve_entry("") is None
    assert iso_for("auto") is None


def test_the_catalogue_holds_the_common_languages():
    codes = {entry.iso639_1 for entry in LANGUAGES}
    for code in ("bn", "hi", "ur", "ar", "en", "es", "fr", "de", "ru", "ja", "ko", "zh"):
        assert code in codes


def test_the_script_of_a_text_names_the_language():
    iso, share, script = detect_language_from_text(BENGALI_TITLE)
    assert iso == "bn"
    assert script == "Bengali"
    assert share > 0.0


def test_a_latin_title_gives_no_answer():
    """Latin letters are shared by too many languages to name one."""
    iso, share, script = detect_language_from_text("Bhober Ei Khela Ghore")
    assert iso is None
    assert share == 0.0
    assert script == "Latin"


def test_a_mixed_title_answers_with_the_strong_script():
    """A title in Latin letters with a Bengali line still names Bangla."""
    iso, _share, _script = detect_language_from_text("Bhober Ei " + BENGALI_TITLE)
    assert iso == "bn"


def test_each_script_names_its_language():
    assert detect_language_from_text(HINDI_TITLE)[0] == "hi"
    assert detect_language_from_text(ARABIC_TITLE)[0] == "ar"
    assert detect_language_from_text(RUSSIAN_TITLE)[0] == "ru"


def test_the_script_of_a_text_is_named():
    assert detect_script(RUSSIAN_TITLE) == "Cyrillic"
    assert detect_script(HINDI_TITLE) == "Devanagari"
    assert detect_script("") == "unknown"


def test_a_language_keeps_the_code_page_of_its_legacy_files():
    assert ansi_codepage("ru") == 1251
    assert ansi_codepage("th") == 874
    assert ansi_codepage("bn") is None


# -- the link is recognised as a link --------------------------------------


def test_a_link_is_told_from_a_path():
    assert is_url("https://www.youtube.com/watch?v=abc")
    assert is_url("http://example.com/a.mp3")
    assert not is_url("song.mp3")
    assert not is_url("")
    assert not is_url("   ")


def test_a_text_that_is_not_a_link_is_refused(monkeypatch):
    from mash_mix import sources

    with pytest.raises(SourceError):
        sources.fetch_metadata("not a link")


# -- what a page says about itself -----------------------------------------


def _page(**overrides) -> dict:
    """A page's metadata, as the fetcher would print it."""
    info = {
        "id": "AU0mxhZcL1w",
        "title": "Bhober Ei | " + BENGALI_TITLE,
        "uploader": "Anupam Movie Songs",
        "duration": 265.32,
        "extractor_key": "Youtube",
        "webpage_url": "https://www.youtube.com/watch?v=AU0mxhZcL1w",
        "language": None,
        "subtitles": {},
        "automatic_captions": {},
        "tags": ["bangla movie song"],
    }
    info.update(overrides)
    return info


def test_the_title_is_read_when_the_page_names_no_language(monkeypatch):
    from mash_mix import sources

    monkeypatch.setattr(sources, "_run", lambda args, timeout=600: json.dumps(_page()))
    source = fetch_metadata("https://www.youtube.com/watch?v=AU0mxhZcL1w")
    assert source.title.startswith("Bhober Ei")
    assert source.guessed_language == "bn"
    assert source.declared_language == ""
    assert source.language_source == "script"
    assert source.script == "Bengali"
    assert source.duration == pytest.approx(265.32)


def test_a_page_that_names_its_language_is_believed(monkeypatch):
    from mash_mix import sources

    monkeypatch.setattr(
        sources,
        "_run",
        lambda args, timeout=600: json.dumps(_page(language="Hindi")),
    )
    source = fetch_metadata("https://example.com/watch")
    assert source.declared_language == "hi"
    assert source.language_source == "declared"
    assert source.language == "hi"


def test_the_subtitle_tracks_the_page_offers_are_listed(monkeypatch):
    from mash_mix import sources

    tracks = {"bn": [{"ext": "vtt"}, {"ext": "srt"}]}
    monkeypatch.setattr(
        sources,
        "_run",
        lambda args, timeout=600: json.dumps(_page(subtitles=tracks)),
    )
    source = fetch_metadata("https://example.com/watch")
    assert len(source.subtitles) == 1
    track = source.subtitles[0]
    assert track["language"] == "bn"
    assert track["automatic"] is False
    assert track["formats"] == ["vtt", "srt"]


def test_the_facts_about_a_source_are_serialisable(monkeypatch):
    from mash_mix import sources

    monkeypatch.setattr(sources, "_run", lambda args, timeout=600: json.dumps(_page()))
    source = fetch_metadata("https://example.com/watch")
    data = source.to_dict()
    assert data["language"] == "bn"
    assert data["language_source"] == "script"
    assert json.dumps(data)


def test_a_fetcher_that_fails_says_so(monkeypatch):
    """A fetch that fails has to raise, not return nothing."""
    from mash_mix import sources

    class _Proc:
        returncode = 1
        stdout = ""
        stderr = "ERROR: Video unavailable"

    def fake_run(command, **kwargs):
        return _Proc()

    monkeypatch.setattr(sources.subprocess, "run", fake_run)
    with pytest.raises(SourceError) as caught:
        sources._run(["--dump-single-json", "https://example.com/watch"])
    assert "Video unavailable" in str(caught.value)


def test_a_fetcher_that_is_not_installed_says_how_to_install_it(monkeypatch):
    from mash_mix import sources

    def fake_run(command, **kwargs):
        raise FileNotFoundError("yt-dlp")

    monkeypatch.setattr(sources.subprocess, "run", fake_run)
    with pytest.raises(SourceError) as caught:
        sources._run(["--version"])
    assert "yt-dlp" in str(caught.value)


def test_a_subtitle_file_is_read_in_whichever_encoding_it_was_written(tmp_path):
    """Older pages write in the code page of their language."""
    target = tmp_path / "subs.srt"
    target.write_bytes("1\n00:00:01,000 --> 00:00:02,000\ncaf\xe9\n".encode("cp1252"))
    assert "caf\xe9" in read_subtitle_file(str(target))

    utf8_target = tmp_path / "utf8.srt"
    utf8_target.write_text("1\n00:00:01,000 --> 00:00:02,000\ncaf\xe9\n", encoding="utf-8")
    assert "caf\xe9" in read_subtitle_file(str(utf8_target))


# -- the language of the audio is worked out on its own --------------------


class _Engine:
    """A recogniser that says what language it hears."""

    def __init__(self, name: str = "faster-whisper", text: str = "ranga mathar chiruni"):
        self.name = name
        self.text = text
        self.calls: list = []
        self._model = ""

    def detect_language(self, audio, sample_rate, *, max_seconds: float = 30.0):
        return "bn", 0.97

    def transcribe(self, audio, sample_rate, **kwargs):
        self.calls.append(kwargs)
        return _Result(
            [
                Segment(
                    start=0.0,
                    end=2.0,
                    text=self.text,
                    words=[Word(start=0.0, end=2.0, text=self.text, probability=0.9)],
                )
            ]
        )


class _Result:
    def __init__(self, segments, language: str = "bn", probability: float = 0.97):
        self.segments = segments
        self.language = language
        self.language_probability = probability
        self.meta = {
            "detected_language": language,
            "language_probability": probability,
        }


def test_the_model_answers_when_no_language_was_named(monkeypatch):
    """Auto mode asks the loaded model what it hears."""
    import mash_mix.tasks.lyrics as lyrics_module

    engine = _Engine()
    monkeypatch.setattr(lyrics_module, "pick_backend", lambda kind, preferred="auto": "faster-whisper")
    monkeypatch.setattr("mash_mix.backends.registry.get_stt", lambda backend, model=None: engine)

    result = lyrics_module.extract_lyrics(
        AudioBuffer(np.zeros(16000 * 5, dtype=np.float32), 16000, 1),
        language="auto",
        quality="auto",
    )
    assert result.meta["language_requested"] == "auto"
    assert result.meta["language_effective"] == "bn"
    assert result.meta["language_source"] == "model"
    assert result.meta["language_probability"] == pytest.approx(0.97)
    assert result.lines


def test_a_hint_is_used_before_the_model_is_asked(monkeypatch):
    """A page that named its language is better than a guess."""
    import mash_mix.tasks.lyrics as lyrics_module

    engine = _Engine()
    monkeypatch.setattr(lyrics_module, "pick_backend", lambda kind, preferred="auto": "faster-whisper")
    monkeypatch.setattr("mash_mix.backends.registry.get_stt", lambda backend, model=None: engine)

    result = lyrics_module.extract_lyrics(
        AudioBuffer(np.zeros(16000 * 5, dtype=np.float32), 16000, 1),
        language="auto",
        language_hint="bn",
        quality="auto",
    )
    assert result.meta["language_source"] == "hint"
    assert result.meta["language_effective"] == "bn"


def test_a_named_language_is_kept_as_it_was_named(monkeypatch):
    """A caller that names a language gets that language, not a guess."""
    import mash_mix.tasks.lyrics as lyrics_module

    engine = _Engine()
    monkeypatch.setattr(lyrics_module, "pick_backend", lambda kind, preferred="auto": "faster-whisper")
    monkeypatch.setattr("mash_mix.backends.registry.get_stt", lambda backend, model=None: engine)

    result = lyrics_module.extract_lyrics(
        AudioBuffer(np.zeros(16000 * 5, dtype=np.float32), 16000, 1),
        language="bn-BD",
        quality="auto",
    )
    assert result.meta["language_requested"] == "bn-BD"
    assert result.meta["language_source"] == "requested"


def test_a_language_the_caller_named_carries_no_certainty(monkeypatch):
    """A model that was told the language reports a certainty of one.

    That number is not a measurement of anything, so it is not reported.
    """
    import mash_mix.tasks.lyrics as lyrics_module

    engine = _Engine()
    monkeypatch.setattr(lyrics_module, "pick_backend", lambda kind, preferred="auto": "faster-whisper")
    monkeypatch.setattr("mash_mix.backends.registry.get_stt", lambda backend, model=None: engine)

    result = lyrics_module.extract_lyrics(
        AudioBuffer(np.zeros(16000 * 5, dtype=np.float32), 16000, 1),
        language="bn-BD",
        quality="auto",
    )
    assert result.meta["language_probability"] is None


def test_a_hint_carries_no_certainty_either(monkeypatch):
    """The page named the language, so the model was told it."""
    import mash_mix.tasks.lyrics as lyrics_module

    engine = _Engine()
    monkeypatch.setattr(lyrics_module, "pick_backend", lambda kind, preferred="auto": "faster-whisper")
    monkeypatch.setattr("mash_mix.backends.registry.get_stt", lambda backend, model=None: engine)

    result = lyrics_module.extract_lyrics(
        AudioBuffer(np.zeros(16000 * 5, dtype=np.float32), 16000, 1),
        language="auto",
        language_hint="bn",
        quality="auto",
    )
    assert result.meta["language_source"] == "hint"
    assert result.meta["language_probability"] is None


def test_a_language_that_is_not_bangla_keeps_its_own_script(monkeypatch):
    """There is nothing to convert a Hindi or an Arabic line to."""
    import mash_mix.tasks.lyrics as lyrics_module

    engine = _Engine(text="\u0939\u093f\u0928\u094d\u0926\u0940")
    monkeypatch.setattr(lyrics_module, "pick_backend", lambda kind, preferred="auto": "faster-whisper")
    monkeypatch.setattr("mash_mix.backends.registry.get_stt", lambda backend, model=None: engine)

    result = lyrics_module.extract_lyrics(
        AudioBuffer(np.zeros(16000 * 5, dtype=np.float32), 16000, 1),
        language="hi",
        quality="auto",
    )
    assert result.language == "hi"
    assert result.meta["output_script"] == "as recognised"


# -- the whole path from a link to the files --------------------------------


def _write_wav(path: Path, seconds: float = 1.0, sample_rate: int = 16000) -> Path:
    """A short silent wav, so the audio decoder has something real to read."""
    import wave

    with wave.open(str(path), "wb") as handle:
        handle.setnchannels(1)
        handle.setsampwidth(2)
        handle.setframerate(sample_rate)
        handle.writeframes(b"\x00\x00" * int(sample_rate * seconds))
    return path


def test_a_link_becomes_the_lyric_files(monkeypatch, tmp_path):
    """A link is fetched, read and written out exactly like a dropped file."""
    import mash_mix.tasks.lyrics as lyrics_module

    engine = _Engine()
    monkeypatch.setattr(lyrics_module, "pick_backend", lambda kind, preferred="auto": "faster-whisper")
    monkeypatch.setattr("mash_mix.backends.registry.get_stt", lambda backend, model=None: engine)

    audio = _write_wav(tmp_path / "source.wav")

    def fake_fetch(url, target_dir, **kwargs):
        return SourceInfo(
            url=url,
            title="Bhober Ei",
            uploader="Anupam Movie Songs",
            duration=265.0,
            extractor="Youtube",
            guessed_language="bn",
            script="Bengali",
            language_source="script",
            audio_path=str(audio),
        )

    monkeypatch.setattr("mash_mix.sources.fetch_metadata", lambda url, **kwargs: fake_fetch(url, ""))
    monkeypatch.setattr("mash_mix.sources.fetch_audio", fake_fetch)

    outcome = lyrics_module.extract_lyrics_from_url(
        "https://www.youtube.com/watch?v=AU0mxhZcL1w",
        str(tmp_path / "work"),
    )
    result = outcome["result"]
    assert result.title == "Bhober Ei"
    assert result.artist == "Anupam Movie Songs"
    assert result.meta["source"]["language"] == "bn"
    assert result.meta["source_language_source"] == "script"
    assert result.lines


# -- the server route -------------------------------------------------------


def _fake_result() -> LyricsResult:
    return LyricsResult(
        lines=[LyricLine(start=0.0, end=2.0, text="ranga mathar chiruni")],
        language="bn-BD",
        duration=2.0,
        title="Bhober Ei",
        artist="Anupam Movie Songs",
        backend="faster-whisper",
        model="large-v3",
        meta={"output_script": "bangla"},
    )


def test_the_route_refuses_a_text_that_is_not_a_link():
    from fastapi import HTTPException

    from mash_mix.server import app as server_app

    with pytest.raises(HTTPException) as caught:
        asyncio.run(server_app.lyrics_from_url(url="not a link"))
    assert caught.value.status_code == 400


def test_the_route_writes_the_same_four_files_as_an_upload(monkeypatch, tmp_path):
    """A link and a dropped file give the same downloads."""
    from mash_mix.server import app as server_app

    audio = tmp_path / "source.wav"
    audio.write_bytes(b"RIFF")
    source = SourceInfo(
        url="https://www.youtube.com/watch?v=AU0mxhZcL1w",
        title="Bhober Ei",
        uploader="Anupam Movie Songs",
        duration=265.0,
        guessed_language="bn",
        language_source="script",
        audio_path=str(audio),
    )

    def fake_extract(url, work_dir, **kwargs):
        return {"result": _fake_result(), "source": source}

    monkeypatch.setattr("mash_mix.tasks.lyrics.extract_lyrics_from_url", fake_extract)
    monkeypatch.setattr(server_app, "_JOB_ROOT", tmp_path / "jobs")

    payload = asyncio.run(
        server_app.lyrics_from_url(url="https://www.youtube.com/watch?v=AU0mxhZcL1w")
    )
    assert payload["language"] == "bn-BD"
    assert payload["title"] == "Bhober Ei"
    assert payload["source"]["url"].startswith("https://")
    assert payload["source"]["language"] == "bn"
    for fmt in ("srt", "txt", "vtt", "lrc"):
        assert fmt in payload["downloads"]

    job_id = payload["job_id"]
    for fmt in ("srt", "txt", "vtt", "lrc"):
        path = tmp_path / "jobs" / job_id / f"Bhober Ei.{fmt}"
        assert path.exists()
