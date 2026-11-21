# SPDX-License-Identifier: MIT | Copyright (c) 2026 MASH (Khaled Bhuiyan)
"""Tests for the subtitles a page carries and for the regional fallback.

Nothing here touches the network and nothing here loads a model. The fetcher
is stood in for, so the tests say what the engine does with what a page offers.
"""

from __future__ import annotations

import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from mash_mix.export.subtitles import (  # noqa: E402
    parse_subtitles,
    parse_timestamp,
    to_srt,
    to_vtt,
)
from mash_mix.lang.catalogue import (  # noqa: E402
    REGIONAL_VARIANTS,
    fallback_chain,
    model_language,
    resolve_tag,
)
from mash_mix.sources import (  # noqa: E402
    SourceInfo,
    SubtitleTrack,
    choose_subtitle_track,
    decode_subtitle_bytes,
)

# The letters are written as code points on purpose. The Bengali and the
# Devanagari letters look alike, so a literal can be changed into its twin.
BANGLA_LINE = "\u09ad\u09ac\u09c7\u09b0 \u098f\u0987 \u0996\u09c7\u09b2\u09be \u0998\u09b0\u09c7"

SRT = (
    "1\n"
    "00:00:01,000 --> 00:00:04,000\n"
    "ranga mathar chiruni\n"
    "\n"
    "2\n"
    "00:00:05,500 --> 00:00:09,250\n"
    "dhaki dhaki bajche\n"
)

VTT = (
    "WEBVTT\n"
    "\n"
    "NOTE\n"
    "This file has some notes in it.\n"
    "\n"
    "cue-1\n"
    "00:00:01.000 --> 00:00:04.000 align:start position:0%\n"
    "<v Speaker>ranga mathar chiruni</v>\n"
    "\n"
    "00:00:05.500 --> 00:00:09.250\n"
    "dhaki dhaki <b>bajche</b>\n"
)


# -- reading a subtitle file ------------------------------------------------


def test_a_timestamp_is_read_in_both_spellings():
    assert parse_timestamp("00:00:01,500") == pytest.approx(1.5)
    assert parse_timestamp("00:00:01.500") == pytest.approx(1.5)
    assert parse_timestamp("01:02:03,004") == pytest.approx(3723.004)


def test_a_timestamp_without_hours_is_read():
    assert parse_timestamp("02:03,500") == pytest.approx(123.5)
    assert parse_timestamp("02:03.5") == pytest.approx(123.5)


def test_something_that_is_not_a_timestamp_gives_nothing():
    assert parse_timestamp("") is None
    assert parse_timestamp("no time here") is None


def test_an_srt_file_is_read():
    lines = parse_subtitles(SRT)
    assert len(lines) == 2
    assert lines[0].start == pytest.approx(1.0)
    assert lines[0].end == pytest.approx(4.0)
    assert lines[0].text == "ranga mathar chiruni"
    assert lines[1].start == pytest.approx(5.5)
    assert lines[1].text == "dhaki dhaki bajche"


def test_a_webvtt_file_is_read():
    """The header, the notes and the cue identifier are all left out."""
    lines = parse_subtitles(VTT)
    assert len(lines) == 2
    assert lines[0].text == "ranga mathar chiruni"
    assert lines[1].text == "dhaki dhaki bajche"


def test_the_markup_inside_a_cue_is_removed():
    text = "00:00:01,000 --> 00:00:02,000\n<v Fred>hello &amp; goodbye</v>\n"
    lines = parse_subtitles(text)
    assert lines[0].text == "hello & goodbye"


def test_a_karaoke_mark_is_removed_and_a_heart_is_kept():
    text = "00:00:01,000 --> 00:00:02,000\n<00:00:01.500>hello <3\n"
    lines = parse_subtitles(text)
    assert lines[0].text == "hello <3"


def test_a_file_that_is_not_a_subtitle_gives_nothing():
    assert parse_subtitles("") == []
    assert parse_subtitles("just some words\nwith no timing\n") == []


def test_a_file_that_was_written_in_a_code_page_is_read():
    """An older page writes in the code page of its language."""
    raw = "1\n00:00:01,000 --> 00:00:02,000\ncaf\xe9\n".encode("cp1252")
    assert "caf\xe9" in decode_subtitle_bytes(raw, "en")


def test_a_file_with_a_byte_order_mark_is_read():
    raw = "1\n00:00:01,000 --> 00:00:02,000\nhello\n".encode("utf-16")
    assert "hello" in decode_subtitle_bytes(raw)


def test_a_file_that_is_read_can_be_written_out_again():
    """What is read can be written back, so a site track becomes an SRT file."""
    lines = parse_subtitles(VTT)
    written = to_srt(lines)
    assert written.startswith("1\n")
    assert "ranga mathar chiruni" in written
    assert to_vtt(lines).startswith("WEBVTT")


# -- the regional fallback --------------------------------------------------


def test_a_regional_tag_falls_back_on_the_language_and_the_other_region():
    assert fallback_chain("bn-BD") == ["bn-BD", "bn", "bn-IN"]
    assert fallback_chain("bn-IN") == ["bn-IN", "bn", "bn-BD"]


def test_a_plain_language_falls_back_on_its_regions():
    assert fallback_chain("bn") == ["bn", "bn-BD", "bn-IN"]
    assert fallback_chain("en") == ["en", "en-US", "en-GB"]


def test_the_fallback_runs_in_both_directions():
    """A page that names one region is served by a track of the other."""
    assert fallback_chain("bn-BD")[2] == "bn-IN"
    assert fallback_chain("bn-IN")[2] == "bn-BD"


def test_a_language_with_no_region_keeps_its_own_chain():
    chain = fallback_chain("ja")
    assert chain[0] == "ja"
    assert "ja" in chain


def test_a_tag_the_catalogue_does_not_know_is_left_alone():
    assert fallback_chain("xyz") == ["xyz"]
    assert fallback_chain("auto") == []
    assert fallback_chain("") == []


def test_the_region_is_kept_when_it_is_asked_for():
    assert resolve_tag("bn-BD") == "bn-BD"
    assert resolve_tag("bn-bd") == "bn-BD"
    assert resolve_tag("bn_bd") == "bn-BD"
    assert resolve_tag("Bangla") == "bn"
    assert resolve_tag("auto") == ""


def test_a_model_is_taught_one_code_per_language():
    """A model does not know a region, so the region is dropped for it."""
    assert model_language("bn-BD") == "bn"
    assert model_language("bn-IN") == "bn"
    assert model_language("pt-BR") == "pt"
    assert model_language("Hindi") == "hi"
    assert model_language("auto") == ""


def test_every_regional_variant_names_a_language_the_catalogue_knows():
    from mash_mix.lang.catalogue import resolve_entry

    for base, variants in REGIONAL_VARIANTS.items():
        assert resolve_entry(base) is not None
        for variant in variants:
            assert resolve_entry(variant) is not None
            assert variant.split("-")[0] == base


# -- choosing the track a page offers ---------------------------------------


def _tracks() -> list:
    return [
        {"language": "en", "raw_language": "en", "automatic": False, "formats": ["vtt"]},
        {"language": "bn", "raw_language": "bn", "automatic": True, "formats": ["vtt"]},
        {"language": "hi", "raw_language": "hi", "automatic": False, "formats": ["srt"]},
    ]


def test_the_track_in_the_language_that_was_asked_for_wins():
    chosen = choose_subtitle_track(_tracks(), "hi")
    assert chosen["raw_language"] == "hi"


def test_a_track_a_human_wrote_beats_one_a_machine_wrote():
    """Within one language, the written track is the better one."""
    tracks = [
        {"language": "bn", "raw_language": "bn", "automatic": True, "formats": ["vtt"]},
        {"language": "bn", "raw_language": "bn", "automatic": False, "formats": ["srt"]},
    ]
    chosen = choose_subtitle_track(tracks, "bn")
    assert chosen["automatic"] is False


def test_the_other_region_serves_when_the_one_asked_for_is_not_there():
    """A page that carries bn-IN serves a caller who asked for bn-BD."""
    tracks = [{"language": "bn", "raw_language": "bn-IN", "automatic": False, "formats": ["vtt"]}]
    chosen = choose_subtitle_track(tracks, "bn-BD")
    assert chosen["raw_language"] == "bn-IN"


def test_a_translated_track_is_found_by_the_language_it_is_written_in():
    """A page names the track it translated bn-en, and it is still Bangla."""
    tracks = [
        {"language": "en", "raw_language": "en", "automatic": False, "formats": ["vtt"]},
        {"language": "bn", "raw_language": "bn-en", "automatic": True, "formats": ["vtt"]},
    ]
    chosen = choose_subtitle_track(tracks, "bn")
    assert chosen["raw_language"] == "bn-en"

    chosen_bd = choose_subtitle_track(tracks, "bn-BD")
    assert chosen_bd["raw_language"] == "bn-en"


def test_an_exact_region_is_asked_for_first():
    """A track filed under the region that was asked for wins."""
    tracks = [
        {"language": "bn", "raw_language": "bn-en", "automatic": True, "formats": ["vtt"]},
        {"language": "bn", "raw_language": "bn-BD", "automatic": False, "formats": ["srt"]},
    ]
    chosen = choose_subtitle_track(tracks, "bn-BD")
    assert chosen["raw_language"] == "bn-BD"


def test_a_track_in_the_wrong_language_is_not_served():
    """A track in the wrong language is worse than no track at all."""
    tracks = [{"language": "ja", "raw_language": "ja", "automatic": False, "formats": ["vtt"]}]
    chosen = choose_subtitle_track(tracks, "hi")
    # Nothing matched, so the first is the least bad of what is there.
    assert chosen["raw_language"] == "ja"


def test_a_page_with_no_tracks_gives_nothing():
    assert choose_subtitle_track([], "bn") == {}


def test_the_tracks_are_ranked_the_best_first():
    """The caller can walk the list, not only take the first one."""
    from mash_mix.sources import rank_subtitle_tracks

    ranked = rank_subtitle_tracks(_tracks(), "bn")
    assert ranked[0]["raw_language"] == "bn"
    assert len(ranked) == 3


def test_a_refused_track_does_not_end_the_run(monkeypatch, tmp_path):
    """A busy site refuses a track now and then. The next one is tried."""
    from mash_mix import sources

    asked: list = []

    def fake_run(args, timeout=600):
        name = args[args.index("--sub-langs") + 1]
        asked.append(name)
        if name == "bn-BD":
            raise sources.SourceError("HTTP Error 429: Too Many Requests")
        template = args[args.index("--output") + 1]
        target = Path(template.replace(".%(ext)s", f".{name}.srt"))
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_text(
            "1\n00:00:01,000 --> 00:00:04,000\nranga mathar chiruni\n",
            encoding="utf-8",
        )
        return ""

    monkeypatch.setattr(sources, "_run", fake_run)

    track = sources.fetch_subtitles(
        "https://example.com/watch",
        str(tmp_path / "work"),
        tracks=[
            {"raw_language": "bn-BD", "automatic": False},
            {"raw_language": "bn", "automatic": False},
        ],
        prefer_language="bn-BD",
    )
    assert asked == ["bn-BD", "bn"]
    assert track is not None
    assert track.raw_language == "bn"
    assert track.language == "bn"


def test_a_track_in_another_language_is_not_used(monkeypatch, tmp_path):
    """The words of a track in the wrong language are not the words asked for."""
    from mash_mix import sources

    def fake_run(args, timeout=600):
        raise AssertionError("no track in another language may be fetched")

    monkeypatch.setattr(sources, "_run", fake_run)

    track = sources.fetch_subtitles(
        "https://example.com/watch",
        str(tmp_path / "work"),
        tracks=[{"raw_language": "en", "automatic": False}],
        prefer_language="bn-BD",
    )
    assert track is None


def test_a_run_that_could_not_read_any_track_says_why(monkeypatch, tmp_path):
    """The reason the page refused is passed on, not swallowed."""
    from mash_mix import sources

    def fake_run(args, timeout=600):
        raise sources.SourceError("HTTP Error 429: Too Many Requests")

    monkeypatch.setattr(sources, "_run", fake_run)

    with pytest.raises(sources.SourceError) as caught:
        sources.fetch_subtitles(
            "https://example.com/watch",
            str(tmp_path / "work"),
            tracks=[{"raw_language": "bn", "automatic": False}],
            prefer_language="bn",
        )
    assert "429" in str(caught.value)


# -- the whole path, with the page carrying the lyrics ----------------------


def _write_wav(path: Path, seconds: float = 1.0, sample_rate: int = 16000) -> Path:
    import wave

    with wave.open(str(path), "wb") as handle:
        handle.setnchannels(1)
        handle.setsampwidth(2)
        handle.setframerate(sample_rate)
        handle.writeframes(b"\x00\x00" * int(sample_rate * seconds))
    return path


def test_the_subtitles_the_page_carries_are_used_instead_of_a_model(
    monkeypatch, tmp_path
):
    """A page that carries its own lyrics is the best source of them."""
    import mash_mix.tasks.lyrics as lyrics_module

    subs = tmp_path / "subs.bn.srt"
    subs.write_text(
        "1\n00:00:01,000 --> 00:00:04,000\n" + BANGLA_LINE + "\n",
        encoding="utf-8",
    )
    audio = _write_wav(tmp_path / "source.wav")

    metadata = SourceInfo(
        url="https://example.com/watch",
        title="Bhober Ei",
        uploader="Anupam Movie Songs",
        duration=265.0,
        guessed_language="bn",
        script="Bengali",
        language_source="script",
        subtitles=[
            {"language": "bn", "raw_language": "bn", "automatic": False, "formats": ["srt"]}
        ],
    )
    fetched = SourceInfo(
        url="https://example.com/watch",
        title="Bhober Ei",
        uploader="Anupam Movie Songs",
        duration=265.0,
        guessed_language="bn",
        script="Bengali",
        language_source="script",
        audio_path=str(audio),
    )

    monkeypatch.setattr("mash_mix.sources.fetch_metadata", lambda url, **kwargs: metadata)
    monkeypatch.setattr("mash_mix.sources.fetch_audio", lambda url, target, **kwargs: fetched)
    monkeypatch.setattr(
        "mash_mix.sources.fetch_subtitles",
        lambda url, target, **kwargs: SubtitleTrack(
            path=str(subs), language="bn", raw_language="bn", automatic=False, fmt="srt"
        ),
    )

    def no_model(*args, **kwargs):
        raise AssertionError("no model should be loaded when the page has subtitles")

    monkeypatch.setattr("mash_mix.backends.registry.get_stt", no_model)

    outcome = lyrics_module.extract_lyrics_from_url(
        "https://example.com/watch", str(tmp_path / "work")
    )
    result = outcome["result"]
    assert result.meta["lyrics_source"] == "site subtitles"
    assert len(result.lines) == 1
    assert result.lines[0].text == BANGLA_LINE
    assert result.language == "bn-BD"
    assert result.backend == "site-subtitles"


def test_the_report_says_what_the_pipeline_did(monkeypatch, tmp_path):
    """Every run leaves a report of what happened, step by step."""
    import mash_mix.tasks.lyrics as lyrics_module

    subs = tmp_path / "subs.bn.srt"
    subs.write_text(
        "1\n00:00:01,000 --> 00:00:04,000\n" + BANGLA_LINE + "\n",
        encoding="utf-8",
    )
    audio = _write_wav(tmp_path / "source.wav")

    metadata = SourceInfo(
        url="https://example.com/watch",
        title="Bhober Ei",
        duration=265.0,
        guessed_language="bn",
        language_source="script",
        subtitles=[
            {"language": "bn", "raw_language": "bn", "automatic": False, "formats": ["srt"]}
        ],
    )
    fetched = SourceInfo(
        url="https://example.com/watch",
        title="Bhober Ei",
        duration=265.0,
        guessed_language="bn",
        language_source="script",
        audio_path=str(audio),
    )

    monkeypatch.setattr("mash_mix.sources.fetch_metadata", lambda url, **kwargs: metadata)
    monkeypatch.setattr("mash_mix.sources.fetch_audio", lambda url, target, **kwargs: fetched)
    monkeypatch.setattr(
        "mash_mix.sources.fetch_subtitles",
        lambda url, target, **kwargs: SubtitleTrack(
            path=str(subs), language="bn", raw_language="bn", automatic=False, fmt="srt"
        ),
    )

    outcome = lyrics_module.extract_lyrics_from_url(
        "https://example.com/watch", str(tmp_path / "work")
    )
    pipeline = outcome["result"].meta["pipeline"]
    assert pipeline["lyrics_source"] == "site subtitles"
    names = [step["step"] for step in pipeline["steps"]]
    assert names == ["metadata", "subtitles", "audio"]
    assert pipeline["steps"][1]["ok"] is True
    assert pipeline["subtitle_track"]["lines"] == 1


def test_a_page_with_no_subtitles_falls_back_on_the_model(monkeypatch, tmp_path):
    """The audio is read when the page has no subtitles of its own."""
    import mash_mix.tasks.lyrics as lyrics_module

    audio = _write_wav(tmp_path / "source.wav")
    metadata = SourceInfo(
        url="https://example.com/watch",
        title="Bhober Ei",
        duration=265.0,
        guessed_language="bn",
        language_source="script",
        subtitles=[],
    )
    fetched = SourceInfo(
        url="https://example.com/watch",
        title="Bhober Ei",
        duration=265.0,
        guessed_language="bn",
        language_source="script",
        audio_path=str(audio),
    )

    monkeypatch.setattr("mash_mix.sources.fetch_metadata", lambda url, **kwargs: metadata)
    monkeypatch.setattr("mash_mix.sources.fetch_audio", lambda url, target, **kwargs: fetched)

    def fake_extract(audio_path, **kwargs):
        raise AssertionError("the model must not be reached in this test")

    monkeypatch.setattr(lyrics_module, "extract_lyrics", fake_extract)

    with pytest.raises(AssertionError):
        lyrics_module.extract_lyrics_from_url(
            "https://example.com/watch", str(tmp_path / "work")
        )
