# SPDX-License-Identifier: MIT | Copyright (c) 2026 MASH (Khaled Bhuiyan)
"""The reader that takes the words of a karaoke off the picture.

The reader of the picture and the video are stood in for, so the rules of the
reading are proved without a video and without the models being downloaded.
"""
from __future__ import annotations

import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from mash_mix import karaoke  # noqa: E402

BANGLA_ONE = "\u0993\u09b0\u09c7 \u09a8\u09c0\u09b2 \u09a6\u09b0\u09bf\u09af\u09bc\u09be"
BANGLA_TWO = "\u0986\u09ae\u09be\u09af\u09bc \u09a6\u09c7\u09b0\u09c7 \u09a6\u09c7 \u099b\u09be\u09a1\u09bc\u09bf\u09af\u09bc\u09be"
BANGLA_THREE = "\u09ac\u09a8\u09cd\u09a6\u09c0 \u09b9\u0987\u09af\u09bc\u09be \u09ae\u09a8\u09cb\u09af\u09bc\u09be"

# The colour of the words before a line is sung, in the middle of the sweep,
# and after it has been sung.
YELLOW = (250.0, 250.0, 20.0)
MIDDLE = (250.0, 135.0, 120.0)
PINK = (250.0, 20.0, 220.0)


class FakeBox:
    def __init__(self, x0, y0, x1, y1, text, conf=0.9):
        self.x0 = x0
        self.y0 = y0
        self.x1 = x1
        self.y1 = y1
        self.text = text
        self.conf = conf


class FakePicture:
    """A stand in for a picture of the video."""

    def __init__(self, height=720, width=1280):
        self.shape = (height, width, 3)


def sample(text, colour=None, left=100, right=400, top=300, bottom=340, conf=0.9):
    """A stand in for what one row of one frame held."""
    return karaoke._Sample(
        text=text,
        conf=conf,
        colour=colour,
        box=(left, top, right, bottom),
    )


def make_reader(pictures):
    """A stand in for the reader of the picture.

    The pictures are given as a list of lists of boxes. Every call returns
    the boxes of the picture that is asked for, in turn. The boxes are given
    in the shape the real reader gives them: the four corners of the text,
    the words and how sure the reader is.
    """
    calls = {"n": 0}

    def readtext(image, detail=1, paragraph=False):
        index = calls["n"]
        calls["n"] += 1
        found = []
        for box in pictures[index % len(pictures)]:
            corners = [
                [box.x0, box.y0],
                [box.x1, box.y0],
                [box.x1, box.y1],
                [box.x0, box.y1],
            ]
            found.append((corners, box.text, box.conf))
        return found

    class Reader:
        def __init__(self, languages, verbose=False):
            self.languages = languages

        def readtext(self, image, detail=1, paragraph=False):
            return readtext(image, detail=detail, paragraph=paragraph)

    return Reader, calls


class FakeCapture:
    """A stand in for the video."""

    def __init__(self, fps=30.0, frames=300):
        self.fps = fps
        self.frames = frames
        self.released = False
        self.position = 0

    def isOpened(self):
        return True

    def get(self, prop):
        if prop == karaoke._prop(self, "FPS"):
            return self.fps
        if prop == karaoke._prop(self, "FRAME_COUNT"):
            return self.frames
        return self.position

    def set(self, prop, value):
        self.position = value

    def read(self):
        return True, None

    def release(self):
        self.released = True


def test_the_pieces_of_text_are_put_into_the_rows_they_sit_in():
    boxes = [
        FakeBox(100, 300, 200, 340, "first"),
        FakeBox(220, 302, 320, 342, "second"),
        FakeBox(100, 400, 260, 440, "third"),
    ]
    rows = karaoke._group_rows(boxes)
    assert len(rows) == 2
    assert rows[0].text == "first second"
    assert rows[1].text == "third"


def test_two_readings_of_the_same_line_are_the_same_line():
    assert karaoke._same_line(BANGLA_ONE, BANGLA_ONE)
    assert karaoke._same_line(BANGLA_ONE, BANGLA_ONE + " ")
    assert not karaoke._same_line(BANGLA_ONE, BANGLA_TWO)
    assert not karaoke._same_line("", "something")


def test_the_reading_the_most_frames_agree_on_is_the_one_that_is_kept():
    readings = [
        (BANGLA_TWO, 0.4),
        (BANGLA_TWO, 0.5),
        (BANGLA_TWO, 0.3),
        (BANGLA_TWO.replace("\u0986", "\u0986\u09b0", 1), 0.6),
        (BANGLA_TWO.replace("\u0986", "\u0986\u09b0", 1), 0.9),
    ]
    assert karaoke._best_reading(readings) == BANGLA_TWO


def test_when_the_frames_are_even_the_surer_reading_is_the_one_that_is_kept():
    readings = [(BANGLA_TWO, 0.4), (BANGLA_TWO.replace("\u0986", "\u0986\u09b0", 1), 0.9)]
    assert karaoke._best_reading(readings) == BANGLA_TWO.replace("\u0986", "\u0986\u09b0", 1)


def test_a_line_that_is_nearly_right_is_kept_when_it_is_the_only_one():
    assert karaoke._best_reading([(BANGLA_ONE, 0.8)]) == BANGLA_ONE
    assert karaoke._best_reading([]) == ""


def test_the_lines_are_put_in_the_order_they_were_sung():
    samples = [
        [sample(BANGLA_ONE, YELLOW)],
        [sample(BANGLA_ONE, MIDDLE)],
        [sample(BANGLA_ONE, PINK)],
        [sample(BANGLA_TWO, YELLOW)],
        [sample(BANGLA_TWO, MIDDLE)],
        [sample(BANGLA_TWO, PINK)],
    ]
    moments = [0.0, 1.0, 2.0, 3.0, 4.0, 5.0]
    lines = karaoke._build_lines(samples, moments)
    assert len(lines) == 2
    judged = [karaoke._judge_line(line, "bn", 1280) for line in lines]
    assert judged[0] == (1.0, 2.0)
    assert judged[1] == (4.0, 5.0)
    for line, times in zip(lines, judged):
        line.sung_start, line.sung_end = times
    result = karaoke._order_lines(lines, 6.0, 1.0)
    assert [line.text for line in result] == [BANGLA_ONE, BANGLA_TWO]
    assert result[0].start == 1.0
    assert result[1].start >= result[0].start


def test_a_line_the_reader_was_unsure_about_does_not_end_the_line():
    samples = [
        [sample(BANGLA_ONE, YELLOW)],
        [None],
        [sample(BANGLA_ONE, MIDDLE, conf=0.8)],
        [sample(BANGLA_ONE, PINK)],
    ]
    moments = [0.0, 1.0, 2.0, 3.0]
    lines = karaoke._build_lines(samples, moments)
    assert len(lines) == 1
    assert lines[0].start == 0.0
    assert len(lines[0].colours) == 4


def test_a_line_that_is_wider_than_the_picture_is_not_a_line_of_the_song():
    # A channel that puts an advertisement over the words puts it across the
    # whole width, and a line of the song sits in the middle.
    assert karaoke._MAX_LINE_WIDTH < 0.88
    line = karaoke._Line(row=0, start=0.0, end=2.0)
    line.readings = [(BANGLA_ONE, 0.9)]
    line.moments = [0.0, 1.0, 2.0]
    line.colours = [YELLOW, MIDDLE, PINK]
    line.widths = [1200, 1200, 1200]
    line.boxes = [(0, 300, 1200, 340)] * 3
    assert karaoke._judge_line(line, "bn", 1280) is None


def test_a_line_that_is_wide_enough_to_hold_the_words_is_kept():
    line = karaoke._Line(row=0, start=0.0, end=2.0)
    line.readings = [(BANGLA_ONE, 0.9)]
    line.moments = [0.0, 1.0, 2.0]
    line.colours = [YELLOW, MIDDLE, PINK]
    line.widths = [400, 400, 400]
    line.boxes = [(100, 300, 500, 340)] * 3
    assert karaoke._judge_line(line, "bn", 1280) == (1.0, 2.0)


def test_words_that_are_not_of_the_script_of_the_song_are_not_the_song():
    # A name of a channel, a web address and a number are not words of a song.
    assert karaoke._is_song_text(BANGLA_ONE, "bn")
    assert not karaoke._is_song_text("DS Karaoke", "bn")
    assert not karaoke._is_song_text("facebook com/dskaraokepage", "bn")
    assert not karaoke._is_song_text("dskaraokemail@gmail.com", "bn")
    assert not karaoke._is_song_text("% % %\u09e8\u09e6\u09e7\u09e9***", "bn")
    assert karaoke._is_song_text("a line of an english song", "latin")


def test_the_script_of_the_song_is_worked_out_from_the_words_that_were_read():
    assert karaoke._dominant_script([BANGLA_ONE, BANGLA_TWO]) == "bn"
    assert karaoke._dominant_script(["a line of an english song"]) == "latin"


def test_a_line_whose_colour_does_not_move_is_not_a_line_of_the_song():
    # A logo or a badge sits on the picture and keeps its colour.
    line = karaoke._Line(row=0, start=0.0, end=2.0)
    line.readings = [(BANGLA_ONE, 0.9)]
    line.moments = [0.0, 1.0, 2.0]
    line.colours = [YELLOW, YELLOW, YELLOW]
    line.widths = [400, 400, 400]
    line.boxes = [(100, 300, 500, 340)] * 3
    assert karaoke._judge_line(line, "bn", 1280) is None


def test_a_line_whose_colour_goes_one_way_and_comes_back_is_not_sung():
    line = karaoke._Line(row=0, start=0.0, end=2.0)
    line.readings = [(BANGLA_ONE, 0.9)]
    line.moments = [0.0, 1.0, 2.0]
    line.colours = [YELLOW, PINK, YELLOW]
    line.widths = [400, 400, 400]
    line.boxes = [(100, 300, 500, 340)] * 3
    assert karaoke._judge_line(line, "bn", 1280) is None


def test_the_same_line_swept_in_both_rows_is_kept_once():
    # A karaoke shows the line that is being sung and the line that comes
    # next, and the same words are swept once in one row and once in the
    # other. The first sweep is the one that was sung.
    samples = [
        [sample(BANGLA_ONE, YELLOW), None],
        [sample(BANGLA_ONE, MIDDLE), None],
        [sample(BANGLA_ONE, PINK), None],
        [None, sample(BANGLA_ONE, YELLOW)],
        [None, sample(BANGLA_ONE, MIDDLE)],
        [None, sample(BANGLA_ONE, PINK)],
    ]
    moments = [0.0, 1.0, 2.0, 3.0, 4.0, 5.0]
    lines = karaoke._build_lines(samples, moments)
    assert len(lines) == 2
    for line in lines:
        judged = karaoke._judge_line(line, "bn", 1280)
        assert judged is not None
        line.sung_start, line.sung_end = judged
    result = karaoke._order_lines(lines, 6.0, 1.0)
    assert len(result) == 1
    assert result[0].text == BANGLA_ONE
    assert result[0].start == 1.0


def test_a_line_of_the_song_that_comes_back_later_is_kept():
    # The same words are sung again much later in the song, and that is a
    # line of the song of its own.
    samples = [
        [sample(BANGLA_ONE, YELLOW)],
        [sample(BANGLA_ONE, MIDDLE)],
        [sample(BANGLA_ONE, PINK)],
        [None],
        [sample(BANGLA_TWO, YELLOW)],
        [sample(BANGLA_TWO, MIDDLE)],
        [sample(BANGLA_TWO, PINK)],
        [None],
        [sample(BANGLA_ONE, YELLOW)],
        [sample(BANGLA_ONE, MIDDLE)],
        [sample(BANGLA_ONE, PINK)],
    ]
    moments = [float(index) for index in range(len(samples))]
    lines = karaoke._build_lines(samples, moments)
    for line in lines:
        judged = karaoke._judge_line(line, "bn", 1280)
        assert judged is not None
        line.sung_start, line.sung_end = judged
    result = karaoke._order_lines(lines, 9.0, 1.0)
    assert [line.text for line in result] == [BANGLA_ONE, BANGLA_TWO, BANGLA_ONE]


def test_the_reader_of_the_picture_is_given_the_language_of_the_song():
    assert karaoke._reader_languages("bn-BD") == ["bn", "en"]
    assert karaoke._reader_languages("bn-IN") == ["bn", "en"]
    assert karaoke._reader_languages("hi") == ["hi", "en"]
    assert karaoke._reader_languages("zh-CN") == ["ch_sim", "en"]
    assert karaoke._reader_languages("auto") == ["bn", "en"]


def test_the_colour_of_the_words_says_how_far_the_sweep_has_gone():
    # The colour moves from one end of the line to the other, so the place
    # of a sample on that line is how far the sweep has gone.
    fractions = karaoke._sweep_fractions([YELLOW, MIDDLE, PINK, YELLOW, MIDDLE, PINK])
    assert fractions
    assert fractions[0] < 0.1
    assert 0.4 < fractions[1] < 0.6
    assert fractions[2] > 0.9


def test_a_row_whose_colour_does_not_move_is_not_swept():
    fractions = karaoke._sweep_fractions([YELLOW] * 6)
    assert fractions == []


def fake_colours(monkeypatch):
    """Stand in for the colour of the places on the picture.

    A row that is high on the picture sweeps, the way a line of a karaoke
    does. A row that is low down keeps its colour, the way a logo does.
    """

    def positions(capture, slots, moments, fps):
        # The colours are held by the moment first and the place second.
        out = []
        for index in range(len(moments)):
            row = []
            for slot in slots:
                if slot.top < 500:
                    row.append((250.0, 250.0 - 20.0 * index, 20.0 + 10.0 * index))
                else:
                    row.append((250.0, 250.0, 20.0))
            out.append(row)
        return out

    monkeypatch.setattr(karaoke, "_sweep_positions", positions)


def test_the_survey_looks_for_the_rows_of_the_lyrics(monkeypatch):
    # Two rows of the song whose words change, and a logo whose words never
    # do. The logo is not a line of the song.
    pictures = [
        [
            FakeBox(100, 300, 400, 340, BANGLA_ONE),
            FakeBox(100, 400, 400, 440, BANGLA_TWO),
            FakeBox(100, 600, 200, 640, "logo"),
        ],
        [
            FakeBox(100, 300, 400, 340, BANGLA_THREE),
            FakeBox(100, 400, 400, 440, BANGLA_ONE),
            FakeBox(100, 600, 200, 640, "logo"),
        ],
    ]
    reader_class, _calls = make_reader(pictures)
    monkeypatch.setattr(karaoke, "_open_reader", lambda language: reader_class(["bn", "en"]))
    monkeypatch.setattr(karaoke, "_open_capture", lambda path: FakeCapture())
    monkeypatch.setattr(karaoke, "_frame_at", lambda capture, moment, fps: FakePicture())
    fake_colours(monkeypatch)

    found = karaoke.survey_burned_in_lyrics("video.mp4", language="bn", frames=2)
    assert found["slots"]
    assert found["band"][0] <= 300
    assert found["band"][1] >= 440
    # The logo sits lower down and never changes, so it is not a row of the
    # song.
    assert found["band"][1] < 600


def test_a_video_with_no_text_that_moves_is_not_a_karaoke(monkeypatch):
    # A logo that never changes is not a line of the song.
    pictures = [[FakeBox(100, 600, 200, 640, "logo")]]
    reader_class, _calls = make_reader(pictures)
    monkeypatch.setattr(karaoke, "_open_reader", lambda language: reader_class(["bn", "en"]))
    monkeypatch.setattr(karaoke, "_open_capture", lambda path: FakeCapture())
    monkeypatch.setattr(karaoke, "_frame_at", lambda capture, moment, fps: FakePicture())
    fake_colours(monkeypatch)

    with pytest.raises(karaoke.KaraokeError):
        karaoke.survey_burned_in_lyrics("video.mp4", language="bn", frames=2)


def test_a_link_with_no_subtitles_is_read_off_the_picture(monkeypatch, tmp_path):
    """A page with no subtitles of its own gets its words off the picture."""
    import mash_mix.sources as sources_module
    import mash_mix.tasks.lyrics as lyrics_module
    from mash_mix.types import LyricLine

    def fake_metadata(url, **kwargs):
        return sources_module.SourceInfo(
            url=url,
            title="A karaoke",
            uploader="A channel",
            duration=232.0,
            extractor="Youtube",
            guessed_language="bn",
            script="Bengali",
            language_source="script",
        )

    monkeypatch.setattr("mash_mix.sources.fetch_metadata", fake_metadata)
    monkeypatch.setattr(
        "mash_mix.sources.fetch_audio",
        lambda url, target_dir, **kwargs: sources_module.SourceInfo(
            url=url,
            title="A karaoke",
            duration=232.0,
            audio_path=str(tmp_path / "source.wav"),
        ),
    )
    monkeypatch.setattr("mash_mix.sources.fetch_video", lambda url, target_dir, **kwargs: "picture.mp4")
    monkeypatch.setattr(karaoke, "have_ocr", lambda: True)
    monkeypatch.setattr(
        karaoke,
        "survey_burned_in_lyrics",
        lambda path, **kwargs: {"band": (0, 100), "slots": [], "duration": 232.0},
    )

    def fake_read(path, **kwargs):
        return karaoke.KaraokeResult(
            lines=[
                LyricLine(start=18.0, end=24.0, text=BANGLA_ONE),
                LyricLine(start=24.0, end=30.0, text=BANGLA_TWO),
            ],
            duration=232.0,
            band=(0, 100),
            rows=2,
            swept=True,
            elapsed=12.0,
            language="bn",
            meta={"frames_read": 232},
        )

    monkeypatch.setattr(karaoke, "read_burned_in_lyrics", fake_read)

    outcome = lyrics_module.extract_lyrics_from_url(
        "https://www.youtube.com/watch?v=O4VgX2jd2Ug",
        str(tmp_path / "work"),
        language="auto",
    )
    result = outcome["result"]
    assert result.meta["lyrics_source"] == "video ocr"
    assert result.backend == "video-ocr"
    assert len(result.lines) == 2
    assert result.lines[0].text == BANGLA_ONE
    steps = [step["step"] for step in result.meta["pipeline"]["steps"]]
    assert "karaoke" in steps


def test_the_karaoke_is_not_read_when_the_page_has_subtitles(monkeypatch, tmp_path):
    """A page that carries its own subtitles is the better answer."""
    import mash_mix.sources as sources_module
    import mash_mix.tasks.lyrics as lyrics_module

    def fake_metadata(url, **kwargs):
        return sources_module.SourceInfo(
            url=url,
            title="A song",
            duration=232.0,
            extractor="Youtube",
            declared_language="bn",
            language_source="declared",
            subtitles=[{"code": "bn", "name": "Bengali", "automatic": False}],
        )

    monkeypatch.setattr("mash_mix.sources.fetch_metadata", fake_metadata)
    monkeypatch.setattr(
        "mash_mix.sources.fetch_audio",
        lambda url, target_dir, **kwargs: sources_module.SourceInfo(
            url=url, title="A song", duration=232.0, audio_path=str(tmp_path / "source.wav")
        ),
    )

    def no_video(url, target_dir, **kwargs):
        raise AssertionError("the picture must not be fetched when the page has subtitles")

    monkeypatch.setattr("mash_mix.sources.fetch_video", no_video)

    class Track:
        path = str(tmp_path / "subs.srt")
        language = "bn"
        automatic = False

        def to_dict(self):
            return {"language": "bn", "automatic": False}

    monkeypatch.setattr(
        "mash_mix.sources.fetch_subtitles", lambda url, target_dir, **kwargs: Track()
    )
    monkeypatch.setattr(
        "mash_mix.sources.read_subtitle_file",
        lambda path, language=None: "1\n00:00:01,000 --> 00:00:03,000\n" + BANGLA_ONE + "\n",
    )

    outcome = lyrics_module.extract_lyrics_from_url(
        "https://www.youtube.com/watch?v=296Gi_nnbho",
        str(tmp_path / "work"),
        language="auto",
    )
    result = outcome["result"]
    assert result.meta["lyrics_source"] == "site subtitles"
