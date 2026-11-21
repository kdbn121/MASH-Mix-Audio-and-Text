# SPDX-License-Identifier: MIT | Copyright (c) 2026 MASH (Khaled Bhuiyan)
"""Read the lyrics that are burned into the picture of a karaoke video.

A karaoke video has no subtitles and often nobody is singing either, so the
only place the words are is the picture. This module reads them off the
picture.

The video is read in three passes.

The first pass is a survey. A few frames are spread over the video and the
text on them is found, which shows where on the picture the words can be. The
colour of every one of those places is then followed over the whole video. A
karaoke sweeps the line along as it is sung, so the colour of the letters
moves from one colour to another and back again as the song goes on, and the
places where the colour moves are the rows of the lyrics. A name and a logo do
not move, so they are not rows of the song. A video with nothing that moves is
not a karaoke, and the survey says so without spending the rest of the work
on it.

The second pass reads the words of those rows at a fixed step and follows the
text of each one. A row that changes its words starts a new line. The words
are read again in every frame they are on the screen in, and the reading that
the most frames agree on is the one that is kept, because a recogniser that
is nearly right is wrong in a different way every time it is tried. The
colour of the letters of the line is kept as well.

The third pass judges every line that was found. The words of the song are in
the script of the song, they sit in the middle of the picture, and the colour
of their letters travels in one direction while they are on the screen. A
channel that puts an advertisement over the words fails the last of those,
because an advertisement is not sung and its colour does not travel. The
moment the colour of the letters starts to travel is the moment the line is
sung, and the moment it stops is the moment the line has been sung, so those
are the times of the lines.

Everything happens on this machine and nothing is sent anywhere.
"""

from __future__ import annotations

import time
from dataclasses import dataclass, field
from difflib import SequenceMatcher
from typing import Any, Callable, Dict, List, Optional, Sequence, Tuple

from .types import LyricLine

__all__ = [
    "KaraokeError",
    "KaraokeResult",
    "have_ocr",
    "read_burned_in_lyrics",
    "survey_burned_in_lyrics",
]


# The languages the reader of the picture knows, by the language code of the
# engine. A language the reader does not know falls back on English, because
# most of the words in a title are in English anyway.
_READER_LANGUAGES: Dict[str, List[str]] = {
    "bn": ["bn", "en"],
    "hi": ["hi", "en"],
    "mr": ["mr", "en"],
    "ne": ["ne", "en"],
    "ta": ["ta", "en"],
    "te": ["te", "en"],
    "kn": ["kn", "en"],
    "ml": ["ml", "en"],
    "gu": ["gu", "en"],
    "pa": ["pa", "en"],
    "ar": ["ar", "en"],
    "fa": ["fa", "en"],
    "ur": ["ur", "en"],
    "ru": ["ru", "en"],
    "uk": ["uk", "en"],
    "el": ["el", "en"],
    "he": ["he", "en"],
    "th": ["th", "en"],
    "vi": ["vi", "en"],
    "zh": ["ch_sim", "en"],
    "ja": ["ja", "en"],
    "ko": ["ko", "en"],
}

# How close two readings of the same line have to be to be the same line.
_SAME_LINE = 0.55

# How near in time two sweeps of the same line have to be for them to be the
# same line of the song sung twice. A karaoke shows the line that is being
# sung and the line that comes next, and the same words are often swept once
# in one row and once in the other. The first sweep of a line is the one that
# was sung, so the later one is dropped.
_SAME_LINE_WINDOW = 12.0

# How tall a place on the picture has to be to hold a row of text.
_SLOT_HEIGHT = 40

# A line of the song is not across the whole width of the picture. A channel
# that puts an advertisement over the words puts it across the whole width,
# while a line of the song sits in the middle.
_MAX_LINE_WIDTH = 0.78

# A line of the song is held on the screen for at least this long.
_MIN_LINE_SECONDS = 0.4

# How much the colour of a row has to move for the row to be a line of the
# song. This is only used to choose the rows of the song in the survey.
_MIN_SWEEP = 0.45

# How far the colour of the letters of a line has to travel for the line to
# count as sung. The letters of a line that is sung go from the colour of the
# words before the line to the colour of the words after it, which is a long
# way in the colours of the picture.
_MIN_TRAVEL = 40.0

# How much of the moves of the colour has to be forwards. A line that is sung
# moves in one direction and stays there, while a line that is only on the
# picture does not move at all.
_MIN_ONE_WAY = 0.55

# A move of the colour that is smaller than this is noise and not a move.
_MOVE_NOISE = 0.02

# A move of the colour this big is the sweep starting.
_SWEEP_STEP = 0.08

# How much of the travel of the colour counts as the line having been sung.
_SWEEP_DONE = 0.95

# The share of the letters of a line that must be of the script of the song.
_MIN_SCRIPT_SHARE = 0.6

# The code points of the Bangla block. The script of the song is found by
# counting them, so the letters are never written out here.
_BANGLA_FIRST = 0x0980
_BANGLA_LAST = 0x09FF

# The code points of the Bangla digits. A number is not a word of a song, so
# the digits are not counted as letters.
_BANGLA_DIGIT_FIRST = 0x09E6
_BANGLA_DIGIT_LAST = 0x09EF

# How fine the search for the moment a line starts being sung is. This part
# of the work reads the colour of the picture only, so it is cheap.
_FINE_STEP = 0.1

# A move of the colour this big, over the fine step, is the sweep starting.
_FINE_MOVE = 0.03

# How long a window the search for that moment looks at. A line that was on
# the screen long before it was sung is a line that was only on the picture.
_FINE_WINDOW = 4.0


class KaraokeError(RuntimeError):
    """Something stopped the words being read off the picture."""


@dataclass
class KaraokeResult:
    """The lines that were read off the picture, and the facts about it."""

    lines: List[LyricLine] = field(default_factory=list)
    duration: float = 0.0
    band: Tuple[int, int] = (0, 0)
    rows: int = 0
    swept: bool = False
    elapsed: float = 0.0
    language: str = ""
    meta: Dict[str, Any] = field(default_factory=dict)


@dataclass
class _Box:
    """One piece of text that was found on the picture."""

    x0: float
    y0: float
    x1: float
    y1: float
    text: str
    conf: float


@dataclass
class _Row:
    """One line of text on the picture."""

    top: int
    bottom: int
    left: int
    right: int
    text: str
    conf: float


@dataclass
class _Slot:
    """One place on the picture where a row of text can sit."""

    top: int
    bottom: int
    left: int
    right: int
    seen: int = 0
    words: set = field(default_factory=set)

    @property
    def middle(self) -> int:
        return (self.top + self.bottom) // 2

    @property
    def key(self) -> int:
        return int(round(self.middle / _SLOT_HEIGHT))


@dataclass
class _Sample:
    """What one row of one frame of the picture held."""

    text: str
    conf: float
    colour: Optional[Tuple[float, float, float]]
    box: Tuple[int, int, int, int]


@dataclass
class _Line:
    """One line of the picture and what was known about it.

    The moments, the colours, the widths and the places are all in step with
    one another, so the colour of a moment is the colour of the words that
    were on the picture at that moment.
    """

    row: int
    start: float
    end: float
    readings: List[Tuple[str, float]] = field(default_factory=list)
    moments: List[float] = field(default_factory=list)
    colours: List[Optional[Tuple[float, float, float]]] = field(default_factory=list)
    widths: List[int] = field(default_factory=list)
    boxes: List[Tuple[int, int, int, int]] = field(default_factory=list)
    sung_start: Optional[float] = None
    sung_end: Optional[float] = None

    def add(
        self,
        moment: float,
        sample: Optional[_Sample],
    ) -> None:
        """Keep what one frame held."""
        self.moments.append(moment)
        self.colours.append(sample.colour if sample is not None else None)
        self.widths.append(sample.box[2] - sample.box[0] if sample is not None else 0)
        self.boxes.append(sample.box if sample is not None else (0, 0, 0, 0))


def have_ocr() -> bool:
    """Say whether the reader of the picture is installed."""
    try:
        import cv2  # noqa: F401
        import easyocr  # noqa: F401
    except ImportError:
        return False
    return True


def _reader_languages(language: str) -> List[str]:
    """The languages the reader of the picture is given for a language."""
    code = (language or "").strip().lower().split("-")[0]
    if not code or code == "auto":
        return ["bn", "en"]
    return list(_READER_LANGUAGES.get(code, [code, "en"]))


def _open_reader(language: str) -> Any:
    """Open the reader of the picture."""
    try:
        import easyocr  # type: ignore
    except ImportError as exc:
        raise KaraokeError(
            "Reading the words off the picture needs the karaoke extra. "
            "Install it with: pip install \"mash-mix-audio-and-text[karaoke]\""
        ) from exc
    try:
        return easyocr.Reader(_reader_languages(language), verbose=False)
    except Exception as exc:  # pragma: no cover - the models are downloaded
        raise KaraokeError(f"The reader of the picture could not be opened: {exc}") from exc


def _open_capture(video_path: str) -> Any:
    """Open the video for reading."""
    try:
        import cv2  # type: ignore
    except ImportError as exc:
        raise KaraokeError(
            "Reading the words off the picture needs the karaoke extra. "
            "Install it with: pip install \"mash-mix-audio-and-text[karaoke]\""
        ) from exc
    capture = cv2.VideoCapture(video_path)
    if not capture.isOpened():
        raise KaraokeError(f"The video could not be opened for reading: {video_path}")
    return capture


def _prop(capture: Any, name: str) -> int:
    """The number of a property of the reader of pictures."""
    import cv2  # type: ignore

    return getattr(cv2, f"CAP_PROP_{name}")


def _frame_at(capture: Any, moment: float, fps: float) -> Optional[Any]:
    """The picture at a moment of the video."""
    capture.set(_prop(capture, "POS_FRAMES"), int(moment * fps))
    ok, frame = capture.read()
    return frame if ok else None


def _video_facts(capture: Any) -> Tuple[float, float]:
    """The length of the video and its rate."""
    fps = float(capture.get(_prop(capture, "FPS")) or 25.0)
    count = int(capture.get(_prop(capture, "FRAME_COUNT")) or 0)
    duration = count / fps if count > 0 and fps > 0 else 0.0
    return duration, fps


def _read_boxes(reader: Any, image: Any) -> List[_Box]:
    """Find every piece of text on a picture."""
    try:
        found = reader.readtext(image, detail=1, paragraph=False)
    except Exception as exc:  # pragma: no cover - depends on the picture
        raise KaraokeError(f"The picture could not be read: {exc}") from exc

    boxes: List[_Box] = []
    for item in found:
        try:
            polygon, text, conf = item
        except (TypeError, ValueError):
            continue
        text = str(text).strip()
        if not text:
            continue
        xs = [float(point[0]) for point in polygon]
        ys = [float(point[1]) for point in polygon]
        boxes.append(
            _Box(
                x0=min(xs),
                y0=min(ys),
                x1=max(xs),
                y1=max(ys),
                text=text,
                conf=float(conf),
            )
        )
    return boxes


def _group_rows(boxes: Sequence[_Box], tolerance: Optional[float] = None) -> List[_Row]:
    """Put the pieces of text into the rows they sit in."""
    if not boxes:
        return []

    heights = sorted(max(1.0, box.y1 - box.y0) for box in boxes)
    middle = heights[len(heights) // 2]
    if tolerance is None:
        tolerance = max(12.0, middle * 0.6)

    ordered = sorted(boxes, key=lambda box: (box.y0 + box.y1) / 2.0)
    groups: List[List[_Box]] = []
    centre = (ordered[0].y0 + ordered[0].y1) / 2.0
    for box in ordered:
        middle_of_box = (box.y0 + box.y1) / 2.0
        if groups and abs(middle_of_box - centre) <= tolerance:
            groups[-1].append(box)
            centre = sum((one.y0 + one.y1) / 2.0 for one in groups[-1]) / len(groups[-1])
        else:
            groups.append([box])
            centre = middle_of_box

    rows: List[_Row] = []
    for group in groups:
        parts = sorted(group, key=lambda box: box.x0)
        rows.append(
            _Row(
                top=int(min(box.y0 for box in group)),
                bottom=int(max(box.y1 for box in group)),
                left=int(min(box.x0 for box in group)),
                right=int(max(box.x1 for box in group)),
                text=" ".join(box.text for box in parts),
                conf=sum(box.conf for box in group) / len(group),
            )
        )
    return rows


def _slot_of(slots: Dict[int, _Slot], row: _Row) -> _Slot:
    """The place on the picture a row of text belongs to."""
    key = int(round((row.top + row.bottom) / 2.0 / _SLOT_HEIGHT))
    slot = slots.get(key)
    if slot is None:
        slot = _Slot(top=row.top, bottom=row.bottom, left=row.left, right=row.right)
        slots[key] = slot
    slot.top = min(slot.top, row.top)
    slot.bottom = max(slot.bottom, row.bottom)
    slot.left = min(slot.left, row.left)
    slot.right = max(slot.right, row.right)
    slot.seen += 1
    slot.words.add(row.text)
    return slot


def _sweep_positions(
    capture: Any,
    slots: Sequence[_Slot],
    moments: Sequence[float],
    fps: float,
) -> List[List[Optional[Tuple[float, float, float]]]]:
    """The colour of the words of each place, at every moment.

    The karaoke sweeps the line along as it is sung, so the colour of the
    letters moves from one colour to another as the line goes. The colour
    that is returned is the mean colour of the letters of the place.
    """
    import cv2  # type: ignore

    samples: List[List[Optional[Tuple[float, float, float]]]] = []
    for moment in moments:
        picture = _frame_at(capture, moment, fps)
        if picture is None:
            samples.append([None] * len(slots))
            continue
        frame = cv2.cvtColor(picture, cv2.COLOR_BGR2RGB)
        row_samples: List[Optional[Tuple[float, float, float]]] = []
        for slot in slots:
            band = frame[max(0, slot.top) : slot.bottom + 1, max(0, slot.left) : slot.right + 1]
            if band.size == 0:
                row_samples.append(None)
                continue
            flat = band.reshape(-1, 3).astype("float32")
            # The letters are the colourful bright parts. The picture behind
            # them is dark on a karaoke, so the mask keeps the letters and
            # throws the rest away.
            high = flat.max(axis=1)
            low = flat.min(axis=1)
            mask = ((high - low) > 40) & (high > 90)
            if int(mask.sum()) < 40:
                row_samples.append(None)
                continue
            picked = flat[mask]
            row_samples.append(
                (
                    float(picked[:, 0].mean()),
                    float(picked[:, 1].mean()),
                    float(picked[:, 2].mean()),
                )
            )
        samples.append(row_samples)
    return samples


def _sweep_fractions(
    samples: Sequence[Optional[Tuple[float, float, float]]]
) -> List[float]:
    """How far along the line the sweep is, for every sample of a place.

    The colour moves along one direction, and the place of a sample on that
    direction is how far the sweep has gone. A place whose colour does not
    move has no sweep, and gives nothing back.
    """
    import numpy as np  # type: ignore

    kept = [sample for sample in samples if sample is not None]
    if len(kept) < 4:
        return []

    points = np.asarray(kept, dtype="float32")
    centred = points - points.mean(axis=0)
    _, _, axes = np.linalg.svd(centred, full_matrices=False)
    direction = axes[0]
    if float(np.linalg.norm(direction)) < 1e-6:
        return []

    projected = centred.dot(direction)
    span = float(projected.max() - projected.min())
    if span < 12.0:
        # The colour does not move, so the line is not swept.
        return []

    out: List[float] = []
    for sample in samples:
        if sample is None:
            out.append(float("nan"))
        else:
            point = np.asarray(sample, dtype="float32") - points.mean(axis=0)
            out.append(float((point.dot(direction) - projected.min()) / span))
    return out


def _survey(
    reader: Any,
    capture: Any,
    duration: float,
    fps: float,
    frames: int,
) -> Dict[str, Any]:
    """Find the rows of the lyrics and the colour that says when they are sung.

    The frames are spread over the middle of the video, so the introduction
    and the end credits do not decide the matter. The colour of every place
    where text was found is then followed over the whole video, and the
    places where the colour moves are the rows of the lyrics.
    """
    if duration <= 0:
        raise KaraokeError("The video has no length to read.")

    moments: List[float] = []
    for index in range(frames):
        share = 0.12 + 0.76 * (index / max(1, frames - 1))
        moments.append(duration * share)

    slots: Dict[int, _Slot] = {}
    seen = 0
    for moment in moments:
        picture = _frame_at(capture, moment, fps)
        if picture is None:
            continue
        seen += 1
        for row in _group_rows(_read_boxes(reader, picture)):
            if row.conf < 0.15:
                continue
            _slot_of(slots, row)

    if not slots:
        raise KaraokeError("No text was found on the picture.")

    wanted = max(2, seen // 2)
    candidates = [
        slot for slot in slots.values() if slot.seen >= wanted and len(slot.words) > 1
    ]
    if not candidates:
        raise KaraokeError("No text that changes was found on the picture.")
    candidates.sort(key=lambda slot: slot.top)

    # The colour of every candidate over the whole video.
    all_moments: List[float] = []
    moment = 0.0
    while moment < duration:
        all_moments.append(round(moment, 3))
        moment += 1.0
    colours = _sweep_positions(capture, candidates, all_moments, fps)

    lyric_slots: List[_Slot] = []
    fractions: List[List[float]] = []
    for index, slot in enumerate(candidates):
        # The colours are held by the moment first and the place second, so
        # the series of one place is taken out of them.
        series = [colour[index] for colour in colours]
        row_fractions = _sweep_fractions(series)
        if not row_fractions:
            continue
        real = [value for value in row_fractions if value == value]
        if not real:
            continue
        if max(real) - min(real) < _MIN_SWEEP:
            continue
        lyric_slots.append(slot)
        # The series is kept only for the places that are rows of the song, so
        # the series and the rows are in step with one another.
        fractions.append(row_fractions)

    if not lyric_slots:
        raise KaraokeError("No text that moves was found on the picture.")

    band = (
        max(0, min(slot.top for slot in lyric_slots) - 10),
        max(slot.bottom for slot in lyric_slots) + 10,
    )
    return {
        "band": band,
        "slots": lyric_slots,
        "moments": all_moments,
        "fractions": fractions,
        "candidates": candidates,
    }


def survey_burned_in_lyrics(
    video_path: str,
    *,
    language: str = "bn",
    frames: int = 8,
) -> Dict[str, Any]:
    """Say whether the words of the song are on the picture of the video.

    This is the cheap look. It reads a few frames, follows the colour of the
    places where the text is and says whether there is text that sweeps, so a
    video that is not a karaoke is not sent through the whole reading.
    """
    reader = _open_reader(language)
    capture = _open_capture(video_path)
    try:
        duration, fps = _video_facts(capture)
        found = _survey(reader, capture, duration, fps, frames)
    finally:
        capture.release()
    found["duration"] = round(duration, 3)
    found["slots"] = [
        {"top": slot.top, "bottom": slot.bottom, "left": slot.left, "right": slot.right}
        for slot in found["slots"]
    ]
    return found


def _same_line(first: str, second: str) -> bool:
    """Say whether two readings are of the same line."""
    if first == second:
        return True
    if not first or not second:
        return False
    if first in second or second in first:
        return True
    return SequenceMatcher(None, first, second).ratio() >= _SAME_LINE


def _best_reading(readings: Sequence[Tuple[str, float]]) -> str:
    """The reading that the most frames agree on.

    A recogniser that is nearly right is wrong in a different way every time
    it is tried, so the reading that shows up most often is the one that is
    kept, and the confidence breaks a tie.
    """
    if not readings:
        return ""
    counts: Dict[str, int] = {}
    best_conf: Dict[str, float] = {}
    for text, conf in readings:
        counts[text] = counts.get(text, 0) + 1
        best_conf[text] = max(best_conf.get(text, 0.0), conf)
    ordered = sorted(counts, key=lambda text: (counts[text], best_conf[text]), reverse=True)
    return ordered[0]


def _letter_counts(text: str) -> Tuple[int, int]:
    """How many letters of the text are Bangla and how many are Latin."""
    bangla = 0
    latin = 0
    for character in text:
        code = ord(character)
        if _BANGLA_FIRST <= code <= _BANGLA_LAST:
            if _BANGLA_DIGIT_FIRST <= code <= _BANGLA_DIGIT_LAST:
                continue
            bangla += 1
        elif ("a" <= character <= "z") or ("A" <= character <= "Z"):
            latin += 1
    return bangla, latin


def _dominant_script(texts: Sequence[str]) -> str:
    """The script that the words of the song are written in.

    The script is worked out from the words that were read, so a karaoke in
    any language is judged by its own script and not by the language that was
    asked for.
    """
    bangla = 0
    latin = 0
    for text in texts:
        one, other = _letter_counts(text)
        bangla += one
        latin += other
    if bangla == 0 and latin == 0:
        return "bn"
    return "bn" if bangla >= latin else "latin"


def _is_song_text(text: str, script: str) -> bool:
    """Say whether the words are words of the song and not something else.

    A name of a channel, a web address and a number are not words of a song,
    and they are told apart by the script they are written in and by what
    they are made of.
    """
    if not text:
        return False
    if "@" in text or "www." in text:
        return False
    bangla, latin = _letter_counts(text)
    total = bangla + latin
    if total == 0:
        return False
    if script == "bn":
        return bangla >= total * _MIN_SCRIPT_SHARE
    return latin >= total * _MIN_SCRIPT_SHARE


def _letters_colour(
    picture: Any,
    box: Tuple[int, int, int, int],
) -> Optional[Tuple[float, float, float]]:
    """The colour of the letters inside a box of the picture.

    The letters are the colourful bright parts. The picture behind them is
    dark on a karaoke, so the mask keeps the letters and throws the rest
    away.
    """
    import cv2  # type: ignore

    left, top, right, bottom = box
    height, width = picture.shape[:2]
    x0 = max(0, int(left))
    x1 = min(width, int(right) + 1)
    y0 = max(0, int(top))
    y1 = min(height, int(bottom) + 1)
    if x1 <= x0 or y1 <= y0:
        return None
    patch = cv2.cvtColor(picture[y0:y1, x0:x1], cv2.COLOR_BGR2RGB)
    flat = patch.reshape(-1, 3).astype("float32")
    high = flat.max(axis=1)
    low = flat.min(axis=1)
    mask = ((high - low) > 40) & (high > 90)
    if int(mask.sum()) < 30:
        return None
    picked = flat[mask]
    return (
        float(picked[:, 0].mean()),
        float(picked[:, 1].mean()),
        float(picked[:, 2].mean()),
    )


def _distance(
    first: Tuple[float, float, float],
    second: Tuple[float, float, float],
) -> float:
    """How far apart two colours are."""
    red = first[0] - second[0]
    green = first[1] - second[1]
    blue = first[2] - second[2]
    return float((red * red + green * green + blue * blue) ** 0.5)


def _sweep_axis(
    points: Sequence[Tuple[float, float, float]],
) -> Optional[Tuple[float, float, float]]:
    """The direction the colour of the letters travels in.

    The karaoke sweeps the line along as it is sung, so the colour of the
    letters moves along one direction in the colours of the picture and stays
    there. The direction is found from the two colours that are furthest apart
    on it, and then found again from those two, which settles it. It is then
    turned to match the way the song goes.
    """
    if len(points) < 3:
        return None

    red = points[-1][0] - points[0][0]
    green = points[-1][1] - points[0][1]
    blue = points[-1][2] - points[0][2]
    length = (red * red + green * green + blue * blue) ** 0.5
    if length < _MIN_TRAVEL:
        return None
    direction = (red / length, green / length, blue / length)

    for _ in range(2):
        lowest: Optional[Tuple[float, float, float]] = None
        highest: Optional[Tuple[float, float, float]] = None
        low_value = 0.0
        high_value = 0.0
        for point in points:
            value = _dot(point, direction)
            if lowest is None or value < low_value:
                low_value = value
                lowest = point
            if highest is None or value > high_value:
                high_value = value
                highest = point
        if lowest is None or highest is None:
            return None
        red = highest[0] - lowest[0]
        green = highest[1] - lowest[1]
        blue = highest[2] - lowest[2]
        length = (red * red + green * green + blue * blue) ** 0.5
        if length < _MIN_TRAVEL:
            return None
        direction = (red / length, green / length, blue / length)

    if _dot(points[-1], direction) < _dot(points[0], direction):
        direction = (-direction[0], -direction[1], -direction[2])
    return direction


def _dot(
    first: Tuple[float, float, float],
    second: Tuple[float, float, float],
) -> float:
    """How much two directions lean on one another."""
    return first[0] * second[0] + first[1] * second[1] + first[2] * second[2]


def _project(
    base: Tuple[float, float, float],
    colour: Tuple[float, float, float],
    axis: Tuple[float, float, float],
) -> float:
    """Where a colour sits along the direction the sweep runs in."""
    red = colour[0] - base[0]
    green = colour[1] - base[1]
    blue = colour[2] - base[2]
    return red * axis[0] + green * axis[1] + blue * axis[2]


def _build_lines(
    samples: Sequence[Sequence[Optional[_Sample]]],
    moments: Sequence[float],
) -> List[_Line]:
    """Follow the text of every row and make a line out of every change."""
    lines: List[_Line] = []
    rows = len(samples[0]) if samples else 0

    for row in range(rows):
        open_line: Optional[_Line] = None
        for index, at in enumerate(moments):
            placed = samples[index][row] if index < len(samples) else None
            text = placed.text if placed is not None else ""
            if open_line is None:
                if not text:
                    continue
                open_line = _Line(row=row, start=at, end=at)
                open_line.readings.append((text, placed.conf if placed else 0.0))
                open_line.add(at, placed)
                continue

            if not text:
                # A frame the reader was not sure about is not the end of the
                # line. The line ends when a different line takes its place.
                open_line.end = at
                open_line.add(at, None)
                continue

            if _same_line(open_line.readings[-1][0], text):
                open_line.readings.append((text, placed.conf))
                open_line.add(at, placed)
                open_line.end = at
                continue

            open_line.end = at
            lines.append(open_line)
            open_line = _Line(row=row, start=at, end=at)
            open_line.readings.append((text, placed.conf))
            open_line.add(at, placed)

        if open_line is not None:
            lines.append(open_line)

    return lines


def _median_width(widths: Sequence[int]) -> int:
    """The width that most of the lines of a row have."""
    kept = sorted(width for width in widths if width > 0)
    if not kept:
        return 0
    return kept[len(kept) // 2]


def _judge_line(
    line: _Line,
    script: str,
    picture_width: int,
) -> Optional[Tuple[float, float]]:
    """Say whether the line is a line of the song, and when it was sung.

    The line is a line of the song when the words are words of the song, when
    they do not run across the whole picture, and when the colour of their
    letters travels in one direction while they are on the screen. The times
    that are given back are the times the line was sung.
    """
    text = _best_reading(line.readings)
    if not _is_song_text(text, script):
        return None

    width = _median_width(line.widths)
    if picture_width > 0 and width > picture_width * _MAX_LINE_WIDTH:
        return None

    points = [
        (moment, colour)
        for moment, colour in zip(line.moments, line.colours)
        if colour is not None
    ]
    if len(points) < 3:
        return None

    axis = _sweep_axis([colour for _, colour in points])
    if axis is None:
        return None

    base = points[0][1]
    values = [(moment, _project(base, colour, axis)) for moment, colour in points]
    span = max(value for _, value in values) - min(value for _, value in values)
    if span < _MIN_TRAVEL:
        return None

    # The colour of a line that is sung moves in one direction and stays
    # there. A line that is only on the picture does not move at all, and a
    # line that goes one way and comes back is not being sung.
    moves = 0
    forwards = 0
    for index in range(1, len(values)):
        step = values[index][1] - values[index - 1][1]
        if abs(step) < span * _MOVE_NOISE:
            continue
        moves += 1
        if step > 0:
            forwards += 1
    if moves == 0 or forwards / moves < _MIN_ONE_WAY:
        return None

    # The sweep starts at the first big move forwards. A line that fades in
    # when it appears moves the other way, because the colour before the
    # fade is dimmer and not yet on the sweep, so the fade is not mistaken
    # for the song.
    start: Optional[float] = None
    for index in range(1, len(values)):
        if values[index][1] - values[index - 1][1] > span * _SWEEP_STEP:
            start = values[index][0]
            break

    # The sweep is done when the colour of the letters has reached the end of
    # its travel.
    low = min(value for _, value in values)
    end: Optional[float] = None
    for moment, value in values:
        if value >= low + span * _SWEEP_DONE:
            end = moment
            break

    if start is None:
        start = line.start
    if end is None:
        end = line.end
    if end < start:
        end = start
    return start, end


def _refine_start(
    capture: Any,
    line: _Line,
    fps: float,
    base: Tuple[float, float, float],
    axis: Tuple[float, float, float],
    span: float,
) -> None:
    """Find the exact moment the line started being sung.

    The words are read once a second, so the moment a line starts being sung
    is only known to the second. The colour of the picture is cheap to read,
    so the moment is looked for again with a finer step. Only a short window
    is looked at, because a line that was on the screen long before it was
    sung is a line that was only on the picture, and the second the words
    were read is the best that can be said about it.
    """
    if line.sung_start is None:
        return
    window = line.sung_start - line.start
    if window < _FINE_STEP * 2 or window > _FINE_WINDOW:
        return
    boxes = [box for box in line.boxes if box[2] > box[0]]
    if not boxes:
        return
    left = sum(box[0] for box in boxes) // len(boxes)
    top = sum(box[1] for box in boxes) // len(boxes)
    right = sum(box[2] for box in boxes) // len(boxes)
    bottom = sum(box[3] for box in boxes) // len(boxes)
    box = (left, top, right, bottom)

    previous: Optional[float] = None
    moment = line.start + _FINE_STEP
    while moment <= line.sung_start:
        picture = _frame_at(capture, moment, fps)
        if picture is not None:
            colour = _letters_colour(picture, box)
            if colour is not None:
                value = _project(base, colour, axis)
                if previous is not None and value - previous > span * _FINE_MOVE:
                    line.sung_start = moment
                    return
                previous = value
        moment += _FINE_STEP


def _order_lines(
    lines: List[_Line],
    duration: float,
    interval: float,
) -> List[LyricLine]:
    """Put the lines in the order they were sung and give each one its time.

    A karaoke shows the line that is being sung and the line that comes
    next, and the same words are often swept once in one row and once in the
    other. The first sweep of a line is the one that was sung, so the later
    sweep of the same words in the other row is dropped and the line of the
    song is kept once. A line that comes back in the same row later is a
    line of the song of its own and is kept.
    """
    if not lines:
        return []

    ordered = sorted(lines, key=lambda line: (line.sung_start or 0.0, line.start))

    result: List[LyricLine] = []
    kept: List[Tuple[str, float, int]] = []
    for line in ordered:
        text = _best_reading(line.readings)
        if not text:
            continue
        start = line.sung_start if line.sung_start is not None else line.start
        end = line.sung_end if line.sung_end is not None else line.end
        if end < start:
            end = start
        if end - start < _MIN_LINE_SECONDS:
            end = start + max(interval, _MIN_LINE_SECONDS)
        if any(
            _same_line(kept_text, text)
            and kept_row != line.row
            and 0 <= start - kept_start <= _SAME_LINE_WINDOW
            for kept_text, kept_start, kept_row in kept
        ):
            # The same words were swept again in the other row a moment
            # later. It is one line of the song.
            continue
        result.append(LyricLine(start=round(start, 3), end=round(end, 3), text=text))
        kept.append((text, start, line.row))

    if not result:
        return []

    if duration > 0 and result[-1].end > duration:
        result[-1].end = duration
    return result


def read_burned_in_lyrics(
    video_path: str,
    *,
    language: str = "bn",
    interval: float = 1.0,
    survey_frames: int = 8,
    progress: Optional[Callable[[float, str], None]] = None,
) -> KaraokeResult:
    """Read the words of the song off the picture of a karaoke video.

    The result is the lines of the song with the time each one is sung.
    """
    started = time.perf_counter()
    reader = _open_reader(language)
    capture = _open_capture(video_path)
    try:
        duration, fps = _video_facts(capture)
        found = _survey(reader, capture, duration, fps, survey_frames)
        band = found["band"]
        slots = found["slots"]
        all_moments = found["moments"]

        if progress:
            progress(0.05, "The words are being read off the picture.")

        samples: List[List[Optional[_Sample]]] = []
        picture_width = 0
        for index, at in enumerate(all_moments):
            picture = _frame_at(capture, at, fps)
            if picture is None:
                samples.append([None] * len(slots))
                continue
            if not picture_width:
                picture_width = int(picture.shape[1])
            crop = picture[band[0] : band[1] + 1]
            rows = _group_rows(_read_boxes(reader, crop)) if crop.size else []
            placed: List[Optional[_Sample]] = []
            for slot in slots:
                middle = (slot.top + slot.bottom) / 2.0 - band[0]
                best: Optional[_Row] = None
                best_gap = 1e9
                for row in rows:
                    gap = abs((row.top + row.bottom) / 2.0 - middle)
                    if gap < best_gap:
                        best_gap = gap
                        best = row
                if best is None or best_gap > 30:
                    placed.append(None)
                    continue
                if best.conf < 0.15:
                    placed.append(None)
                    continue
                box = (best.left, best.top + band[0], best.right, best.bottom + band[0])
                placed.append(
                    _Sample(
                        text=best.text,
                        conf=best.conf,
                        colour=_letters_colour(picture, box),
                        box=box,
                    )
                )
            samples.append(placed)
            if progress and index % 10 == 0:
                progress(
                    0.05 + 0.80 * (index / max(1, len(all_moments))),
                    "The words are being read off the picture.",
                )

        lines = _build_lines(samples, all_moments)

        if progress:
            progress(0.88, "The lines are being judged.")

        script = _dominant_script(
            [text for line in lines for text, _ in line.readings]
        )

        kept: List[_Line] = []
        for line in lines:
            judged = _judge_line(line, script, picture_width)
            if judged is None:
                continue
            line.sung_start, line.sung_end = judged[0], judged[1]
            kept.append(line)

        if progress:
            progress(0.92, "The moment of each line is being found.")

        for line in kept:
            points = [
                (moment, colour)
                for moment, colour in zip(line.moments, line.colours)
                if colour is not None
            ]
            if len(points) < 3:
                continue
            axis = _sweep_axis([colour for _, colour in points])
            if axis is None:
                continue
            base = points[0][1]
            values = [_project(base, colour, axis) for _, colour in points]
            span = max(values) - min(values)
            if span < _MIN_TRAVEL:
                continue
            _refine_start(capture, line, fps, base, axis, span)

        if progress:
            progress(0.95, "The lines are being put in order.")

        result_lines = _order_lines(kept, duration, interval)
        result = KaraokeResult(
            lines=result_lines,
            duration=round(duration, 3),
            band=(int(band[0]), int(band[1])),
            rows=len(slots),
            swept=True,
            elapsed=round(time.perf_counter() - started, 3),
            language=language,
            meta={
                "frames_read": len(all_moments),
                "interval": interval,
                "survey_frames": survey_frames,
                "band": [int(band[0]), int(band[1])],
                "script": script,
                "lines_found": len(result_lines),
            },
        )
    finally:
        capture.release()
    return result
