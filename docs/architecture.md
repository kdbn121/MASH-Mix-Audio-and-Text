# Architecture

How the package is layered, how a request travels from your call to a backend and back, and how to add a backend of your own.

## The layers

```
                         your code
                             |
                 mash_mix.pipeline.MASHMix        one facade, one configuration
                             |
                 mash_mix.tasks.*                 transcribe, lyrics, synthesize,
                             |                    sts, analyze  (the user facing verbs)
             +---------------+--------------------+
             |               |                    |
             v               v                    v
     mash_mix.lang.*   mash_mix.audio.*    mash_mix.backends.registry
     registry          io, resample,       picks one backend, imports lazily
     normalize         vad, chunk,                 |
     numerals          analyze, separate          v
     translit                  |           mash_mix.backends.asr
     dialect                   |           mash_mix.backends.tts
     lexicon                   |           mash_mix.backends.sts
             |                 |                    |
             +-----------------+--------------------+
                             |
                    mash_mix.export.*            subtitles, plaintext
                             |
                    mash_mix.server.app         FastAPI routes and the web page
```

| Layer | Package | Responsibility | Depends on |
|---|---|---|---|
| Language | `mash_mix.lang` | Language tags, text normalisation, numerals, transliteration, regional detection | nothing but the standard library |
| Audio | `mash_mix.audio` | Decoding, encoding, resampling, voice activity detection, chunking, measurement, stem separation | numpy, optionally soundfile, ffmpeg, librosa, demucs |
| Backends | `mash_mix.backends` | Thin adapters over one inference stack each | the stack itself, but only optionally |
| Tasks | `mash_mix.tasks` | The verbs: transcribe, lyrics, synthesize, speech to speech, analyse | lang, audio, backends |
| Export | `mash_mix.export` | SRT, WebVTT, LRC, ASS, plain text, JSON | types only |
| Server | `mash_mix.server` | HTTP API and the web interface | tasks, export |
| Pipeline | `mash_mix.pipeline` | `MASHMix`, one configured engine over every task | everything else, lazily |

The rule the layers enforce is that arrows only point downwards. The language layer never imports audio, audio never imports backends, and the task layer never imports a backend module directly. It asks the registry for one.

## The four design rules

These come straight from the README and they explain most of the design decisions.

1. **No hard dependency on a runtime.** Every backend module imports cleanly without its package installed, and reports itself unavailable instead of crashing.
2. **One API for every task.** The task layer never imports a backend directly. It asks the registry for one.
3. **Bangla logic lives in one place.** No backend file contains Bangla rules. They live in `lang/`, so every backend gets them for free.
4. **No size limits.** Audio is decoded in chunks and written in chunks, so a three hour song is not a memory problem.

## How a request flows

The lyrics task is the clearest example, so here it is end to end.

**1. The call.** You call the task function.

```python
from mash_mix.tasks.lyrics import extract_lyrics

result = extract_lyrics("song.mp3", language="bn-BD")
```

**2. The audio layer decodes.** `tasks.lyrics.extract_lyrics` calls `tasks._common.to_buffer`, which calls `mash_mix.audio.io.load` and then `mash_mix.audio.resample.resample_buffer`. The result is an `AudioBuffer` with mono float32 samples at 16000 Hz.

**3. Optional separation.** When `separate_vocals=True` the task calls `mash_mix.audio.separate.separate`, and replaces the buffer with the `vocals` stem. A failure here is not fatal: the original mix is kept.

**4. The registry picks a backend.** The task calls `tasks._common.pick_backend("stt", backend)`. With `backend="auto"` this walks the preference list in `_common._STT_PREFERENCE` and returns the first backend that reports itself available. With an explicit name the name is used as it is.

**5. The backend is loaded.** `backends.registry.get_stt(name, model=model)` instantiates the backend class, checks `is_available()`, calls `load(model)` and returns the instance.

**6. The audio is chunked.** `mash_mix.audio.chunk.chunk_audio` splits the buffer into overlapping windows. Each chunk is a plain dictionary with `index`, `start`, `end`, `samples`, `start_sample` and `end_sample`.

**7. Each chunk is recognised.** The task calls `engine.transcribe(samples, sample_rate, language=..., word_timestamps=True, vad=True, ...)`. The backend returns a `Transcription` with `Segment` objects and optional `Word` timings. The task shifts every timestamp by the chunk offset so the times are absolute in the original track.

**8. The text is cleaned.** Every segment text goes through `lang.normalize.clean_asr_text`, which removes bracket noise such as `[আড়ালা]`, collapses repeated punctuation and tidies whitespace. If the result is mostly Bangla script it also goes through `lang.normalize.normalize_bangla`.

**9. Repetition control.** `lyrics._drop_repeated_lines` removes decoding loops, and `lyrics._merge_short_lines` joins very short fragments with the next line when the gap is small.

**10. The region is detected.** `lang.dialect.detect_region` runs over the joined text and reports the tag, the confidence and the evidence.

**11. The result is returned.** A `LyricsResult` with the lyric lines, the language, the duration, the backend name, the model name and the detection evidence in `meta`.

**12. Export.** `export.subtitles.to_srt` renders the lines, `export.plaintext.write_text` writes the plain text, and `pipeline.MASHMix.write` does both in one call.

```python
from mash_mix import MASHMix

engine = MASHMix(language="bn-BD")
lyrics = engine.lyrics("song.mp3")
engine.write(lyrics, "out", formats=["srt", "txt"])
```

## The data model

Every result object lives in `mash_mix.types` and every one of them has a `to_dict()` that returns plain builtins only, so a result can be sent over HTTP or written to a report without extra plumbing.

| Class | Fields | JSON friendly |
|---|---|---|
| `AudioBuffer` | `samples` (numpy), `sample_rate`, `channels` | No. `to_dict()` gives the shape and duration, not the audio |
| `Word` | `start`, `end`, `text`, `probability` | Yes |
| `Segment` | `start`, `end`, `text`, `words`, `speaker`, `language`, `probability` | Yes |
| `Transcription` | `text`, `language`, `duration`, `segments`, `backend`, `model`, `elapsed`, `meta` | Yes |
| `LyricLine` | `start`, `end`, `text`, `speaker`, `probability` | Yes |
| `LyricsResult` | `lines`, `language`, `duration`, `title`, `artist`, `backend`, `model`, `elapsed`, `meta` | Yes |
| `SynthesisResult` | `audio` (numpy), `sample_rate`, `backend`, `model`, `voice`, `elapsed`, `meta` | No. `to_dict()` gives the metadata, not the waveform |
| `SeparationResult` | `stems` (dict of numpy), `sample_rate`, `backend`, `model`, `elapsed`, `meta` | No. `to_dict()` gives the frame count of each stem |
| `AnalysisResult` | `duration`, `sample_rate`, `channels`, `bpm`, `key`, `mode`, `loudness_lufs`, `peak_dbfs`, `rms_dbfs`, `spectral_centroid`, `sections`, `beats`, `clipped_samples`, `dc_offset_percent`, `head_silence_seconds`, `crest_db`, `backend`, `meta` | Yes |

The rule of thumb: a result that carries audio is not JSON friendly, but its metadata is. The `to_dict()` method of the three audio carrying classes returns only the descriptive fields.

`AudioBuffer` keeps the sample rate next to the samples, so no backend ever has to guess the tempo of an array it was handed. A one dimensional array is treated as mono and is stored as shape `(1, frames)`. `mono()` averages a multi channel buffer down to a single float32 array.

## Adding a new backend

A backend is a thin adapter. It must import cleanly without its stack installed, report its own availability, and keep a stable call signature so the task layer never needs to know which engine ran.

Write the module under `src/mash_mix/backends/asr/`, `tts/` or `sts/`, then register it.

```python
# src/mash_mix/backends/asr/my_asr.py
"""My own speech to text backend."""

from __future__ import annotations

from typing import Any, Dict, List, Optional, Sequence

import numpy as np

from ...errors import BackendUnavailableError
from ...types import Segment, Transcription, Word
from ..base import BackendInfo, STTBackend
from ..registry import register_stt

try:  # the guard is not optional
    import my_asr_package  # type: ignore

    _HAS_MY_ASR = True
except Exception:  # pragma: no cover - a missing stack must not break the import
    my_asr_package = None  # type: ignore
    _HAS_MY_ASR = False


@register_stt("my-asr")
class MyAsrSTTBackend(STTBackend):
    """Speech to text with my own runtime."""

    info = BackendInfo(
        name="my-asr",
        kind="stt",
        summary="One line description shown by mash-mix backends.",
        package="my-asr-package",
        extra="stt",
        languages=["bn", "bn-BD", "bn-IN", "en"],
        platforms=["windows", "linux", "macos"],
        notes="Anything the user should know before choosing this backend.",
    )

    def __init__(self) -> None:
        super().__init__()
        self._model: Any = None

    def is_available(self) -> bool:
        """True when the runtime dependencies are importable."""
        try:
            return bool(_HAS_MY_ASR and my_asr_package is not None)
        except Exception:  # pragma: no cover - defensive, must never raise
            return False

    def load(self, model: Optional[str] = None, **kwargs: Any) -> None:
        """Load the weights. Safe to call more than once."""
        if self._loaded:
            return
        if not self.is_available():
            raise BackendUnavailableError(
                "The 'my-asr' backend needs the my-asr-package package. "
                "Install it with: pip install 'mash-mix-audio-and-text[stt]'"
            )
        model_name = str(model or kwargs.get("model_id") or "small")
        try:
            self._model = my_asr_package.load(model_name)
        except Exception as exc:
            raise BackendUnavailableError(
                "Could not load the my-asr model '{0}': {1}".format(model_name, exc)
            ) from exc
        self._model_id = model_name
        self._loaded = True

    def unload(self) -> None:
        """Drop the loaded weights."""
        self._model = None
        self._loaded = False

    def transcribe(
        self,
        audio: np.ndarray,
        sample_rate: int,
        *,
        language: Optional[str] = None,
        task: str = "transcribe",
        initial_prompt: Optional[str] = None,
        hotwords: Optional[Sequence[str]] = None,
        word_timestamps: bool = True,
        vad: bool = True,
        progress: bool = False,
        **kwargs: Any,
    ) -> Transcription:
        """Transcribe mono float32 audio sampled at sample_rate."""
        if not self._loaded:
            self.load()

        data = np.asarray(audio, dtype=np.float32).reshape(-1)
        raw = self._model.run(
            data,
            sample_rate,
            language=language,
            prompt=initial_prompt,
            hotwords=list(hotwords) if hotwords else None,
        )

        segments: List[Segment] = []
        for item in raw:
            words: List[Word] = []
            if word_timestamps:
                words = [
                    Word(
                        start=float(w["start"]),
                        end=float(w["end"]),
                        text=str(w["text"]),
                        probability=w.get("probability"),
                    )
                    for w in item.get("words", [])
                ]
            segments.append(
                Segment(
                    start=float(item["start"]),
                    end=float(item["end"]),
                    text=str(item["text"]),
                    words=words,
                    language=language,
                )
            )

        return Transcription(
            text=" ".join(seg.text for seg in segments),
            language=language or "und",
            duration=float(len(data)) / float(sample_rate),
            segments=segments,
            backend=self.name,
            model=self._model_id or "",
            meta={"word_timestamps": bool(word_timestamps), "vad": bool(vad)},
        )
```

Then add the module to `_DISCOVERY_MODULES` in `src/mash_mix/backends/registry.py`, and to `_STT_PREFERENCE` in `src/mash_mix/tasks/_common.py` if it should win the automatic selection.

The contract to respect:

| Method | Rule |
|---|---|
| `is_available()` | Must never raise. Return `False` when the stack is missing. |
| `load(model)` | Must be safe to call twice. Raise `BackendUnavailableError` when the model cannot be loaded. |
| `unload()` | Optional. Release the weights and set `self._loaded = False`. |
| `transcribe(...)` | Receives mono float32 samples and the sample rate. Returns a `Transcription`. |
| `info` | A `BackendInfo` with `name`, `kind`, `summary`, `package`, `extra`, `languages`, `platforms` and `notes`. |

Word timestamps are optional but strongly preferred, because the lyric pipeline needs them for subtitles.

## Thread safety

The package is safe to use from several threads, with two caveats.

What is safe:

- The language layer is pure. `normalize_bangla`, `number_to_bangla_words`, `banglish_to_bangla` and `detect_region` read their arguments and return new values. They hold no state, so they can be called concurrently without a lock.
- The registry tables `_STT`, `_TTS` and `_STS` are only written during `discover()`, which imports modules. After that they are read only.
- `get_stt`, `get_tts` and `get_sts` build a new backend instance for every call, so two threads never share a model object.

What is not safe:

- A single backend instance is not thread safe. Do not call `transcribe` on the same instance from two threads. Use one instance per thread, or a lock.
- The server keeps finished jobs in the module level `_JOBS` dictionary in `server/app.py`. That is safe inside one process, but it is not shared between uvicorn workers. Run the server with one worker if you rely on the job id, or download the files before the worker that created them restarts.
- The language lexicon and the numeral tables are module level constants. They are never mutated after import.

## Lazy import strategy

Importing `mash_mix` must stay cheap, because the package is used on phones where start up time matters. Three mechanisms keep it cheap.

**1. The package re-exports through `__getattr__`.** `mash_mix/__init__.py` does not import the tasks at module level. The names `transcribe`, `extract_lyrics`, `synthesize`, `speak` and `analyze` are resolved on first use.

```python
import mash_mix   # cheap, no heavy imports
mash_mix.transcribe("a.wav")   # the task module is imported now
```

**2. The backend modules guard their own imports.** Every backend wraps its optional import in a try and except and sets a flag. The module still imports when the stack is missing, which is what design rule 1 asks for.

```python
try:
    from faster_whisper import WhisperModel
    _HAS_FASTER_WHISPER = True
except Exception:
    WhisperModel = None
    _HAS_FASTER_WHISPER = False
```

**3. The registry discovers on demand.** `backends.registry.discover()` imports the backend modules the first time something asks for a backend, and the task modules import the registry inside the function body rather than at module level. A machine with no inference stack installed still starts, still runs `mash-mix doctor`, still runs the language tools, and still exports subtitles.

The result is that `import mash_mix` needs only numpy, and everything else is paid for at the moment it is used.

---

Copyright (c) 2026 MASH (Khaled Bhuiyan). MIT Licence. See LICENSE.
