# MASH-Mix-Audio-and-Text

**One speech engine for every device.** Analysis, stem extraction, song to lyric, text to speech, speech to text and speech to speech, running on Windows, Linux, macOS, Android and iOS from the same code.

**Bangla is not an afterthought here.** Bangladeshi Bangla (`bn-BD`) and Indian Bengali (`bn-IN`) are first class: separate language tags, separate text preparation, separate regional detection, and models that actually speak them.

```
pip install mash-mix-audio-and-text[full]
```

---

## Table of contents

- [Why this exists](#why-this-exists)
- [Install](#install)
- [Quick start](#quick-start)
- [Song to lyric](#song-to-lyric)
- [The Bangla layer](#the-bangla-layer)
- [Command line](#command-line)
- [Web interface and API](#web-interface-and-api)
- [Python API](#python-api)
- [Backends](#backends)
- [Platform support](#platform-support)
- [Architecture](#architecture)
- [Development](#development)
- [Licence](#licence)

---

## Why this exists

Most speech toolkits are tied to one hardware vendor. Apple Silicon gets an excellent
engine that is useless on Windows. Edge devices get a different engine with a different
API. Android and iOS get nothing but cloud calls.

This package fixes that with a **backend abstraction**. The task layer is written once.
The inference layer is pluggable, so the same call works on a Windows laptop, a Linux
server, a Mac, a phone running Termux, or an iPhone.

```
                     your code
                         |
              mash_mix.tasks.*  (one API)
                         |
            mash_mix.backends.registry
        /        |         |          |         \
   faster-   whisper   funasr   sherpa-onnx   whisper.cpp
   whisper                                         (mobile)
```

Nothing in the API layer knows which engine ran. That is also why the engine can fall
back: if `faster-whisper` is not installed, it uses `whisper`; if no GPU is present it
uses the CPU path; if only ONNX Runtime is available it uses the edge models.

---

## Install

The base package is small and has one dependency.

```bash
pip install mash-mix-audio-and-text

# with speech to text
pip install 'mash-mix-audio-and-text[stt]'

# with text to speech
pip install 'mash-mix-audio-and-text[tts]'

# with the web interface and API
pip install 'mash-mix-audio-and-text[server]'

# everything a desktop user wants
pip install 'mash-mix-audio-and-text[full]'
```

**ffmpeg is required** for anything other than WAV. It handles mp3, m4a, flac, ogg, opus,
aac, wma, and video containers, on every platform.

| Platform | Install ffmpeg |
|---|---|
| Windows | `winget install Gyan.FFmpeg` or `choco install ffmpeg` |
| macOS | `brew install ffmpeg` |
| Debian and Ubuntu | `sudo apt install ffmpeg` |
| Fedora | `sudo dnf install ffmpeg` |
| Arch | `sudo pacman -S ffmpeg` |
| Android (Termux) | `pkg install ffmpeg` |

Check what is working on your machine:

```bash
mash-mix doctor
```

---

## Quick start

```bash
# what does this machine support
mash-mix doctor

# extract lyrics from a Bangla song, write SRT and TXT
mash-mix lyrics "my song.mp3" -o lyrics-out --lang bn-BD

# transcribe a recording
mash-mix transcribe "interview.wav" -o transcript.txt --lang bn-BD

# speak some Bangla
mash-mix speak "আজকের দিনটা সুন্দর" -o hello.wav --lang bn-BD

# start the web interface
mash-mix serve --host 0.0.0.0 --port 8000
```

---

## Song to lyric

The headline feature. Drop in a song of any length and any format, get lyric lines
with timings, and download them as SRT or plain text.

```bash
mash-mix lyrics "bangla song.mp3" -o out --format srt --format txt --lang bn-BD
```

```
out/song.srt
out/song.txt
```

Or from Python:

```python
from mash_mix.tasks.lyrics import extract_lyrics_to_files

written = extract_lyrics_to_files(
    "bangla song.mp3",
    "out",
    formats=["srt", "txt"],
    language="bn-BD",
    title="Kobita tuku",
    artist="MASH",
)

result = written["result"]
print(result.language)   # bn-BD
print(result.text)      # the lyric lines
```

### What makes it work on songs

Singing voice is hard. Plain speech models repeat themselves, hallucinate lyrics over
instrumentals, and split words across a chorus. The pipeline handles each of those:

| Problem | What the pipeline does |
|---|---|
| Instrumentals get turned into words | Voice activity detection skips silent passages |
| The mix hides the voice | Optional vocal separation, `htdemucs` or the dependency free `mid_side` |
| The model loops on a sung phrase | Consecutive duplicate lines are collapsed |
| The model writes the same syllable fifty times | A line that is one repeated word is dropped |
| A chorus repeats the same line | Duplicate detection keeps the first occurrence and drops the rest |
| The model answers in Devanagari | Script repair maps it back to Bangla script |
| The model answers in romanised Bangla | The script layer writes it out in Bangla |
| The recogniser cuts through a sustained note | The engine's own voice activity detection is off for lyrics |
| Bangla needs different preparation | Bangla normalisation, then a Bangla clean pass on the output |
| Words get split mid phrase | Short fragments merge with the next line when the gap is small |
| A verse comes back as one long line | Lines are cut at the pauses, using the word timings |
| The base model writes Bangla badly | Use a Bangla fine tune, `mozilla-ai/whisper-large-v3-turbo-bn` |

### The model matters most

A general Whisper model writes Bangla poorly. The Bangla fine tunes of Whisper large v3
are far better, and they write the lyrics in Bangla script instead of romanised Bangla.

| Model | Word error rate on Bengali | Note |
|---|---|---|
| `openai/whisper-large-v3` | 55 percent | the general purpose model |
| `mozilla-ai/whisper-large-v3-bn` | 9.7 percent | the best Bangla accuracy available |
| `mozilla-ai/whisper-large-v3-turbo-bn` | 11 percent | four times faster, the recommended one |

```bash
mash-mix lyrics "bangla song.mp3" \
  --backend transformers \
  --model mozilla-ai/whisper-large-v3-turbo-bn
```

### Measuring the result

A lyric engine is easy to build and hard to trust, so the engine can be scored against
the real lyric file of a song.

```bash
python scripts/lyrics_eval.py "song.mp3" "true-lyrics.txt" --backend transformers \
  --model mozilla-ai/whisper-large-v3-turbo-bn
```

It prints the word error rate, the share of true lines that were found, and every line
that was missed, so a change can be measured rather than guessed.

---

## The Bangla layer

This is the part that makes the package more than a wrapper around a model.

### Language tags

Every standard identifier is registered, and `resolve` maps any of them to the canonical
BCP 47 tag.

| Code | Language | Native name | ISO 639-1 | ISO 639-2/B | ISO 639-2/T | ISO 639-3 | UN M.49 | CLDR | Script |
|---|---|---|---|---|---|---|---|---|---|
| `bn` | Bangla (Bengali) | বাংলা (বাঙালি) | bn | ben | beng | ben | | beng | Beng |
| `bn-BD` | Bangladeshi Bangla | বাংলা - বাংলাদেশ | bn | ben | beng | ben | 050 | bng | Beng |
| `bn-IN` | Indian Bengali | বাংলা - ভারত | bn | ben | beng | ben | 356 | beng | Beng |

```python
from mash_mix.lang import resolve, list_languages

resolve("beng")        # LanguageSpec(tag="bn", ...)
resolve("ben-bd")      # LanguageSpec(tag="bn-BD", ...)
resolve("bn_BD")       # LanguageSpec(tag="bn-BD", ...)
```

### Text normalisation

Bangla needs more than NFC. `normalize_bangla` keeps the characters that carry meaning
and folds the ones that do not.

```python
from mash_mix.lang import normalize_bangla

normalize_bangla("ড়  ঢ়  য়")     # canonical nukta decomposition
normalize_bangla("১২৩", digits="native")  # Bangla digits
normalize_bangla("১২৩", digits="ascii")   # 123
```

The important detail: **ZWNJ is kept** and **ZWJ is dropped**. Bangla conjuncts are
inherent in the script, so `ZWJ` carries no meaning there, while `ZWNJ` marks a real
boundary between words. Generic Unicode cleaners get this wrong.

### Numbers read in Bangla

Bangla groups by lakh and crore, not by thousand and million.

```python
from mash_mix.lang import number_to_bangla_words

number_to_bangla_words(12345678)
# এক কোটি তেইশ লক্ষ পঁয়তাল্লিশ হাজার ছয় শত আশি
```

### Romanised Bangla in, Bangla script out

```python
from mash_mix.lang import banglish_to_bangla

banglish_to_bangla("ami tomay bhalobashi")
# আমি তোমায় ভালোবাসি
```

### Bangla or Banglish, the caller decides

Every task that produces text takes a `script` argument. `bangla` keeps the native
script, `banglish` writes the same content in Latin letters.

```python
from mash_mix import extract_lyrics

bangla = extract_lyrics("song.mp3", script="bangla")
banglish = extract_lyrics("song.mp3", script="banglish")
```

```bash
mash-mix lyrics "song.mp3" -o out --script banglish
mash-mix transcribe "speech.wav" --script banglish
mash-mix speak "ami tomay bhalobashi" --script bangla
```

The same choice fixes the two problems that come with romanised Bangla in a Bangla
pipeline:

- a recogniser that answers in romanised Bangla is written out in Bangla, because a
  Bangla voice reads Latin letters as English;
- a user who types Banglish into a Bangla text to speech command still gets Bangla
  speech, because the text is converted before it is spoken.

### Regional detection

Bangla is one language with two standard registers. The detector says which one it is
found, and shows its working.

```python
from mash_mix.lang import detect_region

report = detect_region("ভাই, আপ্নে কেমন আছেন? টাকা পাঠাতে হবে।")
report.tag          # bn-BD
report.confidence   # 0.0 to 1.0
report.evidence     # ["BD:আপ্নে x1", "BD:টাকা x1", ...]
```

It also finds the district dialect cluster: `sylheti`, `chattogram`, `dhaka`, `barishal`,
`rajshahi`, `khulna`, `rangpur`, `mymensingh`, `comilla`. All 64 districts of Bangladesh
are in the lexicon with their Bangla names.

### Script repair

A Whisper model told to recognise Bangla will sometimes answer in Devanagari,
because the two scripts sound the same and the model drifts. The engine
repairs that, so the output is always Bangla script.

```python
from mash_mix.lang import devanagari_to_bangla, dominant_script

devanagari_to_bangla("आमार सोनार बांला")
# আমার সোনার বাংলা
dominant_script("আমার সোনার বাংলা")   # bangla
dominant_script("आमार सोनार बांला")  # devanagari
```

The repair is applied automatically by the lyrics pipeline whenever Bangla was
requested.

### Coverage

| Area | Coverage |
|---|---|
| Script | Bengali `U+0980` to `U+09FF`, including the Assamese range `U+09F0` to `U+09F3` |
| Digits | Bangla `০১২৩৪৫৬৭৮৯`, plus ASCII |
| Punctuation | দাড়ি `।`, ঃ, ৳, and the full ASCII set |
| Nukta letters | ড় ঢ় য়, decomposed and recomposed |
| Joining | ZWNJ kept, ZWJ dropped, ZWSP turned into a space |
| Numerals | Ones, tens, hundreds, thousand, lakh, crore, decimals, negatives |
| Dates and times | Gregorian and Bengali calendar month names, 12 and 24 hour forms |
| Symbols | `%` to শতাংশ, `&` to এবং, `৳` to টাকা, and the rest |
| Stop words | Separate lists for Bangladeshi and Indian Bangla |
| Honorifics | ভাই, আপ্নে, মামা, চাচা for Bangladesh, দা, দি, আপনি for India |
| Dialects | Nine clusters, 64 districts |

---

## Command line

```
mash-mix lyrics        Extract lyrics from a song or spoken track
mash-mix transcribe    Transcribe speech to text
mash-mix speak         Synthesize speech from text
mash-mix sts           Speech to speech
mash-mix analyze       Measure tempo, key, loudness and sections
mash-mix separate      Split a track into stems
mash-mix text          Bangla text tools
mash-mix languages     List the supported language tags
mash-mix backends      List the backends and what is installed
mash-mix models        List the model catalogue with licences
mash-mix doctor        Check this machine
mash-mix serve         Start the web interface and the HTTP API
```

The text tools are useful on their own:

```bash
mash-mix text numbers 12345678
mash-mix text normalize --digits native
mash-mix text translit "ami tomay bhalobashi"
mash-mix text detect "আপনি কেমন আছেন"
mash-mix models --bengali --task tts
```

---

## Web interface and API

```bash
mash-mix serve --host 0.0.0.0 --port 8000
```

Open `http://localhost:8000`. Drop in an audio file, press **Extract lyrics**, then
download the `.srt` or the `.txt`.

The page shows the attached file as soon as it is chosen: the name, the size, the type
and the length, so there is never any doubt about whether the file arrived. When the
engine has opened the file, the card also shows what the server found: the container,
the sample rate and the number of channels.

Next to the language there is an **Output script** choice, `Bangla` or `Banglish`, so
the caller decides which writing system the lyrics come back in.

| Route | Method | What it does |
|---|---|---|
| `/` | GET | The web interface |
| `/api/lyrics` | POST | Upload an audio file, get lyric lines and download links |
| `/api/lyrics/{id}/download/{fmt}` | GET | Download `srt`, `txt`, `vtt` or `lrc` |
| `/api/transcribe` | POST | Transcribe an upload |
| `/api/speech` | POST | OpenAI style text to speech, returns audio |
| `/api/audio/transcriptions` | POST | OpenAI style speech to text |
| `/api/languages` | GET | The language registry |
| `/api/models` | GET | The model catalogue |
| `/api/health` | GET | Liveness and what is installed |

---

## Python API

```python
from mash_mix import MASHMix

engine = MASHMix(language="bn-BD")

lyrics = engine.lyrics("song.mp3")
engine.write(lyrics, "out", formats=["srt", "txt"])

transcript = engine.transcribe("interview.wav")
audio = engine.speak("আজকের দিনটা সুন্দর")
report = engine.analyze("song.mp3")
```

---

## Backends

| Task | Backend | Needs | Bangla |
|---|---|---|---|
| STT | `faster-whisper` | `faster-whisper` | yes, best default |
| STT | `whisper` | `openai-whisper` | yes |
| STT | `funasr` | `funasr` | yes |
| STT | `transformers` | `transformers` | yes |
| STT | `sherpa-onnx` | `sherpa-onnx` | yes, lightest |
| STT | `whisper.cpp` | `pywhispercpp` | yes, smallest |
| TTS | `mms-tts` | `transformers` | yes |
| TTS | `vits` | `TTS` | yes, best for Bangladeshi Bangla |
| TTS | `indic-tts` | `transformers` | yes |
| TTS | `parler-tts` | `transformers` | yes |
| TTS | `piper` | `piper-tts` | yes, lightest |
| TTS | `xtts` | `TTS` | yes, with voice cloning |
| TTS | `chatterbox` | `chatterbox-tts` | no |
| TTS | `edge-tts` | `edge-tts` | yes, online only |
| STS | `pipeline` | an STT and a TTS backend | yes |

Run `mash-mix backends` to see what is installed on your machine, and
`mash-mix models` in the API for the full catalogue with licences.

---

## Platform support

| Platform | Speech to text | Text to speech | How |
|---|---|---|---|
| Windows 10 and 11 | yes | yes | `faster-whisper`, `mms-tts` |
| Linux and servers | yes | yes | `faster-whisper`, `mms-tts`, GPU optional |
| macOS | yes | yes | `faster-whisper`, MPS for local models |
| Android (Termux) | yes | yes | `sherpa-onnx`, `piper`, ONNX Runtime |
| iOS and iPadOS | yes | yes | ONNX Runtime Mobile, CoreML export |
| Raspberry Pi | yes | yes | `sherpa-onnx`, `whisper.cpp` |

Android notes:

```bash
pkg install python ffmpeg rust
pip install 'mash-mix-audio-and-text[edge]'
```

iOS notes: the `edge` extra gives ONNX Runtime Mobile. Convert the models with
`onnxruntime` tooling and run the same task functions inside the app.

---

## Architecture

```
src/mash_mix/
  lang/          language registry, normalisation, numerals, transliteration, dialects
  audio/         decoding, resampling, VAD, chunking, analysis, separation
  backends/      the inference adapters, one folder per task
    asr/         speech to text
    tts/         text to speech
    sts/         speech to speech
  tasks/         transcribe, lyrics, synthesize, speak, analyze
  export/        SRT, WebVTT, LRC, ASS, plain text, JSON
  server/        FastAPI app and the web interface
  pipeline.py    one facade over everything
  models.py      the model catalogue with licences
  doctor.py      the machine check
```

The design rules:

1. **No hard dependency on a runtime.** Every backend module imports cleanly without its
   package installed, and reports itself unavailable instead of crashing.
2. **One API for every task.** The task layer never imports a backend directly. It asks
   the registry for one.
3. **Bangla logic lives in one place.** No backend file contains Bangla rules. They live
   in `lang/`, so every backend gets them for free.
4. **No size limits.** Audio is decoded in chunks and written in chunks, so a three hour
   song is not a memory problem.

---

## Development

```bash
git clone https://github.com/MASH-ai/MASH-Mix-Audio-and-Text.git
cd MASH-Mix-Audio-and-Text
pip install -e '.[full,dev]'
pytest
ruff check src tests
```

The test suite runs without any model weights. Tests that need weights are marked and
skipped by default:

```bash
pytest -m model
```

The self check scripts live in `scripts/`.

| Script | What it does |
|---|---|
| `lyrics_eval.py` | Scores extracted lyrics against the real lyric file of a song |
| `convert_model.py` | Converts a Whisper checkpoint into the CTranslate2 format |
| `verify_lexicon.py` | Checks the curated romanised Bangla dictionary |
| `verify_bangla_maps.py` | Checks the Bangla and Devanagari mapping tables |
| `crosscheck.py` | Runs the whole pipeline over a test track |
| `server_check.py` | Runs the HTTP server and the web interface end to end |
| `sts_check.py` | Runs the speech to speech chain |
| `cuda_probe.py` | Tells the engine whether the GPU path works on this machine |

---

## Licence

MIT. Copyright (c) 2026 MASH (Khaled Bhuiyan).

The models themselves keep their own licences. Several of the best Bangla voices are
released under non commercial terms, so check `mash-mix models` before shipping a
product. `mash-mix doctor` prints the licences of the models it finds locally.
