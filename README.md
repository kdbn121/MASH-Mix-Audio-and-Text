# MASH-Mix-Audio-and-Text

**One speech engine for every device.** Analysis, stem extraction, song to lyric, text to speech, speech to text and speech to speech, running on Windows, Linux, macOS, Android and iOS from the same code.

**Bangla is not an afterthought here.** Bangladeshi Bangla (`bn-BD`) and Indian Bengali (`bn-IN`) are first class: separate language tags, separate text preparation, separate regional detection, and models that actually speak them.

```
pip install "mash-mix-audio-and-text[full] @ git+https://github.com/kdbn121/MASH-Mix-Audio-and-Text.git"
```

---

## Table of contents

- [Why this exists](#why-this-exists)
- [Install](#install)
- [Quick start](#quick-start)
- [Song to lyric](#song-to-lyric)
- [A link instead of a file](#a-link-instead-of-a-file)
- [The subtitles the page already carries](#the-subtitles-the-page-already-carries)
- [The words on the picture of a karaoke](#the-words-on-the-picture-of-a-karaoke)
- [The language of the audio](#the-language-of-the-audio)
- [The report of a run](#the-report-of-a-run)
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

The base package is small and has one dependency. It is not on PyPI, so it is
installed from the repository.

```bash
pip install "mash-mix-audio-and-text @ git+https://github.com/kdbn121/MASH-Mix-Audio-and-Text.git"

# with speech to text
pip install "mash-mix-audio-and-text[stt] @ git+https://github.com/kdbn121/MASH-Mix-Audio-and-Text.git"

# with text to speech
pip install "mash-mix-audio-and-text[tts] @ git+https://github.com/kdbn121/MASH-Mix-Audio-and-Text.git"

# with the web interface and API
pip install "mash-mix-audio-and-text[server] @ git+https://github.com/kdbn121/MASH-Mix-Audio-and-Text.git"

# with the fetcher for links
pip install "mash-mix-audio-and-text[sources] @ git+https://github.com/kdbn121/MASH-Mix-Audio-and-Text.git"

# everything a desktop user wants
pip install "mash-mix-audio-and-text[full] @ git+https://github.com/kdbn121/MASH-Mix-Audio-and-Text.git"
```

The source archive of every release is on the
[release page](https://github.com/kdbn121/MASH-Mix-Audio-and-Text/releases),
for anyone who would rather not install from git.

The same package drives the [Chrome extension](https://github.com/kdbn121/MASH-Mix-Audio-and-Text-Extension),
which measures a song in the browser and sends the file to this engine only
when you ask it for the lyrics.

**ffmpeg is required** for anything other than WAV. It handles mp3, m4a, flac, ogg, opus,
aac, wma, and video containers, on every platform.

**yt-dlp is required for a link.** The `sources` extra installs it. It reads the public
pages of the sites and hands back the audio track, so a URL can be used where a file was
used. The engine works without it, and only a request for a link needs it.

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
| The model writes the same syllable fifty times | A line that repeats is dropped, word by word and character by character |
| A chorus repeats the same line | Duplicate detection keeps the first occurrence and drops the rest |
| The model answers in Devanagari | Script repair maps it back to Bangla script |
| The model answers in romanised Bangla | The script layer writes it out in Bangla |
| The recogniser cuts through a sustained note | The engine's own voice activity detection is off for lyrics |
| Bangla needs different preparation | Bangla normalisation, then a Bangla clean pass on the output |
| Words get split mid phrase | Short fragments merge with the next line when the gap is small |
| A verse comes back as one long line | Lines are cut at the punctuation and at the pauses |
| A broken decode leaves a replacement character | That line is dropped, no lyric contains one |
| The base model writes Bangla badly | Use a Bangla fine tune, `mozilla-ai/whisper-large-v3-turbo-bn` |

### The model matters most

A general Whisper model writes Bangla poorly. The Bangla fine tunes of Whisper large v3
are far better, and they write the lyrics in Bangla script instead of romanised Bangla.

| Model | Word error rate on Bengali | Note |
|---|---|---|
| `openai/whisper-large-v3` | 55 percent | the general purpose model |
| `mozilla-ai/whisper-large-v3-bn` | 9.7 percent | the best Bangla accuracy available |
| `mozilla-ai/whisper-large-v3-turbo-bn` | 11 percent | four times faster, the recommended one |

**The engine no longer starts at the small model.** The small model was the default
once, and a Bangla song read by it comes back mangled: the words run into each other
and the line comes out in romanised Bangla even when the native script was asked for.
That was the single biggest reason the lyrics were not good enough.

The engine now walks a ladder of models and uses the first one that works:

| Rung | Model | Why |
|---|---|---|
| 1 | `faster-whisper` with `large-v3` | the best read of a real song, and the quickest of the strong ones |
| 2 | `transformers` with `mozilla-ai/whisper-large-v3-turbo-bn` | the Bangla fine tune, second because it needs more memory |
| 3 | `faster-whisper` with `small` | the old default, kept as the floor |
| 4 | `faster-whisper` with `tiny` | the last resort |

**A model that cannot be loaded does not end the job.** The strong model needs room in
memory and a download that may not be there. When one cannot be loaded, the engine
moves to the next one and says so in the result, so a request always comes back with
lyrics rather than with an error. The result carries the model that did the work:

```python
result.meta["model_used"]     # the model that read the song
result.meta["model_fallbacks"]  # why the ones above it were skipped
```

The three quality levels choose where the ladder starts:

```bash
mash-mix lyrics "bangla song.mp3" --quality best    # the strongest model
mash-mix lyrics "bangla song.mp3" --quality fast    # the quick one
mash-mix lyrics "bangla song.mp3" --quality auto    # the default
```

A model can still be named outright, which puts it alone on the ladder:

```bash
mash-mix lyrics "bangla song.mp3" \
  --backend transformers \
  --model mozilla-ai/whisper-large-v3-turbo-bn
```

### Reading the song in one pass

A Whisper model reads a whole track on its own, with its own windowing and its own
voice activity detection, and it keeps the thread of the song as it goes. The engine
hands the whole track to the model in one call for that reason.

The chunked path is kept for the models that cannot take a long file. It is the one
that asks the model the same question again and again with no memory of what came
before, and a song is exactly where that shows.

### The hosted recognisers

Four backends read the audio somewhere other than this machine. They are never reached
on their own, because the audio has to leave the machine to reach them, so the caller
has to name one.

**ElevenLabs Scribe** is the strongest hosted recogniser for Bangla. It needs an
ElevenLabs account and the key in the environment:

```bash
# Windows PowerShell
$env:ELEVENLABS_API_KEY = "your key here"
mash-mix lyrics "bangla song.mp3" --backend elevenlabs
```

Without a key the backend reports itself unavailable and stays out of the way. It
gives the time of every word, so the karaoke lines keep their timings.

**Azure Fast Transcription** hears many languages well without a download, and it can
name the speakers without anything but the Azure key:

```bash
$env:AZURE_SPEECH_KEY = "your key here"
$env:AZURE_SPEECH_REGION = "eastus"
mash-mix lyrics "bangla song.mp3" --backend azure-stt
```

**DashScope Qwen3 ASR** gives the time of every word. It fetches the audio itself, so
it needs the address of the file rather than the file:

```bash
$env:DASHSCOPE_API_KEY = "your key here"
mash-mix transcribe "song.wav" --backend dashscope-asr --audio-url "https://example.com/song.wav"
```

**Meta SeamlessM4T** is the multilingual model from Meta. It reads Bengali as well as
most languages and it runs locally, but it needs a large download and a lot of memory:

```bash
mash-mix lyrics "bangla song.mp3" --backend seamless-m4t
```

It gives the time of a sentence and not the time of a word, so the local Whisper
backends are the better choice when a karaoke line needs the time of a word inside it.

### Naming the speakers

**WhisperX** aligns the words of a transcript to the sound, so the time of every word
is exact rather than approximate, and it can tell who is speaking:

```bash
pip install "mash-mix-audio-and-text[align]"
mash-mix lyrics "interview.wav" --backend whisperx
```

The speaker part needs a model from Hugging Face, so it needs a token in `HF_TOKEN`.
Without a token the words are still aligned and the speakers are left empty.

### Measuring the result

A lyric engine is easy to build and hard to trust, so the engine can be scored against
the real lyric file of a song.

```bash
python scripts/lyrics_eval.py "song.mp3" "true-lyrics.txt" --backend transformers \
  --model mozilla-ai/whisper-large-v3-turbo-bn
```

It prints three numbers and every line that was missed, so a change can be measured
rather than guessed.

| Number | What it measures |
|---|---|
| Word error rate | every word in the transcript against every word in the true lyric |
| Line recall | true lines that came back as one whole line |
| Content recall | true lines found somewhere in the transcript, wherever the line breaks fall |

Content recall is the one to read. The line breaks in a transcript are decided by
the segmenter, not by the recogniser, so a lyric that is correct but written as one
long line would score badly on line recall and well on content recall.

#### A measured song

The song *Dol Dol Doloni* by the folk group Paban, 4 minutes 22 seconds, with the
real lyric file as the truth, on a CPU with no graphics card. Every row is the same
scorer over the same truth file.

| Setup | Word error rate | Line recall | Content recall | Time |
|---|---|---|---|---|
| General model, raw mix | 97.4 percent | 22.8 percent | 22.8 percent | 431 s |
| Bangla fine tune, raw mix, 0.1.1 pipeline | 77.1 percent | 0.0 percent | 64.9 percent | 367 s |
| Bangla fine tune, raw mix, 0.1.2 pipeline | 76.2 percent | 26.3 percent | 71.9 percent | 324 s |
| Bangla fine tune, vocals separated, 0.1.2 pipeline | 71.9 percent | 3.5 percent | 47.4 percent | 207 s |

Three lessons are worth keeping.

**The Bangla fine tune is the whole game.** The general model wrote 55 percent wrong
words on Bengali, the fine tune writes about 10 percent. On this song that is the
difference between 22.8 percent and 64.9 percent content recall.

**The pipeline moves the line shape.** The 0.1.2 pipeline cut the lines at the
punctuation the model wrote, so a lyric that used to come back as one long run of
words now comes back as the lines of the song. That is the jump from 0 percent to
26.3 percent line recall.

**Separation is a trade off, not a win.** On the separated voice the words are the
most accurate of the four, 71.9 percent word error rate, and the run is the fastest
because the model has less to listen to. But the separated transcript covers fewer of
the distinct lines, 47.4 percent against 71.9 percent, because Demucs also took the
room reverb and the backing vocal with the voice. Run both and keep the better one.

The word error rate stays high even when the content recall is good, because the
true lyric file lists the chorus nine times and a transcript that catches it four
times is already short of the file, not wrong.

#### The same song, read by the new default

The table above was measured against the old default, the small model. The same song
was read again with the new ladder, and the difference is the whole point of this
release.

| Setup | What came back |
|---|---|
| Old default, `small` model, chunked | `Nudon uton khopati`, `Dhododoloni`, `Ranga mathar chironi`, `Ene debo haath heke man tumi korona` |
| New default, `large-v3`, one pass | `নুতন নুতন হোপাতি তুলে এনে দোপাতি`, `দোল দোল দোলনি রাঙা মাথার ছিরনি`, `এনে দেবো হাত থেকে মান তুমি করো না` |

The old default read the song in romanised Bangla and got the words wrong. The new
default read the same chorus in the native script and got the words right, and it
found the lines the old one missed further into the song.

Two lessons came out of the measurement.

**The model was the problem, not the pipeline.** The filters, the line splitter and
the script repair were all working. They had nothing to work with, because the small
model had already mangled the words before they reached them.

**Reading the song in one pass matters.** The model that reads a whole track keeps
the thread of the song across the whole track. The chunked path asks the same
question again with no memory of what came before, and a song is where that shows.

---

## A link instead of a file

A link is a way of naming a piece of audio. The engine can fetch the audio behind one,
read it and write the lyric files, in the same way as a file that was dropped in.

```bash
mash-mix lyrics "https://www.youtube.com/watch?v=AU0mxhZcL1w" -o out --lang auto
```

```
out/Bhober Ei.srt
out/Bhober Ei.txt
```

Or from Python:

```python
from mash_mix.tasks.lyrics import extract_lyrics_from_url

outcome = extract_lyrics_from_url(
    "https://www.youtube.com/watch?v=AU0mxhZcL1w",
    "out",
    language="auto",
)

result = outcome["result"]
source = outcome["source"]
print(source.title)         # the name the page gave the song
print(source.language)      # the language of it
print(source.language_source)  # where that language came from
```

The fetching is done with `yt-dlp`, which reads the public pages of the sites. It is an
optional dependency, so the engine imports and works without it and only a request for a
link needs it.

**What comes back with the audio.** The engine reports what the page said about itself,
and what it worked out for itself, apart from one another:

| Field | What it means |
|---|---|
| `title` | The name the page gave the song |
| `uploader` | Who put it there |
| `duration` | How long it is, in seconds |
| `declared_language` | The language the page named, when it named one |
| `guessed_language` | The language read from the script of the title and the description |
| `script` | The writing system of the title, for example `Bengali` |
| `language_source` | `declared`, `script`, `model` or `none` |
| `subtitles` | The subtitle tracks the page offers |

A page that names its language is believed. A page that says nothing is read from the
script of its title, so a Bengali title in Bengali letters gives Bangla even when the
page says nothing at all. When the page says nothing and the title says nothing either,
the model is asked what it hears.

The audio is written into the working folder, and the lyric files are written next to it.

---

## The subtitles the page already carries

A page that publishes its own subtitles is the best source of lyrics there is. The words
are right, the timings are right, and nothing has to be recognised. The engine asks for
the track before it reads the audio, and uses it when the page has one.

```bash
mash-mix lyrics "https://www.youtube.com/watch?v=dQw4w9WgXcQ" -o out --lang auto
```

```
language=en lines=61 duration=213.0s elapsed=8.8s model=unknown lyrics_from=site subtitles
```

No model was loaded, nothing was recognised, and the lines came back in the order and at
the timings the page published them. A run that reads the audio instead says so:

```
language=bn-IN lines=24 duration=265.3s elapsed=929.1s model=large-v3 lyrics_from=speech model
```

**How the track is chosen.** The language the caller asked for comes first, because a
track in the wrong language is worse than no track at all. Within one language, a track a
human wrote beats one a machine wrote.

**How the region is found.** Bangla is published as `bn-BD` in Bangladesh and `bn-IN` in
India, and a page may name either one, or neither. The engine keeps a chain of tags for
every language and walks it in both directions:

| Asked for | The chain it tries |
|---|---|
| `bn-BD` | `bn-BD`, `bn`, `bn-IN` |
| `bn-IN` | `bn-IN`, `bn`, `bn-BD` |
| `bn` | `bn`, `bn-BD`, `bn-IN` |
| `en` | `en`, `en-US`, `en-GB` |

A page that carries `bn-IN` serves a caller who asked for `bn-BD`, and the other way
round. A page that publishes a track it made out of another one, such as `bn-en`, is
found by the language it is written in, not by the whole name.

```python
from mash_mix.lang.catalogue import fallback_chain, model_language

fallback_chain("bn-BD")   # ['bn-BD', 'bn', 'bn-IN']
model_language("pt-BR")   # 'pt', because a model is taught one code per language
```

**Reading a subtitle file.** SRT and WebVTT are both read, because a site publishes
both. The cue number, the cue identifier, the voice labels, the styling and the karaoke
marks are all removed, and the words are left. A file written in the code page of its own
language is read with that code page, and a file with a byte order mark is read as what
the mark says it is.

```python
from mash_mix.export.subtitles import parse_subtitles

lines = parse_subtitles(open("subs.en.vtt", encoding="utf-8").read())
```

**The audio is fetched either way.** The panel plays the song it wrote the lyrics for,
so the engine puts a copy of the audio on this machine even when the words came from the
page. Nothing leaves this machine.

---

## The words on the picture of a karaoke

A karaoke has no subtitles, and often nobody is singing either, so the words of the song are on
the screen and nowhere else. The engine downloads the picture and reads it.

```bash
pip install "mash-mix-audio-and-text[karaoke]"
mash-mix lyrics "https://www.youtube.com/watch?v=O4VgX2jd2Ug" -o out --lang bn-BD --karaoke on
```

**The rows of the song are found by the colour sweep.** A karaoke sweeps the line along as
it is sung, so the colour of the letters moves from one colour to another and back again as
the song goes on. The engine follows the colour of every place where text was found over the
whole video, and the places where the colour moves are the rows of the lyrics. A name, a
logo and an advertisement do not sweep, so they are not rows of the song.

**The times of the song come from the same sweep.** The moment the sweep starts is the
moment the line is sung, so the timings are the timings of the song and not a guess.

**The reading that the most frames agree on is the one that is kept.** The words are read
again in every frame they are on the screen in, because a reader that is nearly right is
wrong in a different way every time it is tried.

A video with nothing that sweeps is not a karaoke, and the survey says so without spending
the rest of the work on it. The run carries on to the model instead. The command line has
`--karaoke auto|on|off` and the HTTP route has a `karaoke` field of the same three values.

---

## The language of the audio

The language is worked out in three steps, and each step is reported so a guess is never
shown as a fact.

| Step | Where it comes from | What it means |
|---|---|---|
| 1 | The page named it | The site knows its own language, so that answer is used |
| 2 | The script of the title and the description | A Bengali title in Bengali letters gives Bangla |
| 3 | The model | The loaded recogniser is asked what it hears in the first seconds of the audio |

```bash
mash-mix lyrics "song.mp3" --lang auto     # the engine finds the language
mash-mix lyrics "song.mp3" --lang bn-BD    # the language is forced
mash-mix lyrics "song.mp3" --lang hi       # any ISO 639-1 code
```

The result says which one was used and how sure the model was:

```python
result.meta["language_requested"]     # what the caller asked for, "auto" when open
result.meta["language_effective"]     # the language that was used
result.meta["language_source"]        # requested, hint, model or none
result.meta["language_probability"]   # how sure the model was
```

The ladder of models is built once the language is known, so a Bangla track gets the
Bangla models and an international track gets the general ones. A text that comes back in
the Bengali script is treated as Bangla even when the detector was unsure, because the
words themselves are better evidence than the guess.

**A language other than Bangla keeps the script the recogniser used.** There is nothing
to convert a Hindi, an Arabic or an English line to, so the script repair and the
transliteration apply to Bangla only.

The catalogue of the languages the engine knows lives in `mash_mix.lang.catalogue`. It
holds the ISO 639-1 code, the aliases a person types, the writing script and the Windows
code page of the legacy files of each language.

**A region is not a language to a model.** A model is taught one code per language, so a
request for `pt-BR` reaches it as `pt`. The region the caller asked for is not lost: it
stays in the report, and the fallback chain of the catalogue knows how to get back to it.

---

## The report of a run

Every run leaves a record of what it did, step by step, so a run can be looked at
afterwards without guessing what happened.

```bash
mash-mix lyrics "https://www.youtube.com/watch?v=AU0mxhZcL1w" -o out \
    --report out/report.json
```

```json
{
  "language": "bn-IN",
  "backend": "site-subtitles",
  "meta": {
    "lyrics_source": "site subtitles",
    "language_requested": "auto",
    "language_effective": "bn",
    "language_source": "declared",
    "pipeline": {
      "steps": [
        { "step": "metadata", "ok": true, "subtitle_tracks": 60 },
        { "step": "subtitles", "ok": true, "language": "bn", "lines": 24 },
        { "step": "audio", "ok": true, "duration": 265.0 }
      ]
    }
  }
}
```

Every step says whether it worked and why, so a run that fell back on the model says so
rather than looking like a run that did not need one.

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

### The measurements of the attached file

As soon as a file is attached the server measures it and the card fills in, the same
way the MASH_SUNONA assistant panel does it:

| Measurement | What it means |
|---|---|
| Peak | The loudest sample of the file, in dBFS |
| Loudness | The integrated loudness, in LUFS, measured per channel with the BS.1770 filter |
| Clipped | The number of samples that reached full scale. Any number above zero is a fault the recording cannot repair afterwards |
| DC offset | How far the signal sits above silence, as a percentage of full scale |
| Head silence | The silence at the start of the file, in seconds |
| Crest | The gap between the peak and the average level, in dB |
| Pitch | The key and the mode of the song, for example D minor |
| Speed | The tempo of the song, in beats a minute |

The measurement route is public, so another tool can ask for the same numbers:

```bash
curl -F "file=@my song.mp3" http://localhost:8000/api/analyze
```

### Every device and every operating system

The page is one responsive layout. It runs on a phone, a tablet, a laptop and a wide
desktop, on Windows, Linux, macOS, Android and iOS, in a browser and in a web view.
The type scales with the window, the two columns become one on a narrow screen, the
controls stack their label above them, the tap targets grow on a touch screen, the
motion is dropped when the system asks for less motion, and the page prints cleanly.

| Route | Method | What it does |
|---|---|---|
| `/` | GET | The web interface |
| `/api/lyrics` | POST | Upload an audio file, get lyric lines and download links |
| `/api/lyrics/url` | POST | Give a link, get the audio fetched and its lyric lines and download links |
| `/api/lyrics/{id}/download/{fmt}` | GET | Download `srt`, `txt`, `vtt` or `lrc` |
| `/api/lyrics/{id}/audio` | GET | The audio of a song that was fetched from a link |
| `/api/analyze` | POST | Upload an audio file, get its levels, its key and its tempo |
| `/api/transcribe` | POST | Transcribe an upload |
| `/api/speech` | POST | OpenAI style text to speech, returns audio |
| `/api/audio/transcriptions` | POST | OpenAI style speech to text |
| `/api/languages` | GET | The language registry |
| `/api/models` | GET | The model catalogue |
| `/api/health` | GET | Liveness and what is installed |

The link route works like this:

```bash
curl -F "url=https://www.youtube.com/watch?v=AU0mxhZcL1w" \
     -F "language=auto" \
     http://localhost:8000/api/lyrics/url
```

It returns the same shape as the upload route, with the facts about the source under
`source` and the download links under `downloads`.

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
pip install "mash-mix-audio-and-text[edge] @ git+https://github.com/kdbn121/MASH-Mix-Audio-and-Text.git"
```

iOS notes: the `edge` extra gives ONNX Runtime Mobile. Convert the models with
`onnxruntime` tooling and run the same task functions inside the app.

---

## Architecture

```
src/mash_mix/
  lang/          language registry, normalisation, numerals, transliteration, dialects
  lang/catalogue.py  the international language table, the scripts and the code pages
  audio/         decoding, resampling, VAD, chunking, analysis, separation
  backends/      the inference adapters, one folder per task
    asr/         speech to text
    tts/         text to speech
    sts/         speech to speech
  tasks/         transcribe, lyrics, synthesize, speak, analyze
  export/        SRT, WebVTT, LRC, ASS, plain text, JSON
  sources.py     fetching the audio behind a link
  server/        FastAPI app and the web interface
  pipeline.py    one facade over everything
  models.py      the model catalogue with licences
  doctor.py      the machine check
```

The design rules:

1. **No hard dependency on a runtime.** Every backend module imports cleanly without its
   package installed, and reports itself unavailable instead of crashing. The same is true
   of the fetcher for links.
2. **One API for every task.** The task layer never imports a backend directly. It asks
   the registry for one.
3. **Bangla logic lives in one place.** No backend file contains Bangla rules. They live
   in `lang/`, so every backend gets them for free.
4. **No size limits.** Audio is decoded in chunks and written in chunks, so a three hour
   song is not a memory problem.

---

## Development

```bash
git clone https://github.com/kdbn121/MASH-Mix-Audio-and-Text.git
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

## Privacy

The engine runs on your machine and it talks to no vendor. The audio never leaves the
machine, the models are loaded from disk, and the measurements and the lyrics are written
to files you choose.

**A link is the one thing that reaches the internet.** When you give the engine a link,
the engine visits that page, the way any downloader does, and the site you named sees
your address the way it would see the address of any visitor. The audio and the lyrics
are written onto this machine. Nothing about the song is sent to a vendor, to a
telemetry service or to an account service.

Two backends break that rule on purpose, and they are never reached on their own. The
hosted recognisers, `elevenlabs` and `seamless-m4t`, send the audio away from the
machine, so the caller has to name one. Everything else runs here.

---

## Licence

MIT. Copyright (c) 2026 MASH (Khaled Bhuiyan).

The models themselves keep their own licences. Several of the best Bangla voices are
released under non commercial terms, so check `mash-mix models` before shipping a
product. `mash-mix doctor` prints the licences of the models it finds locally.
