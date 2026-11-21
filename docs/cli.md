# Command line

Every command, every flag and a working example for each one.

## The shape of it

```
mash-mix <command> [options]
```

| Command | What it does |
|---|---|
| `lyrics` | Extract lyrics from a song or a spoken track |
| `transcribe` | Transcribe speech to text |
| `speak` | Synthesise speech from text |
| `sts` | Speech to speech |
| `analyze` | Measure tempo, key, loudness and sections |
| `separate` | Split a track into stems |
| `text` | Bangla text tools |
| `languages` | List the supported language tags |
| `backends` | List the backends and what is installed |
| `doctor` | Check this machine |
| `serve` | Start the web interface and the HTTP API |

The global flag `--version` prints the version and exits.

## lyrics

Extract lyrics from a song or a spoken track.

| Flag | Default | Meaning |
|---|---|---|
| `audio` | required | The audio or video file. Any format, any length. A web address is accepted too |
| `-o`, `--output-dir` | `lyrics-out` | Where to write the files |
| `--stem` | the title, or `lyrics` | The base name for the written files |
| `--format` | `srt` and `txt` | The output format. Repeat the flag for more than one. One of `srt`, `txt`, `vtt`, `lrc`, `json`, `karaoke` |
| `--lang`, `--language` | `auto` | The language tag. `auto` finds the language on its own. Any ISO 639-1 code, and `bn-BD` or `bn-IN` for a region |
| `--script` | `bangla` | The writing system of the lines. `bangla` or `banglish` |
| `--backend` | `auto` | The speech to text backend |
| `--model` | the backend default | The model id for the backend |
| `--title` | none | The title written into the file header |
| `--artist` | none | The artist written into the file header |
| `--separate-vocals` | off | Pull the vocal stem out of the mix before recognition |
| `--separation-model` | `htdemucs` | The separation model. Use `mid_side` for a dependency free vocal removal |
| `--chunk-seconds` | `30.0` | The chunk length in seconds |
| `--overlap-seconds` | `2.0` | The overlap between chunks in seconds |
| `--hotword` | none | A word or phrase to bias the recogniser towards. Repeat the flag |
| `--prompt` | none | An initial prompt to guide the recogniser |
| `--max-repeats` | `2` | How many consecutive identical lines to keep before they are dropped |
| `--print` | off | Print the lyrics to the terminal |
| `--report` | none | Write a JSON report of the whole pipeline to this file |
| `--karaoke` | `auto` | Read the words of a karaoke off the picture: `auto`, `on` or `off` |

```bash
mash-mix lyrics "bangla song.mp3" -o out --format srt --format txt --lang bn-BD
```

```
out/lyrics.srt
out/lyrics.txt
```

### A link instead of a file

The argument can be a web address. The engine fetches the audio behind it with
`yt-dlp`, which is an optional dependency, and reads it exactly like a file that was dropped
in. The engine is what reaches the internet, never the browser.

```bash
mash-mix lyrics "https://www.youtube.com/watch?v=AU0mxhZcL1w" -o out --lang auto
```

```
language=bn-BD lines=45 duration=232.0s elapsed=9.8s model=unknown lyrics_from=site subtitles
```

`lyrics_from` says where the words came from. `site subtitles` means the page carried its own
subtitles and the engine used them, with no model loaded at all. `speech model` means the audio
was read by a recogniser.

The report of the run is written with `--report`:

```bash
mash-mix lyrics "https://www.youtube.com/watch?v=AU0mxhZcL1w" -o out --report out/report.json
```

```json
{
  "steps": [
    { "step": "metadata", "ok": true, "subtitle_tracks": 60 },
    { "step": "subtitles", "ok": true, "language": "bn", "lines": 45 },
    { "step": "audio", "ok": true, "duration": 232.0 }
  ]
}
```

Every step says whether it worked and why, so a run that fell back on the model says so rather
than looking like a run that did not need one.

The base name comes from `--stem`, then from `--title`, then it is `lyrics`. A progress line goes to standard error:

```
language=bn-BD lines=42 duration=214.3s elapsed=61.8s
```

With a title and an artist:

```bash
mash-mix lyrics "song.mp3" -o out --stem kobita-tuku --title "Kobita tuku" --artist "MASH" --print
```

On a dense mix, pull the vocals out first:

```bash
mash-mix lyrics "song.mp3" -o out --separate-vocals --separation-model htdemucs
```

With hotwords and a prompt, which is how you fix a name the model keeps getting wrong:

```bash
mash-mix lyrics "song.mp3" -o out --hotword "কবিতা" --hotword "টুকু" --prompt "একটি বাংলা গানের গান"
```

## transcribe

Transcribe speech to text.

| Flag | Default | Meaning |
|---|---|---|
| `audio` | required | The audio or video file |
| `-o`, `--output` | none | Write the transcript to this file. Without it the text goes to the terminal |
| `--format` | `txt` | The output format. One of `txt`, `json`, `srt`, `vtt`, `karaoke` |
| `--lang`, `--language` | empty | The language tag. Empty means the model decides |
| `--backend` | `auto` | The speech to text backend |
| `--model` | the backend default | The model id for the backend |
| `--task` | `transcribe` | `transcribe` or `translate` |
| `--hotword` | none | A word or phrase to bias the recogniser towards. Repeat the flag |
| `--prompt` | none | An initial prompt to guide the recogniser |
| `--timestamps` | off | Prefix each line with a time |

```bash
mash-mix transcribe "interview.wav" -o transcript.txt --lang bn-BD
```

```bash
mash-mix transcribe "interview.wav" --format json -o transcript.json --lang bn-BD
```

```bash
mash-mix transcribe "interview.wav" --format srt -o interview.srt --timestamps
```

```bash
mash-mix transcribe "interview.wav" --backend sherpa-onnx --model /models/whisper-bn --lang bn-BD
```

The `srt`, `vtt` and `karaoke` formats write their own file. Without `--output` the file is called `transcript.srt`, `transcript.vtt` or `transcript.srt` in the working directory. The `karaoke` format is an SRT with the word being sung marked inside the line, so it needs a backend that gives the time of every word.

## speak

Synthesise speech from text.

| Flag | Default | Meaning |
|---|---|---|
| `text` | required | The text to speak |
| `-o`, `--output` | `speech.wav` | The output audio file |
| `--lang`, `--language` | `bn-BD` | The language tag |
| `--backend` | `auto` | The text to speech backend |
| `--model` | the backend default | The model id for the backend |
| `--voice` | the backend default | The voice identifier |
| `--speed` | `1.0` | The speaking rate |
| `--reference-audio` | none | A reference clip for voice cloning |
| `--no-normalize` | off | Skip the Bangla text preparation |

```bash
mash-mix speak "আজকের দিনটা সুন্দর" -o hello.wav --lang bn-BD
```

```bash
mash-mix speak "আজকের দিনটা সুন্দর" -o hello.wav --backend vits --speed 0.9
```

With a reference clip for voice cloning:

```bash
mash-mix speak "পরীক্ষা করার কথা" -o out.wav --backend xtts --reference-audio my-voice.wav
```

The Bangla preparation expands the symbols and localises the digits. Turn it off when the backend wants the raw text:

```bash
mash-mix speak "100% সফল" -o out.wav --no-normalize
```

## sts

Speech to speech. Speak a recording in another voice or another language.

| Flag | Default | Meaning |
|---|---|---|
| `audio` | required | The recording |
| `-o`, `--output` | `speech-sts.wav` | The output audio file |
| `--target-lang` | `bn-BD` | The target language tag |
| `--target-voice` | the backend default | The target voice |
| `--reference-audio` | none | A reference clip for voice cloning |
| `--stt-backend` | `auto` | The recognition backend |
| `--tts-backend` | `auto` | The synthesis backend |

```bash
mash-mix sts "recording.wav" -o out.wav --target-lang bn-BD --tts-backend vits
```

```bash
mash-mix sts "recording.wav" -o out.wav --stt-backend faster-whisper --tts-backend piper
```

## analyze

Measure tempo, key, loudness and sections.

| Flag | Default | Meaning |
|---|---|---|
| `audio` | required | The track |
| `-o`, `--output` | none | Write the report as JSON to this file |
| `--json` | off | Print the report as JSON |

```bash
mash-mix analyze "song.mp3" --json
```

```bash
mash-mix analyze "song.mp3" -o report.json
```

The report carries the duration, the sample rate, the channels, the tempo in BPM, the key and mode, the loudness in LUFS, the peak and RMS in dBFS, the spectral centroid, the number of clipped samples, the dc offset, the silence at the head of the file, the crest factor, the detected sections and the beat times.

## separate

Split a track into stems.

| Flag | Default | Meaning |
|---|---|---|
| `audio` | required | The track |
| `-o`, `--output-dir` | `stems-out` | Where to write the stem files |
| `--model` | `htdemucs` | The separation model. Use `mid_side` for a dependency free split |

```bash
mash-mix separate "song.mp3" -o stems-out --model htdemucs
```

```bash
mash-mix separate "song.mp3" -o stems-out --model mid_side
```

The stems are written as `vocals.wav` and `no_vocals.wav`.

## text

The Bangla text tools. Every subcommand takes the text as an argument, or reads it from standard input when the argument is omitted.

### text normalize

| Flag | Default | Meaning |
|---|---|---|
| `text` | none | The text. Reads standard input when omitted |
| `--file` | none | Read the text from this file |
| `--digits` | `keep` | `keep`, `native` or `ascii` |

```bash
mash-mix text normalize "ড়  ঢ়  য়"
mash-mix text normalize "123" --digits native
mash-mix text normalize --file notes.txt
cat notes.txt | mash-mix text normalize
```

### text numbers

| Flag | Default | Meaning |
|---|---|---|
| `value` | required | The number to read |

```bash
mash-mix text numbers 12345678
```

```
এক কোটি তেইশ লক্ষ পঁয়তাল্লিশ হাজার ছয়শো আটাত্তর
```

```bash
mash-mix text numbers 42
```

```
বল্লিশ
```

### text translit

| Flag | Default | Meaning |
|---|---|---|
| `text` | none | The text. Reads standard input when omitted |
| `--file` | none | Read the text from this file |
| `--reverse` | off | Bangla script to Latin instead of Latin to Bangla script |

```bash
mash-mix text translit "ami tomay bhalobashi"
mash-mix text translit "আমি তোমায়" --reverse
```

### text detect

| Flag | Default | Meaning |
|---|---|---|
| `text` | none | The text. Reads standard input when omitted |
| `--file` | none | Read the text from this file |

```bash
mash-mix text detect "ভাই, আপ্নে কেমন আছেন? টাকা পাঠাতে হবে।"
```

```json
{
  "tag": "bn-BD",
  "language": "bn-BD",
  "confidence": 0.6265,
  "bd_score": 3.8,
  "in_score": 0.3,
  "dialect": "dhaka",
  "dialect_confidence": 0.5,
  "evidence": ["BD:আপ্নে x1", "BD:ভাই x1"],
  "script": "bangla"
}
```

## languages

List the supported language tags. There are no flags.

```bash
mash-mix languages
```

The output is JSON: an array of three objects, one for `bn`, one for `bn-BD` and one for `bn-IN`. Each object carries the BCP 47 tag, the English and native names, every ISO code, the UN M.49 number, the CLDR language and script codes, the region, the direction and the aliases.

## backends

List the backends and what is installed. There are no flags.

```bash
mash-mix backends
```

```
[ok ] stt faster-whisper   CTranslate2 Whisper, fast on CPU with int8 quantisation.
[no ] stt funasr           FunASR SenseVoice and Paraformer models, fast with Bangla support.
[ok ] stt whisper          Reference OpenAI Whisper implementation with word timestamps.
```

The mark is `ok` when the backend reports itself available and `no` when the runtime is missing. The columns are the kind, the name and the summary.

## doctor

Check this machine for everything the engine needs.

| Flag | Default | Meaning |
|---|---|---|
| `--json` | off | Print the report as JSON instead of the text report |

```bash
mash-mix doctor
```

```
MASH-Mix-Audio-and-Text doctor

platform      desktop (Windows AMD64, Python 3.12.8)
ffmpeg        ffmpeg version 9.0.1-essentials_build
disk free     118.4 GB (model weights need roughly 1 to 4 GB each)

packages      4 of 14 installed

backends
  [ok ] stt faster-whisper   CTranslate2 Whisper, fast on CPU with int8 quantisation.
  [no ] stt funasr           FunASR SenseVoice and Paraformer models, fast with Bangla support.
  [ok ] stt whisper          Reference OpenAI Whisper implementation with word timestamps.

speech to text ready
text to speech not ready

install what is missing
  pip install 'mash-mix-audio-and-text[full]'
```

The JSON form carries the same information plus the raw package list, the backend list and the missing package names.

```bash
mash-mix doctor --json
```

The doctor exits with 0 when ffmpeg is present and at least one speech to text backend and one text to speech backend are ready. It exits with 1 otherwise.

## serve

Start the web interface and the HTTP API.

| Flag | Default | Meaning |
|---|---|---|
| `--host` | `127.0.0.1` | The interface to bind to |
| `--port` | `8000` | The port to listen on |
| `--reload` | off | Reload the server when the source changes |

```bash
mash-mix serve --host 0.0.0.0 --port 8000
```

Open `http://localhost:8000`. Drop in an audio file, press **Extract lyrics**, then download the `.srt` or the `.txt`.

The command needs `fastapi` and `uvicorn`. Without them it prints the install line and exits with 1.

## Exit codes

| Code | Meaning |
|---|---|
| 0 | The command finished successfully |
| 1 | The command failed. The error is printed to standard error as `error: <message>` |
| 1 | `doctor` ran and something is not ready |
| 1 | `serve` could not import `uvicorn` |
| 2 | The arguments were wrong. Argparse printed the usage and the error |
| 130 | The command was interrupted with Ctrl and C |

A missing backend produces exit code 1 with a message that names the extra to install:

```
error: Speech to text backend 'faster-whisper' is not available on this machine. pip install 'mash-mix-audio-and-text[stt]'
```

## Calling the CLI from Python

The same entry point is available as a function, which is handy in a script or a test.

```python
from mash_mix.cli import main

code = main(["text", "numbers", "42"])
print(code)
```

```python
from mash_mix.cli import build_parser

parser = build_parser()
args = parser.parse_args(["lyrics", "song.mp3", "-o", "out"])
print(args.command, args.audio, args.output_dir)
```

---

Copyright (c) 2026 MASH (Khaled Bhuiyan). MIT Licence. See LICENSE.
