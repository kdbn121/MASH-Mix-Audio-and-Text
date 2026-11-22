# HTTP API

Every route of the FastAPI app, with the request body, the response shape and a curl example.

## Starting the server

```bash
pip install 'mash-mix-audio-and-text[server]'
mash-mix serve --host 0.0.0.0 --port 8000
```

Or run uvicorn directly:

```bash
uvicorn mash_mix.server.app:app --host 0.0.0.0 --port 8000
```

The app object is `mash_mix.server.app:app`. The interactive documentation is at `/docs` and the schema is at `/openapi.json`, both served by FastAPI.

The server has no authentication. Bind it to `127.0.0.1` unless it sits behind a proxy that handles that.

## The routes at a glance

| Method | Route | What it does |
|---|---|---|
| GET | `/` | The web interface |
| GET | `/api/health` | Liveness and what is installed |
| GET | `/api/languages` | The language registry |
| GET | `/api/models` | The model catalogue |
| POST | `/api/lyrics` | Upload an audio file, get lyric lines and download links |
| GET | `/api/lyrics/{job_id}/download/{fmt}` | Download `srt`, `txt`, `vtt`, `lrc` or `karaoke` |
| POST | `/api/analyze` | Upload an audio file, get its levels, its key and its tempo |
| POST | `/api/transcribe` | Transcribe an upload |
| POST | `/api/speech` | OpenAI style text to speech, returns audio |
| POST | `/api/audio/transcriptions` | OpenAI style speech to text |
| GET | `/static/*` | The static files of the web interface |

## GET /

The web interface. Returns HTML.

```bash
curl http://localhost:8000/
```

## GET /api/health

Liveness plus a quick view of what is installed.

```bash
curl http://localhost:8000/api/health
```

```json
{
  "status": "ok",
  "version": "0.1.0",
  "stt_backends": ["faster-whisper", "whisper"],
  "tts_backends": ["mms-tts"]
}
```

| Key | Type | Meaning |
|---|---|---|
| `status` | string | Always `ok` when the server answers |
| `version` | string | The package version |
| `stt_backends` | array of strings | The speech to text backends that are available here |
| `tts_backends` | array of strings | The text to speech backends that are available here |

## GET /api/languages

The language registry. Returns the same array as `mash-mix languages`.

```bash
curl http://localhost:8000/api/languages
```

```json
[
  {
    "tag": "bn",
    "name_en": "Bangla (Bengali)",
    "name_native": "বাংলা (বাঙালি)",
    "iso639_1": "bn",
    "iso639_2b": "ben",
    "iso639_2t": "beng",
    "iso639_3": "ben",
    "iso639_5": "ben",
    "un_m49": "",
    "script": "Bengali",
    "cldr_language": "beng",
    "cldr_script": "Beng",
    "region": "",
    "direction": "ltr",
    "notes": "Generic Bangla tag.",
    "aliases": ["beng", "ben", "bangla", "bangali", "bengali", "bang", "bn-beng", "bn-latn-beng"]
  },
  {
    "tag": "bn-BD",
    "name_en": "Bangla - Bangladesh",
    "un_m49": "050",
    "cldr_language": "bng",
    "region": "BD"
  },
  {
    "tag": "bn-IN",
    "name_en": "Bengali - India",
    "un_m49": "356",
    "cldr_language": "beng",
    "region": "IN"
  }
]
```

The array always has three entries. See [bangla.md](bangla.md) for the full identifier table.

## GET /api/models

The model catalogue. Returns the `MODEL_CATALOG` list.

```bash
curl http://localhost:8000/api/models
```

```json
[
  {
    "id": "openai/whisper-large-v3",
    "task": "stt",
    "backend": "whisper",
    "name": "Whisper large v3",
    "size": "1.55 B",
    "languages": "99 languages, Bangla included",
    "bengali": "yes, the strongest general purpose option for Bangla",
    "license": "MIT",
    "platforms": "Windows, Linux, macOS, Android, iOS",
    "notes": "Best quality for Bangla. Needs about 3 GB of memory in float16.",
    "url": "https://huggingface.co/openai/whisper-large-v3"
  }
]
```

Each entry has `id`, `task`, `backend`, `name`, `size`, `languages`, `bengali`, `license`, `platforms`, `notes` and `url`. The `license` field is the one to check before shipping a product. See [models.md](models.md).

## POST /api/lyrics

Upload an audio file, get lyric lines and download links.

The body is multipart form data.

| Field | Type | Default | Meaning |
|---|---|---|---|
| `file` | file | required | The audio or video file |
| `language` | string | `bn-BD` | The language tag |
| `script` | string | `bangla` | The writing system: `bangla` or `banglish` |
| `title` | string | none | The title written into the file header |
| `artist` | string | none | The artist written into the file header |
| `separate_vocals` | boolean | `false` | Pull the vocal stem out before recognition |
| `backend` | string | `auto` | The speech to text backend |
| `model` | string | none | The model id for the backend |

```bash
curl -X POST http://localhost:8000/api/lyrics \
  -F "file=@song.mp3" \
  -F "language=bn-BD" \
  -F "script=bangla" \
  -F "title=Kobita tuku" \
  -F "artist=MASH"
```

```json
{
  "job_id": "a1b2c3d4e5f6",
  "language": "bn-BD",
  "script": "bangla",
  "duration": 214.3,
  "title": "Kobita tuku",
  "artist": "MASH",
  "source": {
    "name": "song.mp3",
    "size_bytes": 6043867,
    "duration": 214.3,
    "sample_rate": 48000,
    "channels": 2,
    "bit_rate": 184067,
    "format_name": "mp3"
  },
  "text": "প্রথম লাইন\nদ্বিতীয় লাইন",
  "lines": [
    {
      "start": 0.0,
      "end": 2.5,
      "text": "প্রথম লাইন",
      "speaker": null,
      "probability": null
    }
  ],
  "meta": {
    "chunks": 8,
    "dialect_confidence": 0.6265,
    "dialect": "dhaka",
    "evidence": ["BD:আপ্নে x1"],
    "bd_score": 3.8,
    "in_score": 0.3,
    "script": "bangla",
    "output_script": "bangla",
    "sample_rate": 16000,
    "channels": 1,
    "dropped_loops": 0,
    "collapsed_repeats": 2
  },
  "downloads": {
    "srt": "/api/lyrics/a1b2c3d4e5f6/download/srt",
    "txt": "/api/lyrics/a1b2c3d4e5f6/download/txt",
    "vtt": "/api/lyrics/a1b2c3d4e5f6/download/vtt",
    "lrc": "/api/lyrics/a1b2c3d4e5f6/download/lrc",
    "karaoke": "/api/lyrics/a1b2c3d4e5f6/download/karaoke"
  }
}
```

| Key | Type | Meaning |
|---|---|---|
| `job_id` | string | The identifier of this job. Use it for the download routes |
| `language` | string | The language of the lyrics |
| `script` | string | The writing system the lines came back in |
| `duration` | number | The length of the audio in seconds |
| `source` | object | What the server found when it opened the file: name, size, container, sample rate, channels |
| `title`, `artist` | string or null | What you passed in |
| `text` | string | The lyric lines joined by newlines |
| `lines` | array | The lyric lines with their times |
| `meta` | object | The chunk count, the regional detection evidence and the pipeline counters |
| `downloads` | object | The five download links |

An error returns status 500 with the message in `detail`.

## POST /api/lyrics/url

Fetch the audio behind a link and get the same lyric lines and download links.

The engine does the fetching, so the browser never touches the site. The body is multipart
form data.

| Field | Type | Default | Meaning |
|---|---|---|---|
| `url` | string | required | The web address of the audio |
| `language` | string | `auto` | The language tag. `auto` finds the language on its own |
| `script` | string | `bangla` | The writing system: `bangla` or `banglish` |
| `title` | string | none | The title written into the file header |
| `artist` | string | none | The artist written into the file header |
| `separate_vocals` | boolean | `false` | Pull the vocal stem out before recognition |
| `backend` | string | `auto` | The speech to text backend |
| `model` | string | none | The model id for the backend |
| `quality` | string | `auto` | Which model to reach for: `auto`, `best` or `fast` |
| `karaoke` | string | `auto` | Read the words of a karaoke off the picture: `auto`, `on` or `off` |

```bash
curl -X POST http://localhost:8000/api/lyrics/url \
  -F "url=https://www.youtube.com/watch?v=AU0mxhZcL1w" \
  -F "language=auto" \
  -F "script=bangla"
```

```json
{
  "job_id": "a1b2c3d4e5f6",
  "language": "bn-BD",
  "lyrics_source": "site subtitles",
  "duration": 232.0,
  "title": "Bolte Bolte Cholte Cholte",
  "artist": "IMRAN MAHMUDUL",
  "source": {
    "url": "https://www.youtube.com/watch?v=AU0mxhZcL1w",
    "title": "Bolte Bolte Cholte Cholte",
    "uploader": "IMRAN MAHMUDUL",
    "duration": 232.0,
    "extractor": "Youtube",
    "declared_language": "bn",
    "language_source": "declared",
    "language": "bn",
    "script": "Bengali",
    "subtitles": [],
    "size_bytes": 44575706
  },
  "text": "বলতে যে মনে হয়\nবলতে তবু দেয় না হৃদয়",
  "lines": [],
  "meta": {
    "lyrics_source": "site subtitles",
    "language_requested": "auto",
    "language_effective": "bn",
    "language_source": "declared",
    "pipeline": {
      "steps": [
        { "step": "metadata", "ok": true, "subtitle_tracks": 60 },
        { "step": "subtitles", "ok": true, "language": "bn", "lines": 45 },
        { "step": "audio", "ok": true, "duration": 232.0 }
      ]
    }
  },
  "audio": "/api/lyrics/a1b2c3d4e5f6/audio",
  "downloads": {
    "srt": "/api/lyrics/a1b2c3d4e5f6/download/srt",
    "txt": "/api/lyrics/a1b2c3d4e5f6/download/txt",
    "vtt": "/api/lyrics/a1b2c3d4e5f6/download/vtt",
    "lrc": "/api/lyrics/a1b2c3d4e5f6/download/lrc",
    "karaoke": "/api/lyrics/a1b2c3d4e5f6/download/karaoke"
  }
}
```

| Key | Type | Meaning |
|---|---|---|
| `lyrics_source` | string | `site subtitles` when the page carried its own words, `speech model` when the audio was read, `video ocr` when the words were read off the picture |
| `source` | object | What the page said about itself: title, uploader, duration, extractor, the language it named, the script of the title and the subtitle tracks it offers |
| `meta.pipeline` | object | Every step of the run, with the reason for each one that did not work |
| `audio` | string | The route that serves the copy of the audio on this machine |

A text that is not a web address returns status 400. A page that cannot be fetched returns
status 502 with the message in `detail`.

## GET /api/lyrics/{job_id}/audio

The audio of a song that was fetched from a link. It is the copy the engine put on this
machine, so a panel can play the song it wrote the lyrics for without anything travelling
anywhere.

```bash
curl -o song.wav http://localhost:8000/api/lyrics/a1b2c3d4e5f6/audio
```

A job with no audio returns status 404.

## GET /api/lyrics/{job_id}/download/{fmt}

Download a finished lyric file. The format is one of `srt`, `txt`, `vtt`, `lrc` or `karaoke`.

```bash
curl -OJ http://localhost:8000/api/lyrics/a1b2c3d4e5f6/download/srt
```

| Format | Media type | File |
|---|---|---|
| `srt` | `application/x-subrip` | The SubRip subtitle file |
| `txt` | `text/plain` | The plain lyric text |
| `vtt` | `text/vtt` | The WebVTT subtitle file |
| `lrc` | `text/plain` | The LRC karaoke file |
| `karaoke` | `application/x-subrip` | The SRT with the word being sung marked inside the line |

### How the job id works

1. The upload route creates a job id, twelve hexadecimal characters from `uuid.uuid4().hex`.
2. The uploaded file is streamed to a folder under the temporary directory, in a folder named `mash-mix-jobs/<job_id>`. The write happens in chunks, so a large upload does not have to fit in memory.
3. The result is written next to it as five files: the SRT, the TXT, the VTT, the LRC and the karaoke SRT.
4. The job id and the paths of those five files are kept in the in memory job table.
5. The download route looks the job id up in that table and serves the file.

Two things follow from that:

- **An unknown job id returns 404** with the detail `Unknown job id`.
- **A format that was not produced returns 404** with the detail `Format 'xyz' was not produced`.

The job table lives in the memory of the server process, and the files live in the temporary folder. Both are gone when the server restarts, so download the files before then. With more than one uvicorn worker, each worker has its own job table, so download from the worker that handled the upload.

## POST /api/analyze

Measure an uploaded file. Returns the levels of the recording, the headroom it has
left, the key of the song and its tempo. The web interface uses this to fill in the
measurement panel of the attached file.

The body is multipart form data with a single field, `file`.

```bash
curl -X POST http://localhost:8000/api/analyze \
  -F "file=@song.mp3"
```

```json
{
  "peak_dbfs": -0.47690773017124866,
  "rms_dbfs": -14.213864439436223,
  "loudness_lufs": -11.296143090320857,
  "clipped_samples": 0,
  "dc_offset_percent": 0.018794408984075372,
  "head_silence_seconds": 2.0833333333333333e-05,
  "crest_db": 13.736956709264975,
  "key": "D",
  "mode": "minor",
  "bpm": 110.99999726717473,
  "backend": "numpy",
  "source": {
    "name": "song.mp3",
    "size_bytes": 6074852,
    "duration": 262.68,
    "sample_rate": 48000,
    "channels": 2,
    "bit_rate": 185000,
    "format_name": "mp3"
  }
}
```

| Field | What it is |
|---|---|
| `peak_dbfs` | The loudest sample of the file, in dBFS |
| `rms_dbfs` | The average level of the file, in dBFS |
| `loudness_lufs` | The integrated loudness, in LUFS, measured with the BS.1770 filter on each channel as it was uploaded |
| `clipped_samples` | The number of samples that reached full scale. Above zero it is a fault the recording cannot repair afterwards |
| `dc_offset_percent` | How far the signal sits above silence, as a percentage of full scale |
| `head_silence_seconds` | The silence at the start of the file, in seconds |
| `crest_db` | The gap between the peak and the average level, in dB |
| `key` | The key of the song, for example `D` |
| `mode` | `major` or `minor` |
| `bpm` | The tempo of the song, in beats a minute |
| `backend` | The analysis engine that produced the numbers |
| `source` | The facts of the container: the name, the size, the length, the sample rate, the number of channels, the bit rate and the format |

Every measurement is optional. A file that cannot be decoded still returns the facts
the probe found, so the caller always has something to show. The uploaded copy is
removed as soon as the measurement is done, so the route holds nothing on disk.

## POST /api/transcribe

Transcribe an uploaded recording.

The body is multipart form data.

| Field | Type | Default | Meaning |
|---|---|---|---|
| `file` | file | required | The recording |
| `language` | string | empty | The language tag. Empty means the model decides |
| `backend` | string | `auto` | The speech to text backend |
| `model` | string | none | The model id for the backend |

```bash
curl -X POST http://localhost:8000/api/transcribe \
  -F "file=@interview.wav" \
  -F "language=bn-BD"
```

```json
{
  "text": "প্রথম লাইন\nদ্বিতীয় লাইন",
  "language": "bn-BD",
  "duration": 12.4,
  "backend": "faster-whisper",
  "model": "small",
  "elapsed": 2.81,
  "segments": [
    {
      "start": 0.0,
      "end": 2.5,
      "text": "প্রথম লাইন",
      "speaker": null,
      "language": null,
      "probability": null,
      "words": [
        {"start": 0.0, "end": 0.4, "text": "প্রথম", "probability": 0.98}
      ]
    }
  ],
  "meta": {"sample_rate": 16000, "channels": 1}
}
```

The shape is the `Transcription.to_dict()` output: the text, the language, the duration, the backend and model names, the elapsed time, the segments with their words, and the metadata.

## POST /api/speech

OpenAI style text to speech. The body is JSON and the response is the audio bytes.

| Field | Type | Default | Meaning |
|---|---|---|---|
| `input` | string | none | The text to speak. `text` is accepted as an alias |
| `language` | string | `bn-BD` | The language tag |
| `backend` | string | `auto` | The text to speech backend |
| `model` | string | none | The model id for the backend |
| `voice` | string | none | The voice identifier |
| `speed` | number | `1.0` | The speaking rate |

```bash
curl -X POST http://localhost:8000/api/speech \
  -H "Content-Type: application/json" \
  -d '{"input": "আজকের দিনটা সুন্দর", "language": "bn-BD"}' \
  --output speech.wav
```

The response has the media type `audio/wav` and the header `Content-Disposition: inline; filename="speech.wav"`.

A missing `input` returns status 400 with the detail `Field 'input' is required`. A backend failure returns status 500 with the message in `detail`.

## POST /api/audio/transcriptions

OpenAI style speech to text.

The body is multipart form data.

| Field | Type | Default | Meaning |
|---|---|---|---|
| `file` | file | required | The recording |
| `model` | string | `auto` | The backend name, or `auto` |
| `language` | string | empty | The language tag |
| `response_format` | string | `json` | `json`, `text`, `vtt` or `srt` |

```bash
curl -X POST http://localhost:8000/api/audio/transcriptions \
  -F "file=@interview.wav" \
  -F "model=faster-whisper" \
  -F "language=bn-BD" \
  -F "response_format=json"
```

The response is the same JSON as `/api/transcribe` when `response_format=json`.

```bash
curl -X POST http://localhost:8000/api/audio/transcriptions \
  -F "file=@interview.wav" -F "response_format=text"
```

```bash
curl -X POST http://localhost:8000/api/audio/transcriptions \
  -F "file=@interview.wav" -F "response_format=srt" --output out.srt
```

| `response_format` | Media type | Body |
|---|---|---|
| `json` | `application/json` | The `Transcription` object |
| `text` | `text/plain` | The plain transcript |
| `vtt` | `text/vtt` | The WebVTT subtitle file |
| `srt` | `application/x-subrip` | The SubRip subtitle file |

### Using it from an OpenAI client

The route follows the OpenAI shape, so an OpenAI client can point at this server. Set the base URL to `http://localhost:8000/v1`. The server does not check the credential value, but the client library insists on one, so pass a value from the environment.

```bash
export OPENAI_BASE_URL="http://localhost:8000/v1"
export OPENAI_API_KEY="$(python -c 'print("local")')"
```

```python
from openai import OpenAI

client = OpenAI(base_url="http://localhost:8000/v1")

with open("interview.wav", "rb") as handle:
    transcript = client.audio.transcriptions.create(
        model="faster-whisper",
        file=handle,
        language="bn",
    )
print(transcript.text)
```

## Error responses

| Status | When |
|---|---|
| 400 | A required field is missing, for example `input` on `/api/speech` |
| 404 | An unknown job id, or a format that was not produced |
| 500 | The task failed. The message from the engine is in `detail` |

```json
{"detail": "Speech to text backend 'faster-whisper' is not available on this machine. pip install 'mash-mix-audio-and-text[stt]'"}
```

## A complete session

```bash
# 1. Is the server up and what is installed
curl http://localhost:8000/api/health

# 2. Upload a song and get the lyrics
JOB=$(curl -s -X POST http://localhost:8000/api/lyrics \
  -F "file=@song.mp3" -F "language=bn-BD" \
  | python -c "import json,sys; print(json.load(sys.stdin)['job_id'])")

# 3. Download the subtitle file
curl -OJ http://localhost:8000/api/lyrics/$JOB/download/srt
```

---

MASH-Mix-Audio-and-Text V0.6.2. Copyright (c) 2026 MASH (Khaled Bhuiyan). MIT Licence. See LICENSE.
