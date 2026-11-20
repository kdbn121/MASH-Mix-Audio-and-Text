# Documentation

The documents for MASH-Mix-Audio-and-Text, one file per topic.

| Document | What it covers |
|---|---|
| [architecture.md](architecture.md) | The layers, the design rules, how a request travels to a backend and back, the data model, and how to add a backend |
| [bangla.md](bangla.md) | The language identifiers, the script rules, the number system, transliteration, regional detection, and how lyric text is prepared |
| [backends.md](backends.md) | Every speech to text, text to speech and speech to speech backend, the automatic selection order, and the memory and speed guidance |
| [models.md](models.md) | The model catalogue with licences, the recommendation order per task, and how to choose for a phone, a laptop, a GPU server or a low bandwidth link |
| [platforms.md](platforms.md) | Windows, Linux, macOS, Android with Termux, iOS, Raspberry Pi, the GPU notes, and running the HTTP API |
| [cli.md](cli.md) | Every command and every flag with a working example, plus the exit codes and what the doctor prints |
| [api.md](api.md) | Every HTTP route with the request body, the response shape and a curl example, including the OpenAI compatible routes |
| [lyrics.md](lyrics.md) | The song to lyric workflow, the pipeline stages, how to improve the result, the output formats, and the troubleshooting guide |

## Where to start

| You want to | Read |
|---|---|
| Install and run something | [cli.md](cli.md), then [platforms.md](platforms.md) for your machine |
| Extract lyrics from a song | [lyrics.md](lyrics.md) |
| Pick a model | [models.md](models.md), then [backends.md](backends.md) |
| Understand the Bangla handling | [bangla.md](bangla.md) |
| Call the engine from Python | [architecture.md](architecture.md) |
| Call it over HTTP | [api.md](api.md) |

The top level [README.md](../README.md) is the front door: the install, the quick start and the feature overview.

---

Copyright (c) 2026 MASH (Khaled Bhuiyan). MIT Licence. See LICENSE.
