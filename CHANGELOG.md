# Changelog

All notable changes to this project are documented here.

The format follows Keep a Changelog, and this project adheres to Semantic Versioning.

## [0.1.1] - 2026-11-20

Song lyric accuracy, the Banglish option, and a visible file panel.

### Added

- **Banglish output option.** Every task that produces text now takes a `script`
  argument, `bangla` or `banglish`, and so does the command line (`--script`) and
  the HTTP API (`script`). The web interface has an **Output script** choice next
  to the language.
- `apply_script`, `banglish_word_ratio` and `resolve_script_target` in the
  language layer, so a document written in romanised Bangla is turned into
  Bangla script and a Bangla document can be turned into Latin letters.
- `scripts/lyrics_eval.py`, which scores extracted lyrics against the real lyric
  file of a song and reports the word error rate, the share of true lines that
  were found, and every line that was missed.
- `scripts/convert_model.py`, which converts a Hugging Face Whisper checkpoint
  into the CTranslate2 format so a Bangla fine tune can run on the fast backend.
- `scripts/verify_lexicon.py`, which checks the curated romanised Bangla
  dictionary for a wrong script, an empty value and a repeated key.
- The Bangla fine tunes `mozilla-ai/whisper-large-v3-bn` and
  `mozilla-ai/whisper-large-v3-turbo-bn` in the model catalogue.
- The `separation` extra, which installs Demucs for vocal separation.
- The web interface shows the attached file as soon as it is chosen: name, size,
  type and length, then the container, sample rate and channel count that the
  server found. A **Model** choice was added next to it.
- Sixty more entries in the curated romanised Bangla dictionary, so a phrase
  such as "ami tomay bhalobashi" comes out in Bangla script.

### Changed

- The `transformers` backend runs the model in `bfloat16` on a CPU. It is about
  three times faster than `float32` and gives the same text.
- Lyric lines are now cut at the pauses, using the word timings, so a verse
  comes back as lines and not as one long run of words.
- A line that is one word repeated over and over is dropped, and a word repeated
  inside a line is collapsed. A speech model that loses the plot on sung audio
  used to write the same syllable until it ran out of tokens.
- Romanised Bangla is written out in Bangla before the text is spoken, so a
  Bangla voice no longer reads Latin letters as English. This applies to text to
  speech and to speech to speech.
- Vocal separation now runs on the stereo mix at its own sample rate. It used to
  run on the mono 16 kHz buffer, which had already thrown away everything the
  separator needs.

### Fixed

- The `transformers` backend no longer fails when the model cannot use a
  generation argument, and falls back to the older spelling of the dtype
  argument.

[0.1.1]: https://github.com/MASH-ai/MASH-Mix-Audio-and-Text/releases/tag/v0.1.1

---

## [0.1.0] - 2026-11-20

First release.

### Added

- Cross platform speech engine with a pluggable backend layer, so the same code
  runs on Windows, Linux, macOS, Android and iOS.
- Song to lyric pipeline: any format, any length, output as SRT, TXT, VTT, LRC
  or JSON, with an optional vocal separation step, voice activity detection and
  repetition control for sung audio.
- Speech to text backends: `whisper`, `faster-whisper`, `transformers`, `funasr`,
  `sherpa-onnx` and `whisper.cpp`.
- Text to speech backends: `mms-tts`, `vits`, `indic-tts`, `parler-tts`, `piper`,
  `xtts`, `chatterbox` and `edge-tts`.
- Speech to speech through a `pipeline` backend built from a recognition and a
  synthesis backend.
- Bangla language layer: BCP 47, ISO 639 and UN M.49 identifiers for `bn`,
  `bn-BD` and `bn-IN`, Bangla text normalisation that preserves ZWNJ and
  decomposes the nukta letters, lakh and crore number reading, symbol
  expansion, romanised Bangla to script conversion, and regional and district
  dialect detection for all 64 districts of Bangladesh.
- Audio layer: decoding and writing through ffmpeg with no size or time limit,
  polyphase resampling, energy and zero crossing voice activity detection,
  silence aware chunking, tempo, key, BS.1770 loudness and section analysis,
  and stem separation with a dependency free mid and side fallback.
- Command line interface, an OpenAI compatible HTTP API and a web interface
  with lyric download buttons.
- `mash-mix doctor` machine check and a model catalogue with licences.

[0.1.0]: https://github.com/MASH-ai/MASH-Mix-Audio-and-Text/releases/tag/v0.1.0

---

Copyright (c) 2026 MASH (Khaled Bhuiyan). MIT Licence. See LICENSE.
