# Changelog

All notable changes to this project are documented here.

The format follows Keep a Changelog, and this project adheres to Semantic Versioning.

## [0.5.0] - 2026-11-21

The words of a karaoke are read off the picture of the video.

### Added

- **A karaoke is read off the picture.** A karaoke video carries no subtitles
  and often has nobody singing either, so the words of the song are on the
  screen and nowhere else. The engine now downloads the picture and reads it.
  A run that used it says so in `meta["lyrics_source"]`, which reads
  `video ocr`, and the panel shows it as "read off the picture of the video".
- **The rows of the song are found by the colour sweep.** A karaoke sweeps the
  line along as it is sung, so the colour of the letters moves from one colour
  to another and back again as the song goes on. The colour of every place
  where text was found is followed over the whole video, and the places where
  the colour moves are the rows of the lyrics. A name and a logo do not move,
  so they are not rows of the song. A video with nothing that moves is not a
  karaoke, and the survey says so without spending the rest of the work on it.
- **Every line that is found is judged on three things.** A karaoke channel
  puts a lot on the picture that is not the song. The words must be in the
  script of the song, so a name of a channel, a web address and a number are
  not words of the song. They must not run across the whole width of the
  picture, because a line of the song sits in the middle. And the colour of
  their letters must travel in one direction while they are on the screen,
  because an advertisement is not sung and its colour does not travel.
- **The times of a karaoke come from the travel of the colour.** The moment the
  colour of the letters starts to move is the moment the line is sung, and the
  moment it stops is the moment the line has been sung. The words are read once
  a second, so the moment is looked for again with a finer step, and that step
  reads the colour of the picture only, which is cheap.
- **Each line of the song is written once.** A karaoke shows the line that is
  being sung and the line that comes next, so the same words are swept twice.
  The first sweep is the one that was sung.
- **The reading that the most frames agree on is the one that is kept.** The
  words are read again in every frame they are on the screen in, and a
  recogniser that is nearly right is wrong in a different way every time it
  is tried, so the reading that shows up most often wins.
- **`mash_mix.karaoke`.** The reader of the picture: `read_burned_in_lyrics`
  for the whole reading, `survey_burned_in_lyrics` for the cheap look that
  says whether a video is a karaoke at all.
- **`mash_mix.sources.fetch_video`.** The picture behind a link, fetched at a
  size that is quick to read and clear enough for the words.
- **The karaoke extra.** `pip install "mash-mix-audio-and-text[karaoke]"`
  brings in EasyOCR and OpenCV. The extra is not needed for anything else,
  and a run without it says so and carries on.
- **`--karaoke auto|on|off` on the command line** and a `karaoke` field on
  `POST /api/lyrics/url`. The default is `auto`, which reads the picture when
  the page carries no subtitles of its own.

### Fixed

- **A line that is wider than the picture is not a line of the song.** A
  channel that puts an advertisement over the words puts it across the whole
  width, while a line of the song sits in the middle.
- **The server keeps answering while a long run is at it.** A route that does
  a long piece of work now runs it in a worker instead of holding the event
  loop, so the health check and the download answer while the engine is busy.
  Reading the words of a karaoke off the picture of a video takes a quarter of
  an hour on an ordinary computer.
- **The series of the sweep and the rows of the song are in step.** A row that
  was left out of the song no longer shifts the timings of the rows after it.

## [0.4.0] - 2026-11-21

The subtitles a page already carries are used, and a region falls back on its twin.

### Added

- **The subtitles the page carries are used instead of a model.** A page that
  publishes its own subtitles is the best source of lyrics there is: the words
  are right, the timings are right, and nothing has to be recognised. The
  engine asks for the track before it reads the audio, and a run that used it
  says so in `meta["lyrics_source"]`. On a live test of a page with sixty one
  cues the run finished with no model loaded at all.
- **A subtitle reader.** `mash_mix.export.subtitles.parse_subtitles` reads SRT
  and WebVTT, because a site publishes both. The cue number, the cue
  identifier, the voice labels, the styling and the karaoke marks are removed,
  and the words are left. `parse_timestamp` reads both spellings of a
  timestamp, with or without the hours.
- **A subtitle file is read in the code page of its own language.** A file
  written in the code page of its language is read with that code page, and a
  file with a byte order mark is read as what the mark says it is.
- **The regional fallback.** The catalogue keeps a chain of tags for every
  language and walks it in both directions, so `bn-BD` falls back on `bn` and
  then on `bn-IN`, and `bn-IN` falls back on `bn` and then on `bn-BD`. A page
  that carries one region serves a caller who asked for the other one.
- **The report of a run.** `mash-mix lyrics --report` writes a JSON report of
  the whole pipeline, step by step, with the reason for every step that did
  not work. The same record is in `meta["pipeline"]` for every caller.
- **Nepali and Sinhala** joined the international catalogue.

### Fixed

- **A regional tag no longer reaches a model as it is.** A model is taught one
  code per language, so a request for `pt-BR` used to reach it as `pt-BR`,
  which it does not know. The language now reaches the model as `pt`, and the
  region the caller asked for stays in the report.
- **A translated subtitle track is found by the language it is written in.** A
  page names the track it made out of another one `bn-en`, and it was not
  matched by a request for Bangla. It is now.

## [0.3.1] - 2026-11-21

The certainty of the language is only reported when it means something.

### Fixed

- **A language the caller named carries no certainty.** A recogniser that was
  told the language reports a certainty of one, and that number measures
  nothing. The result now carries a certainty only when the model worked the
  language out on its own, so a number in the report is always a measurement.

## [0.3.0] - 2026-11-21

A link is now a source, and the language of the audio is found on its own.

### Added

- **A link can be used where a file was used.** `mash-mix lyrics` takes a URL
  and the new `POST /api/lyrics/url` route takes one as well. The audio behind
  the link is fetched with `yt-dlp`, read exactly like a file that was dropped
  in, and written out as the same `.srt`, `.txt`, `.vtt` and `.lrc` files. The
  fetcher is an optional dependency, so the engine works without it and only a
  request for a link needs it.
- **The engine finds the language of the audio.** The language is worked out
  in three steps: the language the page named, then the script of the title and
  the description, then the model itself. Each step is reported, so a guess is
  never shown as a fact. The ladder of models is built once the language is
  known, so a Bangla track gets the Bangla models.
- **A language other than Bangla keeps the script the recogniser used.** The
  script repair and the transliteration apply to Bangla only, because there is
  nothing to convert a Hindi, an Arabic or an English line to.
- **The international language catalogue.** `mash_mix.lang.catalogue` holds the
  ISO 639-1 code, the aliases, the writing script and the Windows code page of
  the legacy files of thirty five languages.
- **The audio of a fetched song is served back.** The new
  `GET /api/lyrics/{id}/audio` route returns the copy the engine put on this
  machine, so a panel can play the song it wrote the lyrics for.

### Fixed

- **The fetcher now writes the file.** The dump flag of the fetcher makes it
  simulate, and a simulation writes nothing, so a link came back with no audio.
  The fetch now says: do the work and then print it.

## [0.2.0] - 2026-11-21

The lyrics of a Bangla song come back right.

### Changed

- **The engine no longer starts at the small model.** The small model was the
  default, and it read a Bangla song badly: the words ran into each other and
  the lines came back in romanised Bangla even when the native script was
  asked for. The same song read by the large model now comes back in the
  native script with the words right. The filters and the line splitter were
  working; they had nothing to work with.
- **The model is chosen from a ladder and the ladder is walked.** The strong
  model needs room in memory and a download that may not be there, so when one
  cannot be loaded the engine moves to the next one instead of failing. A
  request always comes back with lyrics. The result carries the model that
  did the work and the reasons the ones above it were skipped.
- **A whole track is read in one pass** when the model can take a long file.
  A Whisper model keeps the thread of a song across the whole track, and the
  chunked path asks the same question again with no memory of what came
  before. The chunked path is kept for the models that cannot take a long
  file.

### Added

- Three quality levels for the lyric command and for the server: `auto` picks
  the model that fits the language, `best` forces the strongest one and `fast`
  starts at the quick one.
- An ElevenLabs Scribe backend. It is the strongest hosted recogniser for
  Bangla and it gives the time of every word, so the karaoke lines keep their
  timings. It needs an account and the key in `ELEVENLABS_API_KEY`, and
  without a key it keeps out of the way. The audio leaves the machine, so it
  is never reached unless it is named.
- A Meta SeamlessM4T backend. It reads Bengali as well as most languages and
  it runs locally, but it needs a large download and it gives the time of a
  sentence and not the time of a word.
- The catalogue of models now carries the two hosted recognisers and their
  notes.

### Tests

- Tests for the ladder: the order of it, the rule that a named model is the
  only candidate, the rule that a hosted backend is never reached on its own,
  and the rule that a model which cannot be loaded does not end the job.
- Tests for the two hosted backends: the request that is built, the answer
  that is read back, and the rule that a backend without a key stays away.
  None of them reaches the network.

## [0.1.5] - 2026-11-21

The browser extension can reach the engine.

### Added

- The server answers a call from a local origin with the header that lets a
  browser accept it. The extension on the other side of the loopback and the
  web interface on another local port are both a different origin from the
  engine, and a browser applies its own rules to a call that comes from one.
  Only the local origins are let in, so nothing off this machine can reach
  the engine through the browser.

### Tests

- Tests that call the engine the way a browser does, through the ASGI
  interface, so no client library and no running server are needed. They
  check that a local origin is allowed and that a foreign one is shut out.

## [0.1.4] - 2026-11-20

The key of a file no longer depends on the sample rate of the file.

### Fixed

- The frequency of a spectral peak is now worked out from the bins around it,
  with a parabola on the magnitude spectrum, instead of from the bin that
  happens to hold the peak. A tone of 220 Hz falls between two bins of the
  transform, and at the bottom of the range a semitone is narrower than a bin,
  so the bin of a peak can be a quarter of a semitone away from the note it
  belongs to. The key of a file no longer changes when the same recording is
  at 22050 Hz or at 48000 Hz: a C major triad used to read as E minor at
  48 kHz.

### Added

- Tests that read the same chord at 22050 Hz, 44100 Hz and 48000 Hz and expect
  the same key at each of them, and a test that checks the chroma of a single
  tone sits on the pitch class of that tone at every rate.

### Changed

- The version of the package in `pyproject.toml` and the version in the
  package itself now agree. They had drifted apart.

## [0.1.3] - 2026-11-20

The measurements of the attached file, and a page that fits every screen.

### Added

- A measurement panel in the **1. Audio** section of the web interface, in the
  style of the MASH_SUNONA assistant panel. It shows the peak, the integrated
  loudness, the clipped samples, the dc offset, the head silence and the crest
  of the attached file, and under **Own pitch and speed** the key of the song
  and its tempo.
- `POST /api/analyze`, which measures an uploaded file and returns the same
  numbers, together with the facts of the container: the name, the size, the
  length, the sample rate, the number of channels, the bit rate and the
  format. The loudness is measured on each channel of the file as it was
  uploaded, not on a downmix of it.
- The analysis now counts the clipped samples, the dc offset, the head silence
  and the crest factor, and reports them in the result and in its meta block.
- A test file for the measurements, for the analysis route and for the panel
  in the page.

### Changed

- The whole web interface is now one responsive layout. It runs on a phone, a
  tablet, a laptop and a wide desktop, on Windows, Linux, macOS, Android and
  iOS. The type scales with the window, the two columns become one on a
  narrow screen, the controls stack their label above them, the tap targets
  grow on a touch screen, the motion is dropped when the system asks for less
  motion, and the page prints cleanly.
- `mash-mix analyze` measures the channels of a recording as they are. The
  peak of a stereo master is the loudest channel of it and not the average of
  the two, and the BS.1770 loudness is defined per channel. The command line
  and the web panel now report the same numbers for the same file.
- The key of a song is read from the peaks of the spectrum rather than from
  every bin of it. The broadband skirt of a loud note leaks into the pitch
  classes next door, and the harmonic content of a chord does the same. A C
  major triad that carries a few harmonics used to read as E minor, and the
  real song used to read as A major.
- The beat period is searched in steps of a fraction of a frame, and the onsets
  are folded onto the beat grid to tell a beat apart from two beats. The beat
  of a real track does not land on a whole frame, and a search that only looks
  at whole frames reads the wrong tempo whenever it does not. The tempo of the
  real song now reads 111 beats a minute instead of 55, and a click track at
  120, 140 and 180 beats a minute is read at its own tempo.

### Fixed

- The tempo of a track that beats at 120, 140 or 180 beats a minute was read
  at half or a third of its real value. The documented limitation of the
  previous release is gone.

## [0.1.2] - 2026-11-20

Vocal separation that actually runs, and lyric lines that follow the song.

### Added

- `scripts/lyrics_eval.py` now reports a **content recall** figure next to the
  line by line recall. The line by line measure punishes the segmenter for
  the way it cut the transcript, while the content measure asks the question
  that matters: was this true line transcribed at all, wherever the line
  breaks happen to fall.
- A character level repetition filter. A model can repeat a syllable that is
  spelled with several letters, and the word level filter never saw a repeated
  word, so the broken line used to reach the output.
- A test file for the lyric quality filters, the line splitter and the scorer.

### Changed

- The `transformers` backend now runs the model one window at a time instead of
  asking the pipeline to batch several windows together. The batched path proved
  unstable on a CPU and took the whole process down with it.
- The `transformers` backend returns segment level timestamps instead of word
  level ones. Word timestamps ask the model for its full cross attention, and for
  a 30 second window of a large Whisper model that is several gigabytes of
  attention weights. The lyric pipeline cuts its lines from the punctuation and
  the pauses, so it does not need them.
- Lyric lines are cut at the punctuation the model wrote, not only at the
  pauses. A sung line carries almost no pause between words, so the full stop
  and the comma are the more useful signal. A clause break only splits a line
  once enough text has gathered, which keeps short clauses with the phrase
  they belong to.
- A line that carries the unicode replacement character is dropped. A broken
  decode leaves U+FFFD in the string and no lyric contains one.
- A chunk is never longer than the window the model was trained on.

### Fixed

- **Vocal separation never ran.** The Demucs 4.x release changed the call
  signature of `separate_tensor`, so the call failed, the failure was swallowed
  and the error that reached the caller said the package was missing. Both the
  4.x and the 3.x call styles are now supported, so `--separate-vocals` on a
  song now returns the voice instead of the mix.
- The separation model is released before the speech model is loaded, so a song
  with separation no longer needs both networks in memory at the same time.
- The lyric pipeline no longer crashes on a loud passage of a long song.

[0.1.2]: https://github.com/kdbn121/MASH-Mix-Audio-and-Text/releases/tag/v0.1.2

---

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

[0.1.1]: https://github.com/kdbn121/MASH-Mix-Audio-and-Text/releases/tag/v0.1.1

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

[0.1.0]: https://github.com/kdbn121/MASH-Mix-Audio-and-Text/releases/tag/v0.1.0

---

Copyright (c) 2026 MASH (Khaled Bhuiyan). MIT Licence. See LICENSE.
