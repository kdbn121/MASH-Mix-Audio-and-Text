# Lyrics

The song to lyric workflow: how it runs, what each stage does, how to get a better result, and what to expect.

## The workflow

### From the command line

```bash
mash-mix lyrics "bangla song.mp3" -o out --format srt --format txt --lang bn-BD
```

```
out/lyrics.srt
out/lyrics.txt
```

### From Python

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
print(written["srt"])   # out/lyrics.srt
```

The engine facade does the same thing with one configuration:

```python
from mash_mix import MASHMix

engine = MASHMix(language="bn-BD")
lyrics = engine.lyrics("bangla song.mp3")
engine.write(lyrics, "out", formats=["srt", "txt", "vtt", "lrc"])
```

### From a link

A link is a way of naming a piece of audio, so it can be used where a file was used.

```python
from mash_mix.tasks.lyrics import extract_lyrics_from_url

outcome = extract_lyrics_from_url(
    "https://www.youtube.com/watch?v=296Gi_nnbho",
    "out",
    language="auto",
)

result = outcome["result"]
print(result.meta["lyrics_source"])   # site subtitles
print(result.language)                # bn-BD
```

The fetching is done with `yt-dlp`, an optional dependency, so the engine imports and works
without it and only a request for a link needs it.

## The subtitles the page carries

A page that publishes its own subtitles is the best source of lyrics there is. The words are
right, the timings are right, and nothing has to be recognised. The engine asks for the track
before it reads the audio, and uses it when the page has one.

| Step | What happens |
|---|---|
| 1. Read the page | The fetcher reports the tracks the page offers, with the language of each |
| 2. Rank them | The language that was asked for comes first, then the plain language, then the other region |
| 3. Fetch one | The first track that arrives is used. A site that refuses a track does not stop the run |
| 4. Read it | SRT and WebVTT are both read. The cue number, the identifier, the voice labels, the styling and the karaoke marks are removed |
| 5. Work out the language | The track names its own language, so it is known rather than guessed |
| 6. Work out the region | A Bangla text carries `bn-BD` or `bn-IN`, read out of the text itself |

## The words on the picture of a karaoke

A karaoke has no subtitles, and often nobody is singing either, so the words of the song are on the
screen and nowhere else. The engine downloads the picture and reads it.

| Step | What happens |
|---|---|
| 1. Read a few frames | The frames are spread over the middle of the video, so the introduction and the end credits do not decide the matter |
| 2. Follow the colour | The colour of every place where text was found is followed over the whole video |
| 3. Keep the places that sweep | A karaoke sweeps the line along as it is sung, so the colour of the letters moves. A name and a logo do not move |
| 4. Read the words | The band of the rows that sweep is read at every step, and the words of each row are followed |
| 5. Take the reading most frames agree on | A reader that is nearly right is wrong in a different way every time it is tried |
| 6. Judge the line | The words must be in the script of the song, they must not run across the whole width of the picture, and the colour of their letters must travel in one direction while they are on the screen |
| 7. Take the times from the travel | The moment the colour starts to move is the moment the line is sung, and the moment it stops is the moment the line has been sung |

A karaoke usually shows the line that is being sung and the line that comes next, so the same words are
swept twice. The first sweep is the one that was sung, so each line of the song is written once.

A video with nothing that sweeps is not a karaoke, and the survey says so without spending the rest of
the work on it. The run carries on to the model instead.

The reader of the picture needs the karaoke extra:

```bash
pip install "mash-mix-audio-and-text[karaoke]"
```

Without it the run says so in the report and carries on. The command line has `--karaoke auto|on|off`
and the HTTP route has a `karaoke` field of the same three values. The default is `auto`, which reads
the picture when the page carries no subtitles of its own.

A karaoke takes a long time to read, because every second of the video has to be read and every word
has to be recognised. A five minute song takes about seven minutes on an ordinary computer, and the
reading of the picture is the slow part. The video is read through rather than jumped to, so the
reader does not pay for finding it again at every moment, and the report says where the time went.
The panel shows a bar that moves while the engine is at it.

A link that has been read is answered at once the second time, because the words of the picture are
kept.

A track in the wrong language is never used. A track in another language is worse than no track
at all, so the engine falls back on reading the audio rather than on the wrong words.

```python
from mash_mix.export.subtitles import parse_subtitles
from mash_mix.sources import read_subtitle_file

text = read_subtitle_file("subs.bn.srt", language="bn")
lines = parse_subtitles(text)
```

The reader handles both spellings of a timestamp, with or without the hours, and it reads a file
written in the code page of its own language. A file with a byte order mark is read as what the
mark says it is.

## The stages

| Stage | What happens | Why |
|---|---|---|
| 1. Decode | The file is decoded to mono float32 at 16 kHz | Every model wants the same format, and the resampling happens once |
| 2. Separate (optional) | The vocal stem is pulled out of the mix | The model hears the voice, not the instruments |
| 3. Pick a backend | The registry returns the first available speech to text backend | One call works on every platform |
| 4. Chunk | The audio is split into 30 second windows with 2 seconds of overlap | The overlap keeps a word that straddles a boundary from being lost |
| 5. Recognise | Each chunk is transcribed with timestamps and VAD on | VAD skips the silent passages instead of hallucinating lyrics over them |
| 6. Shift | Every timestamp is moved by the chunk offset | The times are absolute in the original track, not in the chunk |
| 7. Clean | `clean_asr_text` removes bracket noise and repeated punctuation | A model that hears an instrumental often writes `[music]` or `[আড়ালা]` |
| 8. Split into lines | The recognised run is cut at the full stops, the commas, the pauses and the length limit | A sung line carries almost no pause, so the punctuation the model wrote is the better signal |
| 9. Drop the noise | A line that repeats itself, word by word or character by character, is dropped, and so is a line carrying a replacement character | A decoder that loses the plot writes the same syllable until it runs out of tokens |
| 10. Normalise | Bangla text goes through `normalize_bangla` | The nukta letters and the digits end up in one canonical form |
| 11. Repetition control | Consecutive identical lines beyond `max_repeats` are dropped | A decoder that loops on a sung phrase repeats it for the rest of the track |
| 12. Merge short lines | A fragment under 1.2 seconds joins the next line when the gap is under 0.8 seconds | A phrase split in two reads badly in a subtitle |
| 13. Detect the region | The marker vote runs over the joined text | The result carries `bn-BD` or `bn-IN` and the evidence behind it |
| 14. Correct | The words a recogniser always gets wrong are replaced | A song that is sung the same way every time gives the same mistake every time |
| 15. Export | SRT, TXT, VTT, LRC, karaoke or JSON is written | The format you need for the player you use |

The whole thing runs with no size limit, because the audio is processed chunk by chunk rather than as one array.

## Separation options

| Setting | Model | What happens | When to use it |
|---|---|---|---|
| Off, the default | none | The mix is recognised as it is | A clean vocal recording, a spoken track, or a quick first look |
| `--separate-vocals` | `htdemucs` | The track is split with Demucs and the vocal stem is recognised | A dense modern mix where the voice competes with the instruments |
| `--separate-vocals` | `mid_side` | A dependency free mid side split | When you cannot install Demucs, or you want a fast rough split |

```bash
mash-mix lyrics "song.mp3" -o out --separate-vocals --separation-model htdemucs
```

```bash
mash-mix lyrics "song.mp3" -o out --separate-vocals --separation-model mid_side
```

The cost is time and disk. Demucs needs an 80 M download and roughly as much memory as the audio, and it runs before the recognition, so a four minute song takes noticeably longer.

Separation also has a downside: a separation model can introduce artefacts of its own, and it can drop quiet consonants. If the mix is already clean, separating it can make the result worse. Try both and keep the better one.

A separation failure is not fatal. The pipeline catches the error and carries on with the original mix.

## Getting a better result

### The language tag

Set it. The default is `bn-BD`, and a wrong tag costs accuracy.

| Tag | When |
|---|---|
| `bn-BD` | Bangladeshi Bangla |
| `bn-IN` | Indian Bengali |
| `bn` | Let the model choose |
| none | Let the model detect, and the regional detector label the result |

```bash
mash-mix lyrics "song.mp3" -o out --lang bn-IN
```

### Hotwords

A hotword is a word or phrase the model is nudged towards. Use it for names, rare words and anything the model keeps getting wrong.

```bash
mash-mix lyrics "song.mp3" -o out --hotword "কবিতা" --hotword "টুকু" --hotword "MASH"
```

The hotwords are folded into the decoder prompt, so they work even on the backends that do not have a separate hotword argument.

### The corrections table

A hotword nudges the model. A correction replaces the word afterwards, so it is certain.

The built in table holds the English spellings that a recogniser is known to get wrong, such as `recieve` for `receive` and `seperate` for `separate`. It is applied to whole words, so `cloud` is not changed inside `cloudy`, and the case and the punctuation of the word are kept.

A table of your own goes on top of the built in one, so your own song can be corrected the way your own song spells it:

```python
from mash_mix.lang.corrections import apply_corrections

text = apply_corrections("Ore nil doriya", {"doriya": "doriya"})
```

The number of corrected lines is in the result under `meta["corrected_words"]`.

### The initial prompt

An initial prompt gives the model the style and the subject of the song, which sharpens the decoder.

```bash
mash-mix lyrics "song.mp3" -o out --prompt "একটি বাংলা গানের গান, ভালোবাসার বিষয়"
```

A prompt in the target language works better than a prompt in English.

### The chunk size

The default is 30 seconds with 2 seconds of overlap.

| Setting | Effect |
|---|---|
| A shorter chunk, say 15 seconds | More context per unit of work, but the model has less history to work with and a chorus can be split badly |
| A longer chunk, say 60 seconds | More context for the decoder, but more memory and more risk of a loop |
| A bigger overlap, say 4 seconds | Fewer words lost at a boundary, at the cost of some duplicated work |

```bash
mash-mix lyrics "song.mp3" -o out --chunk-seconds 20 --overlap-seconds 4
```

### Separating the vocals

Covered above. Use it when the voice is fighting with the mix.

### The repetition control

`--max-repeats` is how many consecutive identical lines are kept before the rest are dropped. The default of 2 keeps a genuine repeated chorus line and removes a decoding loop.

```bash
mash-mix lyrics "song.mp3" -o out --max-repeats 1
```

Lower it to 1 if the model still loops, raise it to 3 if a real chorus is being cut.

### The model

This is the biggest single lever on a Bangla song. A general Whisper model writes
Bangla poorly, and on a song it often answers in romanised Bangla instead of Bangla
script. The Bangla fine tunes of Whisper large v3 do not have that problem.

```bash
mash-mix lyrics "song.mp3" -o out \
  --backend transformers \
  --model mozilla-ai/whisper-large-v3-turbo-bn
```

| Model | Word error rate on Bengali | Speed |
|---|---|---|
| `openai/whisper-large-v3` | 55 percent | slow |
| `mozilla-ai/whisper-large-v3-bn` | 9.7 percent | slow |
| `mozilla-ai/whisper-large-v3-turbo-bn` | 11 percent | four times faster, the recommended one |

The fine tunes run through the `transformers` backend, so `--backend transformers` goes
with them.

### Measuring the result

`scripts/lyrics_eval.py` scores the engine against the real lyric file of a song.

```bash
python scripts/lyrics_eval.py "song.mp3" "true-lyrics.txt" \
  --backend transformers --model mozilla-ai/whisper-large-v3-turbo-bn
```

It reports the word error rate, the share of true lines that were found with a close
match, the share of true lines found anywhere in the transcript, and every line that
was missed. The report is written as JSON when `--out-dir` is given, so two runs can
be compared.

The three numbers answer different questions.

| Number | Question it answers |
|---|---|
| Word error rate | How many words are wrong, over the whole song |
| Line recall | Was this true line returned as one whole line |
| Content recall | Was this true line transcribed at all, wherever the line breaks fall |

Content recall is the fair one for a song, because the line breaks are the
segmenter's decision and not the recogniser's. A lyric that is right but written as
one long line scores badly on line recall and well on content recall.

#### A measured song

The song *Dol Dol Doloni*, 4 minutes 22 seconds, against its real lyric file, on a CPU
with no graphics card. Every row is the same scorer over the same truth file.

| Setup | Word error rate | Line recall | Content recall | Time |
|---|---|---|---|---|
| General model, raw mix | 97.4 percent | 22.8 percent | 22.8 percent | 431 s |
| Bangla fine tune, raw mix, 0.1.1 pipeline | 77.1 percent | 0.0 percent | 64.9 percent | 367 s |
| Bangla fine tune, raw mix, 0.1.2 pipeline | 76.2 percent | 26.3 percent | 71.9 percent | 324 s |
| Bangla fine tune, vocals separated, 0.1.2 pipeline | 71.9 percent | 3.5 percent | 47.4 percent | 207 s |

The fine tune is the whole difference between the first row and the second. The third
row is the line splitting: the lines now follow the punctuation the model wrote, so a
lyric that used to come back as one long run of words comes back as the lines of the
song.

The last row is the interesting one. Separation gives the most accurate words and the
fastest run, because the model has less to listen to, but it covers fewer of the
distinct lines, because Demucs also took the room reverb and the backing vocal with
the voice. That is why separation is a choice and not a default.

### The output script

`--script` chooses the writing system of the lines.

| Value | Result |
|---|---|
| `bangla` | Bangla script. Any romanised Bangla the recogniser produced is written out in Bangla |
| `banglish` | Latin letters |

```bash
mash-mix lyrics "song.mp3" -o out --script banglish
```

The decision is made over the whole song, not line by line, so an English line inside
a Bangla song is not rewritten.

## The output formats

The same lyric lines in four formats.

### SRT

The SubRip subtitle format. The format the lyric download button produces.

```
1
00:00:00,000 --> 00:00:02,500
প্রথম লাইন

2
00:00:02,500 --> 00:00:05,125
দ্বিতীয় লাইন
```

Timestamps are `HH:MM:SS,mmm`, the cues are numbered from 1, and there is a blank line between the cues.

### TXT

The plain text, one line per lyric line, with no timestamps.

```
প্রথম লাইন
দ্বিতীয় লাইন
```

### VTT

The WebVTT format. The same as SRT with a header and dot separated timestamps.

```
WEBVTT
1
00:00:00.000 --> 00:00:02.500
প্রথম লাইন

2
00:00:02.500 --> 00:00:05.125
দ্বিতীয় লাইন
```

With a title, the header carries a NOTE block:

```
WEBVTT
NOTE
Title: Kobita tuku
```

### LRC

The karaoke format. The timestamp is in brackets and the line follows it directly.

```
[00:00.00]প্রথম লাইন
[00:02.50]দ্বিতীয় লাইন
```

The LRC timestamp is in minutes, seconds and hundredths.

### Karaoke SRT

An SRT file with the word that is being sung marked inside the line. The whole line is on the screen the whole time, and the mark travels along it as the song is sung.

This needs the time of every word, so it needs a backend that gives word level timings. The local and hosted recognisers all give them, and a line that came from a page subtitle does not, so a page subtitle is written the plain way.

```
1
00:00:00,000 --> 00:00:00,500
<b>ওরে</b> নীল দরিয়া

2
00:00:00,500 --> 00:00:01,000
ওরে <b>নীল</b> দরিয়া

3
00:00:01,000 --> 00:00:01,600
ওরে নীল <b>দরিয়া</b>
```

Ask for it by name:

```bash
mash-mix lyrics song.mp3 -o out --format karaoke
mash-mix transcribe song.wav --format karaoke
```

The ASS format carries the same thing in the way that format expects, with the `k` tag that tells a player to change the colour of a syllable as it is sung:

```
Dialogue: 0,0:00:00.00,0:00:01.60,Default,,0,0,0,,{\k50}ওরে {\k50}নীল {\k60}দরিয়া
```

### JSON

The full result object, for a program that wants the timings and the metadata.

```json
{
  "text": "প্রথম লাইন\nদ্বিতীয় লাইন",
  "language": "bn-BD",
  "duration": 214.3,
  "title": "Kobita tuku",
  "artist": "MASH",
  "backend": "faster-whisper",
  "model": "large-v3",
  "elapsed": 61.8,
  "lines": [
    {"start": 0.0, "end": 2.5, "text": "প্রথম লাইন", "speaker": null, "probability": null}
  ],
  "meta": {"chunks": 8, "dialect": "dhaka", "script": "bangla"}
}
```

### Line wrapping

A subtitle line is wrapped at 42 characters and 2 lines, on word boundaries. A single word longer than the limit is split by character, because Bangla words can be long.

```
এটি একটি বেশ লম্বা বাংলা লাইন
যা সাবটাইটেলের জন্য দুই লাইনে
ভাঙা উচিত
```

## Troubleshooting

### No lyrics came back

| Cause | Fix |
|---|---|
| No speech to text backend is installed | `mash-mix doctor`, then install the `stt` extra |
| The audio is too quiet or the voice is buried in the mix | Add `--separate-vocals` |
| The language tag is wrong | Try `--lang bn-BD` and `--lang bn-IN` |
| The file is not really audio | Check it plays. The doctor prints the ffmpeg version |
| The song has no vocals at all | Nothing to extract. The detector reports no speech |

The result object tells you what happened even when it is empty:

```python
result = engine.lyrics("song.mp3")
print(result.meta["chunks"])       # how many chunks were processed
print(result.meta["evidence"])     # the regional detection evidence
print(result.backend, result.model)
```

### The language came back wrong

The model returned the wrong language, or the region detector voted for the wrong side.

| Cause | Fix |
|---|---|
| The tag was not set | Pass `--lang bn-BD` or `--lang bn-IN` |
| The text is short | The detector scales its confidence by the length of the text, so a short line is never fully confident |
| The markers are unusual | Look at `result.meta["evidence"]`. It lists the markers that produced the call |

The detection is explainable on purpose. Read the evidence and you can see which word tipped it.

### The same line repeats over and over

That is a decoding loop, which happens when the model hears a phrase it cannot resolve.

| Fix | Effect |
|---|---|
| `--max-repeats 1` | Drops the repeats sooner |
| `--separate-vocals` | The model hears a cleaner voice and loops less |
| A shorter `--chunk-seconds` | A loop inside one chunk does not spread as far |
| `--hotword` | Naming the real lyric steers the decoder away from the loop |

### The timestamps drift

The times are the model's word timings, so they drift when the model mishears the length of a phrase.

| Fix | Effect |
|---|---|
| A shorter `--chunk-seconds` | Less time for the drift to accumulate inside one chunk |
| `--separate-vocals` | The word boundaries are easier to find in a clean vocal |
| A larger model, such as `large-v3` | Better word timings, not only better words |
| Word timestamps on | They are on by default in the lyric pipeline |

### The music is too loud

| Fix | Effect |
|---|---|
| `--separate-vocals --separation-model htdemucs` | The best separation available here |
| `--separation-model mid_side` | A quick, dependency free split |
| A larger model | A stronger model copes with a worse mix |
| A hotword for the key word | Steers the decoder through the noise |

### The Bangla comes out in the wrong script

| Cause | Fix |
|---|---|
| The text was romanised | `mash_mix.lang.banglish_to_bangla` converts it, but the recognition is what needs fixing |
| A typo in the source | Fix the file, then run again |
| A mixed language song | Pass a hotword in the Bangla script so the decoder sees the script it expects |

## The accuracy ceiling

Be honest about this: a speech model trained on spoken voice is being asked to transcribe a sung voice, and it is outside its training distribution.

- **Singing changes the acoustics.** Pitch, duration and vowel length are all stretched, so a model that was tuned on speech makes timing errors.
- **The melody carries the words, not the consonants.** A trained singer can be understood from the tune and the context even when the consonants are barely there. A model has neither.
- **Bangla is a small training share.** Even the large Whisper models saw far less Bangla speech than English speech, and far less Bangla singing than either.
- **A dense mix makes it worse.** The voice competes with the instruments for the model's attention.

So the honest expectation is this:

- On a clean studio vocal, a large model with the language tag set gets most of the words right. Expect to fix some.
- On a dense mix, separation helps a great deal and the result still needs editing.
- On a live recording with a band, expect a rough transcript that identifies the song and its structure, not a publishable lyric sheet.
- The timings are the more reliable part. The words are the hard part.

The way to get closer is the same every time: set the language tag, separate the vocals, use the largest model the hardware allows, give the decoder the right hotwords, and be ready to correct the output by hand.

That is the honest state of running a speech model on a sung voice in 2026. The pipeline makes the result usable. It does not make it perfect.

---

MASH-Mix-Audio-and-Text V0.6.2. Copyright (c) 2026 MASH (Khaled Bhuiyan). MIT Licence. See LICENSE.
