# Backends

Every backend, how the engine chooses one, and how to add your own.

## What a backend is

A backend is a thin adapter around one inference stack. It has three jobs: report whether it can run on this machine, load its weights, and do the inference in a way the task layer can rely on.

Every backend module follows the same contract, defined in `mash_mix.backends.base`.

| Method | Rule |
|---|---|
| `is_available()` | Never raises. Returns `False` when the runtime is missing |
| `load(model)` | Safe to call twice. Raises `BackendUnavailableError` when the model cannot be loaded |
| `unload()` | Optional. Releases the weights |
| `transcribe(...)` or `synthesize(...)` | Receives mono float32 samples or plain text, returns a `Transcription` or a `SynthesisResult` |
| `info` | A `BackendInfo` with the name, kind, summary, package, extra, languages, platforms and notes |

The module level import guard is what makes the whole thing work on a machine with nothing installed:

```python
try:
    from faster_whisper import WhisperModel
    _HAS_FASTER_WHISPER = True
except Exception:
    WhisperModel = None
    _HAS_FASTER_WHISPER = False
```

## Speech to text backends

### faster-whisper

| | |
|---|---|
| Package | `faster-whisper` |
| Install | `pip install 'mash-mix-audio-and-text[stt]'` |
| Default model | `small` |
| Backend name | `faster-whisper` |
| Platforms | Windows, Linux, macOS, Android, iOS |

The recommended default. It runs the same Whisper weights through CTranslate2, so it is several times faster than the reference implementation on a CPU, and it supports int8 quantisation. Use `large-v3` for the best Bangla accuracy.

Strengths: fastest on a CPU, int8 support, VAD filtering on by default, hotword support in newer releases, word timestamps.

Limits: needs a CTranslate2 build for the platform. The model still has to be downloaded once.

```python
from mash_mix.tasks.transcribe import transcribe

result = transcribe(
    "interview.wav",
    backend="faster-whisper",
    model="large-v3",
    language="bn-BD",
)
print(result.text)
```

```bash
mash-mix transcribe "interview.wav" -o transcript.txt --backend faster-whisper --model large-v3 --lang bn-BD
```

### whisper

| | |
|---|---|
| Package | `openai-whisper` |
| Install | `pip install 'mash-mix-audio-and-text[stt]'` |
| Default model | `small` |
| Backend name | `whisper` |
| Platforms | Windows, Linux, macOS |

The reference OpenAI implementation. It is the fallback when CTranslate2 is not available for the platform.

Strengths: the original code path, so it behaves exactly as the OpenAI release does. Word timestamps are supported.

Limits: slower than `faster-whisper` on a CPU, and it pulls in the full PyTorch stack.

```python
result = transcribe("interview.wav", backend="whisper", model="small", language="bn-BD")
```

### funasr

| | |
|---|---|
| Package | `funasr` |
| Install | `pip install funasr` |
| Default model | `FunAudioLLM/SenseVoiceSmall` |
| Backend name | `funasr` |
| Platforms | Windows, Linux, macOS |

FunASR SenseVoice and Paraformer models. SenseVoice covers Chinese, Cantonese, English, Japanese, Korean and Bangla, and it is very fast on short clips.

Strengths: fast, small models, good Bangla coverage on short segments.

Limits: no Android or iOS build in this package, and the Bangla quality is behind the Whisper large models on long sung audio.

```python
result = transcribe("clip.wav", backend="funasr", language="bn-BD")
```

### transformers

| | |
|---|---|
| Package | `transformers` |
| Install | `pip install transformers torch` |
| Default model | `openai/whisper-large-v3-turbo` |
| Backend name | `transformers` |
| Platforms | Windows, Linux, macOS |

The Hugging Face pipeline. It opens up every model on the Hub, including the Bengali specific ones such as `ehzawad/stt_bn_fastconformer_ctc`, which is the strongest public Bengali model on the public benchmarks.

Strengths: the widest model choice, so it is the way to reach a Bangla specialist model that is not a Whisper derivative. The Bangla fine tunes of Whisper large v3, `mozilla-ai/whisper-large-v3-bn` and `mozilla-ai/whisper-large-v3-turbo-bn`, run here and they are the best choice for a Bangla song.

Limits: the largest dependency. On a CPU the model runs in `bfloat16`, which is about three times faster than `float32` and gives the same text.

The audio is sent to the model one 30 second window at a time, and the timestamps that come back are segment level rather than word level. Word timestamps would ask the model for its full cross attention, and for a 30 second window of a large Whisper model that is several gigabytes of attention weights, which is more than a small machine has to give. The lyric pipeline cuts its lines from the punctuation and the pauses, so it does not need word timings.

```python
result = transcribe(
    "interview.wav",
    backend="transformers",
    model="ehzawad/stt_bn_fastconformer_ctc",
    language="bn-BD",
)
```

For a Bangla song:

```python
result = extract_lyrics(
    "song.mp3",
    backend="transformers",
    model="mozilla-ai/whisper-large-v3-turbo-bn",
    language="bn-BD",
)
```

A checkpoint can also be converted into the CTranslate2 format, which the fast
backend loads, with `scripts/convert_model.py`. Measure the conversion with
`scripts/lyrics_eval.py` before you trust it, because a quantised conversion of
a fine tuned model can lose accuracy.

### sherpa-onnx

| | |
|---|---|
| Package | `sherpa-onnx` |
| Install | `pip install 'mash-mix-audio-and-text[edge]'` |
| Default model | None. You point it at a model directory |
| Backend name | `sherpa-onnx` |
| Platforms | Windows, Linux, macOS, Android, iOS |

ONNX models from the sherpa-onnx project: Whisper, Paraformer, Zipformer and NeMo exports in many languages. The model argument is a directory on disk rather than a Hub id.

Strengths: the lightest runtime in the package, no Python model code, it runs on ONNX Runtime, which is the same runtime that reaches a phone.

Limits: the models are not downloaded for you, so you fetch and convert them yourself.

```python
result = transcribe("clip.wav", backend="sherpa-onnx", model="/models/sherpa-onnx-whisper-bn")
```

### whisper.cpp

| | |
|---|---|
| Package | `pywhispercpp` |
| Install | `pip install pywhispercpp` |
| Default model | None. You point it at a ggml file |
| Backend name | `whisper.cpp` |
| Platforms | Windows, Linux, macOS, Android, iOS |

Quantised ggml models from whisper.cpp. The model argument is a file path.

Strengths: the smallest footprint of the Whisper family, which makes it the best choice for a mobile build. Quantised models run on a phone.

Limits: no Python model code at all, so word level control is limited to what the binding exposes.

```python
result = transcribe("clip.wav", backend="whisper.cpp", model="/models/ggml-small.bin")
```

### whisperx

| | | |
|---|---|---|
| Package | `whisperx` | 
| Install | `pip install 'mash-mix-audio-and-text[align]'` | 
| Default model | `small` | 
| Backend name | `whisperx` | 
| Platforms | Windows, Linux, macOS | 

WhisperX does three things a plain recogniser does not. It aligns the words of a transcript to the sound, so the time of every word is exact rather than approximate. It can tell who is speaking, so the `speaker` field of every segment is filled in. And it runs the same Whisper weights through CTranslate2, so it is quick without a graphics card.

Strengths: it is the only backend in this package that names the speakers. A song with two voices and a conversation with four of them both come back with the name of the speaker on every line, which is what a transcript of a conversation needs.

Limits: the speaker part needs a model from Hugging Face, so it needs a token in the `HF_TOKEN` environment variable. Without a token the words are still aligned and the speakers are left empty. Everything runs on this machine, and the model is downloaded once.

```python
result = transcribe("clip.wav", backend="whisperx", model="small")
```

## Hosted recognisers

These three run on somebody else's computer. They are never chosen by the automatic selection, because the audio would leave the machine without being asked. They are named on purpose:

```bash
mash-mix transcribe song.wav --backend elevenlabs
mash-mix transcribe song.wav --backend azure-stt
```

Each one needs an account and a key in the environment. Without the key the backend reports itself unavailable and is never used.

### elevenlabs

| | | |
|---|---|---|
| Package | None, the standard library is enough | 
| Install | Put the key in `ELEVENLABS_API_KEY` | 
| Default model | `scribe_v3` | 
| Backend name | `elevenlabs` | 
| Platforms | Windows, Linux, macOS | 

The hosted Scribe model of ElevenLabs. It is the strongest hosted recogniser that can be pointed at a Bangla song without a download and without the memory a large local model needs. The word level timings come back with the result, so the karaoke subtitle keeps working.

### azure-stt

| | | |
|---|---|---|
| Package | None, the standard library is enough | 
| Install | Put the key in `AZURE_SPEECH_KEY` and the region in `AZURE_SPEECH_REGION` | 
| Default model | `azure-fast-transcription` | 
| Backend name | `azure-stt` | 
| Platforms | Windows, Linux, macOS | 

The Fast Transcription route of Azure AI Speech. It hears many languages well without a download, it gives the time of every word, and it can name the speakers without a Hugging Face token, which is the one thing WhisperX cannot do without one.

Strengths: the speaker labels need nothing but the Azure key. The locales offered when no language is named put Bangla first, because this package is built for Bangla.

Limits: the audio is sent to Azure, so it is opt in. A file longer than about two hours wants the Azure batch route instead.

```python
result = transcribe("song.wav", backend="azure-stt", language="bn")
```

### dashscope-asr

| | | |
|---|---|---|
| Package | None, the standard library is enough | 
| Install | Put the key in `DASHSCOPE_API_KEY` | 
| Default model | `qwen3-asr-flash-filetrans` | 
| Backend name | `dashscope-asr` | 
| Platforms | Windows, Linux, macOS | 

The Qwen3 ASR model of Alibaba DashScope. It gives the time of every word, which is what a karaoke subtitle needs.

Limits: DashScope fetches the audio itself, from a place on the internet it can reach. A file on this machine cannot be named to it, so this backend needs the address of the file:

```python
result = transcribe(
    "song.wav",
    backend="dashscope-asr",
    language="bn",
    audio_url="https://example.com/song.wav",
)
```

Without an address the backend says so plainly and returns nothing, rather than pretending to work.

## Text to speech backends

### mms-tts

| | |
|---|---|
| Package | `transformers` |
| Install | `pip install 'mash-mix-audio-and-text[tts]'` |
| Model | `facebook/mms-tts-ben` |
| Backend name | `mms-tts` |
| Licence | CC BY-NC 4.0, non commercial |
| Platforms | Windows, Linux, macOS, Android, iOS |

A Bangla only VITS model from the MMS family, at 145 M.

Strengths: small, dependable, and it runs everywhere.

Limits: **the licence is non commercial.** Check it before you ship a product. Read [models.md](models.md) for the full licence warning.

```python
from mash_mix.tasks.synthesize import synthesize

result = synthesize("আজকের দিনটা সুন্দর", backend="mms-tts", language="bn-BD")
```

### vits

| | |
|---|---|
| Package | `TTS` |
| Install | `pip install TTS` |
| Model | `EMTIAZZ/bangladeshi-bangla-tts-vits` |
| Backend name | `vits` |
| Licence | Apache 2.0 |
| Platforms | Windows, Linux, macOS, Android, iOS |

A VITS fine tune trained on Bangladeshi Bangla, at 154 M, with local pronunciation and intonation. The model argument is a local checkpoint folder that holds `config.json` and the `.pth` file, so download the weights first and point the backend at the folder.

Strengths: the closest thing to a native Bangladeshi Bangla voice in the open weights space, and the licence permits commercial use.

Limits: one voice, one region. It does not do Indian Bengali, and it needs the checkpoint on disk rather than a model id.

```python
result = synthesize(
    "আজকের দিনটা সুন্দর",
    backend="vits",
    model="/models/bangladeshi-bangla-tts-vits",
    language="bn-BD",
)
```

### indic-tts

| | |
|---|---|
| Package | `transformers` |
| Install | `pip install 'mash-mix-audio-and-text[tts]'` |
| Model | `ai4bharat/IndicTTS` |
| Backend name | `indic-tts` |
| Licence | MIT |
| Platforms | Windows, Linux, macOS |

IndicTTS covers 22 Indic languages including Bangla, at 1.2 B, with a large speaker set.

Strengths: the licence is permissive and the speaker set is wide, so it covers both Bangla regions and several voices.

Limits: large, and desktop only in this package.

### parler-tts

| | |
|---|---|
| Package | `transformers` |
| Install | `pip install 'mash-mix-audio-and-text[tts]'` |
| Model | `ai4bharat/indic-parler-tts` |
| Backend name | `parler-tts` |
| Licence | Apache 2.0 |
| Platforms | Windows, Linux, macOS |

Indic Parler TTS covers 23 Indic languages including Bangla, at 880 M. The voice is steered by a written description, for example a warm Bangladeshi female voice.

Strengths: you choose the voice with words, which is a good way to find a voice that matches a script.

Limits: the largest of the Bangla text to speech options, and it needs a GPU to be comfortable.

```python
result = synthesize(
    "আজকের দিনটা সুন্দর",
    backend="parler-tts",
    language="bn-BD",
    voice="A warm Bangladeshi female voice, slow and clear",
)
```

### piper

| | |
|---|---|
| Package | `piper-tts` |
| Install | `pip install piper-tts` |
| Model | A Piper ONNX voice file |
| Backend name | `piper` |
| Licence | MIT for the runtime, each voice keeps its own licence |
| Platforms | Windows, Linux, macOS, Android, iOS |

Piper voices in ONNX, between 20 M and 120 M each.

Strengths: the lightest option and the one that runs well on a phone. Fast enough to be usable interactively.

Limits: the Bangla voices are community files, so check the licence of the voice you pick.

### xtts

| | |
|---|---|
| Package | `TTS` |
| Install | `pip install TTS` |
| Model | `coqui/XTTS-v2` |
| Backend name | `xtts` |
| Licence | CPML, non commercial |
| Platforms | Windows, Linux, macOS |

XTTS v2 covers 17 languages including Bangla and does zero shot voice cloning from a short reference clip.

Strengths: voice cloning, and it handles a Bangla reference clip well.

Limits: **the CPML licence is non commercial.** It is also large, at 1.8 B.

```python
result = synthesize(
    "আজকের দিনটা সুন্দর",
    backend="xtts",
    language="bn-BD",
    reference_audio="my-voice.wav",
)
```

### chatterbox

| | |
|---|---|
| Package | `chatterbox-tts` |
| Install | `pip install chatterbox-tts` |
| Model | `ResembleAI/chatterbox-multilingual` |
| Backend name | `chatterbox` |
| Licence | MIT |
| Platforms | Windows, Linux, macOS |

A general purpose multilingual voice at 500 M.

Strengths: MIT licensed and easy to install.

Limits: **it has no Bangla voice in the current multilingual release.** It is a fallback for other languages, not a Bangla option.

### edge-tts

| | |
|---|---|
| Package | `edge-tts` |
| Install | `pip install edge-tts` |
| Model | `bn-BD-NabanitaNeural` and `bn-IN-SwaraNeural` |
| Backend name | `edge-tts` |
| Licence | Microsoft online service terms |
| Platforms | Any device with a network connection |

The Microsoft Edge neural voices, one for each Bangla region.

Strengths: the reference quality for Bangladeshi Bangla, with no weights to download and no GPU.

Limits: **online only.** The audio leaves the machine, so it will not work offline and it is not suitable for private content.

```python
result = synthesize("আজকের দিনটা সুন্দর", backend="edge-tts", language="bn-BD")
```

## Speech to speech backends

### pipeline

| | |
|---|---|
| Package | An STT backend and a TTS backend |
| Backend name | `pipeline` |
| Platforms | Windows, Linux, macOS, Android, iOS |

The default speech to speech path. It recognises the speech, optionally translates it, then speaks the text again with a synthesis backend.

Strengths: it inherits the language layer, so Bangla preparation, numeral expansion and symbol expansion all apply to the spoken output. Any STT and TTS pair works.

Limits: it is two models and two passes, so it is slower than a native duplex model and it loses the original prosody.

```python
from mash_mix.tasks.speech_to_speech import speak

result = speak("recording.wav", target_language="bn-BD")
```

## Automatic selection

With `backend="auto"`, which is the default everywhere, the task asks the registry for the list of backends and walks the preference list. The first backend that reports itself available wins.

Speech to text order, from `tasks._common._STT_PREFERENCE`:

1. `faster-whisper`
2. `whisper`
3. `funasr`
4. `transformers`
5. `sherpa-onnx`
6. `whisper.cpp`

Text to speech order, from `tasks._common._TTS_PREFERENCE`:

1. `mms-tts`
2. `vits`
3. `indic-tts`
4. `parler-tts`
5. `piper`
6. `xtts`
7. `chatterbox`
8. `edge-tts`

The order is a quality first order. The light backends are further down the list, so a machine with `faster-whisper` installed never falls back to `whisper.cpp` unless you ask for it.

If none of the preferred backends is available the registry falls back to any available backend of that kind. If there is none at all, the call raises a `RuntimeError` that names the extra to install.

### Forcing a backend

Pass the name on the command line or in the function call.

```bash
mash-mix transcribe "interview.wav" --backend sherpa-onnx --model /models/whisper-bn
mash-mix speak "আজকের দিনটা সুন্দর" --backend piper
```

```python
transcribe("interview.wav", backend="whisper.cpp", model="/models/ggml-small.bin")
synthesize("হ্যালো", backend="edge-tts")
```

An unknown name raises a `ValueError` that lists the known backends.

### Listing what is installed

```bash
mash-mix backends
```

```
[ok ] stt faster-whisper   CTranslate2 Whisper, fast on CPU with int8 quantisation.
[no ] stt funasr           FunASR SenseVoice and Paraformer models, fast with Bangla support.
[ok ] stt whisper          Reference OpenAI Whisper implementation with word timestamps.
```

```python
from mash_mix.backends.registry import backend_infos

for info in backend_infos():
    print(info["name"], info["kind"], info["available"])
```

`mash-mix doctor` prints the same list with the platform, the ffmpeg version, the free disk space and the optional packages that are installed.

## Adding a custom backend

Write the module, register it with the decorator, and add it to the discovery list. The full worked example is in [architecture.md](architecture.md).

The short version:

```python
from mash_mix.backends.base import BackendInfo, STTBackend
from mash_mix.backends.registry import register_stt


@register_stt("my-asr")
class MyAsrSTTBackend(STTBackend):
    info = BackendInfo(
        name="my-asr",
        kind="stt",
        summary="One line description.",
        package="my-asr-package",
        extra="stt",
        languages=["bn", "bn-BD"],
        platforms=["linux"],
    )

    def is_available(self) -> bool:
        return True

    def load(self, model=None, **kwargs) -> None:
        self._loaded = True
        self._model_id = str(model or "default")

    def transcribe(self, audio, sample_rate, **kwargs):
        ...
```

Then add the module path to `_DISCOVERY_MODULES` in `backends/registry.py`, and to `_STT_PREFERENCE` in `tasks/_common.py` if it should win the automatic selection.

## Memory and speed guidance

The numbers below are rough planning figures, not benchmarks. Measure on your own machine before you commit to a hardware plan.

| Model size | Rough weights on disk | Rough memory needed | Speed on a modern CPU | Speed on a GPU |
|---|---|---|---|---|
| tiny, 39 M | 75 MB | 400 MB | fastest, usable for a smoke test | comfortable |
| base, 74 M | 150 MB | 600 MB | fast | comfortable |
| small, 244 M | 500 MB | 1 GB | a good default | comfortable |
| medium, 769 M | 1.5 GB | 2 GB | slow, roughly a quarter of the real time factor | comfortable |
| large-v3, 1.55 B | 3 GB | 4 GB | impractical, several times the real time factor | comfortable |

| Task | Small model | Large model |
|---|---|---|
| Speech to text, one minute of audio | a few seconds on a CPU | a few seconds on a GPU, minutes on a CPU |
| Text to speech, one sentence | well under a second | a second or two on a GPU |

Rules of thumb:

- A phone gets the small model at int8, or the medium model if the phone is new enough.
- A laptop gets the small model by default and the large model when the result has to be right.
- A server with a GPU gets the large model and runs several jobs at once.
- Disk space matters as much as memory. The weights are cached in the model cache directory, and each model is a separate copy.

```bash
mash-mix doctor
```

The doctor prints the free disk space and reminds you that model weights need roughly 1 to 4 GB each.

## Quantisation notes for the Whisper family

The Whisper weights are the same across `whisper`, `faster-whisper`, `whisper.cpp` and `sherpa-onnx`. What changes is how they are stored and how they are executed.

| Backend | Precision | Notes |
|---|---|---|
| `faster-whisper` | int8 by default, float16 on a GPU | The backend picks the compute type automatically: float16 when CTranslate2 reports a CUDA device or when torch reports CUDA, int8 otherwise |
| `whisper` | float32 on a CPU, float16 on a GPU | The reference implementation, no int8 path |
| `whisper.cpp` | ggml quantised, typically q5 or q8 | The smallest download and the smallest memory use |
| `sherpa-onnx` | int8 ONNX | Runs on ONNX Runtime, which is what reaches a phone |

```python
# The compute type is chosen for you. You can override it in the load call.
backend.load("large-v3", compute_type="int8")
```

Practical advice:

- int8 costs a small amount of accuracy and halves the memory. For Bangla, which has a smaller training set than English, stay at int8 or better and do not go below it.
- float16 on a GPU is the best of both. The `faster-whisper` backend selects it automatically when it finds a CUDA device.
- float32 is only worth it on a CPU when you are chasing the last few per cent of accuracy and you have the memory.
- Quantised ggml and ONNX models are the ones to use on a phone, because they are the ones that fit.

---

Copyright (c) 2026 MASH (Khaled Bhuiyan). MIT Licence. See LICENSE.
