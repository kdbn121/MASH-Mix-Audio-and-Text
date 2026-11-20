# Models

Every model the engine knows about, what it is for, which licence it carries and how to choose one.

## The catalogue

The catalogue lives in `mash_mix.models` and is served over HTTP at `/api/models`. Every entry records the task, the backend that drives it, the size, the language coverage, the Bangla support in words, the licence and the platforms.

| Model | Task | Backend | Size | Languages | Bangla | Licence | Platforms |
|---|---|---|---|---|---|---|---|
| `openai/whisper-large-v3` | stt | whisper | 1.55 B | 99 languages | yes, the strongest general purpose option | MIT | Windows, Linux, macOS, Android, iOS |
| `Systran/faster-whisper-large-v3` | stt | faster-whisper | 1.55 B, int8 available | 99 languages | yes, the same weights with CTranslate2 speed | MIT | Windows, Linux, macOS, Android, iOS |
| `ehzawad/stt_bn_fastconformer_ctc` | stt | transformers | 117 M | Bengali | yes, built and tuned for Bengali | MIT | Windows, Linux, macOS, Android, iOS |
| `FunAudioLLM/SenseVoiceSmall` | stt | funasr | 234 M | Chinese, Cantonese, English, Japanese, Korean, Bangla and more | yes, Bangla is in the training mix | Apache 2.0 | Windows, Linux, macOS |
| `Qwen/Qwen3-ASR-1.7B` | stt | transformers | 1.7 B | Chinese, English, Japanese, Korean, European and more | partial, strong on multilingual but not Bangla specialised | Apache 2.0 | Windows, Linux, macOS |
| sherpa-onnx offline models | stt | sherpa-onnx | 40 M to 800 M | Whisper, Paraformer, Zipformer and NeMo exports | yes, through the Whisper and Paraformer exports | Apache 2.0 for the runtime, the weights keep their own licence | Windows, Linux, macOS, Android, iOS |
| whisper.cpp ggml models | stt | whisper.cpp | 75 M to 3 GB | Whisper, same coverage | yes | MIT | Windows, Linux, macOS, Android, iOS |
| `facebook/mms-tts-ben` | tts | mms-tts | 145 M | Bangla | yes, a Bangla only VITS model | **CC BY-NC 4.0, non commercial** | Windows, Linux, macOS, Android, iOS |
| `EMTIAZZ/bangladeshi-bangla-tts-vits` | tts | vits | 154 M | Bangladeshi Bangla | yes, with local pronunciation and intonation | Apache 2.0 | Windows, Linux, macOS, Android, iOS |
| `ai4bharat/indic-parler-tts` | tts | parler-tts | 880 M | 23 Indic languages | yes, with a voice described in words | Apache 2.0 | Windows, Linux, macOS |
| `ai4bharat/IndicTTS` | tts | indic-tts | 1.2 B | 22 Indic languages | yes | MIT | Windows, Linux, macOS |
| `rumik-ai/rumik-oss-1` | tts | transformers | 3 B | 22 Indic languages plus English, with a Bengali accent option | yes, with an explicit Bengali accent control | Apache 2.0 | Windows, Linux, macOS |
| `coqui/XTTS-v2` | tts | xtts | 1.8 B | 17 languages | yes, with zero shot voice cloning | **CPML, non commercial** | Windows, Linux, macOS |
| rhasspy piper voices | tts | piper | 20 M to 120 M per voice | Many, community Bangla voices exist | yes, through community Bangla voice files | MIT for the runtime, each voice keeps its own licence | Windows, Linux, macOS, Android, iOS |
| `ResembleAI/chatterbox-multilingual` | tts | chatterbox | 500 M | 23 languages | **no, the current multilingual release has no Bangla voice** | MIT | Windows, Linux, macOS |
| Microsoft Edge neural voices | tts | edge-tts | no local weights | Bangla for both regions | yes, one voice per region | Microsoft online service terms | any device with a network connection |
| pipeline | sts | pipeline | depends on the chosen pair | whatever the pair supports | yes | follows the two models it uses | Windows, Linux, macOS, Android, iOS |
| `htdemucs` | separation | separate | 80 M | language agnostic | yes | MIT | Windows, Linux, macOS |

### Querying the catalogue

```python
from mash_mix.models import MODEL_CATALOG, bengali_models, models_for, models_for_language

models_for("stt")                    # every speech to text model
models_for_language("bn-BD")         # every model that covers Bangla
bengali_models("tts")               # every Bangla capable synthesis model
len(MODEL_CATALOG)                   # 18
```

## Choosing a model per task

### Speech to text

| Order | Model | Why |
|---|---|---|
| 1 | `Systran/faster-whisper-large-v3` | The best balance of quality and speed. The same weights as Whisper large v3, with int8 on a CPU |
| 2 | `ehzawad/stt_bn_fastconformer_ctc` | The strongest public Bengali specialist model, and it is 117 M rather than 1.55 B |
| 3 | `openai/whisper-large-v3` | The same quality as the first, on the reference implementation, with slower CPU inference |
| 4 | `FunAudioLLM/SenseVoiceSmall` | Very fast and small, good enough for short clips and for a live caption feed |
| 5 | sherpa-onnx and whisper.cpp exports | The same Whisper weights in a form that fits on a phone |

The reasoning: Whisper large v3 covers Bangla well but is heavy. The Bengali FastConformer model is small enough to be practical and is tuned for the language, so it wins when the content is purely Bangla. SenseVoice is the pick for speed. The ONNX and ggml exports are the same weights in a smaller package, so they are not a different model at all.

`Qwen/Qwen3-ASR-1.7B` is a strong multilingual model but it is not Bangla specialised, so it is not a first choice here.

### Text to speech

| Order | Model | Why |
|---|---|---|
| 1 | `EMTIAZZ/bangladeshi-bangla-tts-vits` | The closest to a native Bangladeshi Bangla voice, and Apache 2.0 so it can ship |
| 2 | `facebook/mms-tts-ben` | Small and dependable, but the licence is non commercial |
| 3 | `ai4bharat/IndicTTS` | Wide speaker set and an MIT licence, so it covers both Bangla regions |
| 4 | `ai4bharat/indic-parler-tts` | When you need to describe the voice you want in words rather than pick a name |
| 5 | rhasspy piper voices | When the device is small and the latency matters more than the timbre |
| 6 | `coqui/XTTS-v2` | Only when you need voice cloning, and only when the non commercial licence is acceptable |
| 7 | Microsoft Edge neural voices | As a quality reference while you compare the local models, and nothing else |

The reasoning: for Bangladeshi Bangla the VITS fine tune wins on naturalness and on the licence. MMS TTS is the fallback when the licence does not matter. IndicTTS is the safest choice across both regions. Piper is the only one that fits comfortably on a phone.

`ResembleAI/chatterbox-multilingual` is not a Bangla option. It is listed so the catalogue is complete.

### Speech to speech

There is one entry, `pipeline`, because the default path is built from a speech to text model and a speech to speech model. Pick the pair for the job and the licence follows the pair.

### Separation

`htdemucs` splits a track into stems. The `mid_side` model needs no download at all and is the fallback when the machine cannot run Demucs.

## Licence warning

The licence of the model is separate from the licence of this package. This package is MIT. The models are not.

| Model | Licence | What it means |
|---|---|---|
| `facebook/mms-tts-ben` | CC BY-NC 4.0 | **Non commercial.** Free for research, education and personal use. You need a separate licence for anything that earns money |
| `coqui/XTTS-v2` | CPML | **Non commercial.** The Coqui Public Model License restricts commercial use of the outputs |
| Microsoft Edge neural voices | Microsoft online service terms | **Online service terms.** You are using a Microsoft service, not a model file. The audio leaves the machine and the terms of that service apply |
| rhasspy piper voices | MIT for the runtime, per voice for the weights | The engine is MIT, but each community voice file has its own licence. Check the voice you use |
| sherpa-onnx models | Apache 2.0 for the runtime, per model for the weights | The engine is Apache 2.0. The exported model keeps its own licence |

What to do:

- Shipping a commercial product? Use `EMTIAZZ/bangladeshi-bangla-tts-vits` (Apache 2.0), `ai4bharat/IndicTTS` (MIT), `ai4bharat/indic-parler-tts` (Apache 2.0) or a Piper voice whose licence allows it.
- Doing research or teaching? MMS TTS and XTTS v2 are fine.
- Handling private content? Do not use `edge-tts`. The text is sent to Microsoft.
- Publishing audio that was generated by XTTS v2? Read the CPML terms first. They place conditions on the outputs.

Check the licence before you download, not after. `mash-mix doctor` prints the licences of the models it finds locally.

## Choosing a model for your situation

### A phone

| Setting | Choice |
|---|---|
| Speech to text | `sherpa-onnx` with a Whisper or Paraformer export, or `whisper.cpp` with a ggml model |
| Model size | small at int8, or medium if the phone is new |
| Text to speech | Piper, with a community Bangla voice whose licence you have checked |
| Separation | `mid_side`, which needs no download |
| Server | Do not run the server on a phone. See [platforms.md](platforms.md) |

Budget for the weights: a small quantised model is a few hundred megabytes, the medium one about 1.5 GB.

### A laptop

| Setting | Choice |
|---|---|
| Speech to text | `faster-whisper` with the `small` model at int8 |
| Better accuracy when it matters | `faster-whisper` with `large-v3` |
| Text to speech | `mms-tts` for research, the VITS fine tune for a product |
| Device | CPU is fine. int8 is the default |

The small model at int8 is the sweet spot on a laptop: it runs faster than real time and it fits in memory with room to spare.

### A server with a GPU

| Setting | Choice |
|---|---|
| Speech to text | `faster-whisper` with `large-v3` at float16, which the backend selects automatically |
| Text to speech | `parler-tts` or `indic-tts` |
| Batch | Run several jobs at once, one per worker process, and pin each to the GPU |

On a GPU, move up the size ladder. The large model is fast, and the float16 path keeps the memory use at about 3 GB per worker.

### A low bandwidth environment

| Setting | Choice |
|---|---|
| Speech to text | `whisper.cpp` or `sherpa-onnx` with a quantised model, downloaded once |
| Text to speech | Piper, with the voice file cached locally |
| Avoid | `edge-tts`, which sends the text to Microsoft on every call |
| Avoid | Large models, which are multi gigabyte downloads |

The rule here is that everything runs locally after a single download. A 500 MB model fetched once beats a cloud API that needs a connection every time.

```python
from mash_mix.backends.registry import backend_infos

# What is actually installed and usable right now.
for info in backend_infos():
    if info["available"]:
        print(info["kind"], info["name"])
```

The catalogue is also available over HTTP:

```bash
curl http://localhost:8000/api/models
```

---

Copyright (c) 2026 MASH (Khaled Bhuiyan). MIT Licence. See LICENSE.
