# SPDX-License-Identifier: MIT | Copyright (c) 2026 MASH (Khaled Bhuiyan)
"""Catalogue of the models this engine can drive.

Every entry records what the model is, which languages it covers, the licence
it ships under, and where it runs. The Bangla column is the one that matters
most here, so it is spelled out in words rather than left to a flag.
"""

from __future__ import annotations

from typing import Any, Dict, List, Optional

__all__ = ["MODEL_CATALOG", "models_for", "models_for_language", "bengali_models"]

MODEL_CATALOG: List[Dict[str, Any]] = [
    # ------------------------------------------------------------------ STT
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
        "url": "https://huggingface.co/openai/whisper-large-v3",
    },
    {
        "id": "Systran/faster-whisper-large-v3",
        "task": "stt",
        "backend": "faster-whisper",
        "name": "faster-whisper large v3",
        "size": "1.55 B, int8 available",
        "languages": "99 languages, Bangla included",
        "bengali": "yes, same weights as Whisper with CTranslate2 speed",
        "license": "MIT",
        "platforms": "Windows, Linux, macOS, Android, iOS",
        "notes": "The recommended default. Roughly four times faster than the original on CPU.",
        "url": "https://huggingface.co/Systran/faster-whisper-large-v3",
    },
    {
        "id": "mozilla-ai/whisper-large-v3-bn",
        "task": "stt",
        "backend": "transformers",
        "name": "Whisper large v3, Bangla fine tune",
        "size": "1.55 B",
        "languages": "Bengali",
        "bengali": "yes, the best Bangla accuracy of the open Whisper family",
        "license": "Apache-2.0",
        "platforms": "Windows, Linux, macOS, Android, iOS",
        "notes": (
            "Whisper large v3 fine tuned on 21409 Bengali samples. The base model "
            "scores 55 percent word error rate on Bengali, this one scores 9.7 "
            "percent. The best choice when the audio is Bangla only."
        ),
        "url": "https://huggingface.co/mozilla-ai/whisper-large-v3-bn",
    },
    {
        "id": "mozilla-ai/whisper-large-v3-turbo-bn",
        "task": "stt",
        "backend": "transformers",
        "name": "Whisper large v3 turbo, Bangla fine tune",
        "size": "809 M",
        "languages": "Bengali",
        "bengali": "yes, the same fine tune with the speed of the turbo model",
        "license": "Apache-2.0",
        "platforms": "Windows, Linux, macOS, Android, iOS",
        "notes": (
            "Four times faster than the full model and about as accurate, 11 "
            "percent word error rate on Bengali. The recommended model for Bangla "
            "songs, because it also writes the lyrics in Bangla script instead of "
            "romanised Bangla."
        ),
        "url": "https://huggingface.co/mozilla-ai/whisper-large-v3-turbo-bn",
    },
    {
        "id": "ehzawad/stt_bn_fastconformer_ctc",
        "task": "stt",
        "backend": "transformers",
        "name": "Bengali FastConformer CTC",
        "size": "117 M",
        "languages": "Bengali",
        "bengali": "yes, built and tuned for Bengali, including Bengali script input",
        "license": "MIT",
        "platforms": "Windows, Linux, macOS, Android, iOS",
        "notes": "Best public Bengali model on the FLEURS and Indian sets in public benchmarks.",
        "url": "https://huggingface.co/ehzawad/stt_bn_fastconformer_ctc",
    },
    {
        "id": "FunAudioLLM/SenseVoiceSmall",
        "task": "stt",
        "backend": "funasr",
        "name": "SenseVoice Small",
        "size": "234 M",
        "languages": "Chinese, Cantonese, English, Japanese, Korean, Bangla and more",
        "bengali": "yes, Bangla is in the training mix",
        "license": "Apache 2.0",
        "platforms": "Windows, Linux, macOS",
        "notes": "Very fast, good for Bangla on short clips.",
        "url": "https://huggingface.co/FunAudioLLM/SenseVoiceSmall",
    },
    {
        "id": "Qwen/Qwen3-ASR-1.7B",
        "task": "stt",
        "backend": "transformers",
        "name": "Qwen3 ASR",
        "size": "1.7 B",
        "languages": "Chinese, English, Japanese, Korean, European and more",
        "bengali": "partial, strong on multilingual but not Bangla specialised",
        "license": "Apache 2.0",
        "platforms": "Windows, Linux, macOS",
        "notes": "Good for multilingual audio with a language hint.",
        "url": "https://huggingface.co/Qwen/Qwen3-ASR-1.7B",
    },
    {
        "id": "sherpa-onnx offline models",
        "task": "stt",
        "backend": "sherpa-onnx",
        "name": "sherpa-onnx offline recognisers",
        "size": "40 M to 800 M depending on the model",
        "languages": "Whisper, Paraformer, Zipformer and NeMo exports in many languages",
        "bengali": "yes, through the Whisper and Paraformer exports",
        "license": "Apache 2.0 for sherpa-onnx, the weights keep their own licence",
        "platforms": "Windows, Linux, macOS, Android, iOS",
        "notes": "The lightest option for phones. Runs on ONNX Runtime with no Python model code.",
        "url": "https://github.com/k2-fsa/sherpa-onnx",
    },
    {
        "id": "whisper.cpp ggml models",
        "task": "stt",
        "backend": "whisper.cpp",
        "name": "whisper.cpp ggml",
        "size": "75 M to 3 GB",
        "languages": "Whisper, same coverage",
        "bengali": "yes",
        "license": "MIT",
        "platforms": "Windows, Linux, macOS, Android, iOS",
        "notes": "Best option for mobile builds that want the smallest footprint.",
        "url": "https://github.com/ggerganov/whisper.cpp",
    },
    # ------------------------------------------------------------------ TTS
    {
        "id": "facebook/mms-tts-ben",
        "task": "tts",
        "backend": "mms-tts",
        "name": "MMS TTS for Bangla",
        "size": "145 M",
        "languages": "Bangla",
        "bengali": "yes, a Bangla only VITS model",
        "license": "CC BY-NC 4.0, non commercial",
        "platforms": "Windows, Linux, macOS, Android, iOS",
        "notes": "Small and dependable for Bangla. The non commercial licence matters for products.",
        "url": "https://huggingface.co/facebook/mms-tts-ben",
    },
    {
        "id": "EMTIAZZ/bangladeshi-bangla-tts-vits",
        "task": "tts",
        "backend": "vits",
        "name": "Bangladeshi Bangla TTS, VITS fine tune",
        "size": "154 M",
        "languages": "Bangladeshi Bangla",
        "bengali": "yes, specifically Bangladeshi Bangla with local pronunciation and intonation",
        "license": "Apache 2.0",
        "platforms": "Windows, Linux, macOS, Android, iOS",
        "notes": "The closest thing to a native Bangladeshi Bangla voice in the open weights space.",
        "url": "https://huggingface.co/EMTIAZZ/bangladeshi-bangla-tts-vits",
    },
    {
        "id": "ai4bharat/indic-parler-tts",
        "task": "tts",
        "backend": "parler-tts",
        "name": "Indic Parler TTS",
        "size": "880 M",
        "languages": "23 Indic languages including Bangla",
        "bengali": "yes, with voice described in words",
        "license": "Apache 2.0",
        "platforms": "Windows, Linux, macOS",
        "notes": "Voice is steered by a description, for example a warm Bangladeshi female voice.",
        "url": "https://huggingface.co/ai4bharat/indic-parler-tts",
    },
    {
        "id": "ai4bharat/IndicTTS",
        "task": "tts",
        "backend": "indic-tts",
        "name": "IndicTTS",
        "size": "1.2 B",
        "languages": "22 Indic languages including Bangla",
        "bengali": "yes",
        "license": "MIT",
        "platforms": "Windows, Linux, macOS",
        "notes": "Strong Bangla quality with a large speaker set.",
        "url": "https://huggingface.co/ai4bharat/IndicTTS",
    },
    {
        "id": "rumik-ai/rumik-oss-1",
        "task": "tts",
        "backend": "transformers",
        "name": "rumik oss 1",
        "size": "3 B",
        "languages": "22 Indic languages plus English, with a Bengali accent option",
        "bengali": "yes, with an explicit Bengali accent control",
        "license": "Apache 2.0",
        "platforms": "Windows, Linux, macOS",
        "notes": "Handles code switching between Bangla and English in one utterance.",
        "url": "https://huggingface.co/rumik-ai/rumik-oss-1",
    },
    {
        "id": "coqui/XTTS-v2",
        "task": "tts",
        "backend": "xtts",
        "name": "XTTS v2",
        "size": "1.8 B",
        "languages": "17 languages including Bangla",
        "bengali": "yes, with zero shot voice cloning",
        "license": "CPML, non commercial",
        "platforms": "Windows, Linux, macOS",
        "notes": "Voice cloning from a short reference clip. The licence is non commercial.",
        "url": "https://huggingface.co/coqui/XTTS-v2",
    },
    {
        "id": "rhasspy piper voices",
        "task": "tts",
        "backend": "piper",
        "name": "Piper ONNX voices",
        "size": "20 M to 120 M per voice",
        "languages": "Many, community Bangla voices exist",
        "bengali": "yes, through community Bangla voice files",
        "license": "MIT for the runtime, each voice keeps its own licence",
        "platforms": "Windows, Linux, macOS, Android, iOS",
        "notes": "The lightest option and the one that runs well on a phone.",
        "url": "https://github.com/rhasspy/piper",
    },
    {
        "id": "ResembleAI/chatterbox-multilingual",
        "task": "tts",
        "backend": "chatterbox",
        "name": "Chatterbox Multilingual v3",
        "size": "500 M",
        "languages": "23 languages",
        "bengali": "no, the current multilingual release has no Bangla voice",
        "license": "MIT",
        "platforms": "Windows, Linux, macOS",
        "notes": "A general purpose fallback for other languages. Not a Bangla option.",
        "url": "https://huggingface.co/ResembleAI/chatterbox-multilingual",
    },
    {
        "id": "Microsoft Edge neural voices",
        "task": "tts",
        "backend": "edge-tts",
        "name": "Edge neural voices, bn-BD-NabanitaNeural and bn-IN-SwaraNeural",
        "size": "no local weights",
        "languages": "Bangla for both regions",
        "bengali": "yes, one voice per region, the reference quality for Bangladeshi Bangla",
        "license": "Microsoft online service terms",
        "platforms": "any device with a network connection",
        "notes": "Online only. Useful as a quality reference while comparing local models.",
        "url": "https://speech.platform.bing.com",
    },
    # ------------------------------------------------------------------ STS
    {
        "id": "pipeline",
        "task": "sts",
        "backend": "pipeline",
        "name": "STT plus TTS pipeline",
        "size": "depends on the chosen pair",
        "languages": "Whatever the pair supports",
        "bengali": "yes",
        "license": "follows the two models it uses",
        "platforms": "Windows, Linux, macOS, Android, iOS",
        "notes": "Speech to speech built from the recognition and synthesis backends.",
        "url": "https://github.com/kdbn121/MASH-Mix-Audio-and-Text",
    },
    # ---------------------------------------------------------- separation
    {
        "id": "htdemucs",
        "task": "separation",
        "backend": "separate",
        "name": "Demucs HTDemucs",
        "size": "80 M",
        "languages": "language agnostic",
        "bengali": "yes",
        "license": "MIT",
        "platforms": "Windows, Linux, macOS",
        "notes": "Vocals and instrument separation. The mid_side model needs no download at all.",
        "url": "https://github.com/facebookresearch/demucs",
    },
]


def models_for(task: str) -> List[Dict[str, Any]]:
    """Every model for a task."""
    return [model for model in MODEL_CATALOG if model["task"] == task]


def models_for_language(tag: str) -> List[Dict[str, Any]]:
    """Every model that covers a language tag."""
    from .lang.registry import resolve

    spec = resolve(tag)
    out: List[Dict[str, Any]] = []
    for model in MODEL_CATALOG:
        if spec.is_bangla:
            if model["bengali"].lower().startswith(("yes", "partial")):
                out.append(model)
        elif spec.name_en.lower() in model["languages"].lower():
            out.append(model)
    return out


def bengali_models(task: Optional[str] = None) -> List[Dict[str, Any]]:
    """Every Bangla capable model, optionally filtered by task."""
    models = [
        model
        for model in MODEL_CATALOG
        if model["bengali"].lower().startswith(("yes", "partial"))
    ]
    if task:
        models = [model for model in models if model["task"] == task]
    return models
