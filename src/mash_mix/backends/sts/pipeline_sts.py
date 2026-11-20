# SPDX-License-Identifier: MIT | Copyright (c) 2026 MASH (Khaled Bhuiyan)
"""Speech to speech pipeline backend.

The pipeline backend chains an STT backend with a TTS backend. Speech in,
transcript in the middle, speech out in another language. Keeping the text in
the middle is what lets the whole Bangla layer apply, because the transcript
is normalised and the digits and symbols are spelled out before it is spoken
again.

A translation step is optional. When the detected language differs from the
target language and no translator callable is given, the original text is
spoken in the target voice and the result records that nothing was
translated.
"""

from __future__ import annotations

import time
from typing import Any, Callable, List, Optional, Tuple

import numpy as np

from ...errors import BackendUnavailableError, MashMixError
from ...types import AudioBuffer, SynthesisResult
from ..base import BackendInfo, STSBackend
from ..registry import get_stt, get_tts, register_sts

_STT_PREFERENCE = ["faster-whisper", "whisper", "funasr", "transformers", "sherpa-onnx", "whisper.cpp"]
_TTS_PREFERENCE = ["mms-tts", "vits", "indic-tts", "parler-tts", "piper", "xtts", "chatterbox", "edge-tts"]


def _base_language(tag: Optional[str]) -> str:
    """The language part of a tag, so bn-BD and bn compare equal."""
    if not tag:
        return ""
    return str(tag).lower().replace("_", "-").split("-")[0]


# Whisper writes Bangla in Devanagari script, so a Bangla transcript is
# rewritten into Bangla script before it is spoken again. The two scripts
# share one inventory and many look alike letters, so the table is written
# with codepoints instead of glyphs, which stops a letter from being mapped
# onto its own look alike twin.
_DEVA_NUKTA = {
    "\u095c": "\u09dc",  # the r dotted da to the Bangla r dotted da
    "\u095d": "\u09dd",  # the r dotted dha to the Bangla r dotted dha
    "\u095f": "\u09df",  # the y with nukta to the Bangla y with nukta
}

_DEVA_TO_BANGLA = {
    # consonants
    "\u0915": "\u0985",  # ka
    "\u0916": "\u0996",  # kha
    "\u0917": "\u0997",  # ga
    "\u0918": "\u0998",  # gha
    "\u0919": "\u0999",  # nga
    "\u091a": "\u099a",  # cha
    "\u091b": "\u099b",  # chha
    "\u091c": "\u099c",  # ja
    "\u091d": "\u099d",  # jha
    "\u091e": "\u099e",  # nya
    "\u091f": "\u099f",  # tta
    "\u0920": "\u09a0",  # ttha
    "\u0921": "\u09a1",  # dda
    "\u0922": "\u09a2",  # ddha
    "\u0923": "\u09a3",  # nna
    "\u0924": "\u09a4",  # ta
    "\u0925": "\u09a5",  # tha
    "\u0926": "\u09a6",  # da
    "\u0927": "\u09a7",  # dha
    "\u0928": "\u09a8",  # na
    "\u092a": "\u09aa",  # pa
    "\u092b": "\u09ab",  # pha
    "\u092c": "\u09ac",  # ba
    "\u092d": "\u09ad",  # bha
    "\u092e": "\u09ae",  # ma
    "\u092f": "\u09af",  # ya
    "\u0930": "\u09b0",  # ra
    "\u0932": "\u09b2",  # la
    "\u0935": "\u09ac",  # va
    "\u0936": "\u09b6",  # sha
    "\u0937": "\u09b7",  # ssa
    "\u0938": "\u09b8",  # sa
    "\u0939": "\u09b9",  # ha
    "\u0933": "\u09dc",  # lla
    "\u0915\u09cd\u09b7": "\u0995\u09cd\u09b7",  # ksha
    "\u0924\u09cd\u09b0": "\u09a4\u09cd\u09b0",  # tra
    "\u091c\u09cd\u099e": "\u099c\u09cd\u099e",  # gnya
    "\u0929": "\u099e",  # nna
    "\u0934": "\u099d",  # rha
    "\u0931": "\u09dc",  # rra
    # vowels
    "\u0905": "\u0985",  # a
    "\u0906": "\u0986",  # aa
    "\u0907": "\u0987",  # i
    "\u0908": "\u0988",  # ii
    "\u0909": "\u0989",  # u
    "\u090a": "\u098a",  # uu
    "\u090b": "\u098b",  # vocalic r
    "\u090f": "\u098f",  # e
    "\u0910": "\u0990",  # ai
    "\u0913": "\u0993",  # o
    "\u0914": "\u0994",  # au
    "\u090d": "\u098d",  # vocalic l
    "\u090e": "\u098e",  # vocalic ll
    # matras and signs
    "\u093e": "\u09be",  # aa matra
    "\u093f": "\u09bf",  # i matra
    "\u0940": "\u09c0",  # ii matra
    "\u0941": "\u09c1",  # u matra
    "\u0942": "\u09c2",  # uu matra
    "\u0943": "\u09c3",  # vocalic r matra
    "\u0947": "\u09c7",  # e matra
    "\u0948": "\u09c8",  # ai matra
    "\u094b": "\u09cb",  # o matra
    "\u094c": "\u09cc",  # au matra
    "\u094d": "\u09cd",  # halant
    "\u0902": "\u0982",  # anusvara
    "\u0903": "\u0983",  # visarga
    "\u0901": "\u0981",  # candrabindu
    "\u093c": "",  # a bare nukta is dropped
    "\u093d": "",  # avagraha is dropped
    # digits
    "\u0966": "\u09e6",
    "\u0967": "\u09e7",
    "\u0968": "\u09e8",
    "\u0969": "\u09e9",
    "\u096a": "\u09ea",
    "\u096b": "\u09eb",
    "\u096c": "\u09ec",
    "\u096d": "\u09ed",
    "\u096e": "\u09ee",
    "\u096f": "\u09ef",
    "\u0950": "\u0993\u0981",  # om
}


def _devanagari_to_bangla(text: str) -> str:
    """Rewrite a Devanagari Bangla transcript in Bangla script."""
    for source, target in _DEVA_NUKTA.items():
        text = text.replace(source, target)
    return "".join(_DEVA_TO_BANGLA.get(char, char) for char in text)


def _has_devanagari(text: str) -> bool:
    return any("\u0900" <= char <= "\u097f" for char in text)


@register_sts("pipeline")
class PipelineSTSBackend(STSBackend):
    """Speech to speech by chaining an STT backend with a TTS backend."""

    info = BackendInfo(
        name="pipeline",
        kind="sts",
        summary="Speech to speech by chaining an STT backend with a TTS backend.",
        package=None,
        extra="tts",
        languages=["bn", "bn-BD", "bn-IN", "en", "en-US", "hi", "hi-IN"],
        platforms=["windows", "linux", "macos", "android-termux", "ios"],
        notes=(
            "Bangla is first class here, because the transcript is normalised before it is "
            "spoken again. The backend names can be chosen with the stt and tts keyword "
            "arguments, and a translator callable can be passed to translate the transcript "
            "when the detected language differs from the target language."
        ),
    )

    def __init__(self) -> None:
        super().__init__()
        self._stt_name = ""
        self._tts_name = ""

    # -- lifecycle ------------------------------------------------------
    def is_available(self) -> bool:
        """True when at least one STT backend and one TTS backend are usable."""
        stt_names, tts_names = self._available_names()
        return bool(stt_names) and bool(tts_names)

    @staticmethod
    def _available_names() -> Tuple[List[str], List[str]]:
        try:
            from ..registry import backend_infos

            infos = backend_infos()
        except Exception:
            return [], []
        stt_names = [
            str(info.get("name"))
            for info in infos
            if info.get("kind") == "stt" and info.get("available")
        ]
        tts_names = [
            str(info.get("name"))
            for info in infos
            if info.get("kind") == "tts" and info.get("available")
        ]
        return stt_names, tts_names

    @staticmethod
    def _pick(available: List[str], preference: List[str]) -> Optional[str]:
        for name in preference:
            if name in available:
                return name
        return available[0] if available else None

    def load(self, model: Optional[str] = None) -> None:
        """The pipeline holds no weights of its own, so this only records names."""
        if self._loaded:
            return
        stt_names, tts_names = self._available_names()
        self._stt_name = self._pick(stt_names, _STT_PREFERENCE) or ""
        self._tts_name = self._pick(tts_names, _TTS_PREFERENCE) or ""
        self._loaded = True

    # -- inference ------------------------------------------------------
    def speak(
        self,
        audio: AudioBuffer,
        *,
        target_language: str = "bn-BD",
        target_voice: Optional[str] = None,
        reference_audio: Optional[AudioBuffer] = None,
        **kwargs: Any,
    ) -> SynthesisResult:
        started = time.perf_counter()
        stt_names, tts_names = self._available_names()
        stt_name = kwargs.pop("stt", None) or kwargs.pop("stt_backend", None)
        tts_name = kwargs.pop("tts", None) or kwargs.pop("tts_backend", None)
        translator = kwargs.pop("translator", None)
        source_language = kwargs.pop("source_language", None)
        stt_model = kwargs.pop("stt_model", None)
        tts_model = kwargs.pop("model", None)

        if not stt_name or stt_name == "auto":
            stt_name = self._pick(stt_names, _STT_PREFERENCE)
        if not tts_name or tts_name == "auto":
            tts_name = self._pick(tts_names, _TTS_PREFERENCE)
        if not stt_name or not tts_name:
            raise BackendUnavailableError(
                "The speech to speech pipeline needs one speech to text backend and one text to "
                "speech backend. Install the stt and tts extras, or pass the stt and tts "
                "keyword arguments."
            )

        stt = get_stt(stt_name, model=stt_model)
        tts = get_tts(tts_name, model=tts_model)

        transcription = stt.transcribe(
            audio.mono(),
            audio.sample_rate,
            language=source_language,
            **{key: value for key, value in kwargs.items() if key in _STT_KEYS},
        )
        text = (transcription.text or "").strip()
        detected = transcription.language or source_language or ""
        if not text:
            raise MashMixError(
                "The speech to text backend recognised no speech in the audio"
                + (f" (language {detected})" if detected else "")
                + ", so there is nothing to speak again."
            )

        script_converted = False
        if _base_language(target_language) == "bn" and _has_devanagari(text):
            # Whisper writes Bangla in Devanagari script, and the Bangla
            # voices want Bangla script.
            text = _devanagari_to_bangla(text)
            script_converted = True

        if _base_language(target_language) == "bn":
            # A Bangla voice reads Latin letters as English, so romanised
            # Bangla is written out in Bangla before it is spoken again.
            from ...lang.translit import apply_script, resolve_script_target

            wanted = str(kwargs.pop("script", "bangla") or "bangla").strip().lower()
            if resolve_script_target(text, wanted) == "bangla":
                converted = apply_script(text, "bangla")
                if converted != text:
                    script_converted = True
                text = converted

        translated = False
        if text and _base_language(detected) != _base_language(target_language):
            if callable(translator):
                text = self._translate(translator, text, detected, target_language)
                translated = True

        result = tts.synthesize(
            text,
            language=target_language,
            voice=target_voice,
            reference_audio=reference_audio,
            **{
                key: value
                for key, value in kwargs.items()
                if key in _TTS_KEYS and key not in ("stt", "tts", "translator")
            },
        )

        meta = dict(result.meta)
        meta.update(
            {
                "stt": stt_name,
                "tts": tts_name,
                "source_language": detected,
                "target_language": target_language,
                "translated": translated,
                "script_converted": script_converted,
                "text": text,
            }
        )

        return SynthesisResult(
            audio=np.asarray(result.audio, dtype=np.float32).reshape(-1),
            sample_rate=int(result.sample_rate),
            backend=self.name,
            model=f"{stt_name}+{tts_name}",
            voice=target_voice or result.voice,
            elapsed=time.perf_counter() - started,
            meta=meta,
        )

    # -- helpers --------------------------------------------------------
    @staticmethod
    def _translate(
        translator: Callable[..., Any],
        text: str,
        source: str,
        target: str,
    ) -> str:
        """Call a translator callable in whichever shape it accepts."""
        attempts = [
            lambda: translator(text, source=source, target=target),
            lambda: translator(text, source, target),
            lambda: translator(text, target),
            lambda: translator(text),
        ]
        last_error: Optional[Exception] = None
        for attempt in attempts:
            try:
                out = attempt()
            except TypeError as exc:
                last_error = exc
                continue
            if isinstance(out, str):
                return out
            return str(out)
        raise ValueError(f"The translator callable could not be used: {last_error}")


# Keyword arguments that belong to the STT and the TTS call.
_STT_KEYS = {
    "task",
    "initial_prompt",
    "hotwords",
    "word_timestamps",
    "vad",
    "progress",
    "beam_size",
    "temperature",
    "condition_on_previous_text",
}

_TTS_KEYS = {
    "description",
    "exaggeration",
    "cfg_weight",
    "temperature",
    "reference",
}
