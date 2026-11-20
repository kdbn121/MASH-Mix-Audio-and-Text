# SPDX-License-Identifier: MIT | Copyright (c) 2026 MASH (Khaled Bhuiyan)
"""End to end speech to speech check on real Bangla audio.

TTS speaks a Bangla line, the speech to speech pipeline recognises it and
speaks it again, then the recogniser reads the result back so the text can be
compared with the original.

Usage
-----
    python scripts/sts_check.py
"""

from __future__ import annotations

import sys
import tempfile
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from mash_mix.audio.io import save  # noqa: E402
from mash_mix.backends.registry import get_stt, get_tts  # noqa: E402
from mash_mix.tasks.speech_to_speech import speak  # noqa: E402

LINE = "আমার সোনার বাংলা, আমি তোমায় ভালোবাসি।"


def main() -> int:
    work = Path(tempfile.gettempdir()) / "mash-mix-sts"
    work.mkdir(parents=True, exist_ok=True)

    print("1. speaking the original line")
    tts = get_tts("mms-tts", model="facebook/mms-tts-ben")
    first = tts.synthesize(LINE, language="bn-BD")
    source = save(str(work / "source.wav"), first.audio, first.sample_rate)
    print(f"   {source}")
    print(f"   {len(first.audio) / first.sample_rate:.2f} s of speech")

    print()
    print("2. speech to speech, same language")
    result = speak(
        source,
        target_language="bn-BD",
        stt_backend="faster-whisper",
        tts_backend="mms-tts",
    )
    target = save(str(work / "sts.wav"), result.audio, result.sample_rate)
    print(f"   {target}")
    print(f"   {len(result.audio) / result.sample_rate:.2f} s of speech")
    print(f"   meta: {result.meta}")

    print()
    print("3. reading the result back")
    stt = get_stt("faster-whisper", model="small")
    heard = stt.transcribe(result.audio, result.sample_rate, language="bn-BD")
    print(f"   original: {LINE}")
    print(f"   heard:    {heard.text}")

    ok = bool(heard.text.strip())
    print()
    print("PASSED, the speech to speech chain produced audio and it recognised"
          if ok else "FAILED, nothing came back")
    return 0 if ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
