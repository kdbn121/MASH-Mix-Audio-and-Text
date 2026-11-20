# SPDX-License-Identifier: MIT | Copyright (c) 2026 MASH (Khaled Bhuiyan)
"""Probe whether the CUDA path of faster-whisper works on this machine.

A machine can report an NVIDIA card while the CUDA runtime libraries that
CTranslate2 links against are missing. The model then loads without complaint
and fails at the first inference. This script answers that question in one
second, and it is what the engine checks before it chooses the GPU.

Usage
-----
    python scripts/cuda_probe.py
"""

from __future__ import annotations

import faulthandler
import time

faulthandler.dump_traceback_later(50, exit=True)

import numpy as np  # noqa: E402
from faster_whisper import WhisperModel  # noqa: E402

t0 = time.time()
model = WhisperModel("tiny", device="cuda", compute_type="float16")
print("loaded cuda model in", round(time.time() - t0, 1), flush=True)

audio = np.zeros(16000, dtype="float32")
t0 = time.time()
segments, info = model.transcribe(audio, language="bn")
out = list(segments)
print("generated", len(out), "segments in", round(time.time() - t0, 1), flush=True)
print("CUDA PATH WORKS", flush=True)
