# SPDX-License-Identifier: MIT | Copyright (c) 2026 MASH (Khaled Bhuiyan)
"""Machine check. Tells the user exactly what works here and what is missing."""

from __future__ import annotations

import importlib
import platform
import shutil
import subprocess
import sys
from typing import Any, Dict, List, Optional

__all__ = ["run_doctor", "main"]

_OPTIONAL_PACKAGES = (
    ("numpy", "core"),
    ("torch", "stt"),
    ("torchaudio", "stt"),
    ("whisper", "stt"),
    ("faster_whisper", "stt"),
    ("ctranslate2", "stt"),
    ("transformers", "tts"),
    ("soundfile", "stt"),
    ("librosa", "analysis"),
    ("onnxruntime", "edge"),
    ("sherpa_onnx", "edge"),
    ("fastapi", "server"),
    ("uvicorn", "server"),
    ("demucs", "analysis"),
)


def _ffmpeg_version() -> Optional[str]:
    binary = shutil.which("ffmpeg")
    if not binary:
        return None
    try:
        result = subprocess.run(
            [binary, "-version"],
            capture_output=True,
            text=True,
            timeout=15,
            check=False,
        )
    except Exception:
        return None
    first = (result.stdout or "").splitlines()
    return first[0].strip() if first else "installed"


def _platform_info() -> Dict[str, Any]:
    return {
        "system": platform.system(),
        "release": platform.release(),
        "machine": platform.machine(),
        "python": platform.python_version(),
        "executable": sys.executable,
    }


def _android() -> bool:
    return "ANDROID_ROOT" in __import__("os").environ or "ANDROID_DATA" in __import__("os").environ


def _ios() -> bool:
    return sys.platform == "darwin" and platform.machine() in ("iPhone", "iPad")


def run_doctor() -> Dict[str, Any]:
    """Run every check and return a machine readable report."""
    from .backends.registry import backend_infos

    packages: List[Dict[str, Any]] = []
    missing: List[str] = []
    for name, extra in _OPTIONAL_PACKAGES:
        try:
            importlib.import_module(name)
            packages.append({"name": name, "extra": extra, "installed": True, "version": ""})
        except Exception:
            packages.append({"name": name, "extra": extra, "installed": False, "version": ""})
            missing.append(name)

    backends = backend_infos()

    ffmpeg = _ffmpeg_version()
    disk = shutil.disk_usage(sys.prefix if sys.prefix else ".")

    has_stt = any(b["kind"] == "stt" and b.get("available") for b in backends)
    has_tts = any(b["kind"] == "tts" and b.get("available") for b in backends)

    platform_info = _platform_info()
    if _android():
        platform_info["environment"] = "android-termux"
    elif _ios():
        platform_info["environment"] = "ios"
    else:
        platform_info["environment"] = "desktop"

    lines: List[str] = []
    lines.append("MASH-Mix-Audio-and-Text doctor")
    lines.append("")
    lines.append(
        f"platform      {platform_info['environment']} "
        f"({platform_info['system']} {platform_info['machine']}, Python {platform_info['python']})"
    )
    lines.append(f"ffmpeg        {ffmpeg or 'MISSING, needed for mp3, m4a, flac and video inputs'}")
    lines.append(
        f"disk free     {disk.free / (1024 ** 3):.1f} GB "
        "(model weights need roughly 1 to 4 GB each)"
    )
    lines.append("")
    lines.append(f"packages      {sum(1 for p in packages if p['installed'])} of {len(packages)} installed")
    lines.append("")
    lines.append("backends")
    for info in backends:
        mark = "ok " if info.get("available") else "no "
        lines.append(f"  [{mark}] {info['kind']:3} {info['name']:<16} {info['summary']}")
    lines.append("")
    lines.append(f"speech to text {'ready' if has_stt else 'not ready'}")
    lines.append(f"text to speech {'ready' if has_tts else 'not ready'}")

    if missing:
        lines.append("")
        lines.append("install what is missing")
        lines.append("  pip install 'mash-mix-audio-and-text[full]'")

    ok = bool(ffmpeg) and has_stt and has_tts
    return {
        "ok": ok,
        "platform": platform_info,
        "ffmpeg": ffmpeg,
        "disk_free_bytes": disk.free,
        "packages": packages,
        "backends": backends,
        "stt_ready": has_stt,
        "tts_ready": has_tts,
        "missing_packages": missing,
        "text": "\n".join(lines),
    }


def main(argv: Optional[List[str]] = None) -> int:
    """Entry point for the ``mash-mix-doctor`` command."""
    import json

    report = run_doctor()
    if argv and "--json" in argv:
        print(json.dumps(report, ensure_ascii=False, indent=2))
    else:
        print(report["text"])
    return 0 if report["ok"] else 1


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main(sys.argv[1:]))
