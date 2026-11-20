# SPDX-License-Identifier: MIT | Copyright (c) 2026 MASH (Khaled Bhuiyan)
"""Check the HTTP server and the web interface end to end.

Starts the real server with uvicorn, uploads a Bangla song, and downloads the
lyric files that the web interface offers.

Usage
-----
    python scripts/server_check.py path/to/song.mp3
"""

from __future__ import annotations

import json
import os
import socket
import subprocess
import sys
import time
import urllib.error
import urllib.request
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

HOST = "127.0.0.1"
PORT = 8765
BASE = f"http://{HOST}:{PORT}"


def wait_for_server(timeout: float = 60.0) -> bool:
    deadline = time.time() + timeout
    while time.time() < deadline:
        try:
            with urllib.request.urlopen(f"{BASE}/api/health", timeout=3) as response:
                if response.status == 200:
                    return True
        except Exception:
            time.sleep(1.0)
    return False


def free_port() -> int:
    with socket.socket() as sock:
        sock.bind((HOST, 0))
        return sock.getsockname()[1]


def post_multipart(path: str, fields: dict, file_field: str, file_path: Path) -> dict:
    """Upload a file with a multipart form, without a third party library."""
    boundary = "----mashmixcheck" + str(int(time.time() * 1000))
    body = bytearray()
    for name, value in fields.items():
        body += f"--{boundary}\r\n".encode()
        body += f'Content-Disposition: form-data; name="{name}"\r\n\r\n'.encode()
        body += f"{value}\r\n".encode()
    body += f"--{boundary}\r\n".encode()
    body += (
        f'Content-Disposition: form-data; name="{file_field}"; '
        f'filename="{file_path.name}"\r\n'
    ).encode()
    body += b"Content-Type: application/octet-stream\r\n\r\n"
    body += file_path.read_bytes()
    body += f"\r\n--{boundary}--\r\n".encode()

    request = urllib.request.Request(
        f"{BASE}{path}",
        data=bytes(body),
        headers={"Content-Type": f"multipart/form-data; boundary={boundary}"},
        method="POST",
    )
    with urllib.request.urlopen(request, timeout=900) as response:
        return json.loads(response.read().decode("utf-8"))


def get_text(path: str) -> str:
    with urllib.request.urlopen(f"{BASE}{path}", timeout=120) as response:
        return response.read().decode("utf-8")


def main() -> int:
    if len(sys.argv) < 2:
        print("Give me an audio file to upload.", file=sys.stderr)
        return 1
    audio = Path(sys.argv[1])

    port = free_port()
    global PORT, BASE
    PORT = port
    BASE = f"http://{HOST}:{PORT}"

    env = dict(os.environ)
    env["PYTHONPATH"] = str(ROOT / "src")
    server = subprocess.Popen(
        [
            sys.executable,
            "-m",
            "uvicorn",
            "mash_mix.server.app:app",
            "--host",
            HOST,
            "--port",
            str(port),
            "--log-level",
            "warning",
        ],
        cwd=str(ROOT),
        env=env,
    )

    failures = []
    try:
        print("1. waiting for the server")
        if not wait_for_server():
            print("   the server did not start")
            return 1
        print("   up")

        print()
        print("2. the web interface")
        page = get_text("/")
        for needle in ("Extract lyrics", "Download", "bn-BD", "MASH-Mix-Audio-and-Text"):
            if needle not in page:
                failures.append(f"the web interface is missing '{needle}'")
        print(f"   {len(page)} bytes of html")

        print()
        print("3. the language registry")
        languages = json.loads(get_text("/api/languages"))
        tags = [item["tag"] for item in languages]
        print(f"   {tags}")
        if "bn-BD" not in tags or "bn-IN" not in tags:
            failures.append("the language registry is missing the Bangla tags")

        print()
        print("4. uploading the song")
        started = time.time()
        payload = post_multipart(
            "/api/lyrics",
            {"language": "bn-BD", "title": "Amar Sonar Bangla"},
            "file",
            audio,
        )
        print(f"   took {time.time() - started:.1f} s")
        job = payload["job_id"]
        print(f"   job {job}")
        print(f"   language {payload['language']}, {len(payload['lines'])} lines")
        if not payload["lines"]:
            failures.append("the upload produced no lyric lines")
        for line in payload["lines"][:3]:
            print(f"     {line['start']:.2f}  {line['text']}")

        print()
        print("5. downloading the files")
        for fmt in ("srt", "txt", "vtt", "lrc"):
            content = get_text(payload["downloads"][fmt])
            print(f"   {fmt:<4} {len(content)} bytes")
            if not content.strip():
                failures.append(f"the {fmt} download is empty")
        srt = get_text(payload["downloads"]["srt"])
        if "-->" not in srt:
            failures.append("the srt download has no cue timestamps")

        print()
        print("6. text to speech over the API")
        request = urllib.request.Request(
            f"{BASE}/api/speech",
            data=json.dumps({"input": "আমার সোনার বাংলা", "language": "bn-BD"}).encode("utf-8"),
            headers={"Content-Type": "application/json"},
            method="POST",
        )
        with urllib.request.urlopen(request, timeout=300) as response:
            audio_bytes = response.read()
        print(f"   {len(audio_bytes)} bytes of wav")
        if len(audio_bytes) < 1000:
            failures.append("the speech endpoint returned almost nothing")

    except urllib.error.HTTPError as exc:
        failures.append(f"the server returned HTTP {exc.code}: {exc.read()[:400]!r}")
    finally:
        server.terminate()
        try:
            server.wait(timeout=15)
        except Exception:
            server.kill()

    print()
    if failures:
        print("FAILED")
        for item in failures:
            print(f"  - {item}")
        return 1
    print("PASSED, the server accepted the upload and served every download")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
