# Platforms

How to install and run the engine on Windows, Linux, macOS, Android, iOS and Raspberry Pi, and how to pick the right backends for each one.

## Before you start

One dependency is required everywhere: `ffmpeg`. It handles mp3, m4a, flac, ogg, opus, aac, wma and video containers. WAV works without it.

| Platform | Install ffmpeg |
|---|---|
| Windows | `winget install Gyan.FFmpeg` or `choco install ffmpeg` |
| macOS | `brew install ffmpeg` |
| Debian and Ubuntu | `sudo apt install ffmpeg` |
| Fedora | `sudo dnf install ffmpeg` |
| Arch | `sudo pacman -S ffmpeg` |
| Android (Termux) | `pkg install ffmpeg` |
| iOS | Not applicable, use the app bundle |

Then check the machine:

```bash
mash-mix doctor
```

The doctor prints the platform, the ffmpeg version, the free disk space, the optional packages that are installed, every backend with a mark for whether it is available, and whether speech to text and text to speech are ready. It exits with 0 when everything is ready and 1 when something is missing.

## Windows

| | |
|---|---|
| Install | `pip install 'mash-mix-audio-and-text[stt,tts]'` |
| Backends | `faster-whisper`, `whisper`, `mms-tts`, `vits` |
| Device | CPU by default, CUDA when a NVIDIA GPU is present |

Windows is the best supported desktop platform because every backend has a Windows build.

```powershell
winget install Gyan.FFmpeg
pip install 'mash-mix-audio-and-text[stt,tts]'
mash-mix doctor
mash-mix lyrics "my song.mp3" -o lyrics-out --lang bn-BD
```

Known limits:

- `parler-tts`, `indic-tts` and `xtts` need a GPU to be comfortable.
- The `edge` extra works, but ONNX Runtime Mobile is not a Windows target.
- Long paths and non ASCII file names work, but quote them in the shell.

## Linux

| | |
|---|---|
| Install | `pip install 'mash-mix-audio-and-text[full]'` |
| Backends | everything |
| Device | CPU, CUDA, or MPS nowhere |

Linux is the server platform. It is where the large models earn their keep.

```bash
sudo apt install ffmpeg python3-venv
python3 -m venv .venv
source .venv/bin/activate
pip install 'mash-mix-audio-and-text[full]'
mash-mix doctor
mash-mix lyrics "song.mp3" -o out --format srt --format vtt --lang bn-BD
```

Known limits:

- The audio server route needs `python-multipart` for the file uploads. The `server` extra includes it.
- A headless machine still needs ffmpeg on PATH, otherwise only WAV is readable.
- Run the HTTP API behind a reverse proxy if it is exposed to a network.

## macOS

| | |
|---|---|
| Install | `pip install 'mash-mix-audio-and-text[stt,tts]'` |
| Backends | `faster-whisper`, `whisper`, `mms-tts`, `vits`, `parler-tts` |
| Device | CPU, or MPS on Apple Silicon |

```bash
brew install ffmpeg python@3.11
pip install 'mash-mix-audio-and-text[stt,tts]'
mash-mix doctor
```

On Apple Silicon the local models can use the Metal Performance Shaders backend through PyTorch, which is often faster than the CPU path and much faster than the CUDA path would be on a different machine.

Known limits:

- The reference `whisper` implementation does not use MPS, so use `faster-whisper` if you want the GPU.
- Intel Macs get the CPU path only.

## Android with Termux

| | |
|---|---|
| Install | `pkg install python ffmpeg rust` then `pip install 'mash-mix-audio-and-text[edge]'` |
| Backends | `sherpa-onnx`, `whisper.cpp`, `piper` |
| Device | CPU only |

Termux gives a real Python environment on the phone, which makes it the easiest way to run this engine on Android without writing an app.

```bash
pkg install python ffmpeg rust clang libsndfile
pip install 'mash-mix-audio-and-text[edge]'
mash-mix doctor
mash-mix transcribe "recording.wav" -o out.txt --backend sherpa-onnx --model /sdcard/models/whisper-bn
```

Known limits:

- Only the `edge` backends are practical. The Whisper large models are out of the question on a phone.
- Use the quantised sherpa-onnx or ggml models. A full precision model will not fit.
- The `mid_side` separation model is the only one that needs no download.
- The doctor detects the Termux environment through the `ANDROID_ROOT` and `ANDROID_DATA` environment variables and prints `android-termux` in the platform line.
- Battery and heat are the real constraint. Keep the chunk size small and let the phone cool between jobs.

## iOS and iPadOS

| | |
|---|---|
| Install | Not a pip install. Embed the package in an app or a Pythonista style runtime |
| Backends | ONNX Runtime Mobile, `sherpa-onnx`, `whisper.cpp` |
| Device | Neural Engine through ONNX Runtime Mobile |

iOS is the hardest target and the least automated one. The package gives you the task functions, and the app supplies the runtime.

```bash
pip install 'mash-mix-audio-and-text[edge]'
```

Then, in the app:

```python
from mash_mix.tasks.transcribe import transcribe

result = transcribe("recording.wav", backend="sherpa-onnx", model=model_path)
```

The doctor recognises an iOS environment when the platform is `darwin` and the machine is `iPhone` or `iPad`.

Known limits:

- Convert the models with the `onnxruntime` tooling before they run on a device. The ONNX Runtime Mobile build expects a specific layout.
- The `edge-tts` backend is the only one that needs no local weights, and it needs a network.
- No pip install of the heavy extras. Build the ONNX models on a desktop and copy them across.

## Raspberry Pi

| | |
|---|---|
| Install | `sudo apt install python3-pip ffmpeg` then `pip install 'mash-mix-audio-and-text[edge]'` |
| Backends | `sherpa-onnx`, `whisper.cpp`, `piper` |
| Device | CPU only |

```bash
sudo apt install python3-pip python3-venv ffmpeg
python3 -m venv ~/mashmix && source ~/mashmix/bin/activate
pip install 'mash-mix-audio-and-text[edge]'
mash-mix doctor
mash-mix transcribe "clip.wav" --backend whisper.cpp --model ~/models/ggml-tiny.bin
```

Known limits:

- A Pi 4 with 4 GB can run the small quantised models. The large ones cannot.
- The CPU on a Pi is the bottleneck, so keep the chunk size small and expect several times the real time.
- Use the tiny or base model for a live caption loop and the small model for a batch job.

## Device and GPU notes

### CUDA

An NVIDIA GPU is the fastest path. `faster-whisper` detects it through CTranslate2 or torch and switches to float16 automatically.

```python
from mash_mix.backends.registry import get_stt

backend = get_stt("faster-whisper", model="large-v3")
```

```bash
nvidia-smi
```

What to expect: the large v3 model at float16 needs about 3 GB of VRAM. A second job on the same card needs another 3 GB, so size the card by the number of workers. The reference `whisper` implementation also uses CUDA through PyTorch.

#### A card without the CUDA runtime

This is a common setup and it is worth knowing about. CTranslate2 links against the
CUDA runtime libraries, and a machine can have an NVIDIA card and a working
`nvidia-smi` while those libraries are missing. The model then loads without
complaint and fails at the first inference with
`Library cublas64_12.dll is not found`.

The engine checks for the libraries before it chooses the GPU, so it runs on the
CPU in that case instead of crashing. If the check is wrong for your machine,
force the device:

```bash
set MASH_MIX_DEVICE=cpu     # Windows PowerShell: $env:MASH_MIX_DEVICE="cpu"
```

```python
backend.load("large-v3", device="cpu", compute_type="int8")
```

The same fallback runs at inference time. If a CUDA error appears anyway, the
model is reloaded on the CPU and the transcription is retried, so a broken
GPU setup never costs you the result.

### MPS

On Apple Silicon, PyTorch exposes the Metal Performance Shaders backend. Use it for the `transformers` based models, which is where the big synthesis models live. The CTranslate2 path used by `faster-whisper` does not use MPS, so on a Mac the fast path is the CPU path with int8.

```python
import torch

torch.backends.mps.is_available()   # True on Apple Silicon
```

### CPU threading

The CPU path is the fallback everywhere, and it is the path that matters on a phone and a Pi.

- Set `OMP_NUM_THREADS` to the number of physical cores. On a 4 core machine, `OMP_NUM_THREADS=4` is the usual starting point.
- Do not oversubscribe. On a small machine, more threads than cores makes the audio stutter rather than faster.
- Keep one worker process per physical core at most. More workers than cores makes every worker slower.

```bash
OMP_NUM_THREADS=4 mash-mix lyrics "song.mp3" -o out
```

### ONNX Runtime Mobile

ONNX Runtime Mobile is the build that reaches a phone and an embedded device. It is a small subset of the desktop runtime.

- The models must be converted and laid out the way the mobile tooling expects. Use the `onnxruntime` conversion tools.
- Only the operators that the mobile build implements are available, so not every model converts.
- The `sherpa-onnx` and `whisper.cpp` backends are the ones built for this target.

```bash
pip install onnxruntime
pip install 'mash-mix-audio-and-text[edge]'
```

## Running the HTTP API

### The interface on any screen

The web interface is one responsive page, so the same server serves a phone, a
tablet, a laptop and a wide desktop, and any browser or web view on Windows, Linux,
macOS, Android and iOS opens it without a change.

| Screen | What changes |
|---|---|
| Under 900 pixels wide | The two columns become one, and the result panel comes under the audio panel |
| Under 560 pixels wide | The label of a control sits above it, the measurements sit in two columns, and the details of the file stack |
| Under 380 pixels wide | The measurements sit in one column |
| A touch screen | The buttons, the links and the inputs grow to a tap target of 44 pixels |
| The system asks for less motion | The progress bar and the transitions stop moving |
| The page is printed | The drop box, the buttons and the licence line are left out, and the lyrics are not cut off |

The page carries the colour scheme of the system, so it follows the dark mode of the
machine, and it fits the safe area of a phone that has a notch or a rounded corner.

### On a phone

The server is useful on a phone for a local tool, for example to check a recording before you upload it anywhere. Termux makes this straightforward.

```bash
pkg install python ffmpeg
pip install 'mash-mix-audio-and-text[edge,server]'
mash-mix serve --host 127.0.0.1 --port 8000
```

Then open `http://127.0.0.1:8000` in the phone browser and talk to it from the same device.

Notes for a phone:

- Bind to `127.0.0.1`. The phone is the only client. If you bind to `0.0.0.0` the API is open to the local network, and there is no authentication in this package.
- Keep the model small. A large model will make the phone unusable.
- The server holds finished jobs in memory and in a temporary folder, so the files are gone when the server restarts.

### On a home server

A home server is the right place for the large models and the shared web interface.

```bash
pip install 'mash-mix-audio-and-text[full]'
mash-mix serve --host 0.0.0.0 --port 8000
```

Notes for a home server:

- Put it behind a reverse proxy with TLS and a password. The API has no authentication of its own.
- The upload route streams the file to disk in chunks, so a large upload does not have to fit in memory.
- Finished lyric files live in the temporary folder of the machine, under `mash-mix-jobs`. Clean it up on a schedule.
- If you run more than one worker process, each one keeps its own job table. Download the files from the worker that handled the request.

```bash
# systemd unit sketch
[Service]
ExecStart=/usr/bin/mash-mix serve --host 0.0.0.0 --port 8000
Restart=on-failure
User=mashmix
```

### Running uvicorn directly

```bash
uvicorn mash_mix.server.app:app --host 0.0.0.0 --port 8000 --workers 2
```

The app object is `mash_mix.server.app:app`. See [api.md](api.md) for the routes.

---

MASH-Mix-Audio-and-Text V0.6.2. Copyright (c) 2026 MASH (Khaled Bhuiyan). MIT Licence. See LICENSE.
