# SPDX-License-Identifier: MIT | Copyright (c) 2026 MASH (Khaled Bhuiyan)
"""Convert a Hugging Face Whisper checkpoint into the CTranslate2 format.

The fastest way to run a Whisper checkpoint is the CTranslate2 format, which is
what the ``faster-whisper`` backend loads. The Bangla fine tunes of Whisper large
v3 are only published as plain Hugging Face checkpoints, so this script converts
one and the result can then be used with the fast backend.

Read this before you trust the result. A quantised conversion can lose accuracy,
and a fine tuned model can lose more of it than the general one. Measure the
conversion with ``scripts/lyrics_eval.py`` against the real lyric file of a song
before you rely on it. When a conversion sounds wrong, try ``--quantization
float16``, which keeps the weights as they were.

Usage
-----
    python scripts/convert_model.py mozilla-ai/whisper-large-v3-turbo-bn
    python scripts/convert_model.py mozilla-ai/whisper-large-v3-bn \
        --quantization int8 --output ctranslate2/whisper-large-v3-bn

After the conversion the model is used like this:

    mash-mix lyrics "song.mp3" -o out \
        --backend faster-whisper \
        --model ctranslate2/whisper-large-v3-bn
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

#: The files the fast backend needs next to the converted weights.
TOKENIZER_FILES = (
    "tokenizer.json",
    "vocabulary.json",
    "vocab.json",
    "merges.txt",
    "preprocessor_config.json",
    "tokenizer_config.json",
    "added_tokens.json",
    "special_tokens_map.json",
    "normalizer.json",
    "generation_config.json",
)


def _copy_tokenizer_files(model: str, target: Path) -> list[str]:
    """Copy the tokenizer files of the source model into the converted folder.

    The converter only copies the files it is told about, and the fast backend
    needs the tokenizer of the original model to read the output.
    """
    copied: list[str] = []
    try:
        from huggingface_hub import hf_hub_download  # type: ignore
    except Exception:
        return copied
    for name in TOKENIZER_FILES:
        try:
            source = hf_hub_download(repo_id=model, filename=name)
        except Exception:
            continue
        try:
            (target / name).write_bytes(Path(source).read_bytes())
            copied.append(name)
        except OSError:
            continue
    return copied


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Convert a Whisper checkpoint to CTranslate2.")
    parser.add_argument("model", help="Hugging Face model id or a local checkpoint folder.")
    parser.add_argument(
        "--output",
        default=None,
        help="Where to write the converted model. Default: ctranslate2/<model name>.",
    )
    parser.add_argument(
        "--quantization",
        default="int8",
        choices=["int8", "int8_float16", "float16", "float32"],
        help="How to store the weights. Default: int8, the smallest and the fastest on a CPU.",
    )
    parser.add_argument("--force", action="store_true", help="Overwrite an existing folder.")
    parser.add_argument(
        "--copy-files",
        default=None,
        help="Comma separated extra files to copy into the converted folder.",
    )
    args = parser.parse_args(argv)

    try:
        from ctranslate2.converters import TransformersConverter  # type: ignore
    except Exception as exc:
        print(f"CTranslate2 is not installed: {exc}")
        print("Install it with: pip install ctranslate2 transformers")
        return 2

    name = args.model.strip()
    target = Path(args.output) if args.output else Path("ctranslate2") / name.split("/")[-1]
    if target.exists() and any(target.iterdir()):
        if not args.force:
            print(f"{target} already exists. Use --force to overwrite it.")
            return 1
        print(f"Replacing {target}")

    print(f"Converting {name} to CTranslate2 ({args.quantization})")
    copy_files = None
    if args.copy_files:
        copy_files = [item.strip() for item in args.copy_files.split(",") if item.strip()]
    converter = TransformersConverter(name, copy_files=copy_files)
    converter.convert(
        output_dir=str(target),
        quantization=args.quantization,
        force=bool(args.force),
    )
    copied = _copy_tokenizer_files(name, target)
    print(f"Written to {target}")
    if copied:
        print(f"Tokenizer files copied: {', '.join(copied)}")
    else:
        print(
            "No tokenizer files were copied. Copy them by hand if the fast "
            "backend cannot open the model."
        )
    print()
    print("Use it with the fast backend:")
    print(f'  mash-mix lyrics "song.mp3" -o out --backend faster-whisper --model {target}')
    return 0


if __name__ == "__main__":
    sys.exit(main())
