# SPDX-License-Identifier: MIT | Copyright (c) 2026 MASH (Khaled Bhuiyan)
"""The number of the release is the same everywhere.

The project keeps its number in one place, in ``mash_mix.__version__``, and that
number has to reach the archive, the page in the browser and the footer of every
document. These tests hold the places to that rule, so the number cannot drift
away from one place to another.
"""

from __future__ import annotations

import re
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parent.parent

# The documents that end with a footer of their own.
DOCUMENTS = [
    "CHANGELOG.md",
    "docs/README.md",
    "docs/api.md",
    "docs/architecture.md",
    "docs/backends.md",
    "docs/bangla.md",
    "docs/cli.md",
    "docs/lyrics.md",
    "docs/models.md",
    "docs/platforms.md",
]


def _version() -> str:
    """The number of the release."""
    mash_mix = pytest.importorskip("mash_mix")
    return str(mash_mix.__version__)


def test_the_project_file_and_the_package_carry_the_same_number() -> None:
    """The number in ``pyproject.toml`` is the number of the package."""
    text = (ROOT / "pyproject.toml").read_text(encoding="utf-8")
    found = re.search(r'^version\s*=\s*"([^"]+)"', text, re.MULTILINE)

    assert found, "pyproject.toml names a version"
    assert found.group(1) == _version()


def test_the_web_page_carries_the_number_of_the_release() -> None:
    """The footer of the page in the browser names the release."""
    server = pytest.importorskip("mash_mix.server.app")
    page = (server.STATIC_DIR / "index.html").read_text(encoding="utf-8")

    assert f"MASH-Mix-Audio-and-Text V{_version()}" in page


def test_every_document_carries_the_number_of_the_release() -> None:
    """The footer of every document names the release."""
    version = _version()
    for name in DOCUMENTS:
        text = (ROOT / name).read_text(encoding="utf-8")
        assert f"MASH-Mix-Audio-and-Text V{version}" in text, name


def test_the_readme_carries_the_number_of_the_release() -> None:
    """The licence section of the readme names the release."""
    text = (ROOT / "README.md").read_text(encoding="utf-8")

    assert f"MASH-Mix-Audio-and-Text V{_version()}" in text
