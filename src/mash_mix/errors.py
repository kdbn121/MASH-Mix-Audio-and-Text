# SPDX-License-Identifier: MIT | Copyright (c) 2026 MASH (Khaled Bhuiyan)
"""Exception hierarchy for MASH-Mix-Audio-and-Text."""

from __future__ import annotations


class MashMixError(Exception):
    """Base class for every error raised by this package."""


class BackendUnavailableError(MashMixError):
    """Raised when a requested backend is not installed or not usable on this machine."""


class ModelNotFoundError(MashMixError):
    """Raised when a model id or path cannot be resolved."""


class UnsupportedLanguageError(MashMixError):
    """Raised when a language tag is unknown or unsupported by the selected backend."""


class AudioDecodeError(MashMixError):
    """Raised when an audio file cannot be decoded into samples."""


class DependencyMissingError(MashMixError):
    """Raised when an optional Python dependency is required for the requested feature."""

    def __init__(self, package: str, extra: str, feature: str) -> None:
        self.package = package
        self.extra = extra
        self.feature = feature
        super().__init__(
            f"'{package}' is required for {feature}. "
            f"Install it with: pip install 'mash-mix-audio-and-text[{extra}]'"
        )


class PipelineError(MashMixError):
    """Raised when a pipeline step cannot complete."""
