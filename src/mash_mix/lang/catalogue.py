# SPDX-License-Identifier: MIT | Copyright (c) 2026 MASH (Khaled Bhuiyan)
"""The international language catalogue.

The Bangla registry in :mod:`mash_mix.lang.registry` knows one language in
depth. A song from anywhere in the world needs a second, flatter table: a name
for the language, the code a model understands, the aliases a person types and
the script the language is written in.

The catalogue holds three things per language:

* the ISO 639-1 code, which is the code Whisper and every model in this
  package use,
* the aliases, so ``Hindi``, ``hindi`` and ``hi`` all find the same entry,
* the writing script, which is how the language of a title or a description
  is worked out when the site does not say.

The script of a language is also the way to read a legacy file: several
languages still live on a Windows code page, and the code page of a language is
recorded here next to it.

The Bangla letters are written as code points on purpose. Bengali and
Devanagari letters look alike, so a literal in a source file can be silently
changed into its twin.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple

__all__ = [
    "LanguageEntry",
    "LANGUAGES",
    "SCRIPT_LANGUAGES",
    "REGIONAL_VARIANTS",
    "resolve_entry",
    "resolve_tag",
    "iso_for",
    "entry_for",
    "detect_script",
    "detect_language_from_text",
    "script_language",
    "ansi_codepage",
    "fallback_chain",
    "model_language",
    "list_entries",
]


@dataclass(frozen=True)
class LanguageEntry:
    """One language in the international catalogue."""

    name: str
    iso639_1: str
    script: str
    aliases: Tuple[str, ...] = ()
    ansi_cp: Optional[int] = None
    region: str = ""
    notes: str = ""

    def to_dict(self) -> Dict[str, object]:
        return {
            "name": self.name,
            "iso639_1": self.iso639_1,
            "script": self.script,
            "aliases": list(self.aliases),
            "ansi_cp": self.ansi_cp,
            "region": self.region,
            "notes": self.notes,
        }


#: The writing systems the catalogue can recognise.
SCRIPTS = (
    "Latin",
    "Bengali",
    "Devanagari",
    "Gurmukhi",
    "Gujarati",
    "Oriya",
    "Tamil",
    "Telugu",
    "Kannada",
    "Malayalam",
    "Sinhala",
    "Thai",
    "Lao",
    "Tibetan",
    "Myanmar",
    "Georgian",
    "Khmer",
    "Greek",
    "Cyrillic",
    "Hebrew",
    "Arabic",
    "Hangul",
    "Hiragana",
    "Katakana",
    "Han",
)


def _e(
    name: str,
    iso: str,
    script: str,
    aliases: Tuple[str, ...] = (),
    ansi_cp: Optional[int] = None,
    region: str = "",
    notes: str = "",
) -> LanguageEntry:
    return LanguageEntry(
        name=name,
        iso639_1=iso,
        script=script,
        aliases=aliases,
        ansi_cp=ansi_cp,
        region=region,
        notes=notes,
    )


#: Every language the catalogue knows.
LANGUAGES: Tuple[LanguageEntry, ...] = (
    _e(
        "Bangla",
        "bn",
        "Bengali",
        ("bangla", "bengali", "bangali", "beng", "ben", "bengal", "bengalese"),
        None,
        "",
        "The native script is Bengali. The region is worked out from the text itself.",
    ),
    _e(
        "Hindi",
        "hi",
        "Devanagari",
        ("hindi", "hi", "hindi bhasha"),
        None,
        "IN",
        "Shares the Devanagari script with Marathi, Nepali and Sanskrit.",
    ),
    _e(
        "Urdu",
        "ur",
        "Arabic",
        ("urdu", "ur"),
        1256,
        "PK",
        "Shares the Arabic script with Arabic and Persian.",
    ),
    _e(
        "Arabic",
        "ar",
        "Arabic",
        ("arabic", "ar"),
        1256,
        "",
        "",
    ),
    _e(
        "Persian",
        "fa",
        "Arabic",
        ("persian", "farsi", "fa"),
        1256,
        "IR",
        "",
    ),
    _e(
        "English",
        "en",
        "Latin",
        ("english", "en", "en-us", "en-gb"),
        1252,
        "",
        "",
    ),
    _e(
        "Spanish",
        "es",
        "Latin",
        ("spanish", "espanol", "castilian", "es"),
        1252,
        "",
        "",
    ),
    _e(
        "French",
        "fr",
        "Latin",
        ("french", "francais", "fr"),
        1252,
        "",
        "",
    ),
    _e(
        "German",
        "de",
        "Latin",
        ("german", "deutsch", "de"),
        1252,
        "",
        "",
    ),
    _e(
        "Italian",
        "it",
        "Latin",
        ("italian", "italiano", "it"),
        1252,
        "",
        "",
    ),
    _e(
        "Portuguese",
        "pt",
        "Latin",
        ("portuguese", "pt", "pt-br", "pt-pt"),
        1252,
        "",
        "",
    ),
    _e(
        "Dutch",
        "nl",
        "Latin",
        ("dutch", "flemish", "nl"),
        1252,
        "",
        "",
    ),
    _e(
        "Indonesian",
        "id",
        "Latin",
        ("indonesian", "bahasa indonesia", "id"),
        1252,
        "",
        "",
    ),
    _e(
        "Malayalam",
        "ml",
        "Malayalam",
        ("malayalam", "ml"),
        None,
        "IN",
        "",
    ),
    _e(
        "Tamil",
        "ta",
        "Tamil",
        ("tamil", "ta"),
        None,
        "IN",
        "",
    ),
    _e(
        "Telugu",
        "te",
        "Telugu",
        ("telugu", "te"),
        None,
        "IN",
        "",
    ),
    _e(
        "Kannada",
        "kn",
        "Kannada",
        ("kannada", "kn"),
        None,
        "IN",
        "",
    ),
    _e(
        "Marathi",
        "mr",
        "Devanagari",
        ("marathi", "mr"),
        None,
        "IN",
        "",
    ),
    _e(
        "Nepali",
        "ne",
        "Devanagari",
        ("nepali", "ne"),
        None,
        "NP",
        "Shares the Devanagari script with Hindi and Marathi.",
    ),
    _e(
        "Sinhala",
        "si",
        "Sinhala",
        ("sinhala", "sinhalese", "si"),
        None,
        "LK",
        "",
    ),
    _e(
        "Gujarati",
        "gu",
        "Gujarati",
        ("gujarati", "gu"),
        None,
        "IN",
        "",
    ),
    _e(
        "Punjabi",
        "pa",
        "Gurmukhi",
        ("punjabi", "panjabi", "pa"),
        None,
        "IN",
        "",
    ),
    _e(
        "Russian",
        "ru",
        "Cyrillic",
        ("russian", "ru"),
        1251,
        "",
        "",
    ),
    _e(
        "Ukrainian",
        "uk",
        "Cyrillic",
        ("ukrainian", "uk"),
        1251,
        "",
        "",
    ),
    _e(
        "Greek",
        "el",
        "Greek",
        ("greek", "el"),
        1253,
        "",
        "",
    ),
    _e(
        "Hebrew",
        "he",
        "Hebrew",
        ("hebrew", "he", "iw"),
        1255,
        "",
        "",
    ),
    _e(
        "Thai",
        "th",
        "Thai",
        ("thai", "th"),
        874,
        "",
        "",
    ),
    _e(
        "Vietnamese",
        "vi",
        "Latin",
        ("vietnamese", "vi"),
        1258,
        "",
        "",
    ),
    _e(
        "Turkish",
        "tr",
        "Latin",
        ("turkish", "tr"),
        1254,
        "",
        "",
    ),
    _e(
        "Swahili",
        "sw",
        "Latin",
        ("swahili", "kiswahili", "sw"),
        1252,
        "",
        "",
    ),
    _e(
        "Chinese",
        "zh",
        "Han",
        ("chinese", "mandarin", "zh", "zh-cn", "zh-tw", "chinese simplified"),
        936,
        "",
        "The Han script is shared with Japanese and Korean.",
    ),
    _e(
        "Japanese",
        "ja",
        "Hiragana",
        ("japanese", "nihongo", "ja"),
        932,
        "",
        "",
    ),
    _e(
        "Korean",
        "ko",
        "Hangul",
        ("korean", "hangul", "ko"),
        949,
        "",
        "",
    ),
    _e(
        "Czech",
        "cs",
        "Latin",
        ("czech", "cs"),
        1250,
        "",
        "",
    ),
    _e(
        "Polish",
        "pl",
        "Latin",
        ("polish", "pl"),
        1250,
        "",
        "",
    ),
    _e(
        "Hungarian",
        "hu",
        "Latin",
        ("hungarian", "hu"),
        1250,
        "",
        "",
    ),
    _e(
        "Georgian",
        "ka",
        "Georgian",
        ("georgian", "ka"),
        None,
        "",
        "",
    ),
    _e(
        "Latvian",
        "lv",
        "Latin",
        ("latvian", "lv"),
        1257,
        "",
        "",
    ),
)

_INDEX: Dict[str, LanguageEntry] = {}
for _entry in LANGUAGES:
    _INDEX[_entry.iso639_1.lower()] = _entry
    _INDEX[_entry.name.lower()] = _entry
    for _alias in _entry.aliases:
        _INDEX.setdefault(_alias.lower(), _entry)

#: The languages each writing script points at. The first one is the usual one.
SCRIPT_LANGUAGES: Dict[str, Tuple[str, ...]] = {
    "Latin": ("en", "es", "fr", "de", "it", "pt", "nl", "id", "tr", "vi", "pl", "cs", "hu", "sw"),
    "Bengali": ("bn",),
    "Devanagari": ("hi", "mr"),
    "Gurmukhi": ("pa",),
    "Gujarati": ("gu",),
    "Oriya": ("or",),
    "Tamil": ("ta",),
    "Telugu": ("te",),
    "Kannada": ("kn",),
    "Malayalam": ("ml",),
    "Sinhala": ("si",),
    "Thai": ("th",),
    "Lao": ("lo",),
    "Tibetan": ("bo",),
    "Myanmar": ("my",),
    "Georgian": ("ka",),
    "Khmer": ("km",),
    "Greek": ("el",),
    "Cyrillic": ("ru", "uk"),
    "Hebrew": ("he",),
    "Arabic": ("ar", "ur", "fa"),
    "Hangul": ("ko",),
    "Hiragana": ("ja",),
    "Katakana": ("ja",),
    "Han": ("zh", "ja"),
}

#: The regional forms of a language, the usual one first.
#:
#: A language is one thing in the catalogue and a region is a way of saying
#: which form of it was meant. Bangla is written ``bn-BD`` in Bangladesh and
#: ``bn-IN`` in India, and a site may name either one. The pair is kept
#: together so that a request for one region can fall back on the other, in
#: both directions, when the exact one is not available.
REGIONAL_VARIANTS: Dict[str, Tuple[str, ...]] = {
    "bn": ("bn-BD", "bn-IN"),
    "en": ("en-US", "en-GB"),
    "zh": ("zh-CN", "zh-TW"),
    "pt": ("pt-BR", "pt-PT"),
    "es": ("es-ES", "es-MX"),
    "ar": ("ar-SA", "ar-EG"),
    "hi": ("hi-IN", "hi-NP"),
    "ur": ("ur-PK", "ur-IN"),
    "fa": ("fa-IR", "fa-AF"),
    "tr": ("tr-TR", "tr-CY"),
    "de": ("de-DE", "de-AT", "de-CH"),
    "fr": ("fr-FR", "fr-CA", "fr-BE"),
    "id": ("id-ID", "id-MY"),
    "sw": ("sw-KE", "sw-TZ"),
    "ta": ("ta-IN", "ta-LK"),
    "te": ("te-IN", "te-LK"),
    "kn": ("kn-IN", "kn-LK"),
    "ml": ("ml-IN", "ml-LK"),
    "mr": ("mr-IN",),
    "gu": ("gu-IN",),
    "pa": ("pa-IN",),
    "ne": ("ne-NP",),
    "si": ("si-LK",),
    "th": ("th-TH",),
    "vi": ("vi-VN",),
}

# The unicode blocks of the scripts. Every range is a pair of code points.
_SCRIPT_RANGES: Tuple[Tuple[str, int, int], ...] = (
    ("Latin", 0x0041, 0x024F),
    ("Greek", 0x0370, 0x03FF),
    ("Cyrillic", 0x0400, 0x04FF),
    ("Hebrew", 0x0590, 0x05FF),
    ("Arabic", 0x0600, 0x06FF),
    ("Devanagari", 0x0900, 0x097F),
    ("Bengali", 0x0980, 0x09FF),
    ("Gurmukhi", 0x0A00, 0x0A7F),
    ("Gujarati", 0x0A80, 0x0AFF),
    ("Oriya", 0x0B00, 0x0B7F),
    ("Tamil", 0x0B80, 0x0BFF),
    ("Telugu", 0x0C00, 0x0C7F),
    ("Kannada", 0x0C80, 0x0CFF),
    ("Malayalam", 0x0D00, 0x0D7F),
    ("Sinhala", 0x0D80, 0x0DFF),
    ("Thai", 0x0E00, 0x0E7F),
    ("Lao", 0x0E80, 0x0EFF),
    ("Tibetan", 0x0F00, 0x0FFF),
    ("Myanmar", 0x1000, 0x109F),
    ("Georgian", 0x10A0, 0x10FF),
    ("Khmer", 0x1780, 0x17FF),
    ("Hiragana", 0x3040, 0x309F),
    ("Katakana", 0x30A0, 0x30FF),
    ("Hangul", 0xAC00, 0xD7AF),
    ("Han", 0x4E00, 0x9FFF),
)


def list_entries() -> List[Dict[str, object]]:
    """Every catalogue entry as a plain dictionary."""
    return [entry.to_dict() for entry in LANGUAGES]


def resolve_entry(query: str) -> Optional[LanguageEntry]:
    """Find a language by name, by alias or by ISO code.

    ``auto`` finds nothing on purpose: auto means the language is not known yet.
    """
    cleaned = (query or "").strip().lower().replace("_", "-")
    if not cleaned or cleaned == "auto":
        return None
    entry = _INDEX.get(cleaned)
    if entry is not None:
        return entry
    # A regional tag such as bn-BD or zh-CN still finds its base language.
    base = cleaned.split("-")[0]
    return _INDEX.get(base)


def entry_for(iso: str) -> Optional[LanguageEntry]:
    """Find a language by its ISO code."""
    return resolve_entry(iso)


def iso_for(query: str) -> Optional[str]:
    """The ISO 639-1 code of a language, or ``None`` when it is not known."""
    entry = resolve_entry(query)
    return entry.iso639_1 if entry else None


def _canonical_tag(query: str) -> str:
    """A tag with a lowercase language and an uppercase region.

    ``bn-bd`` and ``BN_bd`` both come back as ``bn-BD``, which is how the
    region is written everywhere else in this package.
    """
    cleaned = (query or "").strip().replace("_", "-")
    if not cleaned:
        return ""
    parts = [part for part in cleaned.split("-") if part]
    if not parts:
        return ""
    head = parts[0].lower()
    rest = [part.upper() for part in parts[1:]]
    return "-".join([head, *rest])


def resolve_tag(query: str) -> str:
    """The canonical tag of a language, with the region kept when one was asked for.

    ``Bangla`` becomes ``bn`` and ``bn-BD`` stays ``bn-BD``. An unknown tag
    comes back empty, and ``auto`` is empty on purpose, because auto means the
    language is not known yet.
    """
    cleaned = _canonical_tag(query)
    if not cleaned or cleaned.lower() == "auto":
        return ""
    entry = resolve_entry(cleaned)
    if entry is None:
        return ""
    if "-" in cleaned and cleaned.split("-")[0] == entry.iso639_1:
        return cleaned
    return entry.iso639_1


def fallback_chain(query: str) -> List[str]:
    """The tags to try for a language, from the exact one to the nearest.

    The chain runs in both directions. A regional tag falls back on the plain
    language and then on the other region, and the plain language falls back
    on its regions, so ``bn-BD`` gives ``bn-BD``, ``bn``, ``bn-IN`` and ``bn``
    gives ``bn``, ``bn-BD``, ``bn-IN``.

    This is what lets a site that names ``bn-IN`` be served by a track filed
    under ``bn-BD`` and the other way round, without either side having to
    know that the other exists.
    """
    cleaned = _canonical_tag(query)
    if not cleaned or cleaned.lower() == "auto":
        return []
    base = cleaned.split("-")[0]
    entry = _INDEX.get(base)
    if entry is None:
        return [cleaned]

    variants = REGIONAL_VARIANTS.get(entry.iso639_1, ())
    chain: List[str] = [cleaned]
    if "-" in cleaned:
        # A regional tag. The plain language comes next, because a model and a
        # plain subtitle track both speak it.
        if entry.iso639_1 not in chain:
            chain.append(entry.iso639_1)
    for variant in variants:
        if variant not in chain:
            chain.append(variant)
    return chain


def model_language(query: str) -> str:
    """The tag a speech model understands, from any regional or plain tag.

    Every Whisper family model is taught one code per language, so a request
    for ``pt-BR`` or ``bn-BD`` has to reach the model as ``pt`` or ``bn``.
    The region is not lost, because the caller keeps the tag it asked for and
    the fallback chain knows how to get back to it.
    """
    chain = fallback_chain(query)
    for tag in chain:
        entry = resolve_entry(tag)
        if entry:
            return entry.iso639_1
    return ""


def script_language(script: str) -> Optional[str]:
    """The usual language of a writing script."""
    codes = SCRIPT_LANGUAGES.get((script or "").strip())
    return codes[0] if codes else None


def script_counts(text: str) -> Dict[str, int]:
    """Count the letters of a text per writing script."""
    counts: Dict[str, int] = {}
    for char in text or "":
        if not char.isalpha():
            continue
        code = ord(char)
        for name, low, high in _SCRIPT_RANGES:
            if low <= code <= high:
                counts[name] = counts.get(name, 0) + 1
                break
    return counts


def detect_script(text: str) -> str:
    """Name the writing script that dominates a text.

    Returns the name of the script, or ``Latin`` when the text holds no
    letters from a block that the catalogue knows, and ``unknown`` when the
    text holds no letters at all.
    """
    counts = script_counts(text)
    if not counts:
        return "unknown"
    return max(counts, key=lambda key: counts[key])


def detect_language_from_text(text: str) -> Tuple[Optional[str], float, str]:
    """Work out the language of a text from the script it is written in.

    Returns the ISO code, the share of the letters that carry it and the name
    of the script.

    A title is often a mixture, with the name of the song in Latin letters and
    the name of the film in its own script. The Latin letters say nothing about
    the language, so the strongest script that is not Latin is the one that
    counts, and Latin text alone gives no answer at all.
    """
    counts = script_counts(text)
    if not counts:
        return None, 0.0, "unknown"
    letters = sum(counts.values())
    strong = {name: count for name, count in counts.items() if name != "Latin"}
    if not strong:
        return None, 0.0, "Latin"
    script = max(strong, key=lambda key: strong[key])
    share = strong[script] / float(letters)
    return script_language(script), round(share, 4), script


def ansi_codepage(iso: str) -> Optional[int]:
    """The Windows code page a language's legacy files use."""
    entry = resolve_entry(iso)
    return entry.ansi_cp if entry else None
