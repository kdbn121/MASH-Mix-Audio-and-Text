# SPDX-License-Identifier: MIT | Copyright (c) 2026 MASH (Khaled Bhuiyan)
"""Language tag registry for Bangla and friends.

Bangla is registered in several standards at once and they do not agree, so
this module keeps every identifier next to its canonical BCP 47 tag:

* BCP 47 (RFC 5646)   : ``bn``, ``bn-BD``, ``bn-IN``
* ISO 639-1          : ``bn``
* ISO 639-2/B        : ``ben``
* ISO 639-2/T        : ``beng``
* ISO 639-3          : ``ben``
* ISO 639-5          : ``ben``
* UN M.49            : ``050`` Bangladesh, ``356`` India
* CLDR language codes: ``beng`` Bengali, ``bng`` Bangla (Bangladesh flavour)
* CLDR script code   : ``Beng`` Bengali script

``resolve`` accepts any of those, in any case, and returns the canonical tag.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, List, Tuple

__all__ = [
    "LanguageSpec",
    "BANGLA",
    "BANGLA_BD",
    "BANGLA_IN",
    "ALL_LANGUAGES",
    "resolve",
    "is_bangla",
    "normalize_tag",
    "script_of",
    "region_of",
    "sibling_tags",
    "list_languages",
]


@dataclass(frozen=True)
class LanguageSpec:
    """One language with every standard identifier attached."""

    tag: str
    name_en: str
    name_native: str
    iso639_1: str
    iso639_2b: str
    iso639_2t: str
    iso639_3: str
    iso639_5: str
    un_m49: str
    script: str
    cldr_language: str
    cldr_script: str
    region: str
    direction: str = "ltr"
    notes: str = ""
    aliases: Tuple[str, ...] = field(default_factory=tuple)

    @property
    def is_bangla(self) -> bool:
        return self.iso639_1 == "bn"

    def to_dict(self) -> Dict[str, object]:
        return {
            "tag": self.tag,
            "name_en": self.name_en,
            "name_native": self.name_native,
            "iso639_1": self.iso639_1,
            "iso639_2b": self.iso639_2b,
            "iso639_2t": self.iso639_2t,
            "iso639_3": self.iso639_3,
            "iso639_5": self.iso639_5,
            "un_m49": self.un_m49,
            "script": self.script,
            "cldr_language": self.cldr_language,
            "cldr_script": self.cldr_script,
            "region": self.region,
            "direction": self.direction,
            "notes": self.notes,
            "aliases": list(self.aliases),
        }


BANGLA = LanguageSpec(
    tag="bn",
    name_en="Bangla (Bengali)",
    name_native="বাংলা (বাঙালি)",
    iso639_1="bn",
    iso639_2b="ben",
    iso639_2t="beng",
    iso639_3="ben",
    iso639_5="ben",
    un_m49="",
    script="Bengali",
    cldr_language="beng",
    cldr_script="Beng",
    region="",
    notes=(
        "Generic Bangla tag. Use bn-BD for Bangladeshi Bangla and bn-IN for Indian Bengali. "
        "CLDR also carries bng as a separate Bangla (Bangladesh) language code."
    ),
    aliases=("beng", "ben", "bangla", "bangali", "bengali", "bang", "bn-beng", "bn-latn-beng"),
)

BANGLA_BD = LanguageSpec(
    tag="bn-BD",
    name_en="Bangla - Bangladesh",
    name_native="বাংলা - বাংলাদেশ (বাঙালি বাংলা)",
    iso639_1="bn",
    iso639_2b="ben",
    iso639_2t="beng",
    iso639_3="ben",
    iso639_5="ben",
    un_m49="050",
    script="Bengali",
    cldr_language="bng",
    cldr_script="Beng",
    region="BD",
    notes=(
        "Bangladeshi Bangla. Eastern Nagari spelling with the "
        "ড় / ঢ় / য় nukta forms kept, the ৳ taka sign, the দাড়ি full stop, "
        "and the Bengali numeric system."
    ),
    aliases=("bn-bd", "beng-bd", "ben-bd", "bangla-bd", "bangladesh", "bn-bd-beng"),
)

BANGLA_IN = LanguageSpec(
    tag="bn-IN",
    name_en="Bengali - India",
    name_native="বাংলা - ভারত (বাঙালি)",
    iso639_1="bn",
    iso639_2b="ben",
    iso639_2t="beng",
    iso639_3="ben",
    iso639_5="ben",
    un_m49="356",
    script="Bengali",
    cldr_language="beng",
    cldr_script="Beng",
    region="IN",
    notes=(
        "Indian Bengali (West Bengal, Tripura, Assam). Slightly different "
        "conventions for the same sounds, for example ড় for the flap sound, "
        "and the colon style of the taka sign."
    ),
    aliases=("bn-in", "beng-in", "ben-in", "bangla-in", "bengali-india", "bn-in-beng"),
)

ALL_LANGUAGES: Tuple[LanguageSpec, ...] = (BANGLA, BANGLA_BD, BANGLA_IN)

_INDEX: Dict[str, LanguageSpec] = {}


def _register(key: str, spec: LanguageSpec, *, overwrite: bool = False) -> None:
    """Add a lookup key unless it is already taken by an earlier registration."""
    key = key.lower()
    if overwrite or key not in _INDEX:
        _INDEX[key] = spec


for _spec in ALL_LANGUAGES:
    _register(_spec.tag, _spec, overwrite=True)
    for _alias in _spec.aliases:
        _register(_alias, _spec)
    _register(_spec.iso639_1, _spec)
    _register(_spec.iso639_2b, _spec)
    _register(_spec.iso639_2t, _spec)
    _register(_spec.iso639_3, _spec)
    _register(_spec.cldr_language, _spec)

# "beng" is the CLDR code for Bengali and the ISO 639-2/T code, both of which
# mean the generic language rather than the Indian variant. The generic entry
# registered first, so it keeps the key.

_BANGLA_SUBTAGS = frozenset({"bn", "ben", "beng", "bng"})


def normalize_tag(tag: str) -> str:
    """Canonicalise a language tag: ``bn_bd`` -> ``bn-BD``, ``BENG`` -> ``bn``."""
    if not tag:
        return "bn"
    cleaned = tag.strip().replace("_", "-")
    parts = [p for p in cleaned.split("-") if p]
    if not parts:
        return "bn"
    base = parts[0].lower()
    spec = _INDEX.get(base)
    rest = parts[1:]
    if spec is None:
        # Unknown base language. Return a best effort canonical tag.
        out = [base]
        for part in rest:
            if len(part) == 4 and part.isalpha():
                out.append(part.title())
            elif len(part) in (2, 3) and part.isalpha():
                out.append(part.upper())
            else:
                out.append(part.lower())
        return "-".join(out)
    if not rest:
        return spec.tag
    region = ""
    for part in rest:
        up = part.upper()
        if len(up) == 2 and up.isalpha():
            region = up
    if region in ("BD", "IN") and spec.tag == "bn":
        return f"bn-{region}"
    if region:
        return f"{spec.tag}-{region}"
    return spec.tag


def resolve(tag: str) -> LanguageSpec:
    """Resolve any known identifier to its :class:`LanguageSpec`.

    An unknown language returns a generic specification built from the tag
    itself, so callers never silently get Bangla back for a language they did
    not ask for.
    """
    key = (tag or "").strip().lower().replace("_", "-")
    spec = _INDEX.get(key)
    if spec is not None:
        return spec
    canonical = normalize_tag(tag)
    spec = _INDEX.get(canonical.lower())
    if spec is not None:
        return spec
    parts = canonical.split("-")
    spec = _INDEX.get(parts[0].lower())
    if spec is not None:
        return spec
    return _generic_spec(canonical)


def _generic_spec(canonical: str) -> LanguageSpec:
    """Build a specification for a language this registry does not know."""
    parts = canonical.split("-")
    base = parts[0].lower() if parts else "und"
    region = ""
    for part in parts[1:]:
        if len(part) == 2 and part.isalpha():
            region = part.upper()
    return LanguageSpec(
        tag=canonical,
        name_en=base,
        name_native=base,
        iso639_1=base if len(base) == 2 else "",
        iso639_2b=base if len(base) == 3 else "",
        iso639_2t=base if len(base) == 3 else "",
        iso639_3=base if len(base) == 3 else "",
        iso639_5=base if len(base) == 3 else "",
        un_m49="",
        script="",
        cldr_language=base,
        cldr_script="",
        region=region,
        notes="Not registered in this package.",
    )


def is_bangla(tag: str) -> bool:
    """True when the tag refers to any Bangla variety."""
    key = (tag or "").strip().lower().replace("_", "-")
    base = key.split("-")[0]
    if base in _BANGLA_SUBTAGS:
        return True
    spec = _INDEX.get(key)
    if spec is not None:
        return spec.is_bangla
    return False


def script_of(tag: str) -> str:
    """Return the script name for a tag, for example ``Bengali``."""
    return resolve(tag).script


def region_of(tag: str) -> str:
    """Return the region code, for example ``BD``, ``IN`` or an empty string."""
    return resolve(tag).region


def sibling_tags(tag: str) -> List[str]:
    """All Bangla tags, with the requested one first."""
    spec = resolve(tag)
    out = [spec.tag]
    for other in ALL_LANGUAGES:
        if other.tag != spec.tag:
            out.append(other.tag)
    return out


def list_languages() -> List[Dict[str, object]]:
    """Every registered language as a plain dictionary."""
    return [spec.to_dict() for spec in ALL_LANGUAGES]
