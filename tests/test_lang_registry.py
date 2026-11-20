# SPDX-License-Identifier: MIT | Copyright (c) 2026 MASH (Khaled Bhuiyan)
"""The language tag registry."""

from __future__ import annotations

import pytest

from mash_mix.lang.registry import (
    ALL_LANGUAGES,
    BANGLA,
    BANGLA_BD,
    BANGLA_IN,
    is_bangla,
    list_languages,
    normalize_tag,
    region_of,
    resolve,
    script_of,
    sibling_tags,
)


def test_resolve_known_identifiers():
    """Every identifier that the registry knows maps to the right tag."""
    assert resolve("beng").tag == "bn"
    assert resolve("ben").tag == "bn"
    assert resolve("bangla").tag == "bn"
    assert resolve("bn").tag == "bn"
    assert resolve("bn-BD").tag == "bn-BD"
    assert resolve("bn_bd").tag == "bn-BD"
    assert resolve("ben-bd").tag == "bn-BD"
    assert resolve("bangladesh").tag == "bn-BD"
    assert resolve("bng").tag == "bn-BD"
    assert resolve("bn-IN").tag == "bn-IN"
    assert resolve("ben-in").tag == "bn-IN"
    assert resolve("bengali-india").tag == "bn-IN"


def test_resolve_is_case_insensitive():
    assert resolve("BENG").tag == resolve("beng").tag
    assert resolve("BN-bd").tag == "bn-BD"
    assert resolve("  bn-BD  ").tag == "bn-BD"


def test_resolve_returns_a_generic_specification_for_an_unknown_tag():
    """An unknown language is not silently turned into Bangla."""
    spec = resolve("zz")
    assert spec.tag == "zz"
    assert spec.is_bangla is False
    assert spec.notes == "Not registered in this package."


def test_is_bangla_for_the_bangla_tags():
    assert is_bangla("bn")
    assert is_bangla("bn-BD")
    assert is_bangla("bn-IN")
    assert is_bangla("bangla")
    assert is_bangla("beng")
    assert is_bangla("bng")


def test_is_bangla_is_false_for_other_scripts():
    assert not is_bangla("en")
    assert not is_bangla("hi")
    assert not is_bangla("ur")
    assert not is_bangla("zz")


def test_iso_codes_for_the_generic_bangla():
    assert BANGLA.iso639_1 == "bn"
    assert BANGLA.iso639_2b == "ben"
    assert BANGLA.iso639_2t == "beng"
    assert BANGLA.iso639_3 == "ben"
    assert BANGLA.iso639_5 == "ben"
    assert BANGLA.un_m49 == ""
    assert BANGLA.cldr_language == "beng"
    assert BANGLA.cldr_script == "Beng"
    assert BANGLA.region == ""


def test_iso_codes_for_bangladeshi_bangla():
    assert BANGLA_BD.iso639_1 == "bn"
    assert BANGLA_BD.iso639_2b == "ben"
    assert BANGLA_BD.iso639_2t == "beng"
    assert BANGLA_BD.iso639_3 == "ben"
    assert BANGLA_BD.iso639_5 == "ben"
    assert BANGLA_BD.un_m49 == "050"
    assert BANGLA_BD.cldr_language == "bng"
    assert BANGLA_BD.cldr_script == "Beng"
    assert BANGLA_BD.region == "BD"


def test_iso_codes_for_indian_bengali():
    assert BANGLA_IN.iso639_1 == "bn"
    assert BANGLA_IN.iso639_2b == "ben"
    assert BANGLA_IN.iso639_2t == "beng"
    assert BANGLA_IN.iso639_3 == "ben"
    assert BANGLA_IN.iso639_5 == "ben"
    assert BANGLA_IN.un_m49 == "356"
    assert BANGLA_IN.cldr_language == "beng"
    assert BANGLA_IN.cldr_script == "Beng"
    assert BANGLA_IN.region == "IN"


def test_un_m49_numbers():
    """The UN M.49 region numbers for Bangladesh and India."""
    numbers = {spec.tag: spec.un_m49 for spec in ALL_LANGUAGES}
    assert numbers == {"bn": "", "bn-BD": "050", "bn-IN": "356"}


def test_script_and_region_helpers():
    assert script_of("bn-BD") == "Bengali"
    assert region_of("bn-BD") == "BD"
    assert region_of("bn-IN") == "IN"
    assert region_of("bn") == ""


def test_sibling_tags():
    assert sibling_tags("bn-BD") == ["bn-BD", "bn", "bn-IN"]
    assert sibling_tags("bn") == ["bn", "bn-BD", "bn-IN"]
    assert len(sibling_tags("bn-IN")) == 3


def test_list_languages_has_three_entries():
    languages = list_languages()
    assert len(languages) == 3
    assert [entry["tag"] for entry in languages] == ["bn", "bn-BD", "bn-IN"]
    for entry in languages:
        assert entry["iso639_1"] == "bn"
        assert entry["cldr_script"] == "Beng"
        assert entry["direction"] == "ltr"
        assert isinstance(entry["aliases"], list)


def test_list_languages_is_json_friendly():
    import json

    payload = json.dumps(list_languages(), ensure_ascii=False)
    assert "bn-BD" in payload


def test_normalize_tag():
    assert normalize_tag("bn_bd") == "bn-BD"
    assert normalize_tag("BN-BD") == "bn-BD"
    assert normalize_tag("bn_in") == "bn-IN"
    assert normalize_tag("") == "bn"
    assert normalize_tag("beng") == "bn"


@pytest.mark.parametrize(
    "tag,expected",
    [("bn-BD", "bn-BD"), ("bn_IN", "bn-IN"), ("bangla", "bn")],
)
def test_resolve_round_trips_through_normalize_tag(tag, expected):
    assert resolve(tag).tag == expected
