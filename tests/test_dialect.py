# SPDX-License-Identifier: MIT | Copyright (c) 2026 MASH (Khaled Bhuiyan)
"""Regional and dialect detection for Bangla."""

from __future__ import annotations

from mash_mix.lang.dialect import detect_dialect, detect_language, detect_region
from mash_mix.lang.lexicon import DIALECT_MARKERS, DISTRICTS, DIVISIONS

BANGLADESHI = "ভাই, আপ্নে কেমন আছেন? টাকা পাঠাতে হবে।"
INDIAN = "আপনি কেমন আছেন? তিনি বলেছিলেন যে কাজটা করেছিলেন, অনুগ্রহ করে বলুন।"
NEUTRAL = "আমি বাড়িতে যাই।"


def test_a_bangladeshi_sentence_detects_bn_bd():
    report = detect_region(BANGLADESHI)
    assert report.tag == "bn-BD"
    assert report.language == "bn-BD"
    assert report.confidence > 0.0
    assert report.bd_score > report.in_score
    assert any(item.startswith("BD:") for item in report.evidence)


def test_an_indian_sentence_detects_bn_in():
    report = detect_region(INDIAN)
    assert report.tag == "bn-IN"
    assert report.language == "bn-IN"
    assert report.confidence > 0.0
    assert report.in_score > report.bd_score
    assert any(item.startswith("IN:") for item in report.evidence)


def test_a_neutral_sentence_keeps_the_default_with_zero_confidence():
    report = detect_region(NEUTRAL)
    assert report.tag == "bn-BD"
    assert report.confidence == 0.0
    assert report.bd_score == 0.0
    assert report.in_score == 0.0
    assert report.evidence == []


def test_a_neutral_sentence_can_keep_another_default():
    report = detect_region(NEUTRAL, default="bn-IN")
    assert report.tag == "bn-IN"
    assert report.confidence == 0.0


def test_empty_text_keeps_the_default():
    report = detect_region("")
    assert report.tag == "bn-BD"
    assert report.confidence == 0.0
    assert report.script == "unknown"


def test_the_script_is_reported():
    assert detect_region(BANGLADESHI).script == "bangla"
    assert detect_region("hello there").script == "latin"


def test_detect_dialect_finds_the_sylheti_cluster():
    cluster, confidence, evidence = detect_dialect("খাওয়া হইছে হাবি কেনা নাই")
    assert cluster == "sylheti"
    assert confidence > 0.0
    assert any(item.startswith("sylheti:") for item in evidence)


def test_detect_dialect_returns_nothing_without_markers():
    cluster, confidence, evidence = detect_dialect("")
    assert cluster is None
    assert confidence == 0.0
    assert evidence == []


def test_detect_region_reports_the_dialect():
    report = detect_region(BANGLADESHI)
    assert report.dialect in DIALECT_MARKERS
    assert report.dialect_confidence > 0.0


def test_detect_language_returns_the_specification():
    spec = detect_language(BANGLADESHI)
    assert spec.tag == "bn-BD"
    assert spec.iso639_1 == "bn"

    fallback = detect_language("")
    assert fallback.tag == "bn-BD"


def test_detect_language_handles_romanised_bangla():
    spec = detect_language("ami bangladesh dhaka theke")
    assert spec.is_bangla


def test_the_report_is_json_friendly():
    import json

    payload = json.dumps(detect_region(BANGLADESHI).to_dict(), ensure_ascii=False)
    assert "bn-BD" in payload


def test_the_district_lexicon_is_complete():
    # Bangladesh has 64 districts, spread over 8 divisions.
    assert len(DISTRICTS) == 64
    assert sum(len(names) for names in DIVISIONS.values()) == 64
    for names in DIVISIONS.values():
        for name in names:
            assert name in DISTRICTS
