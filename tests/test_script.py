# SPDX-License-Identifier: MIT | Copyright (c) 2026 MASH (Khaled Bhuiyan)
"""Tests for the Bangla and Devanagari script repair.

Every expected string is built from code points. Bengali and Devanagari letters
look alike, and editors and chat clients quietly normalise one into the other,
which turns a test into a misleading failure.
"""

from __future__ import annotations

from mash_mix.lang.script import (
    bangla_to_devanagari,
    devanagari_to_bangla,
    dominant_script,
    needs_script_repair,
)


def c(*points: int) -> str:
    """Build a string from code points."""
    return "".join(chr(point) for point in points)


# Devanagari code points.
DKA, DKHA, DGA, DGHA = 0x0915, 0x0916, 0x0917, 0x0918
DCA, DCHA, DJA, DJHA = 0x091A, 0x091B, 0x091C, 0x091D
DTTA, DTTHA, DDA, DDHA, DNNA = 0x091F, 0x0920, 0x0921, 0x0922, 0x0923
DTA, DTHA, DDA2, DDHA2, DNA = 0x0924, 0x0925, 0x0926, 0x0927, 0x0928
DPA, DPHA, DBA, DBHA, DMA, DYA, DRA, DLA = (
    0x092A,
    0x092B,
    0x092C,
    0x092D,
    0x092E,
    0x092F,
    0x0930,
    0x0932,
)
DVA, DSHA, DSSA, DSA, DHA, DLLA = 0x0935, 0x0936, 0x0937, 0x0938, 0x0939, 0x0933
DA, DAA, DI, DII, DU, DUU, DE, DAI, DO, DAU = (
    0x0905,
    0x0906,
    0x0907,
    0x0908,
    0x0909,
    0x090A,
    0x090F,
    0x0910,
    0x0913,
    0x0914,
)
DM_AA, DM_I, DM_II, DM_U, DM_UU, DM_E, DM_AI, DM_O, DM_AU = (
    0x093E,
    0x093F,
    0x0940,
    0x0941,
    0x0942,
    0x0947,
    0x0948,
    0x094B,
    0x094C,
)
DANUSVARA, DVISARGA, DCHANDRABINDU, DDANDA = 0x0902, 0x0903, 0x0901, 0x0964
DRA_NUKTA, DDRHA_NUKTA, DYA_NUKTA = 0x095C, 0x095D, 0x095F

# Bangla code points.
BKA, BKHA, BGA, BGHA = 0x0995, 0x0996, 0x0997, 0x0998
BCA, BCHA, BJA, BJHA = 0x099A, 0x099B, 0x099C, 0x099D
BTTA, BTTHA, BDA, BDHA, BNNA = 0x099F, 0x09A0, 0x09A1, 0x09A2, 0x09A3
BTA, BTHA, BDA2, BDHA2, BNA = 0x09A4, 0x09A5, 0x09A6, 0x09A7, 0x09A8
BPA, BPHA, BBA, BBHA, BMA, BYA, BRA, BLA = (
    0x09AA,
    0x09AB,
    0x09AC,
    0x09AD,
    0x09AE,
    0x09AF,
    0x09B0,
    0x09B2,
)
BSHA, BSSA, BSA, BHA, BLLA = 0x09B6, 0x09B7, 0x09B8, 0x09B9, 0x09B3
BA, BAA, BI, BII, BU, BUU, BE, BAI, BO, BAU = (
    0x0985,
    0x0986,
    0x0987,
    0x0988,
    0x0989,
    0x098A,
    0x098F,
    0x0990,
    0x0993,
    0x0994,
)
BM_AA, BM_I, BM_II, BM_U, BM_UU, BM_E, BM_AI, BM_O, BM_AU = (
    0x09BE,
    0x09BF,
    0x09C0,
    0x09C1,
    0x09C2,
    0x09C7,
    0x09C8,
    0x09CB,
    0x09CC,
)
BANUSVARA, BVISARGA, BCHANDRABINDU, BDANDA = 0x0982, 0x0983, 0x0981, 0x0964
BNUKTA = 0x09BC

ZWJ = 0x200D
ZWNJ = 0x200C
VIRAMA = 0x094D

# "Amar Sonar Bangla, ami tomay bhalobashi." in both scripts.
# The o kar is a sign, not the independent vowel, so BM_O and DM_O are used.
BENGALI_SENTENCE = c(
    BAA, BMA, BM_AA, BRA, 0x20,
    BSHA, BM_O, BNA, BM_AA, BRA, 0x20,
    BBA, BM_AA, BANUSVARA, BLA, BM_AA, 0x2C, 0x20,
    BAA, BMA, BI, 0x20,
    BTA, BM_O, BMA, BM_AA, BYA, 0x20,
    BBHA, BAA, BLA, BM_O, BBA, BI, BSA, BI, BDANDA,
)
DEVANAGARI_SENTENCE = c(
    DAA, DMA, DM_AA, DRA, 0x20,
    DSHA, DM_O, DNA, DM_AA, DRA, 0x20,
    DBA, DM_AA, DANUSVARA, DLA, DM_AA, 0x2C, 0x20,
    DAA, DMA, DI, 0x20,
    DTA, DM_O, DMA, DM_AA, DYA, 0x20,
    DBHA, DAA, DLA, DM_O, DBA, DI, DSA, DI, DDANDA,
)


def test_dominant_script_names_the_script():
    assert dominant_script(BENGALI_SENTENCE) == "bangla"
    assert dominant_script(DEVANAGARI_SENTENCE) == "devanagari"
    assert dominant_script("hello world") == "latin"
    assert dominant_script("") == "unknown"


def test_devanagari_becomes_bangla():
    assert devanagari_to_bangla(DEVANAGARI_SENTENCE) == BENGALI_SENTENCE


def test_bangla_becomes_devanagari():
    result = bangla_to_devanagari(BENGALI_SENTENCE)
    assert c(DAA, DMA, DM_AA, DRA) in result
    assert c(DSHA, DM_O, DNA, DM_AA, DRA) in result
    assert result == DEVANAGARI_SENTENCE


def test_shared_consonants_keep_their_shape():
    assert devanagari_to_bangla(c(DKA, DKHA, DGA, DGHA)) == c(BKA, BKHA, BGA, BGHA)
    assert devanagari_to_bangla(c(DCA, DCHA, DJA, DJHA)) == c(BCA, BCHA, BJA, BJHA)
    assert devanagari_to_bangla(c(DTTA, DTTHA, DDA, DDHA, DNNA)) == c(BTTA, BTTHA, BDA, BDHA, BNNA)
    assert devanagari_to_bangla(c(DTA, DTHA, DDA2, DDHA2, DNA)) == c(BTA, BTHA, BDA2, BDHA2, BNA)
    assert devanagari_to_bangla(c(DPA, DPHA, DBA, DBHA, DMA, DYA, DRA, DLA)) == c(
        BPA, BPHA, BBA, BBHA, BMA, BYA, BRA, BLA
    )
    assert devanagari_to_bangla(c(DSHA, DSSA, DSA, DHA)) == c(BSHA, BSSA, BSA, BHA)
    assert devanagari_to_bangla(c(DVA)) == c(BBA)


def test_vowels_and_matra_signs():
    assert devanagari_to_bangla(c(DA, DAA, DI, DII, DU, DUU)) == c(BA, BAA, BI, BII, BU, BUU)
    assert devanagari_to_bangla(c(DE, DAI, DO, DAU)) == c(BE, BAI, BO, BAU)
    assert devanagari_to_bangla(
        c(DKA, DM_AA, DKA, DM_I, DKA, DM_U, DKA, DM_E, DKA, DM_O, DKA, DM_AU)
    ) == c(BKA, BM_AA, BKA, BM_I, BKA, BM_U, BKA, BM_E, BKA, BM_O, BKA, BM_AU)
    assert devanagari_to_bangla(c(DANUSVARA, 0x20, DVISARGA, 0x20, DCHANDRABINDU)) == c(
        BANUSVARA, 0x20, BVISARGA, 0x20, BCHANDRABINDU
    )


def test_nukta_letters_are_carried_over():
    assert devanagari_to_bangla(c(DRA_NUKTA)) == c(BDA, BNUKTA)
    assert devanagari_to_bangla(c(DDRHA_NUKTA)) == c(BDHA, BNUKTA)
    assert devanagari_to_bangla(c(DYA_NUKTA)) == c(BYA, BNUKTA)


def test_zwj_is_dropped_and_zwnj_is_kept():
    # ZWJ builds Devanagari conjuncts, Bangla does not need it.
    assert devanagari_to_bangla(c(DKA, ZWJ, VIRAMA)) == c(BKA)
    assert c(ZWNJ) in devanagari_to_bangla(c(DKA, ZWNJ, DSSA))


def test_repair_is_only_needed_for_the_wrong_script():
    assert needs_script_repair(DEVANAGARI_SENTENCE, target="bangla") is True
    assert needs_script_repair(BENGALI_SENTENCE, target="bangla") is False
    assert needs_script_repair(BENGALI_SENTENCE, target="devanagari") is True
    assert needs_script_repair(DEVANAGARI_SENTENCE, target="devanagari") is False
