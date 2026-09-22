"""
Canonical vocabularies and text canonicalization.

This module directly addresses the Phase 1 finding (section 3, "Normalization
problems") that the old normalize_address() *deleted* semantically important
words (NO, BLOCK, OPP, NEAR, UNIT, FLOOR -> '') instead of canonicalizing
them. Deletion destroys the evidence a later parsing stage would need (e.g.
you can no longer tell "3rd Floor" was ever a floor once "FLOOR" is gone).

Principle applied throughout this file: every abbreviation maps to a full
canonical form, never to the empty string. If a word is structurally
meaningless as free text (e.g. "NO" as in "House No 5"), that meaning is
still used -- by the *parser* as a keyword anchor -- before any normalized
text is produced for fallback comparison. Nothing is silently thrown away
this early in the pipeline (constraint #4); only the final residual-text
stage (parser.py) decides what's unclassified, and even then the text is
kept, not deleted.
"""

from __future__ import annotations

import re
from typing import Dict, List


# ---------------------------------------------------------------------------
# State canonicalization (kept from the old code -- this part was already
# correct: a closed, known vocabulary, deterministic, no loss of meaning)
# ---------------------------------------------------------------------------

STATE_NAME_TO_CODE: Dict[str, str] = {
    "andhra pradesh": "ap", "arunachal pradesh": "ar", "assam": "as", "bihar": "br",
    "chhattisgarh": "cg", "goa": "ga", "gujarat": "gj", "haryana": "hr",
    "himachal pradesh": "hp", "jammu and kashmir": "jk", "jammu & kashmir": "jk",
    "jharkhand": "jh", "karnataka": "ka", "kerala": "kl", "madhya pradesh": "mp",
    "maharashtra": "mh", "manipur": "mn", "meghalaya": "ml", "mizoram": "mz",
    "nagaland": "nl", "orissa": "or", "odisha": "or", "punjab": "pb", "rajasthan": "rj",
    "sikkim": "sk", "tamil nadu": "tn", "tamilnadu": "tn", "tripura": "tr",
    "uttarakhand": "uk", "uttar pradesh": "up", "west bengal": "wb",
    "andaman and nicobar islands": "an", "chandigarh": "ch",
    "dadra and nagar haveli": "dh", "daman and diu": "dd",
    "delhi": "dl", "new delhi": "dl", "lakshadweep": "ld",
    "pondicherry": "py", "puducherry": "py",
}

# NOTE: the old code's replace_state_names_with_codes() mixed state-name
# folding with unrelated street-suffix folding ("road"->"rd", "street"->"st")
# and a generic "ind"->"in" rule in the SAME substitution dict (Phase 1,
# section 3). That coupling is deliberately undone here: state codes and
# street-suffix canonicalization are now two independent vocabularies, used
# by two independent extractors (state extractor vs. street extractor), so a
# locality that happens to be named "Road" is never at risk of being mangled
# by a state-normalization pass that has nothing to do with it.

STATE_CODE_SET = set(STATE_NAME_TO_CODE.values())

# "india"/"ind" are country markers, not states -- kept as a separate,
# explicitly-named set rather than folded into the state map, since treating
# "India" as if it were a state code was a latent bug risk in the old dict.
COUNTRY_MARKERS = {"india", "ind", "bharat"}


# ---------------------------------------------------------------------------
# General abbreviation canonicalization -- CANONICALIZE, NEVER DELETE
# ---------------------------------------------------------------------------
# Every entry expands an abbreviation to a single canonical full form.
# This canonical form is what downstream text-based comparators (token
# Jaccard, n-gram, residual bag-of-words) operate on, so "RD" and "ROAD"
# always compare as identical without either side's information being lost.

GENERAL_ABBREVIATIONS: Dict[str, str] = {
    "vil": "village", "dist": "district", "distt": "district",
    "po": "post office", "teh": "tehsil", "tehs": "tehsil",
    "bldg": "building", "apt": "apartment", "apts": "apartments",
    "grnd": "ground", "flt": "flat", "soc": "society",
    "chs": "cooperative housing society", "co-op": "cooperative",
    "coop": "cooperative", "resi": "residency", "res": "residency",
    "twr": "tower", "hgts": "heights", "cplx": "complex",
    "encl": "enclave", "nr": "near", "opp": "opposite",
    "behnd": "behind", "adj": "adjacent",
}

# Street/road-suffix canonicalization -- separate vocabulary from state
# codes (see note above). Used specifically by the street extractor.
STREET_SUFFIX_CANONICAL: Dict[str, str] = {
    "rd": "road", "st": "street", "mrg": "marg", "ln": "lane",
    "galli": "lane", "gali": "lane", "chowk": "square", "sqr": "square",
    "sq": "square",
}

# Unit/number-context keywords -- these are semantically meaningful as
# ANCHORS for the parser (they mean "the next number is a flat/unit/house
# number"), so they are canonicalized to a single label form rather than
# deleted. The parser (parser.py) consumes these as anchor keywords; the
# canonical form is what remains if any of this text ends up in residual.
UNIT_KEYWORDS: Dict[str, str] = {
    "flat": "flat", "flt": "flat", "unit": "unit", "house": "house",
    "house no": "house", "hno": "house", "h no": "house",
    "plot": "plot", "plot no": "plot", "room": "room", "rm": "room",
    "no": "number",  # "No" as in "Flat No 5" -- canonicalized, not deleted
}

FLOOR_KEYWORDS = {"floor", "flr", "fl"}

WING_KEYWORDS = {"wing", "block", "blk", "bloak"}  # "bloak" kept: common OCR/typo variant seen in old code's regex

BUILDING_SUFFIXES = {
    "apartment", "apartments", "tower", "towers", "society", "chs",
    "residency", "heights", "complex", "enclave", "nivas", "sadan",
    "chambers", "arcade", "plaza", "park", "gardens", "garden",
    "cooperative housing society", "chsl", "chsltd", "chs ltd",
    "premises", "estate", "mansion", "villa", "bhavan", "vihar",
}

LANDMARK_KEYWORDS = {"near", "opp", "opposite", "behind", "adjacent", "adj", "beside", "next to"}


# ---------------------------------------------------------------------------
# Canonicalization functions
# ---------------------------------------------------------------------------

def canonicalize_abbreviations(text: str, vocab: Dict[str, str]) -> str:
    """
    Word-boundary-safe replacement of each abbreviation with its canonical
    full form. Always expands; never maps to ''.
    Multi-word keys (e.g. "plot no") are applied before single-word keys
    so they are matched as a unit first.
    """
    if not text:
        return text
    # longest keys first, so multi-word phrases match before their
    # single-word prefixes/components do
    for abbr in sorted(vocab.keys(), key=len, reverse=True):
        full = vocab[abbr]
        text = re.sub(rf'\b{re.escape(abbr)}\b', full, text, flags=re.IGNORECASE)
    return text


def canonicalize_state_text(text: str) -> str:
    """Replace full state names with their 2-letter codes. Independent of
    street-suffix canonicalization (see module docstring)."""
    if not text:
        return text
    for name, code in STATE_NAME_TO_CODE.items():
        text = re.sub(rf'\b{re.escape(name)}\b', code, text, flags=re.IGNORECASE)
    return text


def canonicalize_street_suffixes(text: str) -> str:
    if not text:
        return text
    return canonicalize_abbreviations(text, STREET_SUFFIX_CANONICAL)


def canonicalize_general(text: str) -> str:
    if not text:
        return text
    return canonicalize_abbreviations(text, GENERAL_ABBREVIATIONS)


def strip_punctuation_keep_structure(text: str) -> str:
    """
    Uppercase and strip characters outside A-Z0-9 space / - ,
    Comma is now explicitly retained (unlike the old normalize_address,
    which stripped it in the same pass as everything else) because comma
    segmentation is used by the parser as a soft locality/city signal
    (Phase 1, section 5, item 10). '/' and '-' are also retained here --
    the OLD code stripped them at the very end of normalize_address,
    which destroyed the letter-number structure of things like "A-401"
    or "A/401" before any component-specific numeric canonicalization
    could see it. Here, that structure survives until numeric.py
    explicitly canonicalizes it per-field.
    """
    if not text:
        return text
    text = text.upper()
    text = re.sub(r'[^A-Z0-9\s/\-,]', ' ', text)
    text = re.sub(r'\s+', ' ', text).strip()
    return text
