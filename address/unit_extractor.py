"""
Minimal deterministic extractor: UNIT, SUBUNIT, PIN, BASE_ADDRESS.

Built to the finalized candidate->component assignment specification.
This is a SEPARATE module from parser.py -- it does not replace or
modify the existing 12-role parser, which continues to exist unchanged
for its own callers. This module is the new, minimal-scope extractor
approved for the matching layer.

CORE INVARIANT: every span recorded here is an offset into the TRUE,
UNMODIFIED raw_address string passed in. No preprocessing that could
change string length or character positions runs before candidate
detection. This is the direct fix for the span-tracking defect in
parser.py (spans there are recorded against a working string already
mutated by clean_ocr_noise/remove_salutations/format_address, so they
cannot be used to cut the true original string). PIN's regex-based
matching (including the OCR-tolerant path) works directly against
raw_address -- it needs no preprocessing to find its pattern.

RULE SET IMPLEMENTED:
  - PIN: existing mechanism (numeric.py), UNCHANGED, called directly on
    raw_address. No PIN-vs-UNIT precedence rule is implemented -- per
    explicit instruction this was deferred as unproven by any real
    corpus example. A genuine collision must be isolated and reported,
    not silently resolved.
  - UNIT strong keywords (U1, unconditionally decisive):
    Flat, Flat No, Flat No., Unit, Apartment, Apt, Shop, Shop No, Gala, Room
  - SUBUNIT keywords (S1, unconditionally decisive): Wing, Block, Tower
  - Relationship rule (S2): a bare candidate immediately following a
    resolved SUBUNIT, separated only by nothing/single-space/hyphen,
    becomes UNIT at INFERRED confidence. Kept in its own clearly-named
    function (_apply_relationship_rule) because it has ZERO direct
    support in the 70-address validation corpus -- it exists only
    because the original design brief's illustrative example requires
    it structurally, not because real data confirmed it.
  - Negative-context set (N1, unconditional suppression to UNKNOWN):
    Sector, Phase, Pocket, Extension, Ext, Scheme, Stage, Zone, Floor,
    Road, Marg, Street, Lane
  - Compound identifiers: shape ALONE never splits a candidate. A U1
    keyword directly preceding a compound value claims the WHOLE value
    as UNIT (Rule C2). Absent a keyword, a compound stays UNKNOWN
    (Rule C3). Rule C1 (keyword-justified split of a compound) has no
    corpus trigger and is not implemented as an active code path.
  - Multiple independent claims on DIFFERENT candidates (Rule M1) are
    never ambiguous -- each resolves on its own evidence.
  - Multiple claims competing for the SAME role (Rule M2 for UNIT, M3
    for SUBUNIT) produce AMBIGUOUS, with no invented precedence.

DEFERRED, NOT IMPLEMENTED (kept structurally addable, not built):
  - Plot, Plot No as any tier of UNIT keyword.
  - House, House No, H.No as any tier of UNIT keyword, and the
    proper-noun exclusion problem generally.
  - Any unanchored/no-keyword UNIT rule (Rule U3 is the permanent
    default: no keyword, no relationship -> UNKNOWN).
  - Rule C1 (compound splitting via keyword-justified evidence).
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from typing import List, Optional, Tuple

from .datamodel import ComponentType, ParsedComponent, ExtractionConfidence
from .numeric import (
    extract_pin_code, extract_pin_code_ocr_tolerant,
    canonicalize_numeric_identifier,
)


UNIT_STRONG_KEYWORDS = [
    "Flat No.", "Flat No", "Flat",
    "Unit", "Apartment", "Apt",
    "Shop No", "Shop", "Gala", "Room",
]

# Category B (added after auditing the real 133-row corpus for missing_on_one
# cases): explicit "<Word> No." style anchored keywords, NOT bare word
# forms. Per explicit instruction: House/D/Property/Row House are only
# recognized here in their "No"/"No." possessive form -- adding the bare
# words ("House", "Property", "Door") is NOT approved, since a bare
# "House" or "Property" commonly appears as part of a building/business
# NAME (e.g. "Capital Trust House", already flagged as a deferred, unsolved
# proper-noun problem in an earlier session) and would misfire there.
# "<Word> No[.]" is a much narrower, safer syntactic marker.
UNIT_STRONG_KEYWORDS_NO_FORM = [
    "Row House No.", "Row House No",
    "House No.", "House No",
    "D.No.", "D.No", "D No.", "D No",
    "Property No.", "Property No",
]

SUBUNIT_KEYWORDS = ["Wing", "Block", "Tower"]

_ALL_UNIT_STRONG_KEYWORDS = UNIT_STRONG_KEYWORDS + UNIT_STRONG_KEYWORDS_NO_FORM

NEGATIVE_CONTEXT_KEYWORDS = [
    "Sector", "Phase", "Pocket", "Extension", "Ext", "Scheme",
    "Stage", "Zone", "Floor", "Road", "Marg", "Street", "Lane",
]


def _keyword_pattern(keywords: List[str]) -> re.Pattern:
    ordered = sorted(keywords, key=len, reverse=True)
    escaped = [re.escape(k).replace(r'\ ', r'\s+') for k in ordered]
    return re.compile(r'\b(?:' + '|'.join(escaped) + r')\.?', re.IGNORECASE)


_UNIT_STRONG_PATTERN = _keyword_pattern(_ALL_UNIT_STRONG_KEYWORDS)
_SUBUNIT_PATTERN = _keyword_pattern(SUBUNIT_KEYWORDS)

# Reverse-order SUBUNIT keywords: value BEFORE the keyword, e.g. "C WING",
# "E BLOCK" (mirror of "Wing C"/"Block E"). Deliberately excludes TOWER --
# per explicit instruction, Tower's mapping to SUBUNIT is a deferred
# business-rule question, not extended to the reverse-order case in this
# pass. Scoped narrowly to exactly these two keywords, not a generic
# "<value> <word>" pattern, so an arbitrary letter followed by an
# unrelated word can never be mistaken for a wing.
_REVERSE_SUBUNIT_KEYWORDS = ["Wing", "Block"]
_REVERSE_SUBUNIT_PATTERN = re.compile(
    r'^\s*(?:' + '|'.join(re.escape(k) for k in _REVERSE_SUBUNIT_KEYWORDS) + r')\b',
    re.IGNORECASE,
)
_NEGATIVE_CONTEXT_PATTERN = _keyword_pattern(NEGATIVE_CONTEXT_KEYWORDS)

# A candidate value: EITHER a numeric-led alphanumeric cluster (401, 401B,
# 4B-308, 548/D, C-5/32) OR a bare single letter (for SUBUNIT values like
# "Wing B", which carry no digit at all -- the original pattern required a
# digit and silently could never match a bare-letter SUBUNIT candidate;
# found and fixed during validation testing against the required
# representative cases, specifically "Wing B" and "Block B 401").
# Compound letter-prefixed forms (D-817, C-5/32) must capture the FULL
# span including the leading letter and separator, not just the trailing
# digits, or the whole-compound-stays-intact requirement (Rule C3) is
# violated by silently dropping the prefix.
_CANDIDATE_VALUE_PATTERN = re.compile(
    r'\b[A-Za-z]-?\d[\dA-Za-z]*(?:[/\-][A-Za-z0-9]+)*\b'   # letter-prefixed compound: D-817, C-5/32
    r'|\b\d[\dA-Za-z]*(?:[/\-][A-Za-z0-9]+)*\b'             # digit-led: 401, 401B, 4B-308, 548/D
    r'|\b[A-Za-z]\b'                                          # bare single letter: B (for Wing B, Block B)
)

# Narrow, additional pattern for a real user-reported case: a value split
# by a MISTYPED comma acting as an internal separator rather than a field
# break, e.g. "Flat-711,B" meaning the single unit "711B", not two
# separate fields. Distinguishing signal: a digit run immediately
# followed by a comma immediately followed by EXACTLY ONE letter (not a
# whole word) -- a genuine field break virtually always has a space
# after the comma and/or is followed by a real multi-letter word
# ("Flat 401, Mumbai"), so this is deliberately scoped tight rather than
# treating comma as a general-purpose internal separator (which would
# incorrectly fuse genuinely separate fields elsewhere). This is checked
# SEPARATELY from _CANDIDATE_VALUE_PATTERN, as a post-processing merge
# step (see _merge_comma_split_candidates), rather than folded into the
# main regex, so the common case (comma as field break) is never at risk.
_COMMA_SPLIT_COMPOUND_PATTERN = re.compile(r'(\d[\dA-Za-z]*),([A-Za-z])(?![A-Za-z])')

_S2_CONTINUATION_SEP = re.compile(r'^(?:\s|-)*$')
_S2_BREAK_CHARS = set(',./:;()')


@dataclass
class Candidate:
    raw_text: str
    span: Tuple[int, int]
    is_compound: bool
    preceding_keyword: Optional[str] = None
    preceding_keyword_tier: Optional[str] = None
    preceding_keyword_span: Optional[Tuple[int, int]] = None
    resolution: str = "UNKNOWN"
    confidence: Optional[ExtractionConfidence] = None
    type_metadata: Optional[str] = None
    rule_fired: Optional[str] = None
    removal_span: Optional[Tuple[int, int]] = None


@dataclass
class ExtractionResult:
    original_text: str
    unit: Optional[ParsedComponent] = None
    subunit: Optional[ParsedComponent] = None
    pin: Optional[ParsedComponent] = None
    base_address: str = ""
    candidates: List[Candidate] = field(default_factory=list)
    unit_ambiguous_candidates: List[str] = field(default_factory=list)
    subunit_ambiguous_candidates: List[str] = field(default_factory=list)


def _extract_pin(raw_address: str) -> Tuple[Optional[ParsedComponent], Optional[Tuple[int, int]]]:
    pin = extract_pin_code(raw_address)
    if pin == -1:
        all_pins = list(set(re.findall(r'(?<!\d)\d{6}(?!\d)', raw_address)))
        comp = ParsedComponent(
            ComponentType.PIN, None, raw_address, ExtractionConfidence.AMBIGUOUS,
            source="unit_extractor:pin_multiple_candidates", candidates=all_pins,
        )
        return comp, None
    if pin:
        m = re.search(re.escape(pin), raw_address)
        span = m.span() if m else None
        comp = ParsedComponent(
            ComponentType.PIN, pin, pin, ExtractionConfidence.ANCHORED,
            source="unit_extractor:pin_six_digit_isolated", span=span,
        )
        return comp, span

    ocr_pin, raw_token = extract_pin_code_ocr_tolerant(raw_address)
    if ocr_pin:
        m = re.search(re.escape(raw_token), raw_address, flags=re.IGNORECASE)
        span = m.span() if m else None
        corrected_chars = sum(1 for a, b in zip(raw_token.upper(), ocr_pin) if a != b)
        comp = ParsedComponent(
            ComponentType.PIN, ocr_pin, raw_token, ExtractionConfidence.ANCHORED,
            source="unit_extractor:pin_ocr_tolerant", span=span,
            ocr_corrected=corrected_chars > 0, ocr_corrected_chars=corrected_chars,
        )
        return comp, span

    return None, None


def _detect_candidates(raw_address: str, pin_span: Optional[Tuple[int, int]]) -> List[Candidate]:
    candidates = []
    for m in _CANDIDATE_VALUE_PATTERN.finditer(raw_address):
        span = m.span()
        if pin_span and span[0] >= pin_span[0] and span[1] <= pin_span[1]:
            continue
        raw_text = m.group(0)
        is_compound = bool(re.search(r'[/\-]', raw_text)) or bool(re.fullmatch(r'\d+[A-Za-z]+|[A-Za-z]+\d+', raw_text))
        candidates.append(Candidate(raw_text=raw_text, span=span, is_compound=is_compound))
    return candidates


def _find_preceding_keyword(raw_address: str, candidate: Candidate) -> None:
    prefix = raw_address[:candidate.span[0]]
    tail_match = re.search(r'([A-Za-z][A-Za-z .]*?)\s*[-:#]?\s*$', prefix)
    if not tail_match:
        return
    tail = tail_match.group(1)
    tail_start = tail_match.start(1)

    for pattern, tier in ((_UNIT_STRONG_PATTERN, "U1_STRONG"),
                           (_SUBUNIT_PATTERN, "SUBUNIT"),
                           (_NEGATIVE_CONTEXT_PATTERN, "NEGATIVE")):
        km = pattern.search(tail)
        if km and (km.end() == len(tail.rstrip()) or tail.rstrip().endswith(km.group(0).rstrip('.'))):
            keyword_text = km.group(0)
            keyword_span = (tail_start + km.start(), tail_start + km.end())
            candidate.preceding_keyword = keyword_text
            candidate.preceding_keyword_tier = tier
            candidate.preceding_keyword_span = keyword_span
            return


def _find_following_subunit_keyword(raw_address: str, candidate: Candidate) -> None:
    """
    Reverse-order counterpart to _find_preceding_keyword, scoped ONLY to
    SUBUNIT (Wing/Block, NOT Tower -- deferred per explicit instruction).
    Only ever considers a candidate that is a BARE SINGLE LETTER -- a
    multi-character or numeric candidate is never eligible, so this
    cannot misfire on something like "18 STREET" or "PLOT ROAD". Requires
    the literal keyword "Wing"/"Block" to appear immediately after
    (allowing only whitespace/comma/hyphen between), so an arbitrary
    letter followed by an unrelated word is never mistaken for a wing --
    e.g. "C COLONY" or "A APARTMENTS" do NOT match, since neither
    "Colony" nor "Apartments" is in _REVERSE_SUBUNIT_KEYWORDS.

    Does NOT override an existing preceding_keyword_tier already set by
    the forward check -- if a candidate is somehow already claimed
    another way, this reverse check does not run at all (see call site).
    """
    if not re.fullmatch(r'[A-Za-z]', candidate.raw_text):
        return
    tail = raw_address[candidate.span[1]:candidate.span[1] + 20]
    connector_match = re.match(r'\s*[-,]?\s*', tail)
    remainder = tail[connector_match.end():] if connector_match else tail
    km = _REVERSE_SUBUNIT_PATTERN.match(remainder)
    if km:
        keyword_start = candidate.span[1] + connector_match.end()
        keyword_end = keyword_start + len(km.group(0))
        candidate.preceding_keyword = km.group(0)
        candidate.preceding_keyword_tier = "SUBUNIT"
        # keyword_span here actually covers the FOLLOWING keyword, not a
        # preceding one -- reused as-is since _resolve_pass_1 only ever
        # uses preceding_keyword_span to compute a removal_span, and for
        # this reverse case the removal span must extend to the keyword's
        # END (not start), which is exactly what (candidate.span[0], end)
        # produces when _resolve_pass_1 computes
        # (c.preceding_keyword_span[0], c.span[1]) -- see the dedicated
        # override below instead, since that formula assumes the keyword
        # comes BEFORE, not after.
        candidate.preceding_keyword_span = (candidate.span[0], keyword_end)
        candidate.type_metadata = "REVERSE_ORDER_SUBUNIT"


def _resolve_pass_1(candidates: List[Candidate]) -> None:
    for c in candidates:
        if c.preceding_keyword_tier == "U1_STRONG":
            c.resolution = "UNIT"
            c.confidence = ExtractionConfidence.ANCHORED
            c.type_metadata = _classify_unit_keyword(c.preceding_keyword)
            c.rule_fired = "U1+C2" if c.is_compound else "U1"
            c.removal_span = (c.preceding_keyword_span[0], c.span[1])
        elif c.preceding_keyword_tier == "SUBUNIT":
            c.resolution = "SUBUNIT"
            c.confidence = ExtractionConfidence.ANCHORED
            is_reverse_order = c.type_metadata == "REVERSE_ORDER_SUBUNIT"
            c.type_metadata = "WING" if is_reverse_order else _classify_subunit_keyword(c.preceding_keyword)
            c.rule_fired = "S1_reverse" if is_reverse_order else "S1"
            # Forward case ("Wing C"): keyword_span starts BEFORE the
            # candidate, so removal_span = (keyword_start, candidate_end).
            # Reverse case ("C WING"): keyword_span was recorded as
            # (candidate_start, keyword_end) by
            # _find_following_subunit_keyword -- removal_span must be
            # (candidate_start, keyword_end), i.e. the FULL stored span,
            # not just its start paired with candidate.span[1] (which
            # would wrongly truncate the removal to exclude "WING"/"BLOCK"
            # itself, leaving it dangling in BASE_ADDRESS).
            c.removal_span = c.preceding_keyword_span if is_reverse_order else (c.preceding_keyword_span[0], c.span[1])
        elif c.preceding_keyword_tier == "NEGATIVE":
            c.resolution = "UNKNOWN"
            c.rule_fired = "N1"
        elif c.is_compound:
            c.resolution = "UNKNOWN"
            c.rule_fired = "C3"
        else:
            c.resolution = "UNKNOWN"
            c.rule_fired = "U3"


# ---------------------------------------------------------------------------
# Leading-unit inference (Rules L1-L4) -- recovers genuine UNIT identifiers
# that open an address with no keyword at all, a pattern confirmed common
# in the real 133-row corpus ("278/1 SHED NO...", "A-106 MOUNT KAILASH...",
# "711 WEST GURU ANGAD NAGAR", etc.).
#
# This is NOT "first candidate = UNIT". A naive version of that rule was
# tested against the full 133-row corpus and produced confirmed false
# positives: "P" and "O" from "KUSUMAGIRI P.O." (a Post Office abbreviation,
# not an identifier), "5TH" from "5TH FLOOR" (an ordinal describing a
# floor, not a unit), and "D" from "D Y ROAD" (part of a street name).
# Investigating those failures against real candidate output (not assumed)
# showed the false positives share two structural properties genuine
# leading identifiers never had in this corpus:
#   (a) no digit anywhere in the candidate cluster (a bare letter with
#       nothing numeric attached, e.g. isolated "P", "D"), and/or
#   (b) the candidate is immediately followed by a NEGATIVE_CONTEXT
#       keyword ("FLOOR", "ROAD") that already exists in this file's
#       own suppression vocabulary for a different purpose -- reused
#       here rather than inventing a second list.
# Rules L1-L4 below encode exactly these distinguishing properties.
# Genuine identifiers in the corpus (278/1, A-106, 711, D NO 5-276,
# 59/18, D-817, 4B/308, B 003, 201, 801, 56/78) all satisfy L1 and
# never trigger L2/L3/L4; the confirmed false positives all fail L1 or
# trigger L2/L3/L4 -- verified directly, not assumed, against this
# exact corpus (see the accompanying test suite and the 133-row report).
# ---------------------------------------------------------------------------

_ORDINAL_PATTERN = re.compile(r'^\d+(?:ST|ND|RD|TH)$', re.IGNORECASE)

# Words that are explicitly DEFERRED as UNIT keywords (Plot only, as of
# Category B: House/D.No/Row House No/Property No were PROMOTED to real
# anchored keywords -- see UNIT_STRONG_KEYWORDS_NO_FORM -- and so are no
# longer deferred; removed from this pattern accordingly, since leaving
# them here would be stale/misleading even though it was harmless in
# practice -- a promoted keyword is caught by _resolve_pass_1's U1 branch
# before this function ever runs on that candidate). The leading-unit
# inference rule must NOT silently re-introduce a still-deferred keyword
# (Plot) as decisive by guessing at the number that follows it -- that
# would defeat the deliberate deferral. Found and fixed during earlier
# testing: "Plot 8" alone was resolving to UNIT=8 via leading inference,
# because with no keyword TABLE entry for "Plot", nothing stopped the
# leading-candidate rule from treating the following "8" as an ordinary
# unanchored leading number.
_DEFERRED_KEYWORD_PATTERN = re.compile(r'\bPlot(?:\s+No\.?)?\b\.?\s*$', re.IGNORECASE)

# Separate, narrower block: bare words that are NOT approved as UNIT
# keywords in any form and are known (from the earlier proper-noun
# investigation) to commonly appear as part of a building/business NAME
# rather than as a role-anchoring word -- e.g. "Capital Trust House 2".
# Found via testing: with "House" removed from _DEFERRED_KEYWORD_PATTERN
# after Category B promoted "House No"/"House No." specifically, a BARE
# "House" (no "No" suffix) was left with no guard at all, and the
# leading-unit inference rule started silently treating a proper-noun
# collision as a genuine leading identifier. This is the same
# already-flagged, deliberately-unsolved proper-noun problem from an
# earlier session -- addressed here only enough to stop the NEW leading-
# inference code path from reopening it, not as a general solution.
_PROPER_NOUN_RISK_WORD_PATTERN = re.compile(r'\b(?:House|Property|Block|Wing|Tower)\b\.?\s*$', re.IGNORECASE)


def _has_digit(text: str) -> bool:
    return any(ch.isdigit() for ch in text)


# ---------------------------------------------------------------------------
# Category A -- explicit recognized prefix constructions. Derived from
# inspecting the actual failing rows in the 133-row corpus, NOT a generic
# "skip N tokens" or distance-based mechanism. Each pattern below matches
# ONE specific, narrow syntactic shape found in real data; the prefix
# itself is matched and skipped (never treated as UNIT, never removed
# from BASE_ADDRESS beyond what the skipped span itself covers), and the
# SAME L1-L4 structural validation is then applied to whatever candidate
# immediately follows -- no relaxation of those safeguards for prefixed
# candidates.
#
# Patterns implemented, each backed by a specific real corpus example:
#   P1 "<value> BLOCK," -- e.g. "S BLOCK, C36A...", "C BLOCK, 5/48..."
#       A block-identifier-as-prefix construction, syntactically distinct
#       from the SUBUNIT keyword usage "Block <value>" (which fires via
#       Rule S1 already) precisely because BLOCK follows rather than
#       precedes the value here, and is comma-terminated.
#   P2 "GROUND FLOOR-<value>" -- e.g. "GROUND FLOOR- A/27..."
#       A fixed floor-descriptor phrase, hyphen-terminated.
#   P3 "NO <single-letter>, <value>" -- e.g. "NO M, 36..."
#       A bare "NO" marker followed by a single-letter sub-identifier,
#       comma, then the real value -- narrower than the ordinary
#       "NO <value>" keyword-adjacency check already used elsewhere in
#       this file, since here TWO tokens (the letter, then the number)
#       both need to be skipped past, not one.
#
# NOT implemented (Category C, left unresolved per explicit instruction):
#   "C/O <name>, <filler clause>, <value>" -- e.g. "C/O MULTIPURPOSE
#   CORPORATION 30B SHED NO A/1 278/1..." does not reduce to a simple
#   fixed-prefix skip: there is an entire additional candidate-shaped
#   clause ("30B SHED NO A/1") between the C/O marker and the real
#   target, which would require scanning past an arbitrary number of
#   intervening candidates -- exactly the distance-based inference this
#   task explicitly prohibits. Left unresolved.
#   "S/O <name>, ..." -- resolved separately and independently through
#   Category B (the "D NO" keyword it happens to contain), not through
#   any Category A prefix rule -- no A pattern was needed for this case.
# ---------------------------------------------------------------------------

_PREFIX_BLOCK_PATTERN = re.compile(r'^([A-Za-z0-9]{1,4})\s+BLOCK\s*,\s*', re.IGNORECASE)
_PREFIX_GROUND_FLOOR_PATTERN = re.compile(r'^GROUND\s+FLOOR\s*-\s*', re.IGNORECASE)
_PREFIX_NO_LETTER_COMMA_PATTERN = re.compile(r'^NO\s+([A-Za-z])\s*,\s*', re.IGNORECASE)


def _find_category_a_prefix_skip(raw_address: str) -> Optional[int]:
    """Returns the character offset to skip TO (i.e. where the real
    candidate search should begin) if the address opens with one of the
    three recognized Category A prefix constructions, else None. Checked
    only against the very start of the address -- this is still a
    leading-position mechanism, just with one recognized prefix construct
    stepped over first, not a general scan."""
    for pattern in (_PREFIX_BLOCK_PATTERN, _PREFIX_GROUND_FLOOR_PATTERN, _PREFIX_NO_LETTER_COMMA_PATTERN):
        m = pattern.match(raw_address)
        if m:
            return m.end()
    return None


# ---------------------------------------------------------------------------
# Letter+digit compound splitting (single-letter WING prefix/suffix).
#
# Real corpus evidence (see the KAVERI KUNJ CHS pair): the SAME building
# is described three ways across the dataset --
#     "B 003, KAVERI KUNJ CHS..."
#     "B/03, KAVERI KUNJ CHS..."
#     "03, B- wing, KAVERI KUNJ CHS..."
# -- the third form spells "wing" out explicitly, directly confirming
# that the leading letter in the first two forms IS a wing designator,
# not part of a fused flat number. This is the evidence basis for
# splitting the single-letter-prefix shape. The letter-SUFFIX shape
# (701B, 817D) is treated the same way per explicit instruction, though
# no direct corroborating "spelled out" example exists for that specific
# direction in the corpus -- confidence is INFERRED for both directions,
# never ANCHORED, since this is a structural/shape inference, not an
# explicit keyword match.
#
# SCOPE, deliberately narrow (per explicit instruction -- do NOT extend
# to arbitrary multi-part identifiers):
#   Rule W1 (prefix):  <single letter><separator><digits>
#       separator in {"-", "/", " "} (bare, no other characters)
#   Rule W2 (suffix):  <digits><single letter>
#       fused directly, no separator
# Explicitly NOT matched (left as whole/unresolved compounds, unchanged):
#   AB-101, ABC101, 701AB, A1B2, 4B-308, 703-A1-66, C-5/32
#   -- multi-letter prefixes/suffixes, multi-part separator chains, and
#   shapes with more than one letter or more than one separator all fall
#   outside this rule's scope by construction (the regexes below only
#   ever match exactly one letter and one digit run).
# ---------------------------------------------------------------------------

_WING_PREFIX_PATTERN = re.compile(r'^([A-Za-z])([-/ ])(\d{1,5})$')
_WING_SUFFIX_PATTERN = re.compile(r'^(\d{1,5})[-/]?([A-Za-z])$')


def _apply_wing_compound_split(raw_address: str, candidates: List[Candidate]) -> None:
    """
    Runs AFTER _resolve_pass_1 (so it only ever considers candidates
    still UNKNOWN -- anything already keyword-anchored, e.g. "Wing D,
    Flat 817", is left completely untouched, per explicit instruction
    that this rule must never override stronger explicit evidence) and
    BEFORE leading-inference/relationship rules (so a split candidate is
    not also re-processed by those, which only act on still-UNKNOWN
    candidates and would otherwise see the now-resolved pieces).
    """
    i = 0
    category_a_skip_to = _find_category_a_prefix_skip(raw_address)
    while i < len(candidates):
        c = candidates[i]
        # Allow through if UNKNOWN (the ordinary shape-only case), OR if
        # it was resolved via Rule C2 -- a SINGLE keyword (Flat/Unit/etc.)
        # claiming a WHOLE compound value (e.g. "Flat 701B", "Flat
        # D-817"). Per explicit instruction, this case MUST still split
        # (the compound-splitting decision supersedes Rule C2's old
        # "keep whole" default for this specific letter+digit shape).
        # Genuinely separate, independently-anchored fields ("Wing D,
        # Flat 817") are NOT Rule C2 candidates at all -- each keyword
        # there claims its OWN single-value candidate directly (rule
        # "S1"/"U1", not "U1+C2"), so they are correctly excluded by
        # this same check and never reach the split logic, preserving
        # the explicit requirement not to touch that case.
        is_c2_whole_compound = c.rule_fired == "U1+C2"
        if c.resolution != "UNKNOWN" and not is_c2_whole_compound:
            i += 1
            continue

        # Reject ordinals (5TH, 1ST, 2ND, 3RD) and any candidate that
        # is itself an already-excluded shape before even trying to
        # match -- these must never be treated as compound identifiers.
        if _ORDINAL_PATTERN.match(c.raw_text):
            i += 1
            continue

        # Case W2 (suffix): single candidate already contains both
        # parts fused, e.g. "701B" is ONE candidate from detection.
        m_suffix = _WING_SUFFIX_PATTERN.match(c.raw_text)
        if m_suffix:
            digits, letter = m_suffix.group(1), m_suffix.group(2)
            # If this candidate was originally claimed whole by a
            # keyword (Rule C2, e.g. "Flat 701B"), the removal span
            # must extend back to include that keyword too, or it is
            # left dangling in BASE_ADDRESS (found via testing: "Flat
            # 701B, ABC Heights" -> base_address left as "Flat , ABC
            # Heights" without this fix).
            whole = (c.preceding_keyword_span[0], c.span[1]) if is_c2_whole_compound and c.preceding_keyword_span else c.span
            _split_candidate_into_unit_and_subunit(
                raw_address, candidates, i,
                unit_text=digits, subunit_text=letter,
                unit_span=(c.span[0], c.span[0] + len(digits)),
                subunit_span=(c.span[0] + len(digits), c.span[1]),
                whole_span=whole, rule_id="W2_letter_suffix",
            )
            i += 1
            continue

        # Case W1 (prefix), single-candidate form: "D-817"/"A-106" are
        # already captured as one compound candidate by
        # _CANDIDATE_VALUE_PATTERN's letter-prefixed branch.
        m_prefix = _WING_PREFIX_PATTERN.match(c.raw_text)
        if m_prefix:
            letter, sep, digits = m_prefix.groups()
            letter_end = c.span[0] + 1
            whole = (c.preceding_keyword_span[0], c.span[1]) if is_c2_whole_compound and c.preceding_keyword_span else c.span
            _split_candidate_into_unit_and_subunit(
                raw_address, candidates, i,
                unit_text=digits, subunit_text=letter,
                unit_span=(c.span[1] - len(digits), c.span[1]),
                subunit_span=(c.span[0], letter_end),
                whole_span=whole, rule_id="W1_letter_prefix_fused",
            )
            i += 1
            continue

        # Case W1 (prefix), TWO-candidate form: "D/814", "B/03", "B 003"
        # are captured as separate bare-letter + digit candidates by
        # detection (the "/" or " " between them is not part of either
        # candidate's own span). Only merge if this candidate is a bare
        # single letter and the NEXT candidate is a bare digit run with
        # nothing but an allowed separator between them.
        if re.fullmatch(r'[A-Za-z]', c.raw_text) and i + 1 < len(candidates):
            nxt = candidates[i + 1]
            if nxt.resolution == "UNKNOWN" and re.fullmatch(r'\d{1,5}', nxt.raw_text):
                between = raw_address[c.span[1]:nxt.span[0]]
                if re.fullmatch(r'[-/ ]', between):
                    # If this candidate is the one a recognized Category
                    # A prefix skip pointed to (e.g. "GROUND FLOOR-"
                    # immediately preceding "A/27"), extend the removal
                    # span back to cover that prefix too, or it is left
                    # dangling in BASE_ADDRESS -- same orphan-keyword
                    # class of bug already fixed elsewhere in this file,
                    # found here via testing ("GROUND FLOOR- A/27" left
                    # "GROUND FLOOR- ," behind without this).
                    whole = (0, nxt.span[1]) if category_a_skip_to == c.span[0] else (c.span[0], nxt.span[1])
                    _split_candidate_into_unit_and_subunit(
                        raw_address, candidates, i,
                        unit_text=nxt.raw_text, subunit_text=c.raw_text,
                        unit_span=nxt.span, subunit_span=c.span,
                        whole_span=whole, rule_id="W1_letter_prefix_split",
                        absorb_index=i + 1,
                    )
                    i += 2
                    continue

        # Case W2 (suffix), TWO-candidate form: "18 C", "180 A" -- a bare
        # digit run followed by a bare single letter, separated only by
        # whitespace, are captured as two separate candidates by
        # detection. Symmetric to the W1 two-candidate case above, mirror
        # image order (digits first, then letter).
        if re.fullmatch(r'\d{1,5}', c.raw_text) and i + 1 < len(candidates):
            nxt = candidates[i + 1]
            if nxt.resolution == "UNKNOWN" and re.fullmatch(r'[A-Za-z]', nxt.raw_text):
                between = raw_address[c.span[1]:nxt.span[0]]
                if re.fullmatch(r'[-/ ]', between):
                    whole = (0, nxt.span[1]) if category_a_skip_to == c.span[0] else (c.span[0], nxt.span[1])
                    # index MUST be i+1 (the LETTER candidate) since
                    # _split_candidate_into_unit_and_subunit always turns
                    # candidates[index] into the SUBUNIT half -- passing
                    # i here would mutate the DIGIT candidate's identity
                    # into the subunit, leaving a confusing duplicate
                    # entry in the candidate list even though the final
                    # unit/subunit VALUES happened to still come out
                    # correct (found via direct testing before this fix).
                    _split_candidate_into_unit_and_subunit(
                        raw_address, candidates, i + 1,
                        unit_text=c.raw_text, subunit_text=nxt.raw_text,
                        unit_span=c.span, subunit_span=nxt.span,
                        whole_span=whole, rule_id="W2_letter_suffix_split",
                        absorb_index=i,
                    )
                    i += 2
                    continue
        i += 1


def _split_candidate_into_unit_and_subunit(
    raw_address: str, candidates: List[Candidate], index: int,
    unit_text: str, subunit_text: str,
    unit_span: Tuple[int, int], subunit_span: Tuple[int, int],
    whole_span: Tuple[int, int], rule_id: str,
    absorb_index: Optional[int] = None,
) -> None:
    """Resolves candidates[index] AS the SUBUNIT half, and appends a new
    synthetic Candidate for the UNIT half so both feed the existing
    candidate-list-based final-selection logic in extract() unchanged.
    If absorb_index is given (the two-candidate merge case), that
    candidate is marked absorbed so it is never independently resolved
    or double-counted."""
    subunit_candidate = candidates[index]
    subunit_candidate.raw_text = subunit_text
    subunit_candidate.span = subunit_span
    subunit_candidate.resolution = "SUBUNIT"
    subunit_candidate.confidence = ExtractionConfidence.INFERRED
    subunit_candidate.type_metadata = "WING_COMPOUND_SPLIT"
    subunit_candidate.rule_fired = rule_id
    # removal_span covers the WHOLE original compound (both halves),
    # not just this candidate's own piece -- required so BASE_ADDRESS
    # removes "701B"/"D-817" in its entirety, not merely "B"/"D".
    subunit_candidate.removal_span = whole_span

    unit_candidate = Candidate(
        raw_text=unit_text, span=unit_span, is_compound=False,
        resolution="UNIT", confidence=ExtractionConfidence.INFERRED,
        type_metadata="WING_COMPOUND_SPLIT", rule_fired=rule_id,
        removal_span=None,  # the SUBUNIT half's removal_span already covers the whole compound; do not double-remove
    )
    candidates.append(unit_candidate)

    if absorb_index is not None:
        candidates[absorb_index].resolution = "ABSORBED"
        candidates[absorb_index].rule_fired = "absorbed_into_wing_compound_split"


def _apply_leading_unit_inference(raw_address: str, candidates: List[Candidate]) -> None:
    if not candidates:
        return

    skip_to = _find_category_a_prefix_skip(raw_address)
    if skip_to is not None:
        first = next((c for c in candidates if c.span[0] >= skip_to), None)
        is_category_a = True
    else:
        first = candidates[0]
        is_category_a = False
    if first is None:
        return
    if first is not candidates[0] and not is_category_a:
        # safety invariant: outside the recognized Category A prefix
        # skip, this function must NEVER consider anything but the true
        # first candidate -- this branch should be unreachable, kept as
        # an explicit guard rather than silently falling through.
        return

    # Only ever applies to the very first candidate, and only if nothing
    # else already resolved it (a keyword match always takes precedence
    # -- this function runs strictly after _resolve_pass_1).
    if first.resolution != "UNKNOWN":
        return
    # Do not override a candidate N1 already explicitly suppressed --
    # N1 is a stronger, keyword-driven negative signal than this
    # positional inference should ever attempt to overrule. EXCEPTION:
    # if this candidate was reached via a Category A prefix skip, the N1
    # may have fired on the very prefix construction we just deliberately
    # stepped past (found during testing: "GROUND FLOOR- A/27" -- FLOOR
    # is a negative-context keyword and matched immediately before "A",
    # incorrectly blocking the Category A inference that already
    # accounted for "GROUND FLOOR-" as a recognized, intentional skip).
    # In that specific situation the N1 is redundant with the skip
    # itself, not a genuine independent negative signal, so it is safe
    # to proceed.
    if first.rule_fired == "N1" and not is_category_a:
        return
    # Do not guess if a DEFERRED keyword (Plot) immediately precedes
    # this candidate -- deferred means deferred, not "guess anyway
    # because no ANCHORED rule claimed it."
    prefix = raw_address[:first.span[0]]
    if _DEFERRED_KEYWORD_PATTERN.search(prefix):
        return
    # Do not guess if a PROPER-NOUN-RISK word (House, Property, Block,
    # Wing, Tower) immediately precedes this candidate, UNLESS we
    # arrived here via a recognized Category A prefix skip that already
    # explicitly accounted for that exact word (e.g. "S BLOCK," is a
    # recognized, deliberate skip; a bare "Capital Trust House 2" is not
    # -- nothing explicitly decided that "House" here was safe to step
    # past, so it must not be silently guessed through).
    if not is_category_a and _PROPER_NOUN_RISK_WORD_PATTERN.search(prefix):
        return

    # Rule L1 -- digit requirement. The candidate itself must contain a
    # digit, OR (for the bare-single-letter case, e.g. "D NO 5-276",
    # "B 003") the NEXT candidate in the address must be digit-bearing
    # and immediately adjacent (only whitespace, "NO", "No.", or "#"
    # between them -- the same tolerance _find_preceding_keyword already
    # uses elsewhere in this file for keyword/value adjacency).
    first_index = candidates.index(first)
    second = candidates[first_index + 1] if first_index + 1 < len(candidates) else None
    if second is not None and second.preceding_keyword_tier == "U1_STRONG":
        # Bug found via testing: "D NO 879" -- if the SECOND candidate
        # already independently qualifies for a strong ANCHORED keyword
        # claim of its own (here "879" is directly anchored by "D NO",
        # a Category B keyword), the leading-inference merge must NOT
        # absorb it into a weaker INFERRED letter+digit cluster with the
        # first candidate. The stronger, independent claim always wins;
        # simply decline to treat this as a mergeable pair at all, and
        # let _resolve_pass_1's already-computed U1 resolution for
        # `second` stand untouched.
        return
    cluster_text = first.raw_text
    cluster_span = first.span
    cluster_is_letter_plus_digit_pair = False

    if _has_digit(first.raw_text) and second is not None and _has_digit(second.raw_text):
        # Both already contain digits individually, but may still be ONE
        # split identifier with a loose separator between them (found
        # during testing: "5/ 48" -- a space after the slash caused the
        # candidate pattern to split it into "5" and "48" as two
        # separate candidates, even though both already "have a digit"
        # and L1 would otherwise trivially pass on "5" alone, silently
        # truncating the real value). Only merge if the between-text is
        # a plain separator, not ordinary address text (e.g. do NOT
        # merge "711" and "110092" from an unrelated PIN elsewhere).
        between = raw_address[first.span[1]:second.span[0]]
        if re.fullmatch(r'\s*[/\-]\s*', between):
            cluster_is_letter_plus_digit_pair = True  # reuses the same merge/canonicalization path
            connector = "/" if "/" in between else "-"
            cluster_text = f"{first.raw_text}{connector}{second.raw_text}"
            cluster_span = (first.span[0], second.span[1])
    elif _has_digit(first.raw_text):
        pass  # L1 satisfied directly by the first candidate itself
    elif second is not None and _has_digit(second.raw_text):
        between = raw_address[first.span[1]:second.span[0]]
        # Accept whitespace/"NO"/"#" (the original connector set) AND a
        # bare "/" or "-" (found missing during testing: "A/27" was
        # splitting into candidates "A" and "27" with a bare "/" between
        # them, which the original connector check rejected outright,
        # incorrectly leaving a genuine compound identifier unresolved).
        if re.fullmatch(r'\s*(?:NO\.?|#|[/\-])?\s*', between, flags=re.IGNORECASE):
            cluster_is_letter_plus_digit_pair = True
            # Build the cluster's canonical raw_text from just the two
            # meaningful parts (letter + digit-run), not the literal
            # in-between filler text -- capturing the raw slice
            # verbatim (e.g. "D NO 5-276") would feed the connector
            # word itself into canonicalization and corrupt the value
            # (found during testing: produced "DNO5276" instead of a
            # clean D+5-276 identifier).
            # Preserve the ACTUAL connector character (/ or -) if that's
            # what was in the source; otherwise (whitespace/"NO"/"#")
            # default to "-" as the canonical join, consistent with how
            # the rest of this file represents such compounds.
            connector = "/" if "/" in between else ("-" if "-" in between else "-")
            cluster_text = f"{first.raw_text}{connector}{second.raw_text}"
            cluster_span = (first.span[0], second.span[1])
        else:
            return  # L1 fails: bare letter not adjacent to any digit-bearing candidate
    else:
        return  # L1 fails: no digit anywhere in reach

    # Rule L2 -- ordinal exclusion. A candidate that IS purely an ordinal
    # word (5TH, 1ST, 2ND, 3RD) is describing a position (a floor, a
    # cross-street), never a unit identifier on its own.
    if _ORDINAL_PATTERN.match(first.raw_text):
        return

    # Rule L3 -- dotted-abbreviation exclusion. A bare single letter
    # immediately followed by a period and another bare single letter
    # (P.O., P O with a following period, etc.) is an administrative
    # abbreviation pattern, not an identifier -- checked directly against
    # the original string, not just candidate shape, since the period is
    # not itself part of any candidate span.
    if not _has_digit(first.raw_text) and len(first.raw_text) == 1:
        tail = raw_address[first.span[1]:first.span[1] + 4]
        if re.match(r'\.\s*[A-Za-z]\.', tail):
            return

    # Rule L4 -- following negative-context exclusion. If the very next
    # real word after the candidate cluster is one of this file's own
    # NEGATIVE_CONTEXT_KEYWORDS (Floor, Road, Sector, ...), the candidate
    # is describing that concept (a floor number, part of a road name),
    # not a standalone unit -- reuses the existing suppression vocabulary
    # rather than a new one.
    tail_after_cluster = raw_address[cluster_span[1]:cluster_span[1] + 20]
    following_word_match = re.match(r'[\s,]*([A-Za-z]+)', tail_after_cluster)
    if following_word_match:
        following_word = following_word_match.group(1)
        if _NEGATIVE_CONTEXT_PATTERN.fullmatch(following_word.rstrip('.')):
            return

    # All four rules passed -- accept as an INFERRED unit, distinct in
    # both mechanism and confidence from an ANCHORED keyword match.
    first.resolution = "UNIT"
    first.confidence = ExtractionConfidence.INFERRED
    first.type_metadata = "LEADING_INFERRED" if not is_category_a else "LEADING_INFERRED_CATEGORY_A"
    first.removal_span = cluster_span
    if is_category_a:
        # extend the removal span to also cover the recognized prefix
        # itself (e.g. "S BLOCK, ", "GROUND FLOOR- ", "NO M, ") -- the
        # prefix is never turned INTO the unit value, but it must not be
        # left dangling in BASE_ADDRESS either, same principle as the
        # existing orphan-keyword cleanup elsewhere in this file.
        first.removal_span = (0, cluster_span[1])

    if cluster_is_letter_plus_digit_pair:
        first.raw_text = cluster_text
        first.is_compound = True   # route through structure-preserving canonicalization
                                     # (>=2 separator check), not the plain single-value
                                     # canonicalizer -- without this, "D-5-276" collapses
                                     # to "D5276" instead of preserving the D / 5-276 split.
        first.rule_fired = "L1-L4_letter_digit_pair" if not is_category_a else "CategoryA_letter_digit_pair"
        # the second candidate (the digit part) is now folded into the
        # first candidate's cluster and must not be independently
        # resolved or removed a second time.
        second.resolution = "UNKNOWN"
        second.rule_fired = "absorbed_into_leading_cluster"
    else:
        first.rule_fired = "L1-L4_leading_unit" if not is_category_a else "CategoryA_prefix_skip"


def _classify_unit_keyword(keyword_text: str) -> str:
    k = keyword_text.strip().rstrip('.').lower()
    if k.startswith('flat'):
        return "FLAT"
    if k.startswith('unit'):
        return "UNIT"
    if k.startswith('apartment') or k.startswith('apt'):
        return "APARTMENT"
    if k.startswith('shop'):
        return "SHOP"
    if k.startswith('gala'):
        return "GALA"
    if k.startswith('room'):
        return "ROOM"
    if k.startswith('row house'):
        return "ROW_HOUSE"
    if k.startswith('house'):
        return "HOUSE"
    if re.match(r'^d\s*\.?\s*no', k):
        return "DOOR_NO"
    if k.startswith('property'):
        return "PROPERTY"
    return "UNKNOWN"


def _classify_subunit_keyword(keyword_text: str) -> str:
    k = keyword_text.strip().lower()
    if k.startswith('wing'):
        return "WING"
    if k.startswith('block'):
        return "BLOCK"
    if k.startswith('tower'):
        return "TOWER"
    return "UNKNOWN"


def _apply_relationship_rule(raw_address: str, candidates: List[Candidate]) -> None:
    """Rule S2 -- see module docstring: NO CORPUS SUPPORT, implemented
    only because the design brief's illustrative example requires it
    structurally. Kept isolated so it can be disabled/removed
    independently of Pass 1 logic."""
    for c in candidates:
        if c.resolution != "SUBUNIT":
            continue
        following = None
        for other in candidates:
            if other.span[0] > c.span[1]:
                if following is None or other.span[0] < following.span[0]:
                    following = other
        if following is None or following.resolution != "UNKNOWN" or following.is_compound:
            continue
        between = raw_address[c.span[1]:following.span[0]]
        if any(ch in _S2_BREAK_CHARS for ch in between):
            continue
        if not _S2_CONTINUATION_SEP.match(between):
            continue
        following.resolution = "UNIT"
        following.confidence = ExtractionConfidence.INFERRED
        following.type_metadata = c.type_metadata
        following.rule_fired = "S2"
        following.removal_span = following.span


def _resolve_multiple_claims(candidates: List[Candidate]) -> Tuple[List[str], List[str]]:
    unit_claims = [c for c in candidates if c.resolution == "UNIT"]
    subunit_claims = [c for c in candidates if c.resolution == "SUBUNIT"]

    unit_ambiguous_values: List[str] = []
    subunit_ambiguous_values: List[str] = []

    # Include compound-split-derived UNIT claims (W1/W2) in the
    # multi-claim check, not just keyword-anchored ones (U1/U1+C2) --
    # found via testing ("C/O MULTIPURPOSE CORPORATION 30B SHED NO A/1
    # 278/1...") that SUBUNIT already correctly went AMBIGUOUS when two
    # compound-split claims competed, but UNIT did not, because this
    # filter only ever checked keyword-anchored rule IDs. This is a
    # consistency fix to the EXISTING ambiguity-detection mechanism
    # (Rule M2), not a new heuristic restricting where compound-split
    # itself is allowed to fire.
    strong_unit_claims = [c for c in unit_claims if c.rule_fired in ("U1", "U1+C2", "W1_letter_prefix_fused", "W1_letter_prefix_split", "W2_letter_suffix")]
    if len(strong_unit_claims) > 1:
        for c in strong_unit_claims:
            unit_ambiguous_values.append(c.raw_text)
            c.resolution = "AMBIGUOUS"
            c.confidence = None
            c.removal_span = None

    if len(subunit_claims) > 1:
        for c in subunit_claims:
            subunit_ambiguous_values.append(c.raw_text)
            c.resolution = "AMBIGUOUS"
            c.confidence = None
            c.removal_span = None

    return unit_ambiguous_values, subunit_ambiguous_values


def _find_orphaned_keywords(raw_address: str, pin_span: Optional[Tuple[int, int]],
                             candidates: List[Candidate]) -> List[Tuple[int, int]]:
    """
    Fix for the 'Flat 400001' orphan-keyword defect: when a UNIT/SUBUNIT
    keyword's only immediately-following span was entirely consumed by
    PIN blocking, NO candidate is ever generated there (PIN-claimed spans
    are excluded from candidate detection in _detect_candidates), so the
    keyword is invisible to _find_preceding_keyword (which only looks
    BACKWARD from an existing candidate). The keyword then survives,
    dangling with no value, in BASE_ADDRESS.

    This function is a narrow, separate pass: it looks for a UNIT/SUBUNIT
    keyword match whose immediately-following text (allowing only
    whitespace/connector chars, same tolerance as the ordinary
    keyword-adjacency check) starts exactly where a PIN span begins. If
    found, that keyword's span alone is returned for removal -- NOT the
    PIN's span (which is already separately removed via the normal PIN
    removal path), and NO UNIT/SUBUNIT value is created. This is pure
    keyword-orphan cleanup, not a UNIT-vs-PIN precedence rule: the
    keyword is removed only because it demonstrably attempted to claim a
    value that turned out to be a PIN, not because PIN unconditionally
    outranks UNIT in general. If a future extractor version changes how
    candidates are detected near a PIN, this function's condition
    (keyword immediately adjacent to the PIN span, nothing in between)
    simply stops matching and has no effect -- it cannot silently
    misfire on ordinary text, since it requires a PIN span to exist and
    be immediately adjacent to a real keyword match.
    """
    if not pin_span:
        return []
    orphaned: List[Tuple[int, int]] = []
    prefix = raw_address[:pin_span[0]]
    tail_match = re.search(r'([A-Za-z][A-Za-z .]*?)\s*[-:#]?\s*$', prefix)
    if not tail_match:
        return []
    tail = tail_match.group(1)
    tail_start = tail_match.start(1)
    for pattern in (_UNIT_STRONG_PATTERN, _SUBUNIT_PATTERN):
        km = pattern.search(tail)
        if km and (km.end() == len(tail.rstrip()) or tail.rstrip().endswith(km.group(0).rstrip('.'))):
            keyword_span = (tail_start + km.start(), tail_start + km.end())
            orphaned.append(keyword_span)
            break
    return orphaned


def _build_base_address(raw_address: str, removal_spans: List[Tuple[int, int]]) -> str:
    if not removal_spans:
        return re.sub(r'\s+', ' ', raw_address).strip(' ,')

    spans = sorted(removal_spans)
    merged: List[List[int]] = []
    for start, end in spans:
        if merged and start <= merged[-1][1]:
            merged[-1][1] = max(merged[-1][1], end)
            continue
        if merged:
            between = raw_address[merged[-1][1]:start]
            if re.fullmatch(r'[\s,]*', between):
                merged[-1][1] = end
                continue
        merged.append([start, end])

    result = raw_address
    for start, end in sorted(merged, reverse=True):
        result = result[:start] + ' ' + result[end:]

    result = re.sub(r'\s*,\s*,\s*', ', ', result)
    result = re.sub(r'^\s*,\s*', '', result)
    result = re.sub(r'\s*,\s*$', '', result)
    result = re.sub(r'\s+', ' ', result).strip(' ,')
    return result


def extract(raw_address: str) -> ExtractionResult:
    if not raw_address or not raw_address.strip():
        return ExtractionResult(original_text=raw_address or "", base_address="")

    pin_comp, pin_span = _extract_pin(raw_address)

    candidates = _detect_candidates(raw_address, pin_span)
    for c in candidates:
        _find_preceding_keyword(raw_address, c)
        if c.preceding_keyword_tier is None:
            _find_following_subunit_keyword(raw_address, c)

    _resolve_pass_1(candidates)
    _apply_wing_compound_split(raw_address, candidates)
    _apply_leading_unit_inference(raw_address, candidates)
    _apply_relationship_rule(raw_address, candidates)
    unit_ambig, subunit_ambig = _resolve_multiple_claims(candidates)

    # Prefer an ANCHORED (keyword-matched) claim over an INFERRED
    # (leading-position-inferred) one if both exist -- a real keyword
    # match anywhere in the address is always stronger evidence than a
    # positional guess. Fixed here after discovering, during testing,
    # that plain list-order selection could silently prefer the leading
    # INFERRED candidate over a later ANCHORED one purely because it
    # appears first in the string.
    unit_claims_all = [c for c in candidates if c.resolution == "UNIT"]
    unit_candidate = next((c for c in unit_claims_all if c.confidence == ExtractionConfidence.ANCHORED), None) \
        or (unit_claims_all[0] if unit_claims_all else None)
    subunit_claims_all = [c for c in candidates if c.resolution == "SUBUNIT"]
    subunit_candidate = next((c for c in subunit_claims_all if c.confidence == ExtractionConfidence.ANCHORED), None) \
        or (subunit_claims_all[0] if subunit_claims_all else None)

    unit_comp = None
    if unit_candidate:
        # Canonicalization: canonicalize_numeric_identifier was designed
        # for simple single-part flat numbers (strip separators/leading
        # zeros). Applying it to a multi-part compound value (Rule C2's
        # whole-token claim, e.g. "703-A1-66") silently merges/loses
        # structure (confirmed during validation: "703-A1-66" ->
        # "703A166", incorrectly dropping the fact that "A1" and "66"
        # were separate parts) -- a real bug found by running the full
        # corpus, not a hypothetical. For a compound value, canonicalize
        # ONLY for leading-zero stripping, preserve the separator
        # structure otherwise, so comparison is still tolerant to
        # formatting (4B/308 vs 4B-308) without destroying multi-part
        # identity.
        if unit_candidate.is_compound and len(re.findall(r'[/\-]', unit_candidate.raw_text)) >= 2:
            # multi-separator compound (3+ parts): preserve structure,
            # only normalize separators to a single canonical form and
            # strip leading zeros PER PART.
            parts = re.split(r'[/\-]', unit_candidate.raw_text)
            canon_parts = [canonicalize_numeric_identifier(p).canonical for p in parts]
            canon = '-'.join(canon_parts)
        else:
            canon = canonicalize_numeric_identifier(unit_candidate.raw_text).canonical
        unit_comp = ParsedComponent(
            ComponentType.FLAT, canon, unit_candidate.raw_text, unit_candidate.confidence,
            source=f"unit_extractor:{unit_candidate.rule_fired}", span=unit_candidate.span,
            type_metadata=unit_candidate.type_metadata,
        )
    elif unit_ambig:
        unit_comp = ParsedComponent(
            ComponentType.FLAT, None, None, ExtractionConfidence.AMBIGUOUS,
            source="unit_extractor:M2_multiple_claims", candidates=unit_ambig,
        )

    subunit_comp = None
    if subunit_candidate:
        canon = canonicalize_numeric_identifier(subunit_candidate.raw_text).canonical
        subunit_comp = ParsedComponent(
            ComponentType.WING, canon, subunit_candidate.raw_text, subunit_candidate.confidence,
            source=f"unit_extractor:{subunit_candidate.rule_fired}", span=subunit_candidate.span,
            type_metadata=subunit_candidate.type_metadata,
        )
    elif subunit_ambig:
        subunit_comp = ParsedComponent(
            ComponentType.WING, None, None, ExtractionConfidence.AMBIGUOUS,
            source="unit_extractor:M3_multiple_claims", candidates=subunit_ambig,
        )

    removal_spans = []
    if pin_span:
        removal_spans.append(pin_span)
    for c in candidates:
        if c.removal_span:
            removal_spans.append(c.removal_span)
    removal_spans.extend(_find_orphaned_keywords(raw_address, pin_span, candidates))

    base_address = _build_base_address(raw_address, removal_spans)

    return ExtractionResult(
        original_text=raw_address,
        unit=unit_comp, subunit=subunit_comp, pin=pin_comp,
        base_address=base_address,
        candidates=candidates,
        unit_ambiguous_candidates=unit_ambig,
        subunit_ambiguous_candidates=subunit_ambig,
    )