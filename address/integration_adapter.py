"""
Phase 9 integration adapter: ExtractionResult (new unit_extractor.py)
-> MatchResult (existing comparator/contradiction/scorer/policy pipeline).

Deliberately narrow, per the approved Phase 9 scope:
  - Only UNIT->FLAT, SUBUNIT->WING, PIN->PIN enter MatchResult.evidence.
  - BASE_ADDRESS is explicitly NOT wired into scoring at this stage --
    it is preserved on the ExtractionResult objects returned alongside
    MatchResult, available for a future phase, but score_match() never
    sees it, since ParsedAddress has no field for it and this adapter
    does not add one.
  - No changes to comparator.py, contradiction.py, scorer.py,
    policy_engine.py, or policies.py -- this file only constructs the
    ParsedAddress inputs those existing functions already expect, then
    calls them unmodified.

REPORTED, NOT SILENTLY RESOLVED MISMATCH: ComponentEvidence (the
comparison output) has no field for type_metadata, rule/source ID, or
candidate lists -- these live on ParsedComponent (the extraction output)
and are never read by compare_component()/compare_numeric_component()/
compare_exact_component(). Calling those functions unmodified means the
resulting ComponentEvidence objects inside MatchResult.evidence do NOT
carry UNIT_TYPE/SUBUNIT_TYPE or rule-source information -- a real,
structural gap between what the new extractor can express and what
ComponentEvidence can carry. Resolution used here, staying inside the
stated constraints: the ORIGINAL ParsedComponent objects (with
type_metadata intact) are returned alongside MatchResult by match_pair(),
so nothing is discarded -- it is simply not inside ComponentEvidence
itself. A caller needing UNIT_TYPE/SUBUNIT_TYPE reads it from the
returned ParsedComponent pair, not from MatchResult.evidence.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

from .datamodel import ComponentType, ParsedAddress, ParsedComponent, MatchResult
from .scorer import score_match
from .unit_extractor import extract, ExtractionResult


@dataclass
class IntegratedMatchResult:
    match_result: MatchResult
    extraction_a: ExtractionResult
    extraction_b: ExtractionResult
    unit_component_a: Optional[ParsedComponent]
    unit_component_b: Optional[ParsedComponent]
    subunit_component_a: Optional[ParsedComponent]
    subunit_component_b: Optional[ParsedComponent]


def _to_parsed_address(result: ExtractionResult) -> ParsedAddress:
    """
    Minimal ParsedAddress wrapper score_match() requires. Only FLAT/WING/
    PIN are populated -- BASE_ADDRESS is deliberately NOT placed into
    residual_text or any other field, so score_match() (which iterates
    every ComponentType) treats every other component as genuinely
    absent via the existing MISSING_ON_BOTH path, with no special-casing.
    """
    components = {}
    if result.unit is not None:
        components[ComponentType.FLAT] = result.unit
    if result.subunit is not None:
        components[ComponentType.WING] = result.subunit
    if result.pin is not None:
        components[ComponentType.PIN] = result.pin
    return ParsedAddress(original_text=result.original_text, components=components, residual_text="")


def match_pair(address_a: str, address_b: str) -> IntegratedMatchResult:
    """Runs the new extractor on both addresses, adapts into the existing
    ParsedAddress shape, calls the existing, unmodified score_match()."""
    extraction_a = extract(address_a)
    extraction_b = extract(address_b)

    parsed_a = _to_parsed_address(extraction_a)
    parsed_b = _to_parsed_address(extraction_b)

    match_result = score_match(parsed_a, parsed_b)

    return IntegratedMatchResult(
        match_result=match_result,
        extraction_a=extraction_a, extraction_b=extraction_b,
        unit_component_a=extraction_a.unit, unit_component_b=extraction_b.unit,
        subunit_component_a=extraction_a.subunit, subunit_component_b=extraction_b.subunit,
    )
