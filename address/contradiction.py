"""
Contradiction detection and severity classification.

Implements Phase 1 section 9's severity tiers, with the confidence-aware
refinement required by constraint #8: a state/PIN mismatch is only treated
as a hard gate when BOTH sides were extracted with ANCHORED confidence.
A mismatch involving an INFERRED extraction is downgraded a tier, because
the contradiction might be an artifact of a shaky extraction rather than a
genuine fact about the two addresses -- gating a whole match decision on a
guess would be exactly the kind of "overly aggressive rule that rejects
valid matches" the brief explicitly warned against.

Severity tiers (from Phase 1 section 9):
  HARD_GATE            -- PIN mismatch (anchored/anchored), State mismatch
                           (anchored/anchored)
  STRONG_PENALTY        -- Flat/unit mismatch, City mismatch (anchored/anchored)
  MODERATE_PENALTY       -- Wing mismatch, Street mismatch (when locality/
                           city/PIN otherwise agree), State or PIN mismatch
                           where at least one side was only INFERRED
  SUPPORTING_NEGATIVE     -- Building mismatch, Locality mismatch (when
                           city/PIN agree), Landmark (never actually
                           reaches here -- comparator.py never emits
                           MISMATCH for landmark at all)

This module does NOT decide the final score or label -- it only produces
a list of Contradiction objects with severities attached. scorer.py
consumes that list.
"""

from __future__ import annotations

from typing import Dict, List

from .datamodel import ComponentEvidence, ComponentType, Contradiction, ContradictionSeverity, EvidenceState, ExtractionConfidence


def _both_anchored(ev: ComponentEvidence) -> bool:
    return (
        ev.confidence_a == ExtractionConfidence.ANCHORED
        and ev.confidence_b == ExtractionConfidence.ANCHORED
    )


def detect_contradictions(evidence: Dict[ComponentType, ComponentEvidence]) -> List[Contradiction]:
    """
    Walk the per-component evidence and produce the list of contradictions
    with severities assigned. Only MISMATCH states can produce a
    Contradiction -- MISSING_*, AMBIGUOUS, and UNCERTAIN are handled
    entirely by the scorer's evidence-sufficiency logic (scorer.py), not
    here, because "we don't know" (MISSING/AMBIGUOUS) and "we have a
    disagreement we don't trust" (UNCERTAIN) are both categorically
    different from "we found conflicting evidence we DO trust" (Phase 1
    section 9's core distinction, extended to cover UNCERTAIN). Every
    branch below checks `.state == EvidenceState.MISMATCH` specifically
    -- UNCERTAIN never satisfies that check, so it is excluded from
    contradiction generation by construction, not by omission.
    """
    contradictions: List[Contradiction] = []

    pin_ev = evidence.get(ComponentType.PIN)
    if pin_ev and pin_ev.state == EvidenceState.MISMATCH:
        if _both_anchored(pin_ev):
            severity = ContradictionSeverity.HARD_GATE
            detail = "PIN codes differ and both were confidently extracted -- strongest available contradiction"
        else:
            severity = ContradictionSeverity.STRONG_PENALTY
            detail = "PIN codes differ but extraction confidence was not fully anchored on both sides"
        contradictions.append(Contradiction(
            component_type=ComponentType.PIN, severity=severity, detail=detail,
            confidence_a=pin_ev.confidence_a, confidence_b=pin_ev.confidence_b,
        ))

    state_ev = evidence.get(ComponentType.STATE)
    if state_ev and state_ev.state == EvidenceState.MISMATCH:
        if _both_anchored(state_ev):
            severity = ContradictionSeverity.HARD_GATE
            detail = "States differ and both were confidently extracted (full state name matched)"
        else:
            severity = ContradictionSeverity.MODERATE_PENALTY
            detail = "States differ but at least one side's extraction was only inferred (not an anchored full name match)"
        contradictions.append(Contradiction(
            component_type=ComponentType.STATE, severity=severity, detail=detail,
            confidence_a=state_ev.confidence_a, confidence_b=state_ev.confidence_b,
        ))

    flat_ev = evidence.get(ComponentType.FLAT)
    if flat_ev and flat_ev.state == EvidenceState.MISMATCH:
        contradictions.append(Contradiction(
            component_type=ComponentType.FLAT, severity=ContradictionSeverity.STRONG_PENALTY,
            detail="Flat/unit/house/plot numbers differ after canonicalization (and OCR-variant check)",
            confidence_a=flat_ev.confidence_a, confidence_b=flat_ev.confidence_b,
        ))

    city_ev = evidence.get(ComponentType.CITY)
    if city_ev and city_ev.state == EvidenceState.MISMATCH:
        if _both_anchored(city_ev):
            severity = ContradictionSeverity.STRONG_PENALTY
        else:
            severity = ContradictionSeverity.MODERATE_PENALTY
        contradictions.append(Contradiction(
            component_type=ComponentType.CITY, severity=severity,
            detail="City values differ" + ("" if severity == ContradictionSeverity.STRONG_PENALTY
                                            else " (inferred extraction on at least one side)"),
            confidence_a=city_ev.confidence_a, confidence_b=city_ev.confidence_b,
        ))

    wing_ev = evidence.get(ComponentType.WING)
    if wing_ev and wing_ev.state == EvidenceState.MISMATCH:
        contradictions.append(Contradiction(
            component_type=ComponentType.WING, severity=ContradictionSeverity.MODERATE_PENALTY,
            detail="Wing/block letters differ",
            confidence_a=wing_ev.confidence_a, confidence_b=wing_ev.confidence_b,
        ))

    street_ev = evidence.get(ComponentType.STREET)
    if street_ev and street_ev.state == EvidenceState.MISMATCH:
        # Phase 1 section 9: street mismatch when locality/city/PIN
        # otherwise agree could just be a different street within the same
        # known area -- worth a moderate penalty, not a strong one. This is
        # intentionally always MODERATE for now rather than encoding
        # cross-component conditioning logic in this loop. Flagged as a
        # simplification: a context-aware version is a reasonable Phase 3
        # refinement once real data shows whether it matters.
        contradictions.append(Contradiction(
            component_type=ComponentType.STREET, severity=ContradictionSeverity.MODERATE_PENALTY,
            detail="Street/road values differ",
            confidence_a=street_ev.confidence_a, confidence_b=street_ev.confidence_b,
        ))

    building_ev = evidence.get(ComponentType.BUILDING)
    if building_ev and building_ev.state == EvidenceState.MISMATCH:
        contradictions.append(Contradiction(
            component_type=ComponentType.BUILDING, severity=ContradictionSeverity.SUPPORTING_NEGATIVE,
            detail="Building/society names differ (weak evidence -- names are reused across areas and often mis-transliterated)",
            confidence_a=building_ev.confidence_a, confidence_b=building_ev.confidence_b,
        ))

    locality_ev = evidence.get(ComponentType.LOCALITY)
    if locality_ev and locality_ev.state == EvidenceState.MISMATCH:
        contradictions.append(Contradiction(
            component_type=ComponentType.LOCALITY, severity=ContradictionSeverity.SUPPORTING_NEGATIVE,
            detail="Locality values differ (likely a naming-granularity difference rather than a true contradiction)",
            confidence_a=locality_ev.confidence_a, confidence_b=locality_ev.confidence_b,
        ))

    # NOTE: landmark deliberately excluded -- comparator.py's
    # compare_landmark_component() never emits MISMATCH at all (Phase 1
    # section 6/9: landmark differences carry no reliable negative
    # evidence), so there is structurally nothing to add here for it.

    return contradictions
