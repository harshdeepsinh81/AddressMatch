"""
Component-specific comparison logic.

Implements Phase 1 section 6's field-to-algorithm mapping. Each comparator
takes two ParsedComponent (or None) and produces a ComponentEvidence with
one of the six EvidenceState values. This is where the MISSING vs MISMATCH
distinction (constraint #6) is actually enforced -- every comparator checks
presence on both sides BEFORE attempting any similarity comparison.

Numeric fields (PIN, flat, floor) go through numeric.py's canonicalize+
exact-match(+OCR) logic exclusively -- constraint #7, no graduated fuzzy
score is ever applied to them.

Text fields (building, street, locality, city) use a small combination of
the approved similarity primitives, with per-field thresholds. Wing/state
use exact match (small closed vocabularies).
"""

from __future__ import annotations

from typing import Optional

from .datamodel import ComponentEvidence, ComponentType, EvidenceState, ExtractionConfidence, ParsedComponent
from .numeric import numeric_identifiers_match
from . import similarity as sim


# ---------------------------------------------------------------------------
# Thresholds -- PROVISIONAL, explicitly marked as such (constraint #11).
# These govern when a graded text-field similarity counts as MATCH vs
# MISMATCH vs left as a raw score for the scorer to weigh. They are
# starting points based on the general behavior of each algorithm, NOT
# calibrated against real address-pair data -- that calibration is a
# Phase 3 activity once real test data is available (Phase 1, section 19).
# ---------------------------------------------------------------------------

class ComparatorThresholds:
    """Provisional thresholds, overridable at call time or via a future
    config layer -- kept as simple class attributes rather than scattered
    magic numbers so every threshold in the system lives in one inspectable
    place."""
    TEXT_FIELD_MATCH = 0.82        # combined-score >= this -> MATCH
    TEXT_FIELD_MISMATCH = 0.45      # combined-score <  this -> MISMATCH
    # between MISMATCH and MATCH thresholds: still MATCH, but caller can
    # inspect raw_similarity for how strong it was -- we do NOT introduce a
    # third textual evidence state; "weak match" is represented by a MATCH
    # state with a lower raw_similarity, and the scorer (scorer.py) is
    # free to weight it down accordingly rather than the comparator having
    # to invent a new state for it.


def _blank(text: Optional[str]) -> bool:
    return text is None or not text.strip()


def _missing_state(present_a: bool, present_b: bool) -> Optional[EvidenceState]:
    """Returns the MISSING_* state if applicable, else None (meaning both
    sides have something to compare)."""
    if not present_a and not present_b:
        return EvidenceState.MISSING_ON_BOTH
    if not present_a:
        return EvidenceState.MISSING_ON_A
    if not present_b:
        return EvidenceState.MISSING_ON_B
    return None


# ---------------------------------------------------------------------------
# Numeric-field comparator (PIN, FLAT, FLOOR) -- exact-canonical only
# ---------------------------------------------------------------------------

def compare_numeric_component(
    component_type: ComponentType,
    comp_a: Optional[ParsedComponent],
    comp_b: Optional[ParsedComponent],
) -> ComponentEvidence:
    present_a = comp_a is not None and comp_a.is_present() and comp_a.confidence != ExtractionConfidence.AMBIGUOUS
    present_b = comp_b is not None and comp_b.is_present() and comp_b.confidence != ExtractionConfidence.AMBIGUOUS

    # AMBIGUOUS extraction on either side -> AMBIGUOUS evidence, full stop.
    # We do NOT try to force a comparison against an ambiguous candidate
    # list (Phase 1 section 8/10) -- forcing arbitrary pairing is worse
    # than admitting insufficient evidence.
    if (comp_a and comp_a.confidence == ExtractionConfidence.AMBIGUOUS) or \
       (comp_b and comp_b.confidence == ExtractionConfidence.AMBIGUOUS):
        return ComponentEvidence(
            component_type=component_type, state=EvidenceState.AMBIGUOUS,
            value_a=comp_a.value if comp_a else None, value_b=comp_b.value if comp_b else None,
            confidence_a=comp_a.confidence if comp_a else None,
            confidence_b=comp_b.confidence if comp_b else None,
            comparator_used="numeric_exact_canonical",
            detail="one or both sides had multiple unresolved numeric candidates",
        )

    missing = _missing_state(present_a, present_b)
    if missing is not None:
        return ComponentEvidence(
            component_type=component_type, state=missing,
            value_a=comp_a.value if comp_a else None, value_b=comp_b.value if comp_b else None,
            confidence_a=comp_a.confidence if comp_a else None,
            confidence_b=comp_b.confidence if comp_b else None,
            comparator_used="numeric_exact_canonical",
        )

    is_match, used_ocr_here, ocr_chars_here = numeric_identifiers_match(comp_a.value, comp_b.value)

    if not is_match:
        # Confidence-gated disagreement (added after the Step 7 policy
        # review): explicitly scoped to FLAT only, not PIN or FLOOR, even
        # though this function serves all three (NUMERIC_COMPONENTS =
        # {PIN, FLAT, FLOOR}). PIN and FLOOR extraction in parser.py only
        # ever produces ANCHORED confidence today (verified), so this
        # branch is currently inert for them regardless -- but the
        # component-type check below makes that an explicit, visible
        # contract rather than an accidental consequence of today's
        # extractor behavior, which could silently change later. The
        # principle: confidence gates DISAGREEMENT, not agreement -- an
        # INFERRED value that agrees is still trusted as MATCH; only a
        # disagreement involving at least one INFERRED side is downgraded
        # to UNCERTAIN rather than trusted as MISMATCH.
        if component_type == ComponentType.FLAT and (
            comp_a.confidence == ExtractionConfidence.INFERRED
            or comp_b.confidence == ExtractionConfidence.INFERRED
        ):
            return ComponentEvidence(
                component_type=component_type, state=EvidenceState.UNCERTAIN,
                value_a=comp_a.value, value_b=comp_b.value,
                confidence_a=comp_a.confidence, confidence_b=comp_b.confidence,
                comparator_used="numeric_exact_canonical",
                detail="canonical values differ, but at least one side's extraction was only inferred "
                       "(not anchored) -- disagreement not trusted as a genuine contradiction",
            )

    state = EvidenceState.MATCH if is_match else EvidenceState.MISMATCH

    # OCR correction can have already happened UPSTREAM, during parsing
    # (e.g. PIN recovered via extract_pin_code_ocr_tolerant before the two
    # canonical values ever reach this comparator and turn out identical).
    # Surface that too -- constraint #7 requires end-to-end OCR-correction
    # visibility, not just corrections discovered at comparison time.
    upstream_ocr = comp_a.ocr_corrected or comp_b.ocr_corrected
    upstream_chars = max(comp_a.ocr_corrected_chars, comp_b.ocr_corrected_chars)

    used_ocr = used_ocr_here or upstream_ocr
    ocr_chars = max(ocr_chars_here, upstream_chars)

    return ComponentEvidence(
        component_type=component_type, state=state,
        value_a=comp_a.value, value_b=comp_b.value,
        confidence_a=comp_a.confidence, confidence_b=comp_b.confidence,
        comparator_used="numeric_exact_canonical" + ("_ocr" if used_ocr else ""),
        ocr_corrected=used_ocr, ocr_corrected_chars=ocr_chars,
        detail=f"canonical exact match{' via OCR correction' if used_ocr else ''}" if is_match
               else "canonical values differ",
    )


# ---------------------------------------------------------------------------
# Small closed-vocabulary comparator (WING, STATE) -- exact match
# ---------------------------------------------------------------------------

def compare_exact_component(
    component_type: ComponentType,
    comp_a: Optional[ParsedComponent],
    comp_b: Optional[ParsedComponent],
) -> ComponentEvidence:
    present_a = comp_a is not None and comp_a.is_present()
    present_b = comp_b is not None and comp_b.is_present()

    missing = _missing_state(present_a, present_b)
    if missing is not None:
        return ComponentEvidence(
            component_type=component_type, state=missing,
            value_a=comp_a.value if comp_a else None, value_b=comp_b.value if comp_b else None,
            confidence_a=comp_a.confidence if comp_a else None,
            confidence_b=comp_b.confidence if comp_b else None,
            comparator_used="exact_match",
        )

    is_match = sim.exact_match(comp_a.value, comp_b.value)

    if not is_match and component_type == ComponentType.WING and (
        comp_a.confidence == ExtractionConfidence.INFERRED
        or comp_b.confidence == ExtractionConfidence.INFERRED
    ):
        # Confidence-gated disagreement, scoped to WING only -- explicitly
        # NOT applied to STATE, which shares this comparator function
        # (EXACT_COMPONENTS = {WING, STATE}) but already has its own
        # working confidence-aware severity logic in contradiction.py's
        # _both_anchored() check. Changing STATE's EvidenceState output
        # here would alter what that existing, already-correct logic
        # sees, which was not requested and is out of scope for this
        # pass. See compare_numeric_component's equivalent FLAT-only
        # scoping for the same reasoning.
        return ComponentEvidence(
            component_type=component_type, state=EvidenceState.UNCERTAIN,
            value_a=comp_a.value, value_b=comp_b.value,
            confidence_a=comp_a.confidence, confidence_b=comp_b.confidence,
            comparator_used="exact_match",
            detail="values differ, but at least one side's extraction was only inferred "
                   "(not anchored) -- disagreement not trusted as a genuine contradiction",
        )

    return ComponentEvidence(
        component_type=component_type,
        state=EvidenceState.MATCH if is_match else EvidenceState.MISMATCH,
        value_a=comp_a.value, value_b=comp_b.value,
        confidence_a=comp_a.confidence, confidence_b=comp_b.confidence,
        comparator_used="exact_match",
        detail="exact match" if is_match else "values differ",
    )


# ---------------------------------------------------------------------------
# Short text-name comparator (BUILDING, STREET) -- Levenshtein + Jaro-Winkler
# ---------------------------------------------------------------------------

def compare_short_text_component(
    component_type: ComponentType,
    comp_a: Optional[ParsedComponent],
    comp_b: Optional[ParsedComponent],
    thresholds: ComparatorThresholds = ComparatorThresholds,
) -> ComponentEvidence:
    present_a = comp_a is not None and comp_a.is_present()
    present_b = comp_b is not None and comp_b.is_present()

    missing = _missing_state(present_a, present_b)
    if missing is not None:
        return ComponentEvidence(
            component_type=component_type, state=missing,
            value_a=comp_a.value if comp_a else None, value_b=comp_b.value if comp_b else None,
            confidence_a=comp_a.confidence if comp_a else None,
            confidence_b=comp_b.confidence if comp_b else None,
            comparator_used="levenshtein+jaro_winkler",
        )

    lev = sim.levenshtein_ratio(comp_a.value, comp_b.value)
    jw = sim.jaro_winkler(comp_a.value, comp_b.value)
    combined = max(lev, jw)   # best-of, since either can legitimately win depending on where the variation falls

    if combined >= thresholds.TEXT_FIELD_MATCH:
        state = EvidenceState.MATCH
    elif combined < thresholds.TEXT_FIELD_MISMATCH:
        state = EvidenceState.MISMATCH
    else:
        # Deliberately conservative middle ground: per constraint #9 (don't
        # let generic similarity manufacture false confidence), a
        # borderline building/street score is NOT auto-promoted to MATCH.
        # It is reported as AMBIGUOUS so the scorer treats it closer to
        # "insufficient evidence" than to confirmed matching evidence.
        state = EvidenceState.AMBIGUOUS

    return ComponentEvidence(
        component_type=component_type, state=state,
        value_a=comp_a.value, value_b=comp_b.value,
        confidence_a=comp_a.confidence, confidence_b=comp_b.confidence,
        comparator_used="levenshtein+jaro_winkler", raw_similarity=combined,
        detail=f"combined similarity={combined:.3f} (lev={lev:.3f}, jw={jw:.3f})",
    )


# ---------------------------------------------------------------------------
# Locality/City/District comparator -- Jaccard + containment + char n-gram
# ---------------------------------------------------------------------------

def compare_area_component(
    component_type: ComponentType,
    comp_a: Optional[ParsedComponent],
    comp_b: Optional[ParsedComponent],
    thresholds: ComparatorThresholds = ComparatorThresholds,
) -> ComponentEvidence:
    present_a = comp_a is not None and comp_a.is_present()
    present_b = comp_b is not None and comp_b.is_present()

    missing = _missing_state(present_a, present_b)
    if missing is not None:
        return ComponentEvidence(
            component_type=component_type, state=missing,
            value_a=comp_a.value if comp_a else None, value_b=comp_b.value if comp_b else None,
            confidence_a=comp_a.confidence if comp_a else None,
            confidence_b=comp_b.confidence if comp_b else None,
            comparator_used="jaccard+containment+char_ngram",
        )

    jaccard = sim.token_jaccard(comp_a.value, comp_b.value)
    containment = sim.containment_ratio(comp_a.value, comp_b.value)
    ngram = sim.char_ngram_dice(comp_a.value, comp_b.value)
    # containment given more weight than jaccard here, deliberately: this
    # field legitimately varies in granularity (Phase 1 section 7), so
    # penalizing the larger side for extra tokens is usually the wrong
    # call for locality/city specifically. char n-gram catches
    # concatenation/spacing noise that word tokenization would miss.
    combined = max(containment, jaccard, ngram)

    if combined >= thresholds.TEXT_FIELD_MATCH:
        state = EvidenceState.MATCH
    elif combined < thresholds.TEXT_FIELD_MISMATCH:
        state = EvidenceState.MISMATCH
    else:
        state = EvidenceState.AMBIGUOUS

    return ComponentEvidence(
        component_type=component_type, state=state,
        value_a=comp_a.value, value_b=comp_b.value,
        confidence_a=comp_a.confidence, confidence_b=comp_b.confidence,
        comparator_used="jaccard+containment+char_ngram", raw_similarity=combined,
        detail=f"combined={combined:.3f} (jaccard={jaccard:.3f}, containment={containment:.3f}, ngram={ngram:.3f})",
    )


# ---------------------------------------------------------------------------
# Landmark comparator -- supporting evidence only, same algorithm as area
# but NEVER produces MISMATCH (Phase 1 section 6/9: landmark differences
# are not meaningful negative evidence -- different people describe the
# same place with different landmarks)
# ---------------------------------------------------------------------------

def compare_landmark_component(
    comp_a: Optional[ParsedComponent],
    comp_b: Optional[ParsedComponent],
) -> ComponentEvidence:
    present_a = comp_a is not None and comp_a.is_present()
    present_b = comp_b is not None and comp_b.is_present()

    missing = _missing_state(present_a, present_b)
    if missing is not None:
        return ComponentEvidence(
            component_type=ComponentType.LANDMARK, state=missing,
            value_a=comp_a.value if comp_a else None, value_b=comp_b.value if comp_b else None,
            confidence_a=comp_a.confidence if comp_a else None,
            confidence_b=comp_b.confidence if comp_b else None,
            comparator_used="jaccard (landmark, non-gating)",
        )

    jaccard = sim.token_jaccard(comp_a.value, comp_b.value)
    # only MATCH or AMBIGUOUS -- never MISMATCH, by design (see docstring)
    state = EvidenceState.MATCH if jaccard >= ComparatorThresholds.TEXT_FIELD_MATCH else EvidenceState.AMBIGUOUS
    return ComponentEvidence(
        component_type=ComponentType.LANDMARK, state=state,
        value_a=comp_a.value, value_b=comp_b.value,
        confidence_a=comp_a.confidence, confidence_b=comp_b.confidence,
        comparator_used="jaccard (landmark, non-gating)", raw_similarity=jaccard,
        detail="landmark differences are never treated as contradictory evidence",
    )


# ---------------------------------------------------------------------------
# Residual/unclassified text -- token_set_ratio fallback
# ---------------------------------------------------------------------------

def compare_residual(residual_a: str, residual_b: str) -> Optional[float]:
    """
    Returns a 0.0-1.0 similarity for whatever text neither address's
    parser could confidently classify, or None if both sides are empty
    (nothing to compare -- not the same as a mismatch).
    This is the ONE place token_set_ratio-style bag-of-words comparison is
    still used, per Phase 1's conclusion that it's demoted from primary to
    fallback comparator (section 3/21).
    """
    if _blank(residual_a) and _blank(residual_b):
        return None
    return sim.token_set_ratio(residual_a or "", residual_b or "")


# ---------------------------------------------------------------------------
# Dispatch table: component type -> which comparator handles it
# ---------------------------------------------------------------------------

NUMERIC_COMPONENTS = {ComponentType.PIN, ComponentType.FLAT, ComponentType.FLOOR}
EXACT_COMPONENTS = {ComponentType.WING, ComponentType.STATE}
SHORT_TEXT_COMPONENTS = {ComponentType.BUILDING, ComponentType.STREET}
AREA_COMPONENTS = {ComponentType.LOCALITY, ComponentType.CITY, ComponentType.DISTRICT}


def compare_component(
    component_type: ComponentType,
    comp_a: Optional[ParsedComponent],
    comp_b: Optional[ParsedComponent],
) -> ComponentEvidence:
    """Single entry point matcher.py calls per component -- routes to the
    right field-specific comparator so callers don't need to know the
    field-to-algorithm mapping themselves."""
    if component_type in NUMERIC_COMPONENTS:
        return compare_numeric_component(component_type, comp_a, comp_b)
    if component_type in EXACT_COMPONENTS:
        return compare_exact_component(component_type, comp_a, comp_b)
    if component_type in SHORT_TEXT_COMPONENTS:
        return compare_short_text_component(component_type, comp_a, comp_b)
    if component_type in AREA_COMPONENTS:
        return compare_area_component(component_type, comp_a, comp_b)
    if component_type == ComponentType.LANDMARK:
        return compare_landmark_component(comp_a, comp_b)
    raise ValueError(f"No comparator registered for component type: {component_type}")
