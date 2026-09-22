"""
Deterministic scoring: dynamic weighting + evidence sufficiency + gating.

Implements Phase 1 section 11 and constraints #9 and #11.

Three-stage design (Phase 1 section 11's "net recommendation"):
  1. GATING       -- hard contradictions (from contradiction.py) impose a
                     score/label CEILING before any aggregation happens.
                     A pile of agreeing components can never mathematically
                     outweigh a HARD_GATE contradiction -- this is the
                     direct fix for the old system's core failure mode
                     (one strong fuzzy score compensating for a real
                     contradiction).
  2. AGGREGATION  -- dynamic-weighted scoring across only the components
                     that have real (non-missing, non-ambiguous) evidence
                     on at least one side. Weights are relative importance
                     ranks, renormalized over whatever's actually present
                     -- so a pair where only PIN+city+street are
                     extractable doesn't have "wing" implicitly counted
                     as zero out of a budget that assumed its presence.
  3. SUFFICIENCY  -- constraint #9: a pair should NOT receive HIGH merely
                     because the few available fields happen to be very
                     similar, if those fields aren't individually
                     identifying enough. This is evaluated SEPARATELY from
                     the numeric score and can independently cap the label.

ALL weights and thresholds in this module are explicitly marked PROVISIONAL
(constraint #11) -- they encode a defensible relative ordering, not a
calibrated-against-real-data set of numbers. They are grouped in one place
(ScoringConfig) specifically so they're easy to find and override, rather
than scattered as magic numbers through the scoring logic.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, List, Optional

from .datamodel import (
    ComponentEvidence, ComponentType, Contradiction, ContradictionSeverity,
    EvidenceState, ExtractionConfidence, MatchResult, ParsedAddress,
)
from .contradiction import detect_contradictions
from .comparator import compare_component, compare_residual


# ---------------------------------------------------------------------------
# PROVISIONAL configuration -- see module docstring. Every number here is a
# starting point pending calibration against real labeled address pairs
# (Phase 1 section 19 / this file's TODO for Phase 3).
# ---------------------------------------------------------------------------

@dataclass
class ScoringConfig:
    # Relative importance ranks (Phase 1 section 11's ordering:
    # PIN ~= Flat/Unit > City ~= Street > Building > Locality > Wing/Floor
    # > Landmark). These are RELATIVE weights, renormalized at scoring
    # time over whichever components actually have evidence -- see
    # _dynamic_weights() below. Landmark is intentionally absent: it is
    # supporting-only and never contributes to the numeric score directly
    # (Phase 1 section 6), though it still appears in positive_evidence /
    # the explanation output.
    component_weights: Dict[ComponentType, float] = field(default_factory=lambda: {
        ComponentType.PIN: 10.0,
        ComponentType.FLAT: 10.0,
        ComponentType.CITY: 7.0,
        ComponentType.STREET: 7.0,
        ComponentType.BUILDING: 5.0,
        ComponentType.LOCALITY: 4.0,
        ComponentType.WING: 3.0,
        ComponentType.FLOOR: 3.0,
        ComponentType.STATE: 3.0,
    })

    # Point deduction applied per contradiction severity, subtracted from
    # the raw aggregated score (0-100 scale) AFTER aggregation, before the
    # gate ceiling is applied. HARD_GATE contradictions also deduct here
    # (not just gate) so the numeric score itself reflects the
    # contradiction even when inspected outside the label.
    contradiction_penalty: Dict[ContradictionSeverity, float] = field(default_factory=lambda: {
        ContradictionSeverity.HARD_GATE: 40.0,
        ContradictionSeverity.STRONG_PENALTY: 20.0,
        ContradictionSeverity.MODERATE_PENALTY: 10.0,
        ContradictionSeverity.SUPPORTING_NEGATIVE: 4.0,
    })

    # Label ceiling imposed when a contradiction of this severity is
    # present, regardless of numeric score (Phase 1 section 11: "cannot be
    # HIGH" as a gate, not a point deduction that can be outweighed).
    severity_ceiling: Dict[ContradictionSeverity, str] = field(default_factory=lambda: {
        ContradictionSeverity.HARD_GATE: "LOW",
        ContradictionSeverity.STRONG_PENALTY: "MEDIUM",
        ContradictionSeverity.MODERATE_PENALTY: "MEDIUM",
        # SUPPORTING_NEGATIVE imposes no ceiling -- folds into score only,
        # per Phase 1 section 9 ("never gates alone").
    })

    # Score -> label thresholds. Broadly similar to the old system's
    # 75/60 cutoffs (Phase 1 section 11), kept as a starting point since
    # they at least reflect real prior production experience, but this
    # number distribution is genuinely different (component-weighted, not
    # a single fuzzy ratio) so these WILL need recalibration once real
    # test data is available -- explicitly not claimed as final.
    high_threshold: float = 75.0
    medium_threshold: float = 55.0

    # Evidence sufficiency (constraint #9): minimum total weight of
    # non-missing, non-ambiguous components required for HIGH to be
    # reachable at all, and a stricter minimum for MEDIUM. If evidence
    # falls short, the label is capped regardless of how well the
    # available fields agree -- directly implements "a pair should not
    # receive HIGH merely because the available fields happen to be
    # highly similar when there is insufficient identifying information."
    min_evidence_weight_for_high: float = 20.0   # e.g. PIN+FLAT alone (10+10) clears this; PIN alone (10) does not
    min_evidence_weight_for_medium: float = 7.0

    # Additionally (constraint #9's explicit example: "high similarity
    # based only on generic geographic information should not
    # automatically produce HIGH"): HIGH additionally requires that at
    # least one STRONG-identifying component (PIN or FLAT) individually
    # contributed positive (MATCH) evidence -- city/street/locality
    # agreement alone, however strong, cannot reach HIGH on its own.
    strong_identifiers: tuple = (ComponentType.PIN, ComponentType.FLAT)

    residual_weight: float = 3.0   # residual bag-of-words gets a small, fixed weight -- supporting evidence only


DEFAULT_CONFIG = ScoringConfig()


# ---------------------------------------------------------------------------
# Evidence-state -> numeric score contribution (0.0-1.0) for aggregation
# ---------------------------------------------------------------------------

def _evidence_score(ev: ComponentEvidence) -> Optional[float]:
    """
    Returns this component's contribution to the score in [0.0, 1.0], or
    None if it shouldn't contribute at all (missing / ambiguous / uncertain
    -- these are excluded from aggregation entirely per the dynamic-weighting
    design, not scored as zero, since scoring them as zero would wrongly
    penalize a pair for simply not having that field extractable, exactly
    the mistake constraint #6 tells us not to make).
    """
    if ev.state == EvidenceState.MATCH:
        # graded fields (text/area comparators) carry their raw_similarity;
        # exact-match fields (numeric/exact comparators) don't set
        # raw_similarity, so a clean MATCH there is worth full credit (1.0)
        return ev.raw_similarity if ev.raw_similarity is not None else 1.0
    if ev.state == EvidenceState.MISMATCH:
        return 0.0
    if ev.state == EvidenceState.UNCERTAIN:
        # Explicit, not incidental: a disagreement gated by low extraction
        # confidence must contribute neither positive nor negative
        # evidence to the score -- it is withheld evidence, not a weak
        # match or a weak mismatch. Stated as its own branch (rather than
        # left to the catch-all below) precisely so this guarantee is
        # visible and independently testable, not an accident of
        # fallthrough behavior.
        return None
    # MISSING_ON_A / MISSING_ON_B / MISSING_ON_BOTH / AMBIGUOUS all excluded
    return None


def _dynamic_weights(
    evidence: Dict[ComponentType, ComponentEvidence], config: ScoringConfig
) -> Dict[ComponentType, float]:
    """
    Renormalize the configured relative weights over only the components
    that actually contributed a score (see _evidence_score) -- Phase 1
    section 11's "dynamic weighting over a fixed base-importance ranking".
    """
    contributing = {
        ct: config.component_weights[ct]
        for ct, ev in evidence.items()
        if ct in config.component_weights and _evidence_score(ev) is not None
    }
    total = sum(contributing.values())
    if total == 0:
        return {}
    return {ct: w / total for ct, w in contributing.items()}


def _aggregate_score(
    evidence: Dict[ComponentType, ComponentEvidence],
    residual_similarity: Optional[float],
    config: ScoringConfig,
) -> float:
    weights = _dynamic_weights(evidence, config)
    score = 0.0
    for ct, weight in weights.items():
        contribution = _evidence_score(evidence[ct])
        score += weight * contribution

    # residual text folds in with a small fixed weight, separate from the
    # dynamic component redistribution (it's always a fallback signal, not
    # a structural one) -- only if there was something to compare
    if residual_similarity is not None and weights:
        residual_share = config.residual_weight / (config.residual_weight + sum(config.component_weights[ct] for ct in weights))
        score = score * (1 - residual_share) + residual_similarity * residual_share
    elif residual_similarity is not None and not weights:
        # nothing else contributed at all -- residual is literally the
        # only evidence we have
        score = residual_similarity

    return round(score * 100, 2)


def _evidence_weight_present(
    evidence: Dict[ComponentType, ComponentEvidence], config: ScoringConfig
) -> float:
    """Total configured weight of components that had real (non-missing,
    non-ambiguous) evidence on at least one side -- used for the
    sufficiency check, independent of how well they matched."""
    return sum(
        config.component_weights[ct]
        for ct, ev in evidence.items()
        if ct in config.component_weights and _evidence_score(ev) is not None
    )


def _strong_identifier_confirmed(
    evidence: Dict[ComponentType, ComponentEvidence], config: ScoringConfig
) -> bool:
    """Constraint #9's explicit example, guarded against: at least one of
    the strong identifiers (PIN, FLAT) must have produced an actual MATCH
    -- not just be present, not just be absent-and-therefore-not-counted --
    for a HIGH label to be reachable. Prevents 'everything generic agrees'
    from masquerading as a confident identity match."""
    for ct in config.strong_identifiers:
        ev = evidence.get(ct)
        if ev and ev.state == EvidenceState.MATCH:
            return True
    return False


def _apply_gates(
    contradictions: List[Contradiction], config: ScoringConfig
) -> Optional[str]:
    """Returns the most restrictive label ceiling implied by the present
    contradictions, or None if nothing gates. HARD_GATE > STRONG/MODERATE
    in restrictiveness (LOW is more restrictive than MEDIUM)."""
    ceilings = [config.severity_ceiling[c.severity] for c in contradictions if c.severity in config.severity_ceiling]
    if "LOW" in ceilings:
        return "LOW"
    if "MEDIUM" in ceilings:
        return "MEDIUM"
    return None


_LABEL_RANK = {"NO_MATCH": -1, "LOW": 0, "MEDIUM": 1, "HIGH": 2}


def _min_label(a: str, b: str) -> str:
    return a if _LABEL_RANK[a] <= _LABEL_RANK[b] else b


def score_match(parsed_a: ParsedAddress, parsed_b: ParsedAddress, config: ScoringConfig = DEFAULT_CONFIG) -> MatchResult:
    """
    Full scoring pipeline entry point: compare every component, detect
    contradictions, aggregate a score, apply gates and evidence
    sufficiency, and produce a complete MatchResult.

    NOTE: this function does NOT apply business policy (FlatOnlyPolicy
    etc.) -- it produces the raw engine-level MatchResult that
    policies.py / matcher.py then interpret. This separation is
    deliberate and preserves constraint #12 (business policy stays
    separate from core matching logic).
    """
    evidence: Dict[ComponentType, ComponentEvidence] = {}
    for ct in ComponentType:
        if ct in (ComponentType.RESIDUAL, ComponentType.DISTRICT):
            continue  # DISTRICT has no dedicated extractor yet -- see matcher.py note
        ev = compare_component(ct, parsed_a.get(ct), parsed_b.get(ct))
        evidence[ct] = ev

    residual_similarity = compare_residual(parsed_a.residual_text, parsed_b.residual_text)
    contradictions = detect_contradictions(evidence)

    positive_evidence = [ct for ct, ev in evidence.items() if ev.state == EvidenceState.MATCH]
    missing_components = [
        ct for ct, ev in evidence.items()
        if ev.state in (EvidenceState.MISSING_ON_A, EvidenceState.MISSING_ON_B, EvidenceState.MISSING_ON_BOTH)
    ]
    ambiguous_components = [ct for ct, ev in evidence.items() if ev.state == EvidenceState.AMBIGUOUS]

    raw_score = _aggregate_score(evidence, residual_similarity, config)

    evidence_weight = _evidence_weight_present(evidence, config)
    strong_confirmed = _strong_identifier_confirmed(evidence, config)

    evidence_sufficient_for_high = (
        evidence_weight >= config.min_evidence_weight_for_high and strong_confirmed
    )
    evidence_sufficient_for_medium = evidence_weight >= config.min_evidence_weight_for_medium
    evidence_sufficient = evidence_sufficient_for_medium  # overall flag reflects the lower bar; HIGH-specific check applied in label logic below

    # base label purely from score thresholds
    if raw_score >= config.high_threshold:
        label = "HIGH"
    elif raw_score >= config.medium_threshold:
        label = "MEDIUM"
    else:
        label = "LOW"

    # evidence sufficiency can only pull the label DOWN, never up
    if label == "HIGH" and not evidence_sufficient_for_high:
        label = "MEDIUM"
    if label in ("HIGH", "MEDIUM") and not evidence_sufficient_for_medium:
        label = "LOW"

    # gates can only pull the label DOWN, never up
    gate_ceiling = _apply_gates(contradictions, config)
    if gate_ceiling is not None:
        label = _min_label(label, gate_ceiling)

    # numeric score itself also reflects contradiction penalties, kept
    # separate from the label logic above so raw_score stays an honest
    # "how much agreed" number even when the label was capped by a gate
    for c in contradictions:
        raw_score -= config.contradiction_penalty.get(c.severity, 0.0)
    raw_score = max(0.0, round(raw_score, 2))

    reason = _build_reason(label, contradictions, positive_evidence, missing_components, ambiguous_components, evidence_sufficient_for_high, evidence_sufficient_for_medium)

    return MatchResult(
        parsed_a=parsed_a, parsed_b=parsed_b, evidence=evidence,
        contradictions=contradictions, positive_evidence=positive_evidence,
        missing_components=missing_components, ambiguous_components=ambiguous_components,
        residual_similarity=residual_similarity, raw_score=raw_score, label=label,
        evidence_sufficient=evidence_sufficient, score_ceiling=gate_ceiling,
        reason=reason,
    )


def _build_reason(
    label: str, contradictions: List[Contradiction], positive: List[ComponentType],
    missing: List[ComponentType], ambiguous: List[ComponentType],
    sufficient_for_high: bool, sufficient_for_medium: bool,
) -> str:
    parts = [f"Confidence={label}."]
    if contradictions:
        parts.append("Contradictions found: " + ", ".join(f"{c.component_type.value}({c.severity.value})" for c in contradictions) + ".")
    if positive:
        parts.append(f"Matching: {', '.join(ct.value for ct in positive)}.")
    if not sufficient_for_medium:
        parts.append("Insufficient identifying evidence was available for any confident decision.")
    elif not sufficient_for_high:
        parts.append("Insufficient strong identifying evidence (PIN/flat) to support HIGH even though available fields agree.")
    if missing:
        parts.append(f"Missing on one/both sides: {', '.join(ct.value for ct in missing)}.")
    if ambiguous:
        parts.append(f"Ambiguous/unresolved: {', '.join(ct.value for ct in ambiguous)}.")
    return " ".join(parts)
