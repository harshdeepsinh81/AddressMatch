"""
Core data structures for the deterministic address matcher.

Design intent
-------------
Every extracted piece of an address, and every comparison result, is
represented explicitly rather than as bare strings/booleans/ints, so that:

  * nothing is ever "just a string" with meaning implied by position
    (this was the central weakness of the old bag-of-tokens approach),
  * the pipeline is debuggable — you can print/log a ParsedAddress or a
    MatchResult and see exactly what was extracted and why a decision
    was made,
  * downstream stages (comparator, contradiction, scorer, policy) only
    ever talk to these structures, never to raw strings — this keeps
    each stage testable in isolation with plain constructed inputs.

These are intentionally plain dataclasses with no behaviour beyond small
convenience helpers. All actual logic (extraction, comparison, scoring)
lives in the other modules and operates ON these structures.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from typing import Dict, List, Optional


# ---------------------------------------------------------------------------
# Component identity
# ---------------------------------------------------------------------------

class ComponentType(str, Enum):
    """
    Every address component the parser knows how to extract.

    NOTE: This is a closed, explicit list rather than a free-form string
    key, specifically so that a typo in a component name becomes an
    ImportError/AttributeError at development time instead of a silent
    no-op at runtime (a real risk in the old code's dict-of-strings style
    abbreviation tables).
    """
    PIN = "pin"
    STATE = "state"
    FLOOR = "floor"
    FLAT = "flat"          # flat / unit / house / plot number (see numeric.py)
    WING = "wing"           # wing / block letter(s)
    BUILDING = "building"    # building / society / apartment name
    STREET = "street"         # road / street / lane / marg
    LOCALITY = "locality"      # locality / sub-locality / area
    CITY = "city"
    DISTRICT = "district"
    LANDMARK = "landmark"
    RESIDUAL = "residual"      # unclassified leftover text — never discarded


class ExtractionConfidence(str, Enum):
    """
    How sure the parser is that an extracted value is correctly typed
    and correctly bounded (not that the *comparison* will match — that's
    a separate question handled later, in ComponentEvidence).

    ANCHORED   — extracted because of an explicit, unambiguous keyword or
                 pattern anchor (e.g. "FLAT 401", "PIN 400069", a 6-digit
                 isolated PIN pattern). High trust.
    INFERRED   — extracted via a weaker positional/heuristic signal (e.g.
                 an unlabeled numeric token near a building-type word, or
                 a comma-segment guessed to be locality/city). Should be
                 down-weighted relative to ANCHORED evidence of the same
                 component.
    AMBIGUOUS  — the parser found more than one plausible candidate and
                 could not deterministically choose one. The value field
                 may hold a best-guess or may be empty; `candidates` will
                 list what was found. This state must NEVER be silently
                 collapsed into a confident match or mismatch downstream.
    """
    ANCHORED = "anchored"
    INFERRED = "inferred"
    AMBIGUOUS = "ambiguous"


@dataclass
class ParsedComponent:
    """
    One extracted component, with full provenance.

    `value` is the canonicalized value used for comparison (e.g. a flat
    number with separators/leading-zeros normalized). `raw_text` is what
    was actually found in the source string, preserved for debugging and
    for OCR-correction bookkeeping.
    """
    component_type: ComponentType
    value: Optional[str]
    raw_text: Optional[str]
    confidence: ExtractionConfidence
    source: str                                    # which extractor produced this, e.g. "flat_extractor:keyword_anchor"
    span: Optional[tuple] = None                    # (start, end) char offsets into the working string, if available
    candidates: List[str] = field(default_factory=list)  # populated when confidence == AMBIGUOUS
    ocr_corrected: bool = False
    ocr_corrected_chars: int = 0                     # count of characters altered by OCR-confusable correction
    type_metadata: Optional[str] = None              # e.g. "FLAT"/"SHOP"/"WING"/"BLOCK" -- the keyword TYPE that
                                                        # anchored this value, for audit only. NOT a separate
                                                        # matching dimension -- comparator/policy layers never
                                                        # read this field. Added for the new minimal UNIT/SUBUNIT
                                                        # extractor (unit_extractor.py); unset (None) for
                                                        # components produced by the original 12-role parser.py.

    def is_present(self) -> bool:
        return bool(self.value) or bool(self.candidates)

    def __repr__(self) -> str:
        bits = [f"{self.component_type.value}={self.value!r}", f"conf={self.confidence.value}"]
        if self.ocr_corrected:
            bits.append(f"ocr_fix={self.ocr_corrected_chars}")
        if self.candidates:
            bits.append(f"candidates={self.candidates}")
        return f"ParsedComponent({', '.join(bits)})"


@dataclass
class ParsedAddress:
    """
    The full structured result of parsing one address string.

    `components` maps each ComponentType to the (single) ParsedComponent
    extracted for it, if any. Components not found simply aren't present
    as keys — this is the normal case for most addresses, not an error.

    `residual_text` is whatever text remained after all extractors ran;
    it is NEVER dropped (constraint #4), and always participates in the
    fallback bag-of-words comparison (see similarity.py / comparator.py).

    `original_text` is kept for debugging/audit only.
    """
    original_text: str
    components: Dict[ComponentType, ParsedComponent] = field(default_factory=dict)
    residual_text: str = ""

    def get(self, component_type: ComponentType) -> Optional[ParsedComponent]:
        return self.components.get(component_type)

    def has(self, component_type: ComponentType) -> bool:
        c = self.components.get(component_type)
        return c is not None and c.is_present()

    def __repr__(self) -> str:
        comp_str = ", ".join(repr(c) for c in self.components.values())
        return f"ParsedAddress(components=[{comp_str}], residual={self.residual_text!r})"


# ---------------------------------------------------------------------------
# Comparison outcomes
# ---------------------------------------------------------------------------

class EvidenceState(str, Enum):
    """
    The evidence model applied uniformly to every component. MISSING and
    MISMATCH are never conflated -- the single most important modeling
    decision in the whole system (Phase 1, section 10).

    MISSING_ON_A / MISSING_ON_B are kept distinct at comparison time (for
    debugging/explanation) but are treated identically by the scorer —
    see scorer.py — matching how the old evaluate_flat_match already
    collapsed direction into a single "missing_on_one" for decision
    purposes while the raw values remained inspectable.

    UNCERTAIN (added after the Step 7 policy review): distinct from both
    MISMATCH and AMBIGUOUS. A value is present on both sides and genuinely
    disagrees after canonicalization, but at least one side's extraction
    confidence was only INFERRED (not ANCHORED) -- the disagreement might
    be a real contradiction, or might be an artifact of a shaky, guessed
    extraction (e.g. two different unanchored leading numbers). The
    principle: confidence gates disagreement, not agreement -- an
    INFERRED value that AGREES with the other side is still MATCH: it's
    only a disagreement between at-least-one-INFERRED values that is
    downgraded to UNCERTAIN rather than trusted as MISMATCH. UNCERTAIN
    must never generate a Contradiction (see contradiction.py) and must
    never contribute positive or negative evidence to scoring (see
    scorer.py) -- it is evidence explicitly withheld from the decision,
    not a third kind of match/mismatch verdict.
    """
    MATCH = "match"
    MISMATCH = "mismatch"
    MISSING_ON_A = "missing_on_a"
    MISSING_ON_B = "missing_on_b"
    MISSING_ON_BOTH = "missing_on_both"
    AMBIGUOUS = "ambiguous"
    UNCERTAIN = "uncertain"


@dataclass
class ComponentEvidence:
    """
    The result of comparing one component between two parsed addresses.
    """
    component_type: ComponentType
    state: EvidenceState
    value_a: Optional[str]
    value_b: Optional[str]
    confidence_a: Optional[ExtractionConfidence]
    confidence_b: Optional[ExtractionConfidence]
    comparator_used: str            # which comparison method produced this (e.g. "exact_canonical", "token_jaccard")
    raw_similarity: Optional[float] = None   # 0.0-1.0 where the comparator produces a graded score (not for exact-match fields)
    detail: str = ""
    ocr_corrected: bool = False
    ocr_corrected_chars: int = 0

    def __repr__(self) -> str:
        return (f"ComponentEvidence({self.component_type.value}: {self.state.value}, "
                f"a={self.value_a!r}, b={self.value_b!r}, via={self.comparator_used})")


class ContradictionSeverity(str, Enum):
    """
    Phase 1 section 9's severity tiers, as an explicit enum so the scorer
    can branch on it without magic strings.
    """
    HARD_GATE = "hard_gate"                  # cannot be HIGH (or MEDIUM, depending on configured ceiling) — see scorer.py
    STRONG_PENALTY = "strong_penalty"          # can prevent HIGH
    MODERATE_PENALTY = "moderate_penalty"       # softer deduction
    SUPPORTING_NEGATIVE = "supporting_negative"  # folds into score only, never gates alone


@dataclass
class Contradiction:
    """
    A specific, named contradiction found between the two addresses.
    Kept separate from ComponentEvidence so that "this component
    mismatched" and "this mismatch is being treated as a gating
    contradiction at severity X" are independently inspectable — the
    same MISMATCH evidence can, in principle, map to different
    severities depending on extraction confidence (constraint #8).
    """
    component_type: ComponentType
    severity: ContradictionSeverity
    detail: str
    confidence_a: Optional[ExtractionConfidence] = None
    confidence_b: Optional[ExtractionConfidence] = None


# ---------------------------------------------------------------------------
# Final result
# ---------------------------------------------------------------------------

@dataclass
class MatchResult:
    """
    The full output of the deterministic matching engine, before any
    business policy is applied.

    This is intentionally richer than the legacy dict returned by
    Address.get_confidence_level(). The backward-compatible wrapper in
    matcher.py maps this down to the legacy shape; the legacy shape is a
    strict subset of what's available here.
    """
    parsed_a: ParsedAddress
    parsed_b: ParsedAddress
    evidence: Dict[ComponentType, ComponentEvidence] = field(default_factory=dict)
    contradictions: List[Contradiction] = field(default_factory=list)
    positive_evidence: List[ComponentType] = field(default_factory=list)   # components that MATCHed
    missing_components: List[ComponentType] = field(default_factory=list)  # MISSING_ON_* or MISSING_ON_BOTH
    ambiguous_components: List[ComponentType] = field(default_factory=list)
    residual_similarity: Optional[float] = None    # 0.0-1.0 fallback bag-of-words score on unclassified text
    raw_score: float = 0.0                          # 0-100 deterministic aggregate score
    label: str = "LOW"                                # HIGH / MEDIUM / LOW / NO_MATCH — engine-level label, pre-policy
    evidence_sufficient: bool = True                 # see scorer.py — distinguishes "few components matched but
                                                       # they were highly identifying" from "not enough was compared"
    score_ceiling: Optional[str] = None              # label ceiling imposed by a hard gate, if any
    reason: str = ""

    def to_legacy_dict(self) -> dict:
        """Convenience — most of the actual mapping logic lives in matcher.py,
        this is kept here only as a placeholder hook for symmetry/debugging."""
        raise NotImplementedError("Use matcher.py's compatibility wrapper for legacy dict conversion.")


# ---------------------------------------------------------------------------
# Policy layer structures
# ---------------------------------------------------------------------------
# See the Step 7 policy specification: MatchResult (above) represents the
# engine's own generic, client-agnostic assessment. PolicyDecision (below)
# represents a SPECIFIC client's business decision given that evidence.
# These are deliberately NOT the same object and a PolicyDecision.tier is
# NOT required to agree with MatchResult.label in either direction -- see
# policies.py module docstring for the full contract.

class ConfidenceTier(str, Enum):
    HIGH = "HIGH"
    MEDIUM = "MEDIUM"
    LOW = "LOW"


@dataclass
class PolicyDecision:
    """
    A specific policy's business decision, given a MatchResult.

    Qualifiers are deliberately split into four SEPARATE fields rather than
    one generic list (per the approved qualifier vocabulary):

      missing_evidence   -- evidence absent on >=1 side (e.g. PIN wasn't
                             extractable on one/both addresses). This is
                             NOT the same kind of fact as a contradiction.
      contradictions      -- evidence present on both sides that genuinely
                             disagrees (e.g. PIN was extracted on both
                             sides and the values differ).
      policy_concessions   -- this policy chose to tolerate/degrade/ignore
                             something rather than reject on it (e.g.
                             WingDegradationPolicy choosing to degrade one
                             tier instead of rejecting on wing mismatch).
                             This field is what makes a policy's judgment
                             calls inspectable rather than only visible by
                             re-deriving them from the reason string.
      diagnostic_notes     -- anything else worth surfacing that isn't one
                             of the above three.

    `legacy_label` / `legacy_reason` hold the exact string forms the old
    system would have returned, populated by the compatibility mapping in
    matcher.py -- kept on this object (rather than computed ad hoc at the
    API boundary) so the mapping itself is a single, testable function.
    """
    policy_name: str
    tier: ConfidenceTier
    missing_evidence: List[ComponentType] = field(default_factory=list)
    contradictions: List[ComponentType] = field(default_factory=list)
    policy_concessions: List[str] = field(default_factory=list)
    diagnostic_notes: List[str] = field(default_factory=list)
    legacy_label: str = ""
    legacy_reason: str = ""

    def __repr__(self) -> str:
        return (f"PolicyDecision({self.policy_name}: tier={self.tier.value}, "
                f"legacy={self.legacy_label!r}, contradictions={[c.value for c in self.contradictions]}, "
                f"concessions={self.policy_concessions})")
