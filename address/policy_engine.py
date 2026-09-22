"""
Declarative policy requirements schema + shared evaluation engine.

SCOPE (per explicit direction): this module supports the two policies of
the current four that ARE genuinely declarative -- FlatOnlyPolicy and
WingOnlyPolicy. DefaultMatchingPolicy and StrictFlatAndWingPolicy use the
approved escape hatch (direct Python functions in policies.py) instead --
see the architecture-decision note below. WingDegradationPolicy is
DEFERRED entirely -- see policies.py's module docstring. No
degradation-specific machinery is built here, since designing around a
deferred policy's requirements was explicitly ruled out. Re-introducing
it later is additive (a new RequirementAction value + wiring in
policies.py), not a redesign.

ARCHITECTURE DECISION (made after tracing all four policies' exact legacy
branches in matching_policies_old.py): only FlatOnlyPolicy and
WingOnlyPolicy are genuinely decomposable into independent per-component
rules. Both have a SINGLE gate component whose own state (matched /
mismatched / missing-one / missing-both) alone determines the entire
outcome, including the "gate satisfied but similarity not high -> Medium"
floor.

DefaultMatchingPolicy and StrictFlatAndWingPolicy do NOT decompose this
way:
  - DefaultMatchingPolicy's floor is JOINT (flat=match+wing=match+low-sim
    -> Medium, but flat=match+wing=mismatch+low-sim -> Low -- the SAME
    flat outcome produces two different floors depending on wing).
  - StrictFlatAndWingPolicy's floor is also joint at one specific point
    (flat=missing_on_both+wing=exact_match+high-sim -> Medium via a
    dedicated branch, distinct from its otherwise sequential hard-gate
    structure).
Both are implemented via direct Python functions in policies.py mirroring
their legacy trees exactly, rather than growing this schema to express
joint/branch-specific floors declaratively for a pattern only two (soon
possibly three, with WingDegradation) of five total policies need.

CRITICAL EVIDENCE-PRESERVATION RULE (unchanged from the original Step 7
specification, item 6): this module does not collapse missing_on_one vs
missing_on_both, or exact_match vs shared_wing, before a policy sees them.
Both the declarative engine here AND the two escape-hatch functions in
policies.py read the SAME MatchResult.evidence structure -- no
information is lost or collapsed upstream of either evaluation path.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from typing import Dict, List, Optional

from .datamodel import (
    ComponentType, ComponentEvidence, ConfidenceTier, EvidenceState,
    MatchResult, PolicyDecision,
)


class RequirementAction(str, Enum):
    REJECT = "reject"                        # hard-fails the whole decision -> LOW
    CONTRADICTION_CAP = "contradiction_cap"    # tier bounded by a specific ceiling (see contradiction_ceiling)
    TOLERATE = "tolerate"                        # no penalty; may still establish a floor (see satisfied_floor)
    IGNORE = "ignore"                             # this component's evidence isn't consulted by this policy
    # NOTE: DEGRADE_ONE_TIER intentionally omitted -- WingDegradationPolicy
    # is deferred and this action has no current user. See module docstring.


class ComponentRole(str, Enum):
    REQUIRED_GATE = "required_gate"
    IGNORED = "ignored"


@dataclass
class ComponentRequirement:
    """
    A single component's role within ONE genuinely-declarative policy
    (currently: FlatOnlyPolicy, WingOnlyPolicy only).

    satisfied_floor is the tier this component alone GUARANTEES when its
    evidence state is "satisfied" (exact_match/shared_wing) but overall
    similarity does NOT clear the high bar -- traced directly from old
    code's repeated pattern `if satisfied and high_sim: <top> else: <floor>`.

    missing_both_floor is a SEPARATE field for the missing_on_both+high_sim
    case, kept independent of satisfied_floor since old code gives this
    its own branch even where the two currently happen to agree
    numerically -- there is no evidence they must always be equal.
    """
    component: ComponentType
    role: ComponentRole

    on_exact_match: RequirementAction = RequirementAction.TOLERATE
    on_shared_wing: Optional[RequirementAction] = None   # None => same action as on_exact_match
    on_mismatch: RequirementAction = RequirementAction.REJECT
    on_missing_one_side: RequirementAction = RequirementAction.REJECT
    on_missing_both_sides: RequirementAction = RequirementAction.TOLERATE

    contradiction_ceiling: ConfidenceTier = ConfidenceTier.LOW
    satisfied_floor: ConfidenceTier = ConfidenceTier.MEDIUM           # exact_match + !high_sim floor
    missing_both_high_sim_floor: ConfidenceTier = ConfidenceTier.MEDIUM  # missing_on_both + high_sim floor
    missing_both_low_sim_floor: ConfidenceTier = ConfidenceTier.LOW      # missing_on_both + !high_sim floor
                                                                            # (traced separately from satisfied_floor:
                                                                            # old code's missing_on_both branch has its
                                                                            # OWN !high_sim->Low result, distinct from
                                                                            # exact_match's !high_sim->Medium -- these
                                                                            # are NOT the same floor and must not share
                                                                            # one field)


@dataclass
class PolicyRequirements:
    name: str
    components: List[ComponentRequirement] = field(default_factory=list)
    pin_contributes_to_top_tier: bool = True  # old: pins_match required (alongside high_sim) to reach
                                                 # "High" rather than "HighWithoutPin" -- traced, not assumed;
                                                 # both active declarative policies set this True.

    def get(self, component: ComponentType) -> Optional[ComponentRequirement]:
        for c in self.components:
            if c.component == component:
                return c
        return None


def _action_for(req: ComponentRequirement, state: EvidenceState, is_shared_wing: bool = False) -> RequirementAction:
    if state == EvidenceState.MATCH:
        if is_shared_wing and req.on_shared_wing is not None:
            return req.on_shared_wing
        return req.on_exact_match
    if state == EvidenceState.MISMATCH:
        return req.on_mismatch
    if state in (EvidenceState.MISSING_ON_A, EvidenceState.MISSING_ON_B):
        return req.on_missing_one_side
    if state == EvidenceState.MISSING_ON_BOTH:
        return req.on_missing_both_sides
    # AMBIGUOUS has no old-system precedent -- zero effect on the
    # regression comparison. Conservative default: treat like missing-both.
    return req.on_missing_both_sides


def apply_requirements(
    match_result: MatchResult,
    requirements: PolicyRequirements,
    high_similarity: bool,
    pins_match: bool,
) -> PolicyDecision:
    """
    Shared evaluation function for FlatOnlyPolicy / WingOnlyPolicy only.
    Takes high_similarity and pins_match as EXPLICIT BOOLEANS, matching
    old code's exact binary _high_similarity()/_pins_match() semantics,
    so behavior can be verified against the old branch logic directly.
    """
    missing_evidence: List[ComponentType] = []
    contradictions: List[ComponentType] = []
    diagnostic_notes: List[str] = []

    for req in requirements.components:
        if req.role == ComponentRole.IGNORED:
            continue

        ev: Optional[ComponentEvidence] = match_result.evidence.get(req.component)
        if ev is None:
            continue

        state = ev.state
        action = _action_for(req, state, is_shared_wing=False)  # shared_wing plumbing unfinished -- see comparator.py note

        if state in (EvidenceState.MISSING_ON_A, EvidenceState.MISSING_ON_B, EvidenceState.MISSING_ON_BOTH):
            if req.component not in missing_evidence:
                missing_evidence.append(req.component)
        elif state == EvidenceState.MISMATCH:
            if req.component not in contradictions:
                contradictions.append(req.component)
        elif state == EvidenceState.UNCERTAIN:
            # Deliberately NOT added to missing_evidence or contradictions
            # -- it is neither. Surfaced only as a diagnostic note so the
            # fact remains inspectable without conflating it with either
            # existing qualifier field's meaning (per the four-field
            # qualifier vocabulary: missing_evidence = absent,
            # contradictions = present-and-disagreeing-and-trusted;
            # UNCERTAIN is present-and-disagreeing-but-NOT-trusted, which
            # is neither).
            diagnostic_notes.append(f"{req.component.value} disagreement present but extraction confidence "
                                     f"too low to trust (uncertain, not treated as contradiction)")

        if action == RequirementAction.REJECT:
            return PolicyDecision(
                policy_name=requirements.name, tier=ConfidenceTier.LOW,
                missing_evidence=missing_evidence, contradictions=contradictions,
                diagnostic_notes=[f"{req.component.value} required and rejected ({state.value})"],
            )

        if action == RequirementAction.CONTRADICTION_CAP:
            if high_similarity:
                tier = req.contradiction_ceiling
                diagnostic_notes.append(f"{req.component.value} contradiction, high similarity -> capped at {tier.value}")
            else:
                tier = ConfidenceTier.LOW
                diagnostic_notes.append(f"{req.component.value} contradiction, low similarity -> {tier.value}")
            return PolicyDecision(
                policy_name=requirements.name, tier=tier,
                missing_evidence=missing_evidence, contradictions=contradictions,
                diagnostic_notes=diagnostic_notes,
            )

        if action == RequirementAction.TOLERATE:
            # NOTE: UNCERTAIN is intentionally grouped with MISSING_ON_BOTH
            # here, not just at _action_for()'s action-selection level.
            # _action_for() already routes UNCERTAIN to the same TOLERATE
            # action via its conservative fallthrough, but that alone is
            # NOT sufficient -- this ceiling-selection check must ALSO
            # treat them identically, or UNCERTAIN silently falls through
            # to the exact_match-style HIGH ceiling instead of the
            # missing_both floor, which was verified empirically to break
            # policy-level compatibility (FlatOnlyPolicy/WingOnlyPolicy
            # produced HIGH/MEDIUM for UNCERTAIN where MISSING_ON_BOTH
            # produces MEDIUM/LOW). Both represent "no trustworthy gate
            # evidence available" and must share the same ceiling.
            no_trustworthy_gate = state in (EvidenceState.MISSING_ON_BOTH, EvidenceState.UNCERTAIN)
            if high_similarity:
                if no_trustworthy_gate:
                    # old code: missing_on_both + high_sim -> Medium, NOT High.
                    # This is a genuinely different ceiling from exact_match's
                    # High -- missing_on_both never reaches the top tier in
                    # either policy, even with high similarity, because there
                    # was never a gate-component match to begin with.
                    tier = req.missing_both_high_sim_floor
                else:
                    tier = ConfidenceTier.HIGH
                    if not pins_match and requirements.pin_contributes_to_top_tier:
                        diagnostic_notes.append("PIN not matched -- legacy label will be HighWithoutPin, not High")
            else:
                if no_trustworthy_gate:
                    tier = req.missing_both_low_sim_floor
                else:
                    tier = req.satisfied_floor
                diagnostic_notes.append(f"similarity not high -> floor {tier.value}")

            return PolicyDecision(
                policy_name=requirements.name, tier=tier,
                missing_evidence=missing_evidence, contradictions=contradictions,
                diagnostic_notes=diagnostic_notes,
            )

    return PolicyDecision(
        policy_name=requirements.name, tier=ConfidenceTier.LOW,
        missing_evidence=missing_evidence, contradictions=contradictions,
        diagnostic_notes=["no requirement matched -- fallback"],
    )
