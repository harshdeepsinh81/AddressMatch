"""
The four ACTIVE client policies for the current Step 7 scope:

    DefaultMatchingPolicy
    FlatOnlyPolicy
    WingOnlyPolicy
    StrictFlatAndWingPolicy

DEFERRED (explicitly, per direction): WingDegradationPolicy is NOT
implemented in this step. See DEFERRED_WING_DEGRADATION_NOTE below for
the documented reason and future-extension plan. It is not part of the
regression scope, the invariant tests, or ALL_POLICIES.

Two of the four active policies (FlatOnlyPolicy, WingOnlyPolicy) are
implemented via the declarative policy_engine.py, since their legacy
behavior is genuinely single-component-decomposable (confirmed by tracing
every branch of both classes). The other two (DefaultMatchingPolicy,
StrictFlatAndWingPolicy) are implemented as direct functions mirroring
their legacy decision trees exactly, because their floor/gate logic is
JOINT across components in ways a per-component declarative schema cannot
express without adding complexity only they would use -- see
policy_engine.py's module docstring for the specific traced evidence.

Every function/config below is checked against the EXACT legacy branch
structure in matching_policies_old.py, cited by line-range comments at
each point of correspondence, so the mapping from old code to new code is
auditable line-by-line rather than asserted.
"""

from __future__ import annotations

from typing import Tuple

from .datamodel import ComponentType, ConfidenceTier, EvidenceState, MatchResult, PolicyDecision
from .policy_engine import (
    ComponentRequirement, ComponentRole, RequirementAction, PolicyRequirements, apply_requirements,
)


# ---------------------------------------------------------------------------
# DEFERRED: WingDegradationPolicy
# ---------------------------------------------------------------------------
DEFERRED_WING_DEGRADATION_NOTE = """
WingDegradationPolicy:
    status: DEFERRED
    reason: no current business requirement to support wing-degradation
            behavior in this step; deferred by explicit direction rather
            than by default/oversight.
    not part of: current Step 7 implementation, regression scope (1,280
            combinations covers the 4 active policies only), invariant
            tests, ALL_POLICIES registry.
    future extension path: the deferred policy's intended rule --
            "base decision -> wing mismatch? -> HIGH->MEDIUM,
            MEDIUM->LOW, LOW->LOW" -- was already investigated (see prior
            regression findings) and requires either (a) a new
            RequirementAction.DEGRADE_ONE_TIER value added back to
            policy_engine.py plus tier-ordinal arithmetic (straightforward,
            additive, does not require touching the 4 active policies), or
            (b) a fifth escape-hatch function mirroring
            WingDegradationPolicy's legacy tree exactly (same pattern as
            DefaultMatchingPolicy/StrictFlatAndWingPolicy below), if its
            joint base-computation (which mirrors DefaultMatchingPolicy's
            own branch structure, confirmed during investigation) turns
            out not to decompose declaratively either. Both extension
            paths are additive to the current architecture -- neither
            requires redesigning MatchResult, PolicyDecision, or the
            evaluation boundary.
    known old-code issue (found during investigation, NOT fixed, NOT
            currently relevant since the policy isn't implemented): the
            old _DEGRADATION lookup table is missing an entry for
            base_conf == "HighWithoutFlatPlotWing", so wing mismatch
            silently fails to degrade a flat-contradiction-high-similarity
            base case. This remains an open question for whenever
            WingDegradationPolicy is actually implemented -- not resolved
            here, since resolving it now would require designing the
            degradation feature this step is explicitly deferring.
"""


# ---------------------------------------------------------------------------
# Shared helpers
# ---------------------------------------------------------------------------

def _pin_state(match_result: MatchResult) -> Tuple[bool, bool, bool]:
    """Returns (pins_match, pin_mismatch, pin_missing) as independent
    booleans -- PRESERVES the four-way PIN evidence distinction
    structurally (constraint: PIN missing vs PIN mismatch remain
    structurally distinguishable) even though every active policy's
    CONFIGURED business behavior collapses mismatch and missing into the
    same "not pins_match" bucket, exactly matching old _pins_match()
    (verified: returns False for both, zero behavioral difference in old
    code across the full matrix)."""
    ev = match_result.evidence.get(ComponentType.PIN)
    if ev is None:
        return False, False, False
    pins_match = ev.state == EvidenceState.MATCH
    pin_mismatch = ev.state == EvidenceState.MISMATCH
    pin_missing = ev.state in (EvidenceState.MISSING_ON_A, EvidenceState.MISSING_ON_B, EvidenceState.MISSING_ON_BOTH)
    return pins_match, pin_mismatch, pin_missing


def _flat_state(match_result: MatchResult) -> str:
    ev = match_result.evidence.get(ComponentType.FLAT)
    return ev.state.value if ev else "missing_on_both"


def _wing_state(match_result: MatchResult) -> str:
    ev = match_result.evidence.get(ComponentType.WING)
    return ev.state.value if ev else "missing_on_both"


# new-engine EvidenceState values use "missing_on_a"/"missing_on_b"; old
# code's vocabulary is "missing_on_one" (direction-agnostic). This maps
# between them for the escape-hatch functions below, which mirror old
# code's branch conditions literally and therefore need old-shaped state
# strings.
def _old_shaped_state(new_state_value: str) -> str:
    if new_state_value in ("missing_on_a", "missing_on_b"):
        return "missing_on_one"
    return new_state_value  # match/mismatch/missing_on_both pass through as "exact_match" is NOT auto-derived here -- see callers


def _to_old_flat_status(match_result: MatchResult) -> str:
    ev = match_result.evidence.get(ComponentType.FLAT)
    if ev is None:
        return "missing_on_both"
    mapping = {
        EvidenceState.MATCH: "exact_match",
        EvidenceState.MISMATCH: "mismatch",
        EvidenceState.MISSING_ON_A: "missing_on_one",
        EvidenceState.MISSING_ON_B: "missing_on_one",
        EvidenceState.MISSING_ON_BOTH: "missing_on_both",
        EvidenceState.AMBIGUOUS: "missing_on_both",  # no old precedent; conservative mapping, documented
        EvidenceState.UNCERTAIN: "missing_on_both",  # no old precedent -- old code never had a confidence-gated
                                                        # disagreement concept at all. Mapped to missing_on_both
                                                        # (NOT mismatch) because the whole point of UNCERTAIN is
                                                        # that the disagreement is not trusted -- verified
                                                        # empirically (not assumed) to produce identical
                                                        # PolicyDecision.tier to a genuine MISSING_ON_BOTH across
                                                        # all 4 active policies and every combination of the other
                                                        # component's state -- see the dedicated compatibility
                                                        # test suite. Documented as an explicit, verified mapping,
                                                        # not an assumption.
    }
    return mapping[ev.state]


def _to_old_wing_status(match_result: MatchResult) -> str:
    # NOTE: the new engine's WING comparator does not currently emit a
    # separate "shared_wing" state (see comparator.py -- it reports MATCH
    # for any letter overlap). This means _to_old_wing_status can never
    # currently return "shared_wing" even though the old vocabulary
    # includes it -- this is the SAME unfinished-plumbing gap already
    # flagged in policy_engine.py. It does not affect the regression
    # comparison against old code's "shared_wing" test cells (the harness
    # constructs synthetic evidence directly for those cells -- see the
    # regression report), but it IS a real gap for the live parser->engine
    # pipeline, documented here at its second point of relevance.
    ev = match_result.evidence.get(ComponentType.WING)
    if ev is None:
        return "missing_on_both"
    mapping = {
        EvidenceState.MATCH: "exact_match",
        EvidenceState.MISMATCH: "mismatch",
        EvidenceState.MISSING_ON_A: "missing_on_one",
        EvidenceState.MISSING_ON_B: "missing_on_one",
        EvidenceState.MISSING_ON_BOTH: "missing_on_both",
        EvidenceState.AMBIGUOUS: "missing_on_both",
        EvidenceState.UNCERTAIN: "missing_on_both",  # see _to_old_flat_status's identical entry for the
                                                        # full rationale and empirical verification note.
    }
    return mapping[ev.state]


# ---------------------------------------------------------------------------
# 1. DefaultMatchingPolicy -- ESCAPE HATCH (joint flat+wing floor logic)
# ---------------------------------------------------------------------------
# Mirrors matching_policies_old.py lines 96-145 EXACTLY, branch for branch.
# The UNRESOLVED flat=missing_on_both+wing=missing_on_both+high_sim->LOW
# case (old line 141-144's fallthrough) is reproduced verbatim, with no
# attempt to "fix" it toward the other policies' Medium -- per explicit
# instruction, no evidence supports changing it.

def evaluate_default_policy(match_result: MatchResult, high_similarity: bool, pins_match: bool) -> PolicyDecision:
    flat_status = _to_old_flat_status(match_result)
    wing_status = _to_old_wing_status(match_result)

    missing_evidence = []
    contradictions = []
    for ct, status in ((ComponentType.FLAT, flat_status), (ComponentType.WING, wing_status)):
        if status in ("missing_on_one", "missing_on_both"):
            missing_evidence.append(ct)
        elif status == "mismatch":
            contradictions.append(ct)
    pin_ev = match_result.evidence.get(ComponentType.PIN)
    if pin_ev:
        if pin_ev.state == EvidenceState.MISMATCH:
            contradictions.append(ComponentType.PIN)
        elif pin_ev.state in (EvidenceState.MISSING_ON_A, EvidenceState.MISSING_ON_B, EvidenceState.MISSING_ON_BOTH):
            missing_evidence.append(ComponentType.PIN)

    notes = []

    # old lines 102-108: flat mismatch
    if flat_status == "mismatch":
        tier = ConfidenceTier.HIGH if high_similarity else ConfidenceTier.LOW
        notes.append("flat mismatch: HIGH-ceiling if high_sim else LOW (old lines 102-108)")
        return PolicyDecision("DefaultMatchingPolicy", tier, missing_evidence, contradictions, [], notes)

    # old lines 110-111: flat missing on one side
    if flat_status == "missing_on_one":
        notes.append("flat missing on one side -> LOW (old lines 110-111)")
        return PolicyDecision("DefaultMatchingPolicy", ConfidenceTier.LOW, missing_evidence, contradictions, [], notes)

    # old lines 114-133: flat exact match
    if flat_status == "exact_match":
        if wing_status in ("exact_match", "shared_wing"):
            # old lines 116-120
            if high_similarity:
                tier = ConfidenceTier.HIGH  # pins_match distinguishes High vs HighWithoutPin at legacy-mapping layer
            else:
                tier = ConfidenceTier.MEDIUM
            notes.append("flat match + wing match: HIGH if high_sim else MEDIUM (old lines 116-120)")
            return PolicyDecision("DefaultMatchingPolicy", tier, missing_evidence, contradictions, [], notes)

        if wing_status == "mismatch":
            # old lines 122-127: HighWithoutWing/HighWithoutPin/Low
            if high_similarity:
                tier = ConfidenceTier.MEDIUM if pins_match else ConfidenceTier.HIGH
                # old: pins_match+high_sim -> "HighWithoutWing" (NOT top "High"); !pins_match+high_sim -> "HighWithoutPin" (also not top "High")
                # Both are "High"-prefixed legacy strings but neither is the bare "High" -- represented here as:
                #   pins_match  -> MEDIUM internal tier (legacy string is HighWithoutWing, mapped at boundary)
                #   !pins_match -> HIGH internal tier is WRONG per old priority (HighWithoutPin should not outrank HighWithoutWing) --
                # corrected below: old code checks pins_match FIRST, so:
                if pins_match:
                    tier = ConfidenceTier.MEDIUM  # legacy label HighWithoutWing
                    notes.append("flat match + wing mismatch + pins_match + high_sim -> internal MEDIUM, legacy HighWithoutWing (old line 124)")
                else:
                    tier = ConfidenceTier.MEDIUM  # legacy label HighWithoutPin
                    notes.append("flat match + wing mismatch + !pins_match + high_sim -> internal MEDIUM, legacy HighWithoutPin (old line 126)")
            else:
                tier = ConfidenceTier.LOW
                notes.append("flat match + wing mismatch + !high_sim -> LOW (old line 127)")
            return PolicyDecision("DefaultMatchingPolicy", tier, missing_evidence, contradictions, [], notes)

        # wing missing on one or both -- old lines 129-133, "data completeness gap, accepted"
        if high_similarity:
            tier = ConfidenceTier.HIGH  # High or HighWithoutPin, both HIGH-tier internally; distinguished at legacy-mapping layer
        else:
            tier = ConfidenceTier.MEDIUM
        notes.append("flat match + wing missing (tolerated): HIGH if high_sim else MEDIUM (old lines 129-133)")
        return PolicyDecision("DefaultMatchingPolicy", tier, missing_evidence, contradictions, [], notes)

    # old lines 135-141: flat missing on both sides
    if flat_status == "missing_on_both":
        if wing_status in ("exact_match", "shared_wing") and high_similarity:
            notes.append("flat missing_on_both + wing match + high_sim -> HIGH (old line 140)")
            return PolicyDecision("DefaultMatchingPolicy", ConfidenceTier.HIGH, missing_evidence, contradictions, [], notes)
        if wing_status in ("mismatch", "missing_on_one") and high_similarity:
            notes.append("flat missing_on_both + wing mismatch/missing_one + high_sim -> MEDIUM (old line 143)")
            return PolicyDecision("DefaultMatchingPolicy", ConfidenceTier.MEDIUM, missing_evidence, contradictions, [], notes)
        # UNRESOLVED BUSINESS SEMANTICS -- see policies.py module docstring
        # and the Step 7 review exchange. This covers: wing missing_on_both
        # (regardless of high_similarity), AND wing match/mismatch/missing_one
        # when high_similarity is False. Old code's fallthrough to line 145.
        notes.append("UNRESOLVED: flat missing_on_both fallthrough -> LOW (old line 145, preserved exactly, not changed to MEDIUM)")
        return PolicyDecision("DefaultMatchingPolicy", ConfidenceTier.LOW, missing_evidence, contradictions, [], notes)

    # old line 147 (unreachable in practice given the 4-state model, kept for parity)
    return PolicyDecision("DefaultMatchingPolicy", ConfidenceTier.LOW, missing_evidence, contradictions, [],
                           ["fallback: unable to determine (old line 147)"])


# ---------------------------------------------------------------------------
# 2. FlatOnlyPolicy -- DECLARATIVE (single-component-decomposable)
# ---------------------------------------------------------------------------
# Mirrors matching_policies_old.py lines 180-204 exactly via the
# declarative engine: flat is the sole REQUIRED_GATE, wing IGNORED.
FLAT_ONLY_REQUIREMENTS = PolicyRequirements(
    name="FlatOnlyPolicy",
    components=[
        ComponentRequirement(
            component=ComponentType.FLAT, role=ComponentRole.REQUIRED_GATE,
            on_exact_match=RequirementAction.TOLERATE,          # old lines 193-198: exact_match -> High/HighWithoutPin/Medium
            on_mismatch=RequirementAction.CONTRADICTION_CAP,     # old lines 181-186: mismatch -> HighWithoutFlatPlotWing/Low
            contradiction_ceiling=ConfidenceTier.HIGH,             # legacy "HighWithoutFlatPlotWing" is HIGH-tier, preserved exactly
            on_missing_one_side=RequirementAction.REJECT,          # old lines 188-189: missing_on_one -> Low
            on_missing_both_sides=RequirementAction.TOLERATE,       # old lines 200-203: missing_on_both -> Medium/Low
            satisfied_floor=ConfidenceTier.MEDIUM,                    # old line 198: exact_match + !high_sim -> Medium
            missing_both_high_sim_floor=ConfidenceTier.MEDIUM,          # old line 202: missing_on_both + high_sim -> Medium
            missing_both_low_sim_floor=ConfidenceTier.LOW,                # old line 203: missing_on_both + !high_sim -> Low
        ),
        ComponentRequirement(component=ComponentType.WING, role=ComponentRole.IGNORED),
    ],
    pin_contributes_to_top_tier=True,
)


# ---------------------------------------------------------------------------
# 3. WingOnlyPolicy -- DECLARATIVE (single-component-decomposable)
# ---------------------------------------------------------------------------
WING_ONLY_REQUIREMENTS = PolicyRequirements(
    name="WingOnlyPolicy",
    components=[
        ComponentRequirement(component=ComponentType.FLAT, role=ComponentRole.IGNORED),
        ComponentRequirement(
            component=ComponentType.WING, role=ComponentRole.REQUIRED_GATE,
            on_exact_match=RequirementAction.TOLERATE,
            on_shared_wing=RequirementAction.TOLERATE,
            on_mismatch=RequirementAction.CONTRADICTION_CAP,
            contradiction_ceiling=ConfidenceTier.HIGH,   # old lines 243-249: mismatch+high_sim -> HighWithoutFlatPlotWing (HIGH-tier)
            on_missing_one_side=RequirementAction.REJECT,  # old lines 251-252
            on_missing_both_sides=RequirementAction.TOLERATE,  # old lines 262-265
            satisfied_floor=ConfidenceTier.MEDIUM,   # old line 260
            missing_both_high_sim_floor=ConfidenceTier.MEDIUM,  # old line 264
            missing_both_low_sim_floor=ConfidenceTier.LOW,        # old line 265
        ),
    ],
    pin_contributes_to_top_tier=True,
)


# ---------------------------------------------------------------------------
# 4. StrictFlatAndWingPolicy -- ESCAPE HATCH (joint hard-gate + one joint floor)
# ---------------------------------------------------------------------------
# Mirrors matching_policies_old.py lines 304-342 exactly.

def evaluate_strict_flat_and_wing_policy(match_result: MatchResult, high_similarity: bool, pins_match: bool) -> PolicyDecision:
    flat_status = _to_old_flat_status(match_result)
    wing_status = _to_old_wing_status(match_result)

    missing_evidence = []
    contradictions = []
    for ct, status in ((ComponentType.FLAT, flat_status), (ComponentType.WING, wing_status)):
        if status in ("missing_on_one", "missing_on_both"):
            missing_evidence.append(ct)
        elif status == "mismatch":
            contradictions.append(ct)
    pin_ev = match_result.evidence.get(ComponentType.PIN)
    if pin_ev:
        if pin_ev.state == EvidenceState.MISMATCH:
            contradictions.append(ComponentType.PIN)
        elif pin_ev.state in (EvidenceState.MISSING_ON_A, EvidenceState.MISSING_ON_B, EvidenceState.MISSING_ON_BOTH):
            missing_evidence.append(ComponentType.PIN)

    notes = []

    # old lines 309-312: hard flat failures
    if flat_status == "mismatch":
        notes.append("flat mismatch -> LOW, no contradiction tolerance (old line 310)")
        return PolicyDecision("StrictFlatAndWingPolicy", ConfidenceTier.LOW, missing_evidence, contradictions, [], notes)
    if flat_status == "missing_on_one":
        notes.append("flat missing on one side -> LOW (old line 312)")
        return PolicyDecision("StrictFlatAndWingPolicy", ConfidenceTier.LOW, missing_evidence, contradictions, [], notes)

    # old lines 315-318: hard wing failures
    if wing_status == "mismatch":
        notes.append("wing mismatch -> LOW, no contradiction tolerance (old line 316)")
        return PolicyDecision("StrictFlatAndWingPolicy", ConfidenceTier.LOW, missing_evidence, contradictions, [], notes)
    if wing_status == "missing_on_one":
        notes.append("wing missing on one side -> LOW (old line 318)")
        return PolicyDecision("StrictFlatAndWingPolicy", ConfidenceTier.LOW, missing_evidence, contradictions, [], notes)

    # old lines 320-332: both match, or wing absent on both (no gate possible)
    flat_ok = flat_status == "exact_match"
    wing_ok = wing_status in ("exact_match", "shared_wing", "missing_on_both")

    if flat_ok and wing_ok:
        if high_similarity:
            tier = ConfidenceTier.HIGH  # High or HighWithoutPin, both HIGH internally, distinguished at legacy-mapping layer
            notes.append("flat_ok + wing_ok + high_sim -> HIGH (old lines 324-331)")
        else:
            tier = ConfidenceTier.MEDIUM
            notes.append("flat_ok + wing_ok + !high_sim -> MEDIUM (old line 332)")
        return PolicyDecision("StrictFlatAndWingPolicy", tier, missing_evidence, contradictions, [], notes)

    # old lines 334-341: no flat on either side (flat_status == missing_on_both, since
    # flat_ok is False here and the mismatch/missing_one flat cases already returned above)
    if flat_status == "missing_on_both":
        if wing_status in ("exact_match", "shared_wing") and high_similarity:
            notes.append("flat missing_on_both + wing match + high_sim -> MEDIUM (old line 339)")
            return PolicyDecision("StrictFlatAndWingPolicy", ConfidenceTier.MEDIUM, missing_evidence, contradictions, [], notes)
        if high_similarity:
            notes.append("flat missing_on_both + wing not matching + high_sim -> MEDIUM (old line 341)")
            return PolicyDecision("StrictFlatAndWingPolicy", ConfidenceTier.MEDIUM, missing_evidence, contradictions, [], notes)
        notes.append("flat missing_on_both + !high_sim -> LOW (old line 342)")
        return PolicyDecision("StrictFlatAndWingPolicy", ConfidenceTier.LOW, missing_evidence, contradictions, [], notes)

    return PolicyDecision("StrictFlatAndWingPolicy", ConfidenceTier.LOW, missing_evidence, contradictions, [],
                           ["fallback: unable to determine (old line 344)"])


# ---------------------------------------------------------------------------
# Registry -- ONLY the 4 active policies (WingDegradationPolicy excluded)
# ---------------------------------------------------------------------------

def evaluate_policy(policy_name: str, match_result: MatchResult, high_similarity: bool, pins_match: bool) -> PolicyDecision:
    if policy_name == "DefaultMatchingPolicy":
        return evaluate_default_policy(match_result, high_similarity, pins_match)
    if policy_name == "FlatOnlyPolicy":
        return apply_requirements(match_result, FLAT_ONLY_REQUIREMENTS, high_similarity, pins_match)
    if policy_name == "WingOnlyPolicy":
        return apply_requirements(match_result, WING_ONLY_REQUIREMENTS, high_similarity, pins_match)
    if policy_name == "StrictFlatAndWingPolicy":
        return evaluate_strict_flat_and_wing_policy(match_result, high_similarity, pins_match)
    raise ValueError(f"Unknown or deferred policy: {policy_name}. "
                      f"If this is WingDegradationPolicy, see DEFERRED_WING_DEGRADATION_NOTE.")


ACTIVE_POLICY_NAMES = [
    "DefaultMatchingPolicy", "FlatOnlyPolicy", "WingOnlyPolicy", "StrictFlatAndWingPolicy",
]
