"""
Maps a PolicyDecision (new architecture, 4 active policies) back to the
exact legacy label and reason strings matching_policies_old.py returned.

This module intentionally re-derives old-shaped flat/wing status strings
and pins_match/high_similarity booleans from MatchResult, then applies the
EXACT same branch conditions old code used to pick a label string. This
mirrors, rather than infers from, the tier -- because the tier alone is
not always sufficient to recover which of several old strings sharing a
tier applies (e.g. "High" vs "HighWithoutPin" vs "HighWithoutFlatPlotWing"
can all correspond to an internal HIGH tier; "HighWithoutWing" corresponds
to internal MEDIUM per the DefaultMatchingPolicy escape-hatch function's
own comments). Re-deriving from the same evidence the escape-hatch
functions used keeps this mapping auditable against the same line-cited
old branches.
"""

from __future__ import annotations

from .datamodel import ComponentType, ConfidenceTier, EvidenceState, MatchResult, PolicyDecision
from .policies import _to_old_flat_status, _to_old_wing_status, _pin_state


def to_legacy_label_and_reason(
    policy_name: str, decision: PolicyDecision, match_result: MatchResult,
    high_similarity: bool, pins_match: bool,
) -> tuple:
    flat_status = _to_old_flat_status(match_result)
    wing_status = _to_old_wing_status(match_result)

    if policy_name == "DefaultMatchingPolicy":
        return _default_legacy(flat_status, wing_status, high_similarity, pins_match, decision)
    if policy_name == "FlatOnlyPolicy":
        return _flat_only_legacy(flat_status, high_similarity, pins_match)
    if policy_name == "WingOnlyPolicy":
        return _wing_only_legacy(wing_status, high_similarity, pins_match)
    if policy_name == "StrictFlatAndWingPolicy":
        return _strict_legacy(flat_status, wing_status, high_similarity, pins_match)
    raise ValueError(f"No legacy mapping for policy: {policy_name}")


def _default_legacy(flat_status, wing_status, high_similarity, pins_match, decision):
    if flat_status == "mismatch":
        if high_similarity:
            return "HighWithoutFlatPlotWing", "Flat/Plot/House numbers mismatch but base address similar."
        return "Low", "Flat/Plot/House number mismatch."
    if flat_status == "missing_on_one":
        return "Low", "One address has flat/plot number, other doesn't."
    if flat_status == "exact_match":
        if wing_status in ("exact_match", "shared_wing"):
            if pins_match and high_similarity:
                return "High", "Flat and wing match, PINs match, base address similar."
            if not pins_match and high_similarity:
                return "HighWithoutPin", "Flat and wing match, but PINs missing/mismatch; base similar."
            return "Medium", "Flat and wing match but base address similarity low."
        if wing_status == "mismatch":
            if pins_match and high_similarity:
                return "HighWithoutWing", "Flat matches, PINs match, base similar, but wings differ."
            if not pins_match and high_similarity:
                return "HighWithoutPin", "Flat matches but wings differ and PINs missing/mismatch."
            return "Low", "Flat matches but wings mismatch and other factors weak."
        # wing missing on one or both
        if pins_match and high_similarity:
            return "High", "Flat matches, PINs match, base similar."
        if not pins_match and high_similarity:
            return "HighWithoutPin", "Flat matches, base similar, but PINs missing/mismatch."
        return "Medium", "Flat matches but low base address similarity."
    if flat_status == "missing_on_both":
        if wing_status in ("exact_match", "shared_wing") and high_similarity:
            return "High", "Wings match and base address similar."
        if wing_status in ("mismatch", "missing_on_one") and high_similarity:
            return "Medium", "No flat info on either side; base address similar."
        return "Low", "Insufficient address components for high confidence."
    return "Low", "Unable to determine confidence level."


def _flat_only_legacy(flat_status, high_similarity, pins_match):
    if flat_status == "mismatch":
        if high_similarity:
            return "HighWithoutFlatPlotWing", "Flat/Plot/House numbers mismatch but base address similar."
        return "Low", "Flat/Plot/House number mismatch."
    if flat_status == "missing_on_one":
        return "Low", "One address has flat/plot number, other doesn't."
    if flat_status == "exact_match":
        if pins_match and high_similarity:
            return "High", "Flat matches, PINs match, base address similar."
        if not pins_match and high_similarity:
            return "HighWithoutPin", "Flat matches, base similar, but PINs missing/mismatch."
        return "Medium", "Flat matches but low base address similarity."
    if flat_status == "missing_on_both":
        if high_similarity:
            return "Medium", "No flat info on either side; base address similar."
        return "Low", "Insufficient address components for high confidence."
    return "Low", "Unable to determine confidence level."


def _wing_only_legacy(wing_status, high_similarity, pins_match):
    if wing_status == "mismatch":
        if high_similarity:
            return "HighWithoutFlatPlotWing", "Wing mismatch but base address similar."
        return "Low", "Wing mismatch."
    if wing_status == "missing_on_one":
        return "Low", "One address has wing information, other doesn't."
    if wing_status in ("exact_match", "shared_wing"):
        if pins_match and high_similarity:
            return "High", "Wing matches, PINs match, base address similar."
        if not pins_match and high_similarity:
            return "HighWithoutPin", "Wing matches, base similar, but PINs missing/mismatch."
        return "Medium", "Wing matches but low base address similarity."
    if wing_status == "missing_on_both":
        if high_similarity:
            return "Medium", "No wing info on either side; base address similar."
        return "Low", "Insufficient address components for high confidence."
    return "Low", "Unable to determine confidence level."


def _strict_legacy(flat_status, wing_status, high_similarity, pins_match):
    if flat_status == "mismatch":
        return "Low", "Flat/Plot/House number mismatch."
    if flat_status == "missing_on_one":
        return "Low", "One address has flat/plot number, other doesn't."
    if wing_status == "mismatch":
        return "Low", "Wing mismatch."
    if wing_status == "missing_on_one":
        return "Low", "One address has wing information, other doesn't."

    flat_ok = flat_status == "exact_match"
    wing_ok = wing_status in ("exact_match", "shared_wing", "missing_on_both")

    if flat_ok and wing_ok:
        if pins_match and high_similarity:
            label = ("Flat and wing match, PINs match, base address similar."
                      if wing_status != "missing_on_both"
                      else "Flat matches, PINs match, base address similar (no wing data).")
            return "High", label
        if not pins_match and high_similarity:
            return "HighWithoutPin", "Components match but PINs missing/mismatch; base similar."
        return "Medium", "Components match but low base address similarity."

    if flat_status == "missing_on_both":
        if wing_status in ("exact_match", "shared_wing") and high_similarity:
            return "Medium", "No flat info; wing matches and base address similar."
        if high_similarity:
            return "Medium", "No flat or wing info on either side; base address similar."
        return "Low", "Insufficient address components for high confidence."

    return "Low", "Unable to determine confidence level."
