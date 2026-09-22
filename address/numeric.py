"""
Numeric identifier canonicalization and OCR-confusable handling.

Implements Phase 1 section 8. Principle: numeric identifiers (PIN, flat,
house, plot, floor) never receive a graduated "closeness" score -- only a
canonicalize-then-exact-match outcome, optionally resolved through a
bounded, auditable OCR-confusable correction. See constraint #7: OCR
correction info (whether correction was needed, which characters, how many)
must be preserved, not just a boolean.

The OCR-confusable map and the "only short digit-bearing tokens" scoping
rule are ported from the old code's ocr_normalize_confusables /
OCR_CONFUSABLE_TO_DIGIT -- that scoping was a correct, deliberate safety
constraint in the original (it stops "CHENNAI600078" from having letters
rewritten) and is preserved unchanged here.
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from typing import Optional, Tuple


# ---------------------------------------------------------------------------
# OCR confusable map -- ported verbatim from the old code
# ---------------------------------------------------------------------------

OCR_CONFUSABLE_TO_DIGIT = {
    "O": "0", "Q": "0", "D": "0",
    "I": "1", "L": "1",
    "Z": "2",
    "S": "5",
    "G": "6",
    "T": "7",
    "B": "8",
}
_OCR_TRANSLATION = str.maketrans(OCR_CONFUSABLE_TO_DIGIT)

# Only tokens this short are eligible for OCR-confusable rewriting -- see
# module docstring; unchanged from the old MAX_OCR_NORMALIZED_TOKEN_LENGTH.
MAX_OCR_TOKEN_LENGTH = 6

# A 6-digit-ish PIN candidate (digits or confusable letters) standing alone.
OCR_TOLERANT_PIN_PATTERN = re.compile(
    r'(?<![A-Za-z0-9])[0-9OQDILZSBGT]{6}(?![A-Za-z0-9])', re.IGNORECASE
)
MIN_REAL_DIGITS_IN_OCR_PIN = 4


@dataclass
class NumericCanonicalizationResult:
    canonical: str
    ocr_corrected: bool
    ocr_corrected_chars: int
    raw: str


def _corrected_char_count(raw_upper: str, corrected: str) -> int:
    return sum(1 for a, b in zip(raw_upper, corrected) if a != b)


def ocr_normalize_token(token: str) -> Tuple[str, int]:
    """
    Apply OCR-confusable correction to a single short, digit-bearing token.
    Returns (corrected_token, num_chars_changed). If the token is too long
    or has no digit, returns it unchanged with 0 corrections -- this is the
    same scoping the old code used to avoid rewriting ordinary words.
    """
    if not token:
        return token, 0
    upper = token.upper()
    if len(upper) > MAX_OCR_TOKEN_LENGTH or not any(c.isdigit() for c in upper):
        return upper, 0
    corrected = upper.translate(_OCR_TRANSLATION)
    return corrected, _corrected_char_count(upper, corrected)


def canonicalize_numeric_identifier(raw: str) -> NumericCanonicalizationResult:
    """
    Canonicalize a flat/unit/house/plot/floor identifier for exact-match
    comparison (Phase 1 section 8, canonicalization steps 1-4):

      1. Strip separators (-, /, spaces) between parts but KEEP the
         letter-number structure -- "A-401", "A/401", "A 401" all collapse
         to "A401", but "A401" and "401" remain genuinely distinct (a wing
         prefix difference is real signal, not noise).
      2. Normalize leading zeros ("0401" -> "401") on the numeric run.
      3. Uppercase any letter component.

    OCR-confusable correction is NOT applied here automatically -- it is
    offered as a separate, explicit alternative candidate (see
    ocr_normalize_token / comparator.py) so the caller can distinguish a
    clean exact match from one that only matched after OCR correction, and
    can report how many characters were corrected (constraint #7).
    """
    if raw is None:
        return NumericCanonicalizationResult("", False, 0, "")
    raw_stripped = raw.strip()
    if not raw_stripped:
        return NumericCanonicalizationResult("", False, 0, raw_stripped)

    # remove separators between alnum runs, keep letters+digits
    collapsed = re.sub(r'[\s\-/]+', '', raw_stripped).upper()

    # split into leading letters / digits / trailing letters to normalize
    # leading zeros on the numeric run without disturbing letter parts,
    # e.g. "A0401B" -> letters "A", digits "0401" -> "401", letters "B"
    m = re.match(r'^([A-Z]*)(\d+)([A-Z]*)$', collapsed)
    if m:
        prefix, digits, suffix = m.groups()
        digits_norm = str(int(digits)) if digits else digits
        canonical = f"{prefix}{digits_norm}{suffix}"
    else:
        # doesn't fit the simple letter-digit-letter shape (e.g. multiple
        # digit runs like "12A34") -- leave as collapsed/uppercased rather
        # than guessing at a canonical form; comparator will fall back to
        # exact string comparison on this value, no OCR-magic applied.
        canonical = collapsed

    return NumericCanonicalizationResult(canonical, False, 0, raw_stripped)


def numeric_identifiers_match(
    value_a: str, value_b: str
) -> Tuple[bool, bool, int]:
    """
    Compare two RAW (not yet canonicalized) numeric identifiers.

    Returns (is_match, used_ocr_correction, ocr_corrected_chars).

    Tries, in order:
      1. canonical exact match (no OCR correction)
      2. canonical match after OCR-confusable correction on whichever
         side(s) are short/digit-bearing enough to be eligible
    This mirrors the old code's two-pass approach in evaluate_flat_match
    (try exact, then retry with ocr_normalize=True) but now reports how
    many characters were corrected rather than a bare boolean.
    """
    canon_a = canonicalize_numeric_identifier(value_a)
    canon_b = canonicalize_numeric_identifier(value_b)

    if canon_a.canonical == canon_b.canonical:
        return True, False, 0

    ocr_a, chars_a = ocr_normalize_token(canon_a.canonical)
    ocr_b, chars_b = ocr_normalize_token(canon_b.canonical)

    if ocr_a == ocr_b and (chars_a > 0 or chars_b > 0):
        return True, True, max(chars_a, chars_b)

    return False, False, 0


def extract_pin_code(address: str):
    """
    Ported unchanged from the old address.py: find a six-digit PIN.
    Returns the PIN string, -1 if multiple distinct 6-digit sequences are
    found (ambiguous -- caller must treat as a hard "cannot determine PIN"
    case, exactly as before), or None if none found.
    """
    if not isinstance(address, str) or not address.strip():
        return None
    pattern = re.compile(r'(?<!\d)\d{6}(?!\d)')
    matches = pattern.findall(address)
    if matches:
        unique = set(matches)
        return matches[0] if len(unique) == 1 else -1
    clean = re.sub(r'[\s-]', '', address)
    matches = pattern.findall(clean)
    if matches:
        unique = set(matches)
        return matches[0] if len(unique) == 1 else -1
    return None


def extract_pin_code_ocr_tolerant(address: str) -> Tuple[Optional[str], Optional[str]]:
    """Ported unchanged from the old address.py's OCR-tolerant PIN recovery."""
    if not isinstance(address, str) or not address.strip():
        return None, None
    candidates = []
    for match in OCR_TOLERANT_PIN_PATTERN.finditer(address):
        raw_token = match.group(0)
        real_digits = sum(c.isdigit() for c in raw_token)
        if real_digits < MIN_REAL_DIGITS_IN_OCR_PIN:
            continue
        corrected = raw_token.upper().translate(_OCR_TRANSLATION)
        if corrected.isdigit():
            candidates.append((corrected, raw_token))
    unique_pins = {pin for pin, _ in candidates}
    if len(unique_pins) == 1:
        return candidates[0]
    return None, None
