"""
Character-level cleanup, applied once, before component parsing.

This module is a near-verbatim port of the *good* parts of the old
address.py's cosmetic formatting layer (clean_ocr_address, format_address,
and the small text_processing helpers it depended on). Their job is purely
cosmetic: fix spacing, casing, label-prefix noise, OCR scan-line artifacts.

None of this makes semantic decisions (nothing here decides "this token is
a flat number") — that is entirely the job of parser.py. Keeping this
separate is what makes the parser's precedence rules (constraint #3)
possible to reason about: the parser always receives already-cleaned,
already-cased text and never has to fight punctuation noise itself.
"""

from __future__ import annotations

import re

# --- compiled once, module load time -------------------------------------

_ADDR_LABEL_PREFIX    = re.compile(r'^\s*add(?:r(?:ess)?)?\s*[#:.\-]?\s*', re.IGNORECASE)
_OCR_NOISE_CHARS      = re.compile(r'[|\\~`^@!*<>{}\[\]"]+')
_REPEATED_SEPARATORS  = re.compile(r'([,./])\1+')
_EXCESS_HYPHENS       = re.compile(r'-{3,}')
_LEADING_TRAILING_SEP = re.compile(r'^[\s,./\-:#]+|[\s,./\-:#]+$')

_CAP_BOUNDARY   = re.compile(r'([a-z])([A-Z])')
_COMMA_SPACING  = re.compile(r',\s*')
_HYPHEN_SPACING = re.compile(r'\s*-\s*')
_WORD_DIGIT_1   = re.compile(r'([a-zA-Z])(\d)')
_WORD_DIGIT_2   = re.compile(r'(\d)([a-zA-Z])')
_LEADING_SYMBOL = re.compile(r'^[^\w]+')
_MULTI_SPACE    = re.compile(r'\s+')
_LETTER_HYPHEN_LETTER = re.compile(r'([a-zA-Z])-(?=[a-zA-Z])')
_LETTER_AMP_HYPHEN    = re.compile(r'([a-zA-Z])([&-])')

# Salutation / relationship-indicator stripping (D/O, C/O, S/O, Mr, Mrs, ...)
_RELATION_PREFIX = re.compile(r"^[A-Za-z]/[A-Za-z]\.?\s*:?\s*")
_SALUTATION_PREFIX = re.compile(
    r"^(Mr|Mrs|Ms|Miss|Dr|Prof|Rev|Sr|Jr|Shri|Smt|Ku|M/s)\.?\s*", re.IGNORECASE
)


def _capitalize_text(text: str) -> str:
    if not text:
        return text
    return ' '.join(w.capitalize() if not w.isupper() else w for w in text.split())


def clean_ocr_noise(text: str) -> str:
    """
    Strip field-label prefixes and OCR scan-line artifacts.
    Behavior-identical port of the old clean_ocr_address().
    """
    if not text:
        return text
    text = _ADDR_LABEL_PREFIX.sub('', text)
    text = _OCR_NOISE_CHARS.sub(' ', text)
    text = _REPEATED_SEPARATORS.sub(r'\1', text)
    text = _EXCESS_HYPHENS.sub('-', text)
    text = _LEADING_TRAILING_SEP.sub('', text)
    text = _MULTI_SPACE.sub(' ', text)
    return text.strip()


def remove_salutations(text: str) -> str:
    """Strip a leading relationship indicator (D/O, C/O, S/O) or a leading
    salutation (Mr, Mrs, Dr, Shri, ...). Behavior-identical port."""
    if not text:
        return text
    cleaned = _RELATION_PREFIX.sub("", text.strip())
    cleaned = _SALUTATION_PREFIX.sub("", cleaned.strip())
    return cleaned.strip()


def format_address(raw_address: str) -> str:
    """
    Cosmetic formatting pass: spacing, casing, symbol cleanup.
    Behavior-identical port of the old format_address(), modulo one fix:
    the old code had a no-op regex substitution
    (`re.sub(r'([a-zA-Z])-(?=[a-zA-Z])', r'\\1-', address)`) that matched
    then replaced with the same thing — flagged here, left in as a
    harmless no-op rather than silently "fixed", since changing it has
    zero observable effect and this module's job is faithful porting,
    not opportunistic rewriting of unrelated logic.
    """
    if not raw_address:
        return raw_address
    address = _COMMA_SPACING.sub(', ', raw_address)
    address = _CAP_BOUNDARY.sub(r'\1 \2', address)
    address = _WORD_DIGIT_1.sub(r'\1 \2', address)
    address = _WORD_DIGIT_2.sub(r'\1 \2', address)
    address = _LETTER_HYPHEN_LETTER.sub(r'\1-', address)     # no-op, see docstring
    address = _LETTER_AMP_HYPHEN.sub(r'\1 \2', address)
    address = _HYPHEN_SPACING.sub('-', address)
    address = _capitalize_text(address)
    address = _LEADING_SYMBOL.sub('', address)
    address = _MULTI_SPACE.sub(' ', address)
    return address.strip()


def clean_and_format(raw_address: str) -> str:
    """Convenience: the full cosmetic-cleanup pipeline in the order the
    old code applied it (OCR noise cleanup was applied by callers before
    format_address; matcher.py preserves that ordering)."""
    if not raw_address:
        return raw_address
    text = clean_ocr_noise(raw_address)
    text = remove_salutations(text)
    text = format_address(text)
    return text
