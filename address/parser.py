"""
Deterministic address parser.

Implements Phase 1 section 5 and constraint #3: a MODULAR cascade of
independent extractors, not one giant regex or one huge function full of
special cases. Each extractor is a small, separately-testable function
with a single job. `parse_address()` orchestrates them in a fixed
PRECEDENCE ORDER and, critically, REMOVES matched text from the working
string as it goes -- this is what prevents e.g. a floor number from later
being picked up as a flat number (constraint #3's explicit example).

Precedence order and why:
  1. PIN            -- cheapest, highest-precision, closed 6-digit pattern.
                        Removing it first prevents its digits from ever
                        being mistaken for a flat/house/floor number.
  2. State           -- closed vocabulary, removing it prevents state names
                        from polluting the locality/city residual.
  3. Floor           -- MUST run before flat/unit extraction, specifically
                        so "3rd Floor" doesn't leave a bare "3" for the
                        flat extractor to grab (this is the exact ordering
                        constraint #3 calls out by name).
  4. Flat/Unit/House/Plot -- keyword-anchored first; unlabeled numeric
                        fallback only after anchored extraction has had
                        first claim on the text.
  5. Wing/Block      -- small closed alphabet, runs after flat so it isn't
                        confused with a flat's letter prefix.
  6. Building/Society -- suffix-anchored span capture.
  7. Street/Road     -- suffix-anchored span capture.
  8. Landmark        -- keyword-anchored span capture (supporting evidence
                        only downstream, but still extracted here so it
                        doesn't dilute the locality/city residual).
  9. Locality/City/District -- weakest deterministic tier; uses comma
                        segmentation as a soft signal over whatever
                        remains.
 10. Residual        -- everything left over, kept verbatim, NEVER dropped
                        (constraint #4).

Each extractor returns zero or one ParsedComponent (never fabricates a
value it didn't find) and reports its own ExtractionConfidence, so
downstream stages can weight ANCHORED extraction above INFERRED guesses.
"""

from __future__ import annotations

import re
from typing import List, Optional, Tuple

from .datamodel import ComponentType, ExtractionConfidence, ParsedAddress, ParsedComponent
from .canonical import (
    STATE_NAME_TO_CODE, STATE_CODE_SET, COUNTRY_MARKERS,
    BUILDING_SUFFIXES, STREET_SUFFIX_CANONICAL, WING_KEYWORDS,
    FLOOR_KEYWORDS, LANDMARK_KEYWORDS, UNIT_KEYWORDS,
)
from .numeric import extract_pin_code, extract_pin_code_ocr_tolerant, canonicalize_numeric_identifier
from .ocr_cleanup import clean_ocr_noise, remove_salutations, format_address


# ---------------------------------------------------------------------------
# Small shared helpers
# ---------------------------------------------------------------------------

def _blank_span(text: str, start: int, end: int) -> str:
    """Replace a matched span with spaces (not remove it), so remaining
    character offsets in the working string stay stable for subsequent
    extractors. This mirrors the old code's approach of string-replacing
    the PIN text out before further processing, generalized to every
    extractor."""
    return text[:start] + (' ' * (end - start)) + text[end:]


def _word_pattern(words) -> str:
    return r'(?:' + '|'.join(re.escape(w) for w in sorted(words, key=len, reverse=True)) + r')'


# ---------------------------------------------------------------------------
# 1. PIN extractor
# ---------------------------------------------------------------------------

def extract_pin(working_text: str) -> Tuple[Optional[ParsedComponent], str]:
    pin = extract_pin_code(working_text)
    if pin == -1:
        # Ambiguous: multiple distinct 6-digit runs. Old code aborted the
        # whole comparison here; we instead represent it explicitly as an
        # AMBIGUOUS component and let the scorer/policy layer decide what
        # to do with that (a strict policy can still choose to treat this
        # as fatal -- but that's now a business decision, not baked into
        # the parser). This is exactly the kind of "flag rather than
        # silently choose" situation called out in the brief.
        all_pins = re.findall(r'(?<!\d)\d{6}(?!\d)', working_text)
        comp = ParsedComponent(
            ComponentType.PIN, None, working_text, ExtractionConfidence.AMBIGUOUS,
            source="pin_extractor:multiple_candidates", candidates=list(set(all_pins)),
        )
        return comp, working_text
    if pin:
        m = re.search(re.escape(pin), working_text)
        span = m.span() if m else None
        cleaned = _blank_span(working_text, *span) if span else working_text
        comp = ParsedComponent(
            ComponentType.PIN, pin, pin, ExtractionConfidence.ANCHORED,
            source="pin_extractor:six_digit_isolated", span=span,
        )
        return comp, cleaned

    # OCR-tolerant fallback
    ocr_pin, raw_token = extract_pin_code_ocr_tolerant(working_text)
    if ocr_pin:
        m = re.search(re.escape(raw_token), working_text, flags=re.IGNORECASE)
        span = m.span() if m else None
        cleaned = _blank_span(working_text, *span) if span else working_text
        corrected_chars = sum(1 for a, b in zip(raw_token.upper(), ocr_pin) if a != b)
        comp = ParsedComponent(
            ComponentType.PIN, ocr_pin, raw_token, ExtractionConfidence.ANCHORED,
            source="pin_extractor:ocr_tolerant", span=span,
            ocr_corrected=corrected_chars > 0, ocr_corrected_chars=corrected_chars,
        )
        return comp, cleaned

    return None, working_text


# ---------------------------------------------------------------------------
# 2. State extractor
# ---------------------------------------------------------------------------

_STATE_NAME_PATTERN = re.compile(
    r'\b(' + '|'.join(re.escape(k) for k in sorted(STATE_NAME_TO_CODE.keys(), key=len, reverse=True)) + r')\b',
    re.IGNORECASE,
)
_STATE_CODE_PATTERN = re.compile(
    r'\b(' + '|'.join(re.escape(c.upper()) for c in STATE_CODE_SET) + r')\b'
)


def extract_state(working_text: str) -> Tuple[Optional[ParsedComponent], str]:
    m = _STATE_NAME_PATTERN.search(working_text)
    if m:
        code = STATE_NAME_TO_CODE[m.group(1).lower()]
        cleaned = _blank_span(working_text, *m.span())
        comp = ParsedComponent(
            ComponentType.STATE, code, m.group(1), ExtractionConfidence.ANCHORED,
            source="state_extractor:full_name", span=m.span(),
        )
        return comp, cleaned

    # bare 2-letter state code as its own token (lower precision -- lots of
    # 2-letter tokens exist that aren't state codes, so this is INFERRED,
    # not ANCHORED)
    m2 = _STATE_CODE_PATTERN.search(working_text)
    if m2:
        cleaned = _blank_span(working_text, *m2.span())
        comp = ParsedComponent(
            ComponentType.STATE, m2.group(1).lower(), m2.group(1), ExtractionConfidence.INFERRED,
            source="state_extractor:bare_code", span=m2.span(),
        )
        return comp, cleaned

    return None, working_text


# ---------------------------------------------------------------------------
# 3. Floor extractor -- MUST run before flat/unit (see module docstring)
# ---------------------------------------------------------------------------

_FLOOR_PATTERN = re.compile(
    # NOTE: the ordinal suffix (st/nd/rd/th) is matched as OPTIONALLY
    # space-separated from its digit ("3rd" or "3 rd") because the shared
    # cosmetic cleanup pass (ocr_cleanup.format_address, ported from the
    # old code) splits letter-digit boundaries -- "3rd" -> "3 rd" -- before
    # this parser ever sees the text. Discovered via testing, not assumed;
    # documenting here rather than silently special-casing it invisibly.
    r'\b(\d+\s?(?:st|nd|rd|th)?\s*(?:' + _word_pattern(FLOOR_KEYWORDS) + r')|'
    r'(?:' + _word_pattern(FLOOR_KEYWORDS) + r')\s*(?:no\.?\s*)?(\d+))\b',
    re.IGNORECASE,
)
_GROUND_FLOOR_PATTERN = re.compile(r'\bground\s*(?:' + _word_pattern(FLOOR_KEYWORDS) + r')\b', re.IGNORECASE)


def extract_floor(working_text: str) -> Tuple[Optional[ParsedComponent], str]:
    m = _GROUND_FLOOR_PATTERN.search(working_text)
    if m:
        cleaned = _blank_span(working_text, *m.span())
        comp = ParsedComponent(
            ComponentType.FLOOR, "0", m.group(0), ExtractionConfidence.ANCHORED,
            source="floor_extractor:ground", span=m.span(),
        )
        return comp, cleaned

    m = _FLOOR_PATTERN.search(working_text)
    if m:
        digits = m.group(1) if m.group(1) and m.group(1)[0].isdigit() else m.group(2)
        digits = re.sub(r'\D', '', digits) if digits else None
        if digits:
            cleaned = _blank_span(working_text, *m.span())
            comp = ParsedComponent(
                ComponentType.FLOOR, str(int(digits)), m.group(0), ExtractionConfidence.ANCHORED,
                source="floor_extractor:keyword_anchor", span=m.span(),
            )
            return comp, cleaned

    return None, working_text


# ---------------------------------------------------------------------------
# 4. Flat / Unit / House / Plot extractor
# ---------------------------------------------------------------------------
# Fixes the Phase-1-identified weakness: the old regex matched ANY
# digit-bearing token because the keyword group was optional. Here,
# keyword-anchored extraction is tried FIRST and exclusively; only if
# nothing anchors do we fall back to a single unlabeled numeric candidate,
# explicitly marked INFERRED (or AMBIGUOUS if there are several).

_UNIT_KEYWORD_PATTERN = _word_pattern(set(UNIT_KEYWORDS.keys()) - {"no"})  # "no" alone is too ambiguous as a standalone anchor
_FLAT_ANCHORED_PATTERN = re.compile(
    r'\b(?:' + _UNIT_KEYWORD_PATTERN + r')\.?\s*(?:no\.?)?\s*[-:#]?\s*([A-Z]?[-/]?\d+[A-Z]?)\b',
    re.IGNORECASE,
)
# a leading bare unit pattern e.g. "A-401, Sky Towers..." at the very start
_LEADING_UNIT_PATTERN = re.compile(r'^\s*([A-Z]-?\d+[A-Z]?|\d+[A-Z]?)\s*[,]', re.IGNORECASE)
_BARE_NUMERIC_TOKEN = re.compile(r'\b([A-Z]?-?\d{1,5}[A-Z]?)\b', re.IGNORECASE)

# Numbers immediately preceded by these words are almost never a flat/unit
# identifier in Indian addresses -- they're a locality/layout numbering
# scheme (Sector 5, Phase 2, Block 3-as-area-not-wing, Extension 12).
# Discovered via testing ("Sector 5, Gurgaon" was being misread as flat=5)
# and handled explicitly rather than silently left as a latent false
# positive: the unanchored-numeric fallback must not claim a number that
# is itself anchored to a DIFFERENT, non-unit keyword.
_NON_UNIT_NUMERIC_CONTEXT = re.compile(
    r'\b(?:sector|phase|extension|ext|zone|stage)\.?\s*\d{1,5}[A-Z]?\b', re.IGNORECASE
)


def extract_flat(working_text: str) -> Tuple[Optional[ParsedComponent], str]:
    m = _FLAT_ANCHORED_PATTERN.search(working_text)
    if m:
        raw = m.group(1)
        canon = canonicalize_numeric_identifier(raw).canonical
        cleaned = _blank_span(working_text, *m.span())
        comp = ParsedComponent(
            ComponentType.FLAT, canon, raw, ExtractionConfidence.ANCHORED,
            source="flat_extractor:keyword_anchor", span=m.span(),
        )
        return comp, cleaned

    m = _LEADING_UNIT_PATTERN.match(working_text)
    if m:
        raw = m.group(1)
        canon = canonicalize_numeric_identifier(raw).canonical
        cleaned = _blank_span(working_text, m.start(1), m.end(1))
        comp = ParsedComponent(
            ComponentType.FLAT, canon, raw, ExtractionConfidence.INFERRED,
            source="flat_extractor:leading_position", span=m.span(1),
        )
        return comp, cleaned

    # spans claimed by a non-unit numeric context (Sector 5, Phase 2, ...)
    # are off-limits to the unanchored flat fallback below
    excluded_spans = [mm.span() for mm in _NON_UNIT_NUMERIC_CONTEXT.finditer(working_text)]

    def _in_excluded_span(match) -> bool:
        return any(s <= match.start() and match.end() <= e for s, e in excluded_spans)

    # unanchored fallback: collect remaining bare numeric-ish tokens
    candidates = [
        mm.group(1) for mm in _BARE_NUMERIC_TOKEN.finditer(working_text)
        if not _in_excluded_span(mm)
    ]
    # ignore tokens that are actually just plain long numbers unlikely to be
    # unit identifiers (e.g. leftover partial PIN fragments) -- keep it
    # simple and conservative rather than trying to guess further
    candidates = [c for c in candidates if len(re.sub(r'\D', '', c)) <= 5]

    if len(candidates) == 1:
        raw = candidates[0]
        canon = canonicalize_numeric_identifier(raw).canonical
        m2 = re.search(re.escape(raw), working_text)
        cleaned = _blank_span(working_text, *m2.span()) if m2 else working_text
        comp = ParsedComponent(
            ComponentType.FLAT, canon, raw, ExtractionConfidence.INFERRED,
            source="flat_extractor:unanchored_single_candidate", span=m2.span() if m2 else None,
        )
        return comp, cleaned

    if len(candidates) > 1:
        # AMBIGUOUS: multiple unlabeled numeric candidates, no reliable way
        # to choose one (Phase 1 section 8's AMBIGUOUS state, explicitly
        # modeled rather than forced via the old sort-and-truncate trick).
        comp = ParsedComponent(
            ComponentType.FLAT, None, working_text, ExtractionConfidence.AMBIGUOUS,
            source="flat_extractor:multiple_unanchored_candidates", candidates=candidates,
        )
        return comp, working_text  # do not blank anything -- we didn't confidently claim any of it

    return None, working_text


# ---------------------------------------------------------------------------
# 5. Wing / Block extractor
# ---------------------------------------------------------------------------
# Ported logic from the old extract_wing_info: single letter near a
# wing/block keyword, or a letter joined to a number by / or -.

_WING_KEYWORD_PATTERN = re.compile(
    r'\b(?:' + _word_pattern(WING_KEYWORDS) + r')\.?\s*[-:#]?\s*([A-H])\b', re.IGNORECASE
)
_WING_PREFIX_PATTERN = re.compile(r'\b([A-H])[-/]\d', re.IGNORECASE)


def extract_wing(working_text: str) -> Tuple[Optional[ParsedComponent], str]:
    m = _WING_KEYWORD_PATTERN.search(working_text)
    if m:
        letter = m.group(1).upper()
        cleaned = _blank_span(working_text, *m.span())
        comp = ParsedComponent(
            ComponentType.WING, letter, m.group(0), ExtractionConfidence.ANCHORED,
            source="wing_extractor:keyword_anchor", span=m.span(),
        )
        return comp, cleaned

    m = _WING_PREFIX_PATTERN.search(working_text)
    if m:
        letter = m.group(1).upper()
        # don't blank this span -- the digit part still needs to be
        # available to the flat extractor in cases where ordering means
        # wing runs after flat in a given pipeline configuration; matcher.py
        # controls actual call order (flat before wing, per precedence list)
        comp = ParsedComponent(
            ComponentType.WING, letter, m.group(0), ExtractionConfidence.INFERRED,
            source="wing_extractor:letter_number_prefix", span=m.span(1),
        )
        return comp, working_text

    return None, working_text


# ---------------------------------------------------------------------------
# 6. Building / Society extractor
# ---------------------------------------------------------------------------

_BUILDING_SUFFIX_PATTERN = re.compile(
    r'((?:[A-Z][A-Za-z0-9\'\.]*\s+){0,3}(?:' + _word_pattern(BUILDING_SUFFIXES) + r'))\b',
    re.IGNORECASE,
)


def extract_building(working_text: str) -> Tuple[Optional[ParsedComponent], str]:
    m = _BUILDING_SUFFIX_PATTERN.search(working_text)
    if m:
        raw = m.group(1).strip()
        cleaned = _blank_span(working_text, *m.span())
        comp = ParsedComponent(
            ComponentType.BUILDING, raw.upper(), raw, ExtractionConfidence.ANCHORED,
            source="building_extractor:suffix_anchor", span=m.span(),
        )
        return comp, cleaned
    return None, working_text


# ---------------------------------------------------------------------------
# 7. Street / Road extractor
# ---------------------------------------------------------------------------

_ALL_STREET_WORDS = set(STREET_SUFFIX_CANONICAL.keys()) | set(STREET_SUFFIX_CANONICAL.values())
_STREET_SUFFIX_PATTERN = re.compile(
    r'((?:[A-Z][A-Za-z0-9\'\.]*\s+){0,4}(?:' + _word_pattern(_ALL_STREET_WORDS) + r'))\b',
    re.IGNORECASE,
)


def extract_street(working_text: str) -> Tuple[Optional[ParsedComponent], str]:
    m = _STREET_SUFFIX_PATTERN.search(working_text)
    if m:
        raw = m.group(1).strip()
        cleaned = _blank_span(working_text, *m.span())
        comp = ParsedComponent(
            ComponentType.STREET, raw.upper(), raw, ExtractionConfidence.ANCHORED,
            source="street_extractor:suffix_anchor", span=m.span(),
        )
        return comp, cleaned
    return None, working_text


# ---------------------------------------------------------------------------
# 8. Landmark extractor
# ---------------------------------------------------------------------------

_LANDMARK_PATTERN = re.compile(
    r'\b(?:' + _word_pattern(LANDMARK_KEYWORDS) + r')\.?\s+([A-Za-z0-9\'\.\s]{2,40}?)(?=,|$)',
    re.IGNORECASE,
)


def extract_landmark(working_text: str) -> Tuple[Optional[ParsedComponent], str]:
    m = _LANDMARK_PATTERN.search(working_text)
    if m:
        raw = m.group(1).strip()
        if raw:
            cleaned = _blank_span(working_text, *m.span())
            comp = ParsedComponent(
                ComponentType.LANDMARK, raw.upper(), raw, ExtractionConfidence.ANCHORED,
                source="landmark_extractor:keyword_anchor", span=m.span(),
            )
            return comp, cleaned
    return None, working_text


# ---------------------------------------------------------------------------
# 9. Locality / City / District -- weakest deterministic tier
# ---------------------------------------------------------------------------
# No reliable keyword anchors exist for this tier in Indian addresses.
# Deterministic best-effort: use comma-segmentation over whatever remains.
# The LAST non-empty comma segment before end-of-string is treated as the
# strongest city/locality candidate (Indian addresses conventionally end
# ...Locality, City - PIN, with PIN/state already removed by now). This is
# explicitly INFERRED confidence, never ANCHORED -- it is a positional
# heuristic, not a keyword match.

def extract_locality_city(working_text: str) -> Tuple[Optional[ParsedComponent], Optional[ParsedComponent], str]:
    segments = [s.strip() for s in working_text.split(',') if s.strip()]
    if not segments:
        return None, None, working_text

    city_comp = None
    locality_comp = None

    if len(segments) >= 1:
        city_raw = segments[-1]
        if city_raw and re.search(r'[A-Za-z]', city_raw):
            city_comp = ParsedComponent(
                ComponentType.CITY, city_raw.upper(), city_raw, ExtractionConfidence.INFERRED,
                source="locality_city_extractor:last_comma_segment",
            )
    if len(segments) >= 2:
        locality_raw = segments[-2]
        if locality_raw and re.search(r'[A-Za-z]', locality_raw):
            locality_comp = ParsedComponent(
                ComponentType.LOCALITY, locality_raw.upper(), locality_raw, ExtractionConfidence.INFERRED,
                source="locality_city_extractor:second_last_comma_segment",
            )

    # Remove only what we actually claimed (last one or two segments);
    # everything before stays in the working text for the residual bucket.
    claimed = min(2, len(segments))
    remaining_segments = segments[:-claimed] if claimed else segments
    cleaned = ', '.join(remaining_segments)
    return city_comp, locality_comp, cleaned


# ---------------------------------------------------------------------------
# Orchestration
# ---------------------------------------------------------------------------

def parse_address(raw_address: str) -> ParsedAddress:
    """
    Run the full extractor cascade in precedence order, producing a
    ParsedAddress. Never raises on malformed/partial input -- missing
    components are simply absent, and unrecognized text always survives
    into residual_text (constraint #4).
    """
    if not raw_address or not raw_address.strip():
        return ParsedAddress(original_text=raw_address or "", components={}, residual_text="")

    # IMPORTANT ORDERING NOTE (discovered via testing, not assumed):
    # PIN extraction -- specifically the OCR-tolerant path -- must run
    # BEFORE format_address()'s letter/digit-boundary splitting
    # (e.g. "4OOO69" -> "4 OOO 69"), because that splitting destroys the
    # contiguous 6-character run the OCR-tolerant PIN pattern needs to
    # match. Salutation/noise cleanup (clean_ocr_noise, remove_salutations)
    # is safe to run first since it doesn't touch digit-letter boundaries.
    # This means PIN extraction runs on lightly-cleaned text, and the
    # heavier cosmetic formatting (format_address) is applied AFTER, to
    # whatever's left once the PIN has been pulled out and blanked.
    lightly_cleaned = remove_salutations(clean_ocr_noise(raw_address))

    components = {}

    pin_comp, working = extract_pin(lightly_cleaned)
    if pin_comp:
        components[ComponentType.PIN] = pin_comp

    # now apply the full cosmetic formatting pass to what's left (PIN
    # already blanked out, so its digits can't be re-split or re-matched
    # by anything downstream)
    working = format_address(working)

    state_comp, working = extract_state(working)
    if state_comp:
        components[ComponentType.STATE] = state_comp

    floor_comp, working = extract_floor(working)  # MUST precede flat -- see docstring
    if floor_comp:
        components[ComponentType.FLOOR] = floor_comp

    flat_comp, working = extract_flat(working)
    if flat_comp:
        components[ComponentType.FLAT] = flat_comp

    wing_comp, working = extract_wing(working)
    if wing_comp:
        components[ComponentType.WING] = wing_comp

    building_comp, working = extract_building(working)
    if building_comp:
        components[ComponentType.BUILDING] = building_comp

    street_comp, working = extract_street(working)
    if street_comp:
        components[ComponentType.STREET] = street_comp

    landmark_comp, working = extract_landmark(working)
    if landmark_comp:
        components[ComponentType.LANDMARK] = landmark_comp

    city_comp, locality_comp, working = extract_locality_city(working)
    if city_comp:
        components[ComponentType.CITY] = city_comp
    if locality_comp:
        components[ComponentType.LOCALITY] = locality_comp

    # whatever remains -- never dropped
    residual = re.sub(r'[\s,]+', ' ', working).strip(' ,')

    return ParsedAddress(original_text=raw_address, components=components, residual_text=residual)
