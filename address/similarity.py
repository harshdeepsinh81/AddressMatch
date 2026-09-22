"""
Deterministic string-similarity algorithm primitives.

This module implements exactly the trimmed set approved in Phase 1 /
constraint #10 -- no more:

  * canonical exact match            -> see numeric.py for numeric fields;
                                         exact_match() here for text fields
  * normalized Levenshtein distance  -> levenshtein_ratio()
  * Jaro-Winkler                     -> jaro_winkler()
  * token Jaccard                    -> token_jaccard()
  * containment                      -> containment_ratio()
  * character n-gram similarity      -> char_ngram_dice()
  * residual token-based similarity  -> token_set_ratio() (kept, demoted to
                                         fallback/residual-text use only --
                                         see comparator.py)

LCS, Damerau-Levenshtein, weighted Jaccard, phonetic algorithms etc. are
deliberately NOT implemented, per constraint #10 -- add only on a
demonstrated failure case against real data, not speculatively.

IMPLEMENTATION NOTE (flagged, not hidden): the approved Phase 1 design
called for `rapidfuzz`/`jellyfish` as the underlying implementations of
these algorithms. Neither package is installable in this sandbox (no
network egress for pip). Every function below is therefore a small,
straightforward, pure-Python/stdlib implementation of the SAME algorithm
-- the input/output contract (function name, arguments, 0.0-1.0 range)
is written so that swapping the body for a rapidfuzz/jellyfish call later
is a one-function, no-caller-changes edit. This does not affect
determinism (these are still the same textbook algorithms, not a
different technique) but it does mean these implementations are not as
heavily battle-tested/optimized as the C-backed libraries would be --
worth re-validating against rapidfuzz's output on your real data in
Phase 3 if exact numeric parity matters.
"""

from __future__ import annotations

from typing import List, Set


# ---------------------------------------------------------------------------
# Exact match (text fields)
# ---------------------------------------------------------------------------

def exact_match(a: str, b: str) -> bool:
    if a is None or b is None:
        return False
    return a.strip().upper() == b.strip().upper()


# ---------------------------------------------------------------------------
# Levenshtein distance / normalized ratio
# ---------------------------------------------------------------------------

def _levenshtein_distance(a: str, b: str) -> int:
    if a == b:
        return 0
    if not a:
        return len(b)
    if not b:
        return len(a)
    prev_row = list(range(len(b) + 1))
    for i, ca in enumerate(a, start=1):
        curr_row = [i] + [0] * len(b)
        for j, cb in enumerate(b, start=1):
            cost = 0 if ca == cb else 1
            curr_row[j] = min(
                curr_row[j - 1] + 1,      # insertion
                prev_row[j] + 1,           # deletion
                prev_row[j - 1] + cost,     # substitution
            )
        prev_row = curr_row
    return prev_row[-1]


def levenshtein_ratio(a: str, b: str) -> float:
    """Normalized similarity in [0.0, 1.0]: 1 - (edit_distance / max_len).
    Suitable for short textual fields (building/street/locality names) --
    NOT for long residual text or numeric identifiers (see module/Phase 1
    guidance: numeric fields use numeric.py's exact-match logic instead)."""
    if a is None or b is None:
        return 0.0
    a, b = a.strip().upper(), b.strip().upper()
    if not a and not b:
        return 1.0
    if not a or not b:
        return 0.0
    max_len = max(len(a), len(b))
    if max_len == 0:
        return 1.0
    dist = _levenshtein_distance(a, b)
    return 1.0 - (dist / max_len)


# ---------------------------------------------------------------------------
# Jaro-Winkler
# ---------------------------------------------------------------------------

def _jaro_similarity(a: str, b: str) -> float:
    if a == b:
        return 1.0
    len_a, len_b = len(a), len(b)
    if len_a == 0 or len_b == 0:
        return 0.0

    match_distance = max(len_a, len_b) // 2 - 1
    match_distance = max(match_distance, 0)

    a_matches = [False] * len_a
    b_matches = [False] * len_b

    matches = 0
    transpositions = 0

    for i in range(len_a):
        start = max(0, i - match_distance)
        end = min(i + match_distance + 1, len_b)
        for j in range(start, end):
            if b_matches[j] or a[i] != b[j]:
                continue
            a_matches[i] = True
            b_matches[j] = True
            matches += 1
            break

    if matches == 0:
        return 0.0

    k = 0
    for i in range(len_a):
        if not a_matches[i]:
            continue
        while not b_matches[k]:
            k += 1
        if a[i] != b[k]:
            transpositions += 1
        k += 1
    transpositions //= 2

    return (matches / len_a + matches / len_b + (matches - transpositions) / matches) / 3.0


def jaro_winkler(a: str, b: str, prefix_weight: float = 0.1, max_prefix: int = 4) -> float:
    """
    Jaro-Winkler similarity in [0.0, 1.0]. Chosen (Phase 1 section 7) as a
    complement to Levenshtein specifically for short name fields, since it
    weights common-prefix agreement more heavily -- useful for Indian
    place-name transliteration variance where the start of a word tends
    to be stable and later syllables vary more.
    """
    if a is None or b is None:
        return 0.0
    a, b = a.strip().upper(), b.strip().upper()
    if not a and not b:
        return 1.0
    if not a or not b:
        return 0.0
    jaro = _jaro_similarity(a, b)
    prefix_len = 0
    for ca, cb in zip(a, b):
        if ca != cb:
            break
        prefix_len += 1
        if prefix_len == max_prefix:
            break
    return jaro + prefix_len * prefix_weight * (1 - jaro)


# ---------------------------------------------------------------------------
# Token-based: Jaccard, containment, token_set_ratio (residual fallback)
# ---------------------------------------------------------------------------

def _tokenize(text: str) -> List[str]:
    if not text:
        return []
    return [t for t in text.strip().upper().split() if t]


def token_jaccard(a: str, b: str) -> float:
    """|intersection| / |union| of token sets. Order-independent, the core
    comparator for building/street/locality spans (Phase 1 section 6)."""
    tokens_a, tokens_b = set(_tokenize(a)), set(_tokenize(b))
    if not tokens_a and not tokens_b:
        return 1.0
    if not tokens_a or not tokens_b:
        return 0.0
    intersection = tokens_a & tokens_b
    union = tokens_a | tokens_b
    return len(intersection) / len(union) if union else 0.0


def containment_ratio(a: str, b: str) -> float:
    """
    |intersection| / |smaller token set|. Unlike Jaccard, does not penalize
    the LARGER side for having extra tokens -- important for locality/city
    fields where granularity legitimately differs ("Andheri" fully
    contained in "Andheri East, Mumbai"), per Phase 1 section 7.
    """
    tokens_a, tokens_b = set(_tokenize(a)), set(_tokenize(b))
    if not tokens_a and not tokens_b:
        return 1.0
    if not tokens_a or not tokens_b:
        return 0.0
    smaller = min(len(tokens_a), len(tokens_b))
    if smaller == 0:
        return 0.0
    return len(tokens_a & tokens_b) / smaller


def token_set_ratio(a: str, b: str) -> float:
    """
    Bag-of-words comparator, functionally equivalent in spirit to
    fuzzywuzzy/rapidfuzz's token_set_ratio: build the intersection and
    per-side-difference token sets, compare the resulting strings with a
    Levenshtein-based ratio, take the best of the three combinations.

    Per Phase 1's conclusion, this is DEMOTED to a fallback comparator for
    genuinely unstructured residual text only (see comparator.py) -- it is
    no longer the primary address-matching signal.
    """
    tokens_a, tokens_b = set(_tokenize(a)), set(_tokenize(b))
    if not tokens_a and not tokens_b:
        return 1.0
    if not tokens_a or not tokens_b:
        return 0.0

    intersection = tokens_a & tokens_b
    diff_a = tokens_a - tokens_b
    diff_b = tokens_b - tokens_a

    sorted_intersection = ' '.join(sorted(intersection))
    combined_a = ' '.join(sorted(intersection | diff_a))
    combined_b = ' '.join(sorted(intersection | diff_b))

    scores = [
        levenshtein_ratio(sorted_intersection, combined_a),
        levenshtein_ratio(sorted_intersection, combined_b),
        levenshtein_ratio(combined_a, combined_b),
    ]
    return max(scores)


# ---------------------------------------------------------------------------
# Character n-gram similarity (Dice coefficient)
# ---------------------------------------------------------------------------

def _char_ngrams(text: str, n: int = 3) -> Set[str]:
    if not text:
        return set()
    padded = text.strip().upper().replace(' ', '')
    if len(padded) < n:
        return {padded} if padded else set()
    return {padded[i:i + n] for i in range(len(padded) - n + 1)}


def char_ngram_dice(a: str, b: str, n: int = 3) -> float:
    """
    Dice coefficient over character n-grams (trigrams by default).
    Chosen over character n-gram cosine (Phase 1 section 7: redundant,
    pick one) for simplicity/explainability. Useful specifically for
    locality/city names where word-boundary tokenization itself is
    unreliable due to spacing/concatenation inconsistencies (e.g.
    "Andheri West" vs "AndheriWest").
    """
    ngrams_a, ngrams_b = _char_ngrams(a, n), _char_ngrams(b, n)
    if not ngrams_a and not ngrams_b:
        return 1.0
    if not ngrams_a or not ngrams_b:
        return 0.0
    intersection = ngrams_a & ngrams_b
    return (2 * len(intersection)) / (len(ngrams_a) + len(ngrams_b))
