from __future__ import annotations

import hashlib
import re
from typing import Any

_TOKEN_RE = re.compile(r"[A-Za-z0-9_'-]+")
_SPACE_RE = re.compile(r"\s+")
_URL_RE = re.compile(r"https?://\S+")
_MENTION_RE = re.compile(r"[@#]")

_SIMHASH_BITS = 64


def normalize_text_for_dedupe(text: str) -> str:
    text = (text or "").lower()
    text = _URL_RE.sub(" ", text)
    text = _MENTION_RE.sub(" ", text)
    text = text.replace("’", "'").replace("“", '"').replace("”", '"')
    text = " ".join(_TOKEN_RE.findall(text))
    text = _SPACE_RE.sub(" ", text).strip()
    return text


def exact_text_hash(text: str) -> str:
    normalized = normalize_text_for_dedupe(text)
    return hashlib.sha1(normalized.encode("utf-8")).hexdigest()


def token_ngrams(text: str, n: int = 3) -> set[str]:
    normalized = normalize_text_for_dedupe(text)
    tokens = normalized.split()
    if not tokens:
        return set()
    if len(tokens) < n:
        return {" ".join(tokens)}
    return {" ".join(tokens[i : i + n]) for i in range(len(tokens) - n + 1)}


def jaccard_similarity(left: set[str], right: set[str]) -> float:
    if not left and not right:
        return 1.0
    if not left or not right:
        return 0.0
    intersection = len(left & right)
    union = len(left | right)
    if union == 0:
        return 0.0
    return intersection / union


def simhash_int(text: str) -> int:
    normalized = normalize_text_for_dedupe(text)
    tokens = normalized.split()
    if not tokens:
        return 0

    weights = [0] * _SIMHASH_BITS
    for token in tokens:
        digest = hashlib.blake2b(token.encode("utf-8"), digest_size=8).digest()
        value = int.from_bytes(digest, byteorder="big", signed=False)
        for bit_idx in range(_SIMHASH_BITS):
            bit = (value >> bit_idx) & 1
            weights[bit_idx] += 1 if bit else -1

    fingerprint = 0
    for bit_idx, score in enumerate(weights):
        if score >= 0:
            fingerprint |= 1 << bit_idx
    return fingerprint


def simhash_hex(text: str) -> str:
    return f"{simhash_int(text):016x}"


def hamming_distance_hex(left: str, right: str) -> int:
    try:
        left_int = int(left, 16)
        right_int = int(right, 16)
    except (TypeError, ValueError):
        return _SIMHASH_BITS
    return (left_int ^ right_int).bit_count()


def are_near_duplicates(
    left_doc: dict[str, Any],
    right_doc: dict[str, Any],
    *,
    left_ngrams: set[str] | None = None,
    right_ngrams: set[str] | None = None,
    max_hamming: int = 3,
    min_jaccard: float = 0.88,
    ngram_size: int = 3,
) -> bool:
    left_hash = str(left_doc.get("dedupe_text_hash", "") or "")
    right_hash = str(right_doc.get("dedupe_text_hash", "") or "")
    if left_hash and right_hash and left_hash == right_hash:
        return True

    left_simhash = str(left_doc.get("dedupe_simhash", "") or "")
    right_simhash = str(right_doc.get("dedupe_simhash", "") or "")
    if left_simhash and right_simhash:
        if hamming_distance_hex(left_simhash, right_simhash) > max_hamming:
            return False

    if left_ngrams is None:
        left_ngrams = token_ngrams(str(left_doc.get("text", "")), n=ngram_size)
    if right_ngrams is None:
        right_ngrams = token_ngrams(str(right_doc.get("text", "")), n=ngram_size)

    return jaccard_similarity(left_ngrams, right_ngrams) >= min_jaccard


def suppress_near_duplicates(
    candidate_ids: list[str],
    docs_by_id: dict[str, dict[str, Any]],
    *,
    target_n: int,
    max_hamming: int = 3,
    min_jaccard: float = 0.88,
    ngram_size: int = 3,
) -> tuple[list[str], list[str]]:
    selected: list[str] = []
    suppressed: list[str] = []
    ngram_cache: dict[str, set[str]] = {}

    def get_ngrams(doc_id: str) -> set[str]:
        if doc_id not in ngram_cache:
            ngram_cache[doc_id] = token_ngrams(
                str(docs_by_id[doc_id].get("text", "")),
                n=ngram_size,
            )
        return ngram_cache[doc_id]

    for doc_id in candidate_ids:
        current = docs_by_id[doc_id]
        current_ngrams = get_ngrams(doc_id)

        is_duplicate = False
        for kept_id in selected:
            kept = docs_by_id[kept_id]
            if are_near_duplicates(
                current,
                kept,
                left_ngrams=current_ngrams,
                right_ngrams=get_ngrams(kept_id),
                max_hamming=max_hamming,
                min_jaccard=min_jaccard,
                ngram_size=ngram_size,
            ):
                is_duplicate = True
                break

        if is_duplicate:
            suppressed.append(doc_id)
            continue

        selected.append(doc_id)
        if len(selected) >= target_n:
            break

    if len(selected) < target_n:
        for doc_id in suppressed:
            if doc_id not in selected:
                selected.append(doc_id)
            if len(selected) >= target_n:
                break

    return selected[:target_n], suppressed
