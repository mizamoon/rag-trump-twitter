from __future__ import annotations

import json
import re
from dataclasses import dataclass
from functools import lru_cache
from typing import Iterable

import numpy as np

from config import (
    ENABLE_EMBEDDING_QUERY_EXPANSION,
    KEYWORD_QUERY_WEIGHT,
    LOWER_QUERY_WEIGHT,
    ORIGINAL_QUERY_WEIGHT,
    TERM_EMBEDDINGS_PATH,
    TERM_EXPANSION_MAX_VARIANTS,
    TERM_EXPANSION_TOP_K,
    TERM_VOCAB_PATH,
)

_STOPWORDS = {
    "a", "an", "and", "are", "as", "at", "be", "by", "did", "do", "does",
    "donald", "for", "from", "how", "in", "is", "it", "of", "on", "or",
    "regarding", "say", "said", "says", "tell", "that", "the", "their", "them",
    "think", "this", "to", "trump", "tweets", "tweet", "what", "when", "where",
    "which", "who", "why", "with", "about", "comments", "comment", "statement",
    "statements",
}

_TOKEN_RE = re.compile(r"[A-Za-z0-9_'-]+")
_SPACE_RE = re.compile(r"\s+")

_SEMANTIC_TERM_WEIGHT = 0.58
_SEMANTIC_APPEND_TO_KEYWORD_WEIGHT = 0.62

_GENERIC_EXPANSION_TERMS = {
    "sex",
    "president",
    "presidential",
    "administration",
    "people",
    "country",
    "countries",
    "government",
    "media",
    "news",
    "today",
    "tomorrow",
    "tonight",
    "year",
    "years",
    "time",
    "times",
    "thing",
    "things",
    "support",
    "america",
    "american",
    "americans",
    "campaign",
    "election",
}

_ALLOWED_SEMANTIC_UNIGRAMS = {
    "gay",
    "lgbt",
    "lgbtq",
    "transgender",
    "abortion",
    "border",
    "immigration",
    "tariffs",
    "trade",
    "china",
    "iran",
    "ukraine",
    "nato",
    "covid",
    "isis",
}


@dataclass(frozen=True, slots=True)
class QueryVariant:
    text: str
    weight: float
    source: str


def normalize_query(query: str) -> str:
    q = (query or "").strip()
    q = _SPACE_RE.sub(" ", q)
    return q


def tokenize_for_bm25(text: str) -> list[str]:
    return [tok.lower() for tok in _TOKEN_RE.findall(text)]


def keyword_query(query: str) -> str:
    tokens = tokenize_for_bm25(query)
    keywords = [t for t in tokens if t not in _STOPWORDS]
    return " ".join(keywords).strip()


def _canonical_variant_key(text: str) -> str:
    return normalize_query(text).lower()


def _dedupe_keep_best(variants: list[QueryVariant]) -> list[QueryVariant]:
    deduped: dict[str, QueryVariant] = {}

    for variant in variants:
        cleaned_text = normalize_query(variant.text)
        if not cleaned_text:
            continue

        cleaned_variant = QueryVariant(
            text=cleaned_text,
            weight=float(variant.weight),
            source=variant.source,
        )
        key = _canonical_variant_key(cleaned_text)

        existing = deduped.get(key)
        if existing is None or cleaned_variant.weight > existing.weight:
            deduped[key] = cleaned_variant

    return list(deduped.values())


@lru_cache(maxsize=1)
def _load_term_expansion_artifacts() -> tuple[list[str] | None, np.ndarray | None]:
    if not TERM_VOCAB_PATH.exists() or not TERM_EMBEDDINGS_PATH.exists():
        return None, None

    with open(TERM_VOCAB_PATH, "r", encoding="utf-8") as f:
        vocab = json.load(f)

    embeddings = np.load(TERM_EMBEDDINGS_PATH).astype("float32")

    if len(vocab) != len(embeddings):
        return None, None

    return vocab, embeddings


def _embedding_guided_expansions(
    query: str,
    embed_backend,
    top_k: int = TERM_EXPANSION_TOP_K,
) -> list[str]:
    if not ENABLE_EMBEDDING_QUERY_EXPANSION:
        return []

    if embed_backend is None:
        return []

    vocab, term_embeddings = _load_term_expansion_artifacts()
    if vocab is None or term_embeddings is None or len(vocab) == 0:
        return []

    normalized_query = normalize_query(query)
    if not normalized_query:
        return []

    encoded = embed_backend.encode_queries([normalized_query])
    query_vec = np.asarray(encoded.dense[0], dtype="float32")

    scores = term_embeddings @ query_vec

    shortlist_size = min(len(vocab), max(top_k * 12, 32))
    shortlist_idx = np.argpartition(scores, -shortlist_size)[-shortlist_size:]
    shortlist_idx = shortlist_idx[np.argsort(scores[shortlist_idx])[::-1]]

    keyword_tokens = set(tokenize_for_bm25(keyword_query(normalized_query)))
    query_tokens = set(tokenize_for_bm25(normalized_query))

    expansions: list[str] = []

    for idx in shortlist_idx:
        term = str(vocab[int(idx)]).strip()
        if not term:
            continue

        term_tokens = tokenize_for_bm25(term)
        if not term_tokens:
            continue

        if any(tok in query_tokens for tok in term_tokens):
            continue

        if len(term_tokens) == 1:
            tok = term_tokens[0]

            if tok in _GENERIC_EXPANSION_TERMS:
                continue

            if len(keyword_tokens) >= 2 and tok not in _ALLOWED_SEMANTIC_UNIGRAMS:
                continue

        else:
            if len(keyword_tokens) >= 2 and not (set(term_tokens) & keyword_tokens):
                continue

            if any(tok in _GENERIC_EXPANSION_TERMS for tok in term_tokens):
                continue

        expansions.append(term)
        if len(expansions) >= top_k:
            break

    return expansions


def generate_weighted_query_variants(
    query: str,
    embed_backend=None,
) -> list[QueryVariant]:
    original = normalize_query(query)
    if not original:
        return []

    lowered = original.lower()
    keyword_only = keyword_query(original)

    raw_variants: list[QueryVariant] = [
        QueryVariant(text=original, weight=ORIGINAL_QUERY_WEIGHT, source="original"),
        QueryVariant(text=lowered, weight=LOWER_QUERY_WEIGHT, source="lower"),
    ]

    if keyword_only and _canonical_variant_key(keyword_only) not in {
        _canonical_variant_key(original),
        _canonical_variant_key(lowered),
    }:
        raw_variants.append(
            QueryVariant(
                text=keyword_only,
                weight=KEYWORD_QUERY_WEIGHT,
                source="keyword",
            )
        )

    semantic_terms = _embedding_guided_expansions(
        original,
        embed_backend=embed_backend,
    )

    for term in semantic_terms:
        term = normalize_query(term)
        if not term:
            continue

        term_tokens = tokenize_for_bm25(term)

        raw_variants.append(
            QueryVariant(
                text=term,
                weight=_SEMANTIC_TERM_WEIGHT,
                source="semantic_term",
            )
        )

        if len(term_tokens) >= 2 and keyword_only:
            raw_variants.append(
                QueryVariant(
                    text=f"{keyword_only} {term}",
                    weight=_SEMANTIC_APPEND_TO_KEYWORD_WEIGHT,
                    source="semantic_keyword_plus_term",
                )
            )

    variants = _dedupe_keep_best(raw_variants)
    return variants[:TERM_EXPANSION_MAX_VARIANTS]


def generate_query_variants(
    query: str,
    embed_backend=None,
) -> list[str]:
    return [
        variant.text
        for variant in generate_weighted_query_variants(query, embed_backend=embed_backend)
    ]


def weighted_rrf(
    rankings: Iterable[tuple[list[str], float]],
    k: int = 60,
) -> dict[str, float]:
    scores: dict[str, float] = {}
    for ranking, ranking_weight in rankings:
        if not ranking:
            continue
        for rank, doc_id in enumerate(ranking, start=1):
            scores[doc_id] = scores.get(doc_id, 0.0) + float(ranking_weight) / (k + rank)
    return scores


def rrf(rankings: Iterable[list[str]], k: int = 60) -> dict[str, float]:
    return weighted_rrf(((ranking, 1.0) for ranking in rankings), k=k)