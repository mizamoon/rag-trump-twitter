from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
from rank_bm25 import BM25Okapi
from sentence_transformers import CrossEncoder

from config import (
    BGE_SPARSE_RRF_WEIGHT,
    BGE_SPARSE_TOP_N,
    BM25_RRF_WEIGHT,
    BM25_TOP_N,
    DEDUPE_MAX_HAMMING,
    DEDUPE_MIN_JACCARD,
    DENSE_MATRIX_PATH,
    DENSE_RRF_WEIGHT,
    DENSE_TOP_N,
    DEDUP_NGRAM_SIZE,
    DOCSTORE_PATH,
    EMBED_MODEL,
    ENABLE_NEAR_DUPLICATE_SUPPRESSION,
    FUSION_TOP_N,
    INDEX_META_PATH,
    NORMALIZE_EMBEDDINGS,
    RERANK_BATCH_SIZE,
    RERANK_MODEL,
    RERANK_TOP_N,
    RRF_K,
    SPARSE_WEIGHTS_PATH,
)
from rag.dedupe import suppress_near_duplicates
from rag.embeddings import EmbeddingBackend, get_embedding_backend, is_bge_m3_model
from rag.query_utils import (
    QueryVariant,
    generate_weighted_query_variants,
    normalize_query,
    tokenize_for_bm25,
    weighted_rrf,
)


@dataclass(slots=True)
class RetrievedDoc:
    page_content: str
    metadata: dict[str, Any]


# --- global variables ---
_encoder: EmbeddingBackend | None = None
_reranker: CrossEncoder | None = None
_docstore: list[dict[str, Any]] | None = None
_doc_by_id: dict[str, dict[str, Any]] | None = None
_dense_matrix: np.ndarray | None = None
_bm25: BM25Okapi | None = None
_bm25_tokens: list[list[str]] | None = None
_sparse_weights: list[dict[int, float]] | None = None
_sparse_postings: dict[int, list[tuple[int, float]]] | None = None
_index_validated = False


# --- embedder / reranker ---
def get_encoder() -> EmbeddingBackend:
    global _encoder
    if _encoder is None:
        _encoder = get_embedding_backend(EMBED_MODEL)
    return _encoder


def get_reranker() -> CrossEncoder:
    global _reranker
    if _reranker is None:
        _reranker = CrossEncoder(RERANK_MODEL)
    return _reranker


# --- index loaders ---
def _validate_paths() -> None:
    required = [DOCSTORE_PATH, DENSE_MATRIX_PATH, INDEX_META_PATH]
    if is_bge_m3_model(EMBED_MODEL):
        required.append(SPARSE_WEIGHTS_PATH)

    missing = [str(path) for path in required if not Path(path).exists()]
    if missing:
        joined = "\n".join(missing)
        raise FileNotFoundError(
            "Hybrid index artifacts are missing. Run `python build_index.py` first.\n"
            f"Missing:\n{joined}"
        )


def validate_index_compatibility() -> None:
    _validate_paths()
    with open(INDEX_META_PATH, "r", encoding="utf-8") as f:
        meta = json.load(f)

    built_with = meta.get("embed_model")
    if built_with != EMBED_MODEL:
        raise ValueError(
            f"Index was built with EMBED_MODEL='{built_with}', but current EMBED_MODEL='{EMBED_MODEL}'. "
            "Rebuild the index."
        )

    if bool(meta.get("normalize_embeddings")) != bool(NORMALIZE_EMBEDDINGS):
        raise ValueError("Embedding normalization setting changed. Rebuild the index.")

    if is_bge_m3_model(EMBED_MODEL):
        if meta.get("embed_backend") != "bge-m3":
            raise ValueError(
                "Current EMBED_MODEL expects native BGE-M3 indexing, but the stored index "
                "was not built with embed_backend='bge-m3'. Rebuild the index."
            )
        if not bool(meta.get("sparse_available")):
            raise ValueError(
                "Current EMBED_MODEL expects sparse lexical weights, but the stored index "
                "does not have them. Rebuild the index."
            )


def ensure_index_ready() -> None:
    global _index_validated
    if not _index_validated:
        validate_index_compatibility()
        _index_validated = True


def get_docstore() -> list[dict[str, Any]]:
    global _docstore, _doc_by_id
    if _docstore is None:
        ensure_index_ready()
        docs: list[dict[str, Any]] = []
        with open(DOCSTORE_PATH, "r", encoding="utf-8") as f:
            for line in f:
                docs.append(json.loads(line))
        _docstore = docs
        _doc_by_id = {doc["doc_id"]: doc for doc in docs}
    return _docstore


def get_dense_matrix() -> np.ndarray:
    global _dense_matrix
    if _dense_matrix is None:
        ensure_index_ready()
        _dense_matrix = np.load(DENSE_MATRIX_PATH)
        if _dense_matrix.dtype != np.float32:
            _dense_matrix = _dense_matrix.astype("float32")
    return _dense_matrix


def get_bm25() -> tuple[BM25Okapi, list[list[str]]]:
    global _bm25, _bm25_tokens
    if _bm25 is None or _bm25_tokens is None:
        docs = get_docstore()
        _bm25_tokens = [tokenize_for_bm25(doc["text"]) for doc in docs]
        _bm25 = BM25Okapi(_bm25_tokens)
    return _bm25, _bm25_tokens


def get_sparse_weights() -> list[dict[int, float]]:
    global _sparse_weights
    if _sparse_weights is None:
        ensure_index_ready()
        weights: list[dict[int, float]] = []
        with open(SPARSE_WEIGHTS_PATH, "r", encoding="utf-8") as f:
            for line in f:
                payload = json.loads(line)
                raw_weights = payload.get("lexical_weights", {}) or {}
                weights.append({int(k): float(v) for k, v in raw_weights.items()})
        _sparse_weights = weights
    return _sparse_weights


def get_sparse_postings() -> dict[int, list[tuple[int, float]]]:
    global _sparse_postings
    if _sparse_postings is None:
        postings: dict[int, list[tuple[int, float]]] = {}
        for doc_global_idx, weights in enumerate(get_sparse_weights()):
            for token_id, token_weight in weights.items():
                postings.setdefault(int(token_id), []).append((doc_global_idx, float(token_weight)))
        _sparse_postings = postings
    return _sparse_postings


# --- filtering by date ---
def in_date_range(doc_date: str, date_from: str | None = None, date_to: str | None = None) -> bool:
    if not doc_date:
        return False if (date_from or date_to) else True
    if date_from and doc_date < date_from:
        return False
    if date_to and doc_date > date_to:
        return False
    return True


def filtered_doc_indices(date_from: str | None = None, date_to: str | None = None) -> np.ndarray:
    docs = get_docstore()
    keep = [
        idx
        for idx, doc in enumerate(docs)
        if in_date_range(str(doc.get("date", "")), date_from=date_from, date_to=date_to)
    ]
    return np.asarray(keep, dtype=np.int32)


# --- utilities ---
def _top_indices(scores: np.ndarray, top_n: int) -> np.ndarray:
    if scores.size == 0:
        return np.asarray([], dtype=np.int32)
    top_n = min(top_n, scores.size)
    part = np.argpartition(scores, -top_n)[-top_n:]
    ordered = part[np.argsort(scores[part])[::-1]]
    return ordered.astype(np.int32)


def _encode_query_variants(
    query_variants: list[QueryVariant],
    encoder: EmbeddingBackend | None = None,
) -> tuple[np.ndarray, list[dict[int, float]] | None]:
    encoder = encoder or get_encoder()
    encoded = encoder.encode_queries([variant.text for variant in query_variants])
    return encoded.dense.astype("float32"), encoded.lexical_weights


# --- dense retrieval ---
def dense_rankings(
    query_variants: list[QueryVariant],
    doc_idx: np.ndarray,
    query_dense: np.ndarray,
) -> tuple[list[list[str]], dict[str, dict[str, Any]]]:
    docs = get_docstore()
    matrix = get_dense_matrix()[doc_idx]
    rankings: list[list[str]] = []
    stats: dict[str, dict[str, Any]] = {}

    for variant, qvec in zip(query_variants, query_dense, strict=False):
        scores = matrix @ qvec
        order_local = _top_indices(scores, DENSE_TOP_N)

        ranked_ids: list[str] = []
        for rank, local_idx in enumerate(order_local, start=1):
            global_idx = int(doc_idx[local_idx])
            doc_id = str(docs[global_idx]["doc_id"])
            score = float(scores[local_idx])
            ranked_ids.append(doc_id)

            entry = stats.setdefault(doc_id, {})
            if score > entry.get("dense_score", float("-inf")):
                entry["dense_score"] = score
                entry["dense_rank"] = rank
                entry["dense_query"] = variant.text
                entry["dense_query_source"] = variant.source
                entry["dense_query_weight"] = variant.weight

        rankings.append(ranked_ids)

    return rankings, stats


# --- BM25 retrieval ---
def bm25_rankings(
    query_variants: list[QueryVariant],
    doc_idx: np.ndarray,
) -> tuple[list[list[str]], dict[str, dict[str, Any]]]:
    bm25, _ = get_bm25()
    docs = get_docstore()

    rankings: list[list[str]] = []
    stats: dict[str, dict[str, Any]] = {}

    for variant in query_variants:
        q_tokens = tokenize_for_bm25(variant.text)
        if not q_tokens:
            rankings.append([])
            continue

        scores_all = np.asarray(bm25.get_scores(q_tokens), dtype="float32")
        scores = scores_all[doc_idx]
        order_local = _top_indices(scores, BM25_TOP_N)

        ranked_ids: list[str] = []
        for rank, local_idx in enumerate(order_local, start=1):
            global_idx = int(doc_idx[local_idx])
            doc_id = str(docs[global_idx]["doc_id"])
            score = float(scores[local_idx])
            ranked_ids.append(doc_id)

            entry = stats.setdefault(doc_id, {})
            if score > entry.get("bm25_score", float("-inf")):
                entry["bm25_score"] = score
                entry["bm25_rank"] = rank
                entry["bm25_query"] = variant.text
                entry["bm25_query_source"] = variant.source
                entry["bm25_query_weight"] = variant.weight

        rankings.append(ranked_ids)

    return rankings, stats


# --- BGE-M3 sparse retrieval ---
def bge_sparse_rankings(
    query_variants: list[QueryVariant],
    doc_idx: np.ndarray,
    query_lexical: list[dict[int, float]] | None,
) -> tuple[list[list[str]], dict[str, dict[str, Any]]]:
    encoder = get_encoder()
    if not encoder.supports_sparse:
        return [], {}

    docs = get_docstore()
    postings = get_sparse_postings()
    allowed = np.zeros(len(docs), dtype=bool)
    allowed[doc_idx] = True

    if not query_lexical:
        return [[] for _ in query_variants], {}

    rankings: list[list[str]] = []
    stats: dict[str, dict[str, Any]] = {}

    for variant, weights in zip(query_variants, query_lexical, strict=False):
        if not weights:
            rankings.append([])
            continue

        score_by_global: dict[int, float] = {}
        for token_id, query_weight in weights.items():
            for global_idx, doc_weight in postings.get(int(token_id), []):
                if not allowed[global_idx]:
                    continue
                score_by_global[global_idx] = score_by_global.get(global_idx, 0.0) + float(query_weight) * float(doc_weight)

        ranked_global = sorted(
            score_by_global.items(),
            key=lambda item: item[1],
            reverse=True,
        )[:BGE_SPARSE_TOP_N]

        ranked_ids: list[str] = []
        for rank, (global_idx, score) in enumerate(ranked_global, start=1):
            doc_id = str(docs[int(global_idx)]["doc_id"])
            ranked_ids.append(doc_id)

            entry = stats.setdefault(doc_id, {})
            if score > entry.get("bge_sparse_score", float("-inf")):
                entry["bge_sparse_score"] = float(score)
                entry["bge_sparse_rank"] = rank
                entry["bge_sparse_query"] = variant.text
                entry["bge_sparse_query_source"] = variant.source
                entry["bge_sparse_query_weight"] = variant.weight

        rankings.append(ranked_ids)

    return rankings, stats


# --- rerank with cross-encoder ---
def rerank(query: str, candidate_ids: list[str], top_k: int) -> list[RetrievedDoc]:
    if not candidate_ids:
        return []

    reranker = get_reranker()
    docs_by_id = _doc_by_id or {doc["doc_id"]: doc for doc in get_docstore()}

    pairs = [(query, docs_by_id[doc_id]["text"]) for doc_id in candidate_ids]
    scores = reranker.predict(pairs, batch_size=RERANK_BATCH_SIZE)

    ranked = sorted(zip(candidate_ids, scores), key=lambda x: float(x[1]), reverse=True)

    output: list[RetrievedDoc] = []
    for rank, (doc_id, score) in enumerate(ranked[:top_k], start=1):
        raw = docs_by_id[doc_id]
        metadata = {
            "doc_id": doc_id,
            "date": raw.get("date", ""),
            "post_url": raw.get("post_url", ""),
            "platform": raw.get("platform", ""),
            "rerank_score": float(score),
            "rerank_rank": rank,
        }
        output.append(RetrievedDoc(page_content=raw["text"], metadata=metadata))
    return output


# --- main retrieval ---
def retrieve_tweets(
    query: str,
    k: int = 5,
    date_from: str | None = None,
    date_to: str | None = None,
):
    retrieval_query = normalize_query(query)
    encoder = get_encoder()

    weighted_variants = generate_weighted_query_variants(
        retrieval_query,
        embed_backend=encoder,
    )
    query_variants = [variant.text for variant in weighted_variants]

    doc_idx = filtered_doc_indices(date_from=date_from, date_to=date_to)

    empty_response = {
        "retrieval_query": retrieval_query,
        "query_variants": query_variants,
        "query_variants_debug": [
            {
                "text": variant.text,
                "weight": variant.weight,
                "source": variant.source,
            }
            for variant in weighted_variants
        ],
        "docs": [],
    }

    if doc_idx.size == 0:
        return empty_response

    query_dense, query_lexical = _encode_query_variants(weighted_variants, encoder=encoder)

    dense_lists, dense_stats = dense_rankings(weighted_variants, doc_idx, query_dense)
    bm25_lists, bm25_stats = bm25_rankings(weighted_variants, doc_idx)
    sparse_lists, sparse_stats = bge_sparse_rankings(weighted_variants, doc_idx, query_lexical)

    fusion_inputs: list[tuple[list[str], float]] = []

    for ranking, variant in zip(dense_lists, weighted_variants, strict=False):
        fusion_inputs.append((ranking, variant.weight * DENSE_RRF_WEIGHT))

    for ranking, variant in zip(bm25_lists, weighted_variants, strict=False):
        fusion_inputs.append((ranking, variant.weight * BM25_RRF_WEIGHT))

    for ranking, variant in zip(sparse_lists, weighted_variants, strict=False):
        fusion_inputs.append((ranking, variant.weight * BGE_SPARSE_RRF_WEIGHT))

    fused_scores = weighted_rrf(fusion_inputs, k=RRF_K)
    if not fused_scores:
        return empty_response

    fused_top = sorted(fused_scores.items(), key=lambda x: x[1], reverse=True)[:FUSION_TOP_N]
    fused_ids = [doc_id for doc_id, _ in fused_top]

    docs_by_id = _doc_by_id or {doc["doc_id"]: doc for doc in get_docstore()}
    suppressed_duplicate_ids: list[str] = []

    rerank_pool = fused_ids[:RERANK_TOP_N]
    if ENABLE_NEAR_DUPLICATE_SUPPRESSION:
        rerank_pool, suppressed_duplicate_ids = suppress_near_duplicates(
            fused_ids,
            docs_by_id,
            target_n=min(RERANK_TOP_N, len(fused_ids)),
            max_hamming=DEDUPE_MAX_HAMMING,
            min_jaccard=DEDUPE_MIN_JACCARD,
            ngram_size=DEDUP_NGRAM_SIZE,
        )

    docs = rerank(retrieval_query, rerank_pool, top_k=k)

    for doc in docs:
        doc_id = doc.metadata["doc_id"]
        doc.metadata["fused_score"] = float(fused_scores.get(doc_id, 0.0))

        if doc_id in dense_stats:
            doc.metadata.update(dense_stats[doc_id])

        if doc_id in bm25_stats:
            doc.metadata.update(bm25_stats[doc_id])

        if doc_id in sparse_stats:
            doc.metadata.update(sparse_stats[doc_id])

    return {
        "retrieval_query": retrieval_query,
        "query_variants": query_variants,
        "query_variants_debug": [
            {
                "text": variant.text,
                "weight": variant.weight,
                "source": variant.source,
            }
            for variant in weighted_variants
        ],
        "suppressed_duplicate_ids": suppressed_duplicate_ids,
        "rerank_pool_size": len(rerank_pool),
        "docs": docs,
    }