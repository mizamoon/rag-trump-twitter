from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Protocol

import numpy as np
from sentence_transformers import SentenceTransformer

from config import (
    BGE_M3_PASSAGE_MAX_LENGTH,
    BGE_M3_QUERY_MAX_LENGTH,
    BGE_M3_USE_FP16,
    EMBED_BATCH_SIZE,
    EMBED_MODEL,
    NORMALIZE_EMBEDDINGS,
)

try:
    from FlagEmbedding import BGEM3FlagModel
except Exception:
    BGEM3FlagModel = None


def is_bge_m3_model(model_name: str) -> bool:
    return model_name.strip().lower() == "baai/bge-m3"


def _normalize_rows(vectors: np.ndarray) -> np.ndarray:
    if vectors.ndim == 1:
        denom = np.linalg.norm(vectors)
        if denom <= 1e-12:
            return vectors
        return vectors / denom

    denom = np.linalg.norm(vectors, axis=1, keepdims=True)
    denom = np.clip(denom, 1e-12, None)
    return vectors / denom


def _coerce_dense(vectors: Any) -> np.ndarray:
    out = np.asarray(vectors, dtype="float32")
    if NORMALIZE_EMBEDDINGS:
        out = _normalize_rows(out).astype("float32")
    return out.astype("float32")


def _coerce_lexical_weights(items: Any) -> list[dict[int, float]]:
    if not items:
        return []
    output: list[dict[int, float]] = []
    for item in items:
        cleaned: dict[int, float] = {}
        for key, value in dict(item).items():
            cleaned[int(key)] = float(value)
        output.append(cleaned)
    return output


@dataclass(slots=True)
class EncodedBatch:
    dense: np.ndarray
    lexical_weights: list[dict[int, float]] | None = None


class EmbeddingBackend(Protocol):
    backend_name: str
    supports_sparse: bool

    def encode_queries(self, queries: list[str]) -> EncodedBatch:
        ...

    def encode_corpus(self, corpus: list[str]) -> EncodedBatch:
        ...

    def lexical_matching_score(
        self,
        query_weights: dict[int, float],
        doc_weights: dict[int, float],
    ) -> float:
        ...


class SentenceTransformerBackend:
    backend_name = "sentence-transformers"
    supports_sparse = False

    def __init__(self, model_name: str) -> None:
        self.model = SentenceTransformer(model_name)

    def encode_queries(self, queries: list[str]) -> EncodedBatch:
        vectors = self.model.encode(
            queries,
            batch_size=EMBED_BATCH_SIZE,
            normalize_embeddings=NORMALIZE_EMBEDDINGS,
            convert_to_numpy=True,
        )
        return EncodedBatch(dense=np.asarray(vectors, dtype="float32"))

    def encode_corpus(self, texts):
        output = self.model.encode(
            texts,
            batch_size=32,
            max_length=8192,
            return_dense=True,
            return_sparse=True
        )

        dense_vectors = output["dense_vecs"]
        lexical_weights = output["lexical_weights"]

        return dense_vectors, lexical_weights

    def lexical_matching_score(
        self,
        query_weights: dict[int, float],
        doc_weights: dict[int, float],
    ) -> float:
        return 0.0


class BGEM3Backend:
    backend_name = "bge-m3"
    supports_sparse = True

    def __init__(self, model_name: str) -> None:
        if BGEM3FlagModel is None:
            raise ImportError(
                "FlagEmbedding is required for BGE-M3. Install it with `pip install FlagEmbedding`."
            )
        self.model = BGEM3FlagModel(model_name, use_fp16=BGE_M3_USE_FP16)

    def encode_queries(self, queries: list[str]) -> EncodedBatch:
        output = self.model.encode(
            queries,
            batch_size=EMBED_BATCH_SIZE,
            max_length=BGE_M3_QUERY_MAX_LENGTH,
            return_dense=True,
            return_sparse=True,
        )

        return EncodedBatch(
            dense=_coerce_dense(output["dense_vecs"]),
            lexical_weights=_coerce_lexical_weights(output.get("lexical_weights")),
        )

    def encode_corpus(self, corpus: list[str]) -> EncodedBatch:
        output = self.model.encode(
            corpus,
            batch_size=EMBED_BATCH_SIZE,
            max_length=BGE_M3_PASSAGE_MAX_LENGTH,
            return_dense=True,
            return_sparse=True,
        )

        return EncodedBatch(
            dense=_coerce_dense(output["dense_vecs"]),
            lexical_weights=_coerce_lexical_weights(output.get("lexical_weights")),
        )

    def lexical_matching_score(
        self,
        query_weights: dict[int, float],
        doc_weights: dict[int, float],
    ) -> float:
        return float(self.model.compute_lexical_matching_score(query_weights, doc_weights))


def get_embedding_backend(model_name: str = EMBED_MODEL) -> EmbeddingBackend:
    if is_bge_m3_model(model_name):
        return BGEM3Backend(model_name)
    return SentenceTransformerBackend(model_name)
