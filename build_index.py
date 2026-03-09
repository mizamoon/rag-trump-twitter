from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd
from dotenv import load_dotenv

from config import (
    DOCSTORE_PATH,
    DENSE_MATRIX_PATH,
    EMBED_MODEL,
    INDEX_META_PATH,
    INDEX_PATH,
    INPUT_PATH,
    NORMALIZE_EMBEDDINGS,
    SPARSE_WEIGHTS_PATH,
)
from rag.dedupe import exact_text_hash, normalize_text_for_dedupe, simhash_hex
from rag.embeddings import get_embedding_backend

load_dotenv()


def load_data(path: Path) -> pd.DataFrame:
    df = pd.read_csv(path)
    df = df.dropna(subset=["text"])
    df["text"] = df["text"].astype(str).str.strip()
    df = df[df["text"] != ""]

    if "date" not in df.columns:
        df["date"] = ""
    else:
        df["date"] = df["date"].fillna("").astype(str)

    if "post_url" not in df.columns:
        df["post_url"] = ""
    else:
        df["post_url"] = df["post_url"].fillna("").astype(str)

    if "platform" not in df.columns:
        df["platform"] = ""
    else:
        df["platform"] = df["platform"].fillna("").astype(str)

    return df.reset_index(drop=True)


def make_docstore(df: pd.DataFrame) -> list[dict]:
    docs: list[dict] = []
    for idx, row in df.iterrows():
        text = str(row["text"]).strip()
        normalized = normalize_text_for_dedupe(text)
        docs.append(
            {
                "doc_id": str(idx),
                "text": text,
                "date": str(row.get("date", "")),
                "post_url": str(row.get("post_url", "")),
                "platform": str(row.get("platform", "")),
                "dedupe_text_hash": exact_text_hash(text),
                "dedupe_simhash": simhash_hex(text),
                "dedupe_norm_length": len(normalized),
            }
        )
    return docs


def save_docstore(docs: list[dict]) -> None:
    DOCSTORE_PATH.parent.mkdir(parents=True, exist_ok=True)
    with open(DOCSTORE_PATH, "w", encoding="utf-8") as f:
        for doc in docs:
            f.write(json.dumps(doc, ensure_ascii=False) + "\n")


def save_sparse_weights(
    doc_ids: list[str],
    lexical_weights: list[dict[int, float]] | None,
) -> bool:
    if not lexical_weights:
        if SPARSE_WEIGHTS_PATH.exists():
            SPARSE_WEIGHTS_PATH.unlink()
        return False

    with open(SPARSE_WEIGHTS_PATH, "w", encoding="utf-8") as f:
        for doc_id, weights in zip(doc_ids, lexical_weights, strict=False):
            payload = {
                "doc_id": doc_id,
                "lexical_weights": {str(k): float(v) for k, v in weights.items()},
            }
            f.write(json.dumps(payload, ensure_ascii=False) + "\n")
    return True


def embed_texts(texts: list[str]) -> tuple[np.ndarray, list[dict[int, float]] | None, str]:
    backend = get_embedding_backend(EMBED_MODEL)
    encoded = backend.encode_corpus(texts)
    return encoded.dense.astype("float32"), encoded.lexical_weights, backend.backend_name


def save_meta(
    vectors: np.ndarray,
    docs_count: int,
    *,
    embed_backend: str,
    sparse_available: bool,
) -> None:
    meta = {
        "embed_model": EMBED_MODEL,
        "embed_backend": embed_backend,
        "docs_count": docs_count,
        "embedding_dim": int(vectors.shape[1]),
        "normalize_embeddings": NORMALIZE_EMBEDDINGS,
        "storage": "numpy_exact_dense + jsonl_docstore",
        "sparse_available": bool(sparse_available),
        "sparse_weights_path": str(SPARSE_WEIGHTS_PATH) if sparse_available else "",
        "dedupe_metadata": True,
    }
    with open(INDEX_META_PATH, "w", encoding="utf-8") as f:
        json.dump(meta, f, ensure_ascii=False, indent=2)


def main() -> None:
    print(f"Loading data from {INPUT_PATH} ...")
    df = load_data(INPUT_PATH)
    docs = make_docstore(df)

    print(f"Loaded {len(docs)} documents")
    print(f"Embedding with {EMBED_MODEL} ...")
    vectors, lexical_weights, embed_backend = embed_texts([doc["text"] for doc in docs])

    INDEX_PATH.mkdir(parents=True, exist_ok=True)
    save_docstore(docs)
    np.save(DENSE_MATRIX_PATH, vectors)
    sparse_available = save_sparse_weights([doc["doc_id"] for doc in docs], lexical_weights)
    save_meta(
        vectors,
        len(docs),
        embed_backend=embed_backend,
        sparse_available=sparse_available,
    )

    print(f"Saved docstore to {DOCSTORE_PATH}")
    print(f"Saved dense matrix to {DENSE_MATRIX_PATH}")
    if sparse_available:
        print(f"Saved sparse weights to {SPARSE_WEIGHTS_PATH}")
    print(f"Saved metadata to {INDEX_META_PATH}")


if __name__ == "__main__":
    main()
