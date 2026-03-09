from __future__ import annotations

import json
import math
import re
from collections import Counter
from typing import Iterable

import numpy as np

from config import (
    DOCSTORE_PATH,
    EMBED_MODEL,
    TERM_EMBEDDINGS_PATH,
    TERM_MAX_DOC_FREQ_RATIO,
    TERM_MIN_DOC_FREQ,
    TERM_VOCAB_MAX_TERMS,
    TERM_VOCAB_PATH,
)
from rag.embeddings import get_embedding_backend

TOKEN_RE = re.compile(r"[A-Za-z][A-Za-z0-9_+'-]{1,31}")
SPACE_RE = re.compile(r"\s+")

STOPWORDS = {
    "a", "an", "and", "are", "as", "at", "be", "been", "being", "but", "by",
    "can", "could", "did", "do", "does", "doing", "done", "for", "from",
    "had", "has", "have", "here", "how", "in", "into", "is", "it", "its",
    "itself", "just", "may", "might", "more", "most", "much", "must", "not",
    "of", "on", "onto", "or", "over", "same", "shall", "should", "some",
    "such", "than", "that", "the", "their", "them", "there", "these", "they",
    "this", "those", "through", "to", "under", "very", "was", "were", "what",
    "when", "where", "which", "who", "whom", "why", "will", "with", "would",
    "you", "your",
    "about", "comment", "comments", "feel", "felt", "post", "posts", "said",
    "say", "saying", "says", "statement", "statements", "tell", "tweet", "tweets",
}

CORPUS_NOISE_TERMS = {
    "trump", "trumps", "donald", "donaldtrump", "presidenttrump",
    "trump2020", "maga", "kag", "americafirst", "makeamericagreatagain",
    "thankyou", "realdonaldtrump",
}

GENERIC_UNIGRAMS = {
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

SHORT_WHITELIST = {
    "uk", "us", "eu", "un", "g7", "g20", "gop", "dnc", "rnc",
    "fbi", "cia", "irs", "doj", "cdc", "who", "ice",
    "lgbt", "lgbtq", "isis", "iran", "iraq", "nato", "nafta",
    "covid",
}

BAD_PATTERNS = [
    re.compile(r"^@"),
    re.compile(r"^http"),
    re.compile(r"^www"),
    re.compile(r"^amp$"),
    re.compile(r"^[a-z]*\d+[a-z\d]*$"),
    re.compile(r"^[^a-z]+$"),
]

MAX_TOKEN_LEN = 24
MAX_BIGRAM_SHARE = 0.45
BIGRAM_SCORE_BONUS = 1.15


def normalize_text(text: str) -> str:
    return SPACE_RE.sub(" ", (text or "").strip())


def normalize_token(token: str) -> str:
    t = token.strip().lower()
    t = t.strip("'-_+")
    if t.endswith("'s"):
        t = t[:-2]
    elif t.endswith("s'"):
        t = t[:-1]
    return t


def is_valid_token(token: str) -> bool:
    t = normalize_token(token)
    if not t:
        return False

    if t in SHORT_WHITELIST:
        return True

    if len(t) < 3 or len(t) > MAX_TOKEN_LEN:
        return False

    if t in STOPWORDS or t in CORPUS_NOISE_TERMS or t in GENERIC_UNIGRAMS:
        return False

    if "trump" in t:
        return False

    if any(ch.isdigit() for ch in t):
        return False

    for pattern in BAD_PATTERNS:
        if pattern.search(t):
            return False

    return True


def iter_document_texts() -> Iterable[str]:
    if not DOCSTORE_PATH.exists():
        raise FileNotFoundError(
            f"{DOCSTORE_PATH} not found. Build the main index first."
        )

    with open(DOCSTORE_PATH, "r", encoding="utf-8") as f:
        for line in f:
            row = json.loads(line)
            text = normalize_text(row.get("text") or "")
            if text:
                yield text


def tokenize_document(text: str) -> list[str]:
    raw = TOKEN_RE.findall(text)
    tokens: list[str] = []
    for tok in raw:
        norm = normalize_token(tok)
        if is_valid_token(norm):
            tokens.append(norm)
    return tokens


def extract_bigrams(tokens: list[str]) -> set[str]:
    phrases: set[str] = set()
    if len(tokens) < 2:
        return phrases

    for left, right in zip(tokens, tokens[1:]):
        if left == right:
            continue
        if left in GENERIC_UNIGRAMS or right in GENERIC_UNIGRAMS:
            continue
        if left in CORPUS_NOISE_TERMS or right in CORPUS_NOISE_TERMS:
            continue
        phrase = f"{left} {right}"
        phrases.add(phrase)

    return phrases


def bigram_score(
    phrase_df: int,
    left_df: int,
    right_df: int,
    n_docs: int,
) -> float:
    numerator = max(1, phrase_df) * max(1, n_docs)
    denominator = max(1, left_df) * max(1, right_df)
    pmi = math.log(numerator / denominator)
    pmi = max(0.0, min(pmi, 2.5))
    return float(phrase_df) * (1.0 + pmi) * BIGRAM_SCORE_BONUS


def build_term_vocab(texts: Iterable[str]) -> list[str]:
    unigram_df = Counter()
    bigram_df = Counter()
    n_docs = 0

    for text in texts:
        tokens = tokenize_document(text)
        if not tokens:
            continue

        n_docs += 1
        unigram_df.update(set(tokens))
        bigram_df.update(extract_bigrams(tokens))

    if n_docs == 0:
        return []

    max_df = max(TERM_MIN_DOC_FREQ + 1, int(n_docs * TERM_MAX_DOC_FREQ_RATIO))
    bigram_min_df = max(3, TERM_MIN_DOC_FREQ - 1)

    unigram_candidates: list[tuple[str, float, int]] = []
    for term, df in unigram_df.items():
        if TERM_MIN_DOC_FREQ <= df <= max_df:
            score = float(df)
            unigram_candidates.append((term, score, df))

    bigram_candidates: list[tuple[str, float, int]] = []
    for phrase, df in bigram_df.items():
        if not (bigram_min_df <= df <= max_df):
            continue

        left, right = phrase.split(" ", 1)
        if left not in unigram_df or right not in unigram_df:
            continue

        score = bigram_score(
            phrase_df=df,
            left_df=unigram_df[left],
            right_df=unigram_df[right],
            n_docs=n_docs,
        )
        bigram_candidates.append((phrase, score, df))

    unigram_candidates.sort(key=lambda x: (-x[1], -x[2], x[0]))
    bigram_candidates.sort(key=lambda x: (-x[1], -x[2], x[0]))

    max_bigrams = int(TERM_VOCAB_MAX_TERMS * MAX_BIGRAM_SHARE)
    selected_bigrams = bigram_candidates[:max_bigrams]

    remaining_slots = max(0, TERM_VOCAB_MAX_TERMS - len(selected_bigrams))
    selected_unigrams = unigram_candidates[:remaining_slots]

    merged = selected_unigrams + selected_bigrams
    merged.sort(key=lambda x: (-x[1], -x[2], x[0]))

    return [term for term, _, _ in merged[:TERM_VOCAB_MAX_TERMS]]


def main() -> None:
    print(f"Reading documents from {DOCSTORE_PATH} ...")
    terms = build_term_vocab(iter_document_texts())

    if not terms:
        raise RuntimeError("No valid expansion terms were produced.")

    print(f"Selected {len(terms)} terms for expansion vocab")
    print(f"Preview: {terms[:20]}")

    backend = get_embedding_backend(EMBED_MODEL)
    print(f"Encoding term vocabulary with {backend.backend_name} ...")

    encoded = backend.encode_queries(terms)
    term_vectors = np.asarray(encoded.dense, dtype="float32")

    TERM_VOCAB_PATH.parent.mkdir(parents=True, exist_ok=True)

    with open(TERM_VOCAB_PATH, "w", encoding="utf-8") as f:
        json.dump(terms, f, ensure_ascii=False)

    np.save(TERM_EMBEDDINGS_PATH, term_vectors)

    print(f"Saved vocab to {TERM_VOCAB_PATH}")
    print(f"Saved term embeddings to {TERM_EMBEDDINGS_PATH}")
    print("Done.")


if __name__ == "__main__":
    main()