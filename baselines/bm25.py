"""Lightweight BM25-style scoring for hybrid retrieval baselines."""

from __future__ import annotations

import math
import re
from collections import Counter


def _tokenize(text: str) -> list[str]:
    return [t for t in re.findall(r"[a-z0-9]+", text.lower()) if len(t) > 1]


def bm25_scores(
    query: str,
    documents: list[str],
    *,
    k1: float = 1.5,
    b: float = 0.75,
) -> list[float]:
    """Okapi BM25 scores aligned with ``documents`` order."""
    if not documents:
        return []
    tokenized = [_tokenize(d) for d in documents]
    query_tokens = _tokenize(query)
    if not query_tokens:
        return [0.0] * len(documents)

    df: Counter[str] = Counter()
    for doc_tokens in tokenized:
        for term in set(doc_tokens):
            df[term] += 1

    n_docs = len(documents)
    avgdl = sum(len(t) for t in tokenized) / n_docs
    scores: list[float] = []

    for doc_tokens in tokenized:
        tf = Counter(doc_tokens)
        dl = len(doc_tokens)
        score = 0.0
        for term in query_tokens:
            if term not in df:
                continue
            idf = math.log(1 + (n_docs - df[term] + 0.5) / (df[term] + 0.5))
            freq = tf.get(term, 0)
            denom = freq + k1 * (1 - b + b * dl / avgdl)
            score += idf * (freq * (k1 + 1)) / (denom + 1e-9)
        scores.append(score)
    return scores
