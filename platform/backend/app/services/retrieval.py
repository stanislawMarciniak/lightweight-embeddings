"""Hybrid (lexical BM25 + dense) retrieval with out-of-domain rejection.

Why hybrid: on ultra-short FAQ strings the dense embedding space is anisotropic /
collapsed (every query is ~0.97 cosine to the "When is the contest?" hub), so pure
cosine cannot tell where/when/teams/prize apart. BM25 adds the lexical signal that
disambiguates them, and a content-term gate rejects genuinely out-of-domain queries
("where is the France?") instead of returning the nearest hub.

Scoring: per-query min-max normalize each signal across candidates, then
    score = w_lex * bm25_norm + w_dense * dense_norm
Lexical is weighted higher because the dense space is unreliable on short text.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional

import numpy as np

from app.utils.text_norm import content_terms, lexical_terms


@dataclass
class Candidate:
    kind: str              # "faq" | "doc"
    text: str              # text indexed for BM25 (faq: question+answer, doc: content)
    answer: str            # faq answer or doc content (used to build the response)
    row: Dict[str, Any]


@dataclass
class ScoredCandidate:
    candidate: Candidate
    score: float
    dense: float
    bm25: float


class BM25:
    def __init__(self, docs_terms: List[List[str]], k1: float = 1.5, b: float = 0.75) -> None:
        self.k1 = k1
        self.b = b
        self.N = len(docs_terms)
        self.doc_len = [len(d) for d in docs_terms]
        self.avgdl = (sum(self.doc_len) / self.N) if self.N else 0.0
        self.tf: List[Dict[str, int]] = []
        self.df: Dict[str, int] = {}
        for terms in docs_terms:
            freqs: Dict[str, int] = {}
            for t in terms:
                freqs[t] = freqs.get(t, 0) + 1
            self.tf.append(freqs)
            for t in freqs:
                self.df[t] = self.df.get(t, 0) + 1

    def idf(self, term: str) -> float:
        n = self.df.get(term, 0)
        if n == 0:
            return 0.0
        return math.log(1.0 + (self.N - n + 0.5) / (n + 0.5))

    def scores(self, query_terms: List[str]) -> np.ndarray:
        out = np.zeros(self.N, dtype=np.float32)
        if not self.avgdl:
            return out
        for i in range(self.N):
            freqs = self.tf[i]
            dl = self.doc_len[i]
            s = 0.0
            for t in query_terms:
                f = freqs.get(t, 0)
                if f == 0:
                    continue
                idf = self.idf(t)
                denom = f + self.k1 * (1 - self.b + self.b * dl / self.avgdl)
                s += idf * (f * (self.k1 + 1)) / denom
            out[i] = s
        return out

    def has_any(self, terms: List[str]) -> bool:
        return any(self.df.get(t, 0) > 0 for t in terms)


@dataclass
class Corpus:
    candidates: List[Candidate]
    matrix: np.ndarray            # (N, D) L2-normalized dense embeddings
    bm25: BM25
    _content_index: BM25 = field(default=None)  # BM25 over content terms only (for the gate)

    @property
    def size(self) -> int:
        return len(self.candidates)


def build_corpus(
    faq_rows: List[Dict[str, Any]], faq_mat: np.ndarray,
    chunk_rows: List[Dict[str, Any]], chunk_mat: np.ndarray,
) -> Corpus:
    candidates: List[Candidate] = []
    mats: List[np.ndarray] = []
    if faq_rows:
        for i, r in enumerate(faq_rows):
            q, a = r.get("question", ""), r.get("answer", "")
            candidates.append(Candidate("faq", f"{q} {a}".strip(), a, r))
        mats.append(faq_mat)
    if chunk_rows:
        for i, r in enumerate(chunk_rows):
            c = r.get("content", "")
            candidates.append(Candidate("doc", c, c, r))
        mats.append(chunk_mat)

    if mats and all(m.size for m in mats) and len({m.shape[1] for m in mats}) == 1:
        matrix = np.concatenate(mats, axis=0)
    elif len(mats) == 1 and mats[0].size:
        matrix = mats[0]
    else:
        matrix = np.zeros((len(candidates), 0), dtype=np.float32)

    lex = [lexical_terms(c.text) for c in candidates]
    con = [content_terms(c.text) for c in candidates]
    corpus = Corpus(candidates=candidates, matrix=matrix, bm25=BM25(lex))
    corpus._content_index = BM25(con)
    return corpus


def _minmax(x: np.ndarray) -> np.ndarray:
    if x.size == 0:
        return x
    lo, hi = float(x.min()), float(x.max())
    if hi - lo < 1e-9:
        return np.zeros_like(x)
    return (x - lo) / (hi - lo)


def hybrid_search(
    corpus: Corpus,
    query: str,
    query_embedding: Optional[np.ndarray],
    top_k: int = 5,
    w_lex: float = 0.6,
    w_dense: float = 0.4,
) -> List[ScoredCandidate]:
    """Return ranked candidates. Empty list => out-of-domain (no content-term match)."""
    n = corpus.size
    if n == 0:
        return []

    # Out-of-domain gate: at least one *content* term (not a stopword / wh-word)
    # must occur somewhere in the corpus. Rejects "where is the France?".
    q_content = content_terms(query)
    if not q_content or not corpus._content_index.has_any(q_content):
        return []

    bm25 = corpus.bm25.scores(lexical_terms(query))
    if query_embedding is not None and corpus.matrix.shape[1] == query_embedding.shape[0]:
        q = query_embedding / (np.linalg.norm(query_embedding) + 1e-12)
        dense = corpus.matrix @ q
    else:
        dense = np.zeros(n, dtype=np.float32)

    combined = w_lex * _minmax(bm25) + w_dense * _minmax(dense)
    order = np.argsort(-combined)[:top_k]
    return [
        ScoredCandidate(corpus.candidates[i], float(combined[i]), float(dense[i]), float(bm25[i]))
        for i in order
    ]
