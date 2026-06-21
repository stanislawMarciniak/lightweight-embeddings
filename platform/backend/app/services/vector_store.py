"""In-process per-user vector cache for FAQ and document chunks.

The previous implementation fetched *all* of a user's FAQ/chunk rows from Supabase
and ran a Python cosine loop on *every* chat request. This caches the embedding
matrices in process memory and does retrieval as a single matmul. The cache is
invalidated whenever the user's FAQ/documents change.

Embeddings are stored in Supabase as REAL[] (variable length), so the 100-d ->
128-d switch needs no schema change; rows whose dimension does not match the
current query embedding are skipped (and should be re-embedded via the migration).
"""

from __future__ import annotations

import logging
import threading
import time
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
from supabase import Client

logger = logging.getLogger(__name__)


@dataclass
class _Cached:
    rows: List[Dict[str, Any]]
    matrix: np.ndarray  # (N, D) L2-normalized
    built_at: float


def _normalize_rows(rows: List[Dict[str, Any]]) -> Tuple[List[Dict[str, Any]], np.ndarray]:
    vecs: List[np.ndarray] = []
    kept: List[Dict[str, Any]] = []
    dim: Optional[int] = None
    for r in rows:
        emb = r.get("embedding")
        if not emb:
            continue
        v = np.asarray(emb, dtype=np.float32)
        if dim is None:
            dim = v.shape[0]
        if v.shape[0] != dim:
            continue
        kept.append(r)
        vecs.append(v)
    if not vecs:
        return [], np.zeros((0, 0), dtype=np.float32)
    mat = np.stack(vecs, axis=0)
    norms = np.linalg.norm(mat, axis=1, keepdims=True)
    mat = mat / np.clip(norms, 1e-12, None)
    return kept, mat


class UserVectorStore:
    """Caches FAQ and chunk embedding matrices per user; thread-safe."""

    def __init__(self) -> None:
        self._faq: Dict[str, _Cached] = {}
        self._chunks: Dict[str, _Cached] = {}
        self._corpus: Dict[str, Any] = {}
        self._lock = threading.Lock()

    # --- FAQ ---

    def _load_faq(self, user_id: str, supabase: Client) -> _Cached:
        resp = (
            supabase.table("faq").select("*").eq("user_id", user_id)
            .order("created_at", desc=False).execute()
        )
        rows = resp.data or []
        kept, mat = _normalize_rows(rows)
        return _Cached(rows=kept, matrix=mat, built_at=time.time())

    def get_faq(self, user_id: str, supabase: Client) -> _Cached:
        with self._lock:
            c = self._faq.get(user_id)
        if c is None:
            c = self._load_faq(user_id, supabase)
            with self._lock:
                self._faq[user_id] = c
        return c

    # --- chunks ---

    def _load_chunks(self, user_id: str, supabase: Client) -> _Cached:
        resp = (
            supabase.table("document_chunks").select("*").eq("user_id", user_id)
            .order("document_id", desc=False).order("chunk_index", desc=False).execute()
        )
        rows = resp.data or []
        kept, mat = _normalize_rows(rows)
        return _Cached(rows=kept, matrix=mat, built_at=time.time())

    def get_chunks(self, user_id: str, supabase: Client) -> _Cached:
        with self._lock:
            c = self._chunks.get(user_id)
        if c is None:
            c = self._load_chunks(user_id, supabase)
            with self._lock:
                self._chunks[user_id] = c
        return c

    # --- unified hybrid corpus (FAQ + doc chunks) ---

    def get_corpus(self, user_id: str, supabase: Client):
        with self._lock:
            c = self._corpus.get(user_id)
        if c is not None:
            return c
        from app.services.retrieval import build_corpus

        faq = self.get_faq(user_id, supabase)
        chunks = self.get_chunks(user_id, supabase)
        corpus = build_corpus(faq.rows, faq.matrix, chunks.rows, chunks.matrix)
        with self._lock:
            self._corpus[user_id] = corpus
        return corpus

    # --- invalidation ---

    def invalidate(self, user_id: str) -> None:
        with self._lock:
            self._faq.pop(user_id, None)
            self._chunks.pop(user_id, None)
            self._corpus.pop(user_id, None)


def search(matrix: np.ndarray, query: np.ndarray, top_k: int = 5) -> List[Tuple[int, float]]:
    """Return [(row_index, cosine), ...] top_k by similarity. query may be unnormalized."""
    if matrix.size == 0 or matrix.shape[1] != query.shape[0]:
        return []
    q = query / (np.linalg.norm(query) + 1e-12)
    sims = matrix @ q
    k = min(top_k, sims.shape[0])
    top = np.argpartition(-sims, k - 1)[:k]
    top = top[np.argsort(-sims[top])]
    return [(int(i), float(sims[i])) for i in top]


_STORE: Optional[UserVectorStore] = None
_LOCK = threading.Lock()


def get_vector_store() -> UserVectorStore:
    global _STORE
    if _STORE is None:
        with _LOCK:
            if _STORE is None:
                _STORE = UserVectorStore()
    return _STORE
