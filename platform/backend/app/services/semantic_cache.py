"""In-process semantic cache for chat queries.

If a new query is semantically near a previously served one
(cosine_similarity > threshold), the cached retrieval result is returned without
re-running FAQ/document retrieval. Bounded by size (LRU eviction) and TTL.

Designed to be very fast: a single matmul over at most `max_size` cached vectors
(no database access). Per-user namespacing avoids cross-tenant leakage.
"""

from __future__ import annotations

import logging
import threading
import time
from collections import OrderedDict
from dataclasses import dataclass, field
from typing import Any, Dict, Optional, Tuple

import numpy as np

logger = logging.getLogger("cache")


@dataclass
class CacheEntry:
    embedding: np.ndarray  # L2-normalized
    result: Any
    created_at: float
    last_used: float
    terms: frozenset = frozenset()
    hits: int = 0


def _jaccard(a: frozenset, b: frozenset) -> float:
    if not a and not b:
        return 1.0
    if not a or not b:
        return 0.0
    return len(a & b) / len(a | b)


class SemanticCache:
    def __init__(self, threshold: float = 0.98, max_size: int = 1024, ttl_seconds: float = 3600.0,
                 lexical_threshold: float = 0.6) -> None:
        self.threshold = threshold
        self.lexical_threshold = lexical_threshold
        self.max_size = max_size
        self.ttl = ttl_seconds
        self._store: Dict[str, "OrderedDict[str, CacheEntry]"] = {}
        self._lock = threading.Lock()
        self._lookups = 0
        self._hits = 0

    @staticmethod
    def _norm(v: np.ndarray) -> np.ndarray:
        n = np.linalg.norm(v)
        return v / n if n > 0 else v

    def _purge_expired(self, ns: "OrderedDict[str, CacheEntry]", now: float) -> None:
        expired = [k for k, e in ns.items() if now - e.created_at > self.ttl]
        for k in expired:
            del ns[k]

    def lookup(self, user_id: str, embedding: np.ndarray, query: Optional[str] = None) -> Tuple[Optional[Any], float]:
        """Return (cached_result | None, best_similarity). Logs HIT/MISS, similarity, lookup time.

        A hit requires cosine > threshold AND lexical token overlap (Jaccard) above
        lexical_threshold. The lexical guard is essential because the dense space is
        anisotropic: "where is the contest?" and "when is the contest?" sit at cosine
        ~0.99, so cosine alone would wrongly merge them."""
        start = time.perf_counter()
        from app.utils.text_norm import lexical_terms

        q_terms = frozenset(lexical_terms(query)) if query is not None else None
        q = self._norm(np.asarray(embedding, dtype=np.float32))
        with self._lock:
            self._lookups += 1
            ns = self._store.get(user_id)
            if not ns:
                logger.info("CACHE MISS (empty) | CACHE SIMILARITY 0.000 | CACHE LOOKUP TIME %.2f ms",
                            (time.perf_counter() - start) * 1000.0)
                return None, 0.0
            now = time.time()
            self._purge_expired(ns, now)
            if not ns:
                logger.info("CACHE MISS (expired) | CACHE SIMILARITY 0.000 | CACHE LOOKUP TIME %.2f ms",
                            (time.perf_counter() - start) * 1000.0)
                return None, 0.0

            keys = list(ns.keys())
            mat = np.stack([ns[k].embedding for k in keys], axis=0)  # (N, D), already normalized
            # Dimension guard (e.g. after switching embedding backend)
            if mat.shape[1] != q.shape[0]:
                logger.info("CACHE MISS (dim change) | CACHE LOOKUP TIME %.2f ms",
                            (time.perf_counter() - start) * 1000.0)
                return None, 0.0
            sims = mat @ q
            order = np.argsort(-sims)
            best_sim = float(sims[order[0]])
            lookup_ms = (time.perf_counter() - start) * 1000.0
            for j in order:
                sim = float(sims[j])
                if sim <= self.threshold:
                    break  # sorted desc: no further candidate can pass
                entry = ns[keys[j]]
                if q_terms is not None and _jaccard(entry.terms, q_terms) < self.lexical_threshold:
                    continue  # cosine ok but different words (where vs when) -> not a hit
                entry.hits += 1
                entry.last_used = now
                ns.move_to_end(keys[j])
                self._hits += 1
                logger.info("CACHE HIT | CACHE SIMILARITY %.4f | CACHE LOOKUP TIME %.2f ms",
                            sim, lookup_ms)
                return entry.result, sim
            logger.info("CACHE MISS | CACHE SIMILARITY %.4f | CACHE LOOKUP TIME %.2f ms",
                        best_sim, lookup_ms)
            return None, best_sim

    def store(self, user_id: str, query: str, embedding: np.ndarray, result: Any) -> None:
        from app.utils.text_norm import lexical_terms

        q = self._norm(np.asarray(embedding, dtype=np.float32))
        now = time.time()
        terms = frozenset(lexical_terms(query))
        with self._lock:
            ns = self._store.setdefault(user_id, OrderedDict())
            ns[query] = CacheEntry(embedding=q, result=result, created_at=now, last_used=now, terms=terms)
            ns.move_to_end(query)
            while len(ns) > self.max_size:
                ns.popitem(last=False)  # evict least-recently-used

    def invalidate(self, user_id: str) -> None:
        """Drop cached results for a user (call when their FAQ/documents change)."""
        with self._lock:
            self._store.pop(user_id, None)

    def hit_rate(self) -> float:
        return (self._hits / self._lookups) if self._lookups else 0.0

    def stats(self) -> Dict[str, float]:
        return {"lookups": self._lookups, "hits": self._hits, "hit_rate": round(self.hit_rate(), 4)}


_CACHE: Optional[SemanticCache] = None
_LOCK = threading.Lock()


def get_semantic_cache() -> SemanticCache:
    global _CACHE
    if _CACHE is None:
        with _LOCK:
            if _CACHE is None:
                from app.config import get_settings

                s = get_settings()
                _CACHE = SemanticCache(
                    threshold=s.SEMANTIC_CACHE_THRESHOLD,
                    max_size=s.SEMANTIC_CACHE_MAX_SIZE,
                    ttl_seconds=s.SEMANTIC_CACHE_TTL_SECONDS,
                )
    return _CACHE
