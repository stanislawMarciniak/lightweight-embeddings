from __future__ import annotations

import logging
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
from supabase import Client

from app.config import get_settings
from app.models.schemas import FAQCreate
from app.services.embedding_service import EmbeddingService
from app.services.semantic_cache import get_semantic_cache
from app.services.vector_store import get_vector_store, search
from app.utils.perf import RequestTimer

logger = logging.getLogger("perf")


def faq_text(question: str, answer: str) -> str:
    """Text embedded for a FAQ row. question + answer aligns with the fine-tuned
    query->passage retrieval geometry (the answer is the 'passage')."""
    return f"{question} {answer}".strip()


class FAQService:
    def __init__(self, supabase: Client, embedding_service: EmbeddingService) -> None:
        self.supabase = supabase
        self.embedding_service = embedding_service
        self.settings = get_settings()
        self.store = get_vector_store()

    def create_faq(self, user_id: str, payload: FAQCreate) -> Dict[str, Any]:
        timer = RequestTimer(label="FAQ create")
        with timer.stage("Embedding"):
            emb = self.embedding_service.get_sentence_embedding(
                faq_text(payload.question, payload.answer)
            )
        data = {
            "user_id": user_id,
            "question": payload.question,
            "answer": payload.answer,
            "embedding": emb.astype(float).tolist(),
        }
        with timer.stage("Database insert"):
            resp = self.supabase.table("faq").insert(data).execute()
        self.store.invalidate(user_id)
        get_semantic_cache().invalidate(user_id)
        timings = timer.finish()
        logger.info(
            "FAQ saved (1 row): question=%r | total=%.1f ms (embed=%.1f, db=%.1f)",
            payload.question[:60],
            timings["total"],
            timings.get("Embedding", 0.0),
            timings.get("Database insert", 0.0),
        )
        return resp.data[0]

    def bulk_create(self, user_id: str, items: List[Dict[str, str]]) -> List[Dict[str, Any]]:
        """Insert many FAQ at once. Embeds all rows in a single batched pass.

        `items` is a list of {"question", "answer"}. Rows missing either field are
        skipped by the caller; here we assume they are valid."""
        if not items:
            return []
        timer = RequestTimer(label="FAQ bulk import")
        texts = [faq_text(it["question"], it["answer"]) for it in items]
        with timer.stage("Embedding"):
            embeddings = self.embedding_service.embed_batch(texts)
        rows = [
            {
                "user_id": user_id,
                "question": it["question"],
                "answer": it["answer"],
                "embedding": embeddings[i].astype(float).tolist(),
            }
            for i, it in enumerate(items)
        ]
        with timer.stage("Database insert"):
            resp = self.supabase.table("faq").insert(rows).execute()
        self.store.invalidate(user_id)
        get_semantic_cache().invalidate(user_id)
        timings = timer.finish()
        logger.info(
            "FAQ bulk saved (%d rows) | total=%.1f ms (embed=%.1f, db=%.1f)",
            len(rows),
            timings["total"],
            timings.get("Embedding", 0.0),
            timings.get("Database insert", 0.0),
        )
        return resp.data or []

    def list_faqs(self, user_id: str) -> List[Dict[str, Any]]:
        resp = (
            self.supabase.table("faq")
            .select("*")
            .eq("user_id", user_id)
            .order("created_at", desc=False)
            .execute()
        )
        return resp.data or []

    def delete_faq(self, user_id: str, faq_id: str) -> None:
        (
            self.supabase.table("faq")
            .delete()
            .eq("user_id", user_id)
            .eq("id", faq_id)
            .execute()
        )
        self.store.invalidate(user_id)
        get_semantic_cache().invalidate(user_id)

    def find_best_match(
        self, user_id: str, message_embedding: np.ndarray
    ) -> Optional[Tuple[Dict[str, Any], float]]:
        cached = self.store.get_faq(user_id, self.supabase)
        hits = search(cached.matrix, message_embedding, top_k=1)
        if not hits:
            return None
        idx, sim = hits[0]
        return cached.rows[idx], sim

    def autocomplete(
        self, user_id: str, partial: str, partial_embedding: np.ndarray, limit: int = 5
    ) -> List[Dict[str, Any]]:
        """Semantic + prefix autocomplete over FAQ *questions* for partial keystrokes."""
        from app.utils.text_norm import word_tokens

        cached = self.store.get_faq(user_id, self.supabase)
        rows, mat = cached.rows, cached.matrix
        if not rows:
            return []

        qtok = word_tokens(partial)
        prefix = qtok[-1] if qtok else ""
        full = set(qtok[:-1])

        if mat.size and mat.shape[1] == partial_embedding.shape[0]:
            q = partial_embedding / (np.linalg.norm(partial_embedding) + 1e-12)
            dense = mat @ q
        else:
            dense = np.zeros(len(rows), dtype=np.float32)

        scored = []
        for i, row in enumerate(rows):
            toks = word_tokens(row.get("question", ""))
            lex = len(full & set(toks))
            if prefix and any(t.startswith(prefix) for t in toks):
                lex += 1
            scored.append((lex, float(dense[i]) if i < len(dense) else 0.0, i))
        scored.sort(key=lambda x: (x[0], x[1]), reverse=True)
        return [
            {"id": str(rows[i].get("id")), "question": rows[i].get("question"),
             "score": round(0.5 * lex + 0.5 * d, 4)}
            for lex, d, i in scored[:limit]
        ]

    def suggest(
        self, user_id: str, context_embedding: np.ndarray, top_k: int = 3
    ) -> List[Dict[str, Any]]:
        """Top-k FAQ for a context embedding (used by proactive suggestions)."""
        cached = self.store.get_faq(user_id, self.supabase)
        hits = search(cached.matrix, context_embedding, top_k=top_k)
        out: List[Dict[str, Any]] = []
        for idx, sim in hits:
            row = cached.rows[idx]
            out.append({"id": str(row.get("id")), "question": row.get("question"),
                        "answer": row.get("answer"), "similarity": round(sim, 4)})
        return out
