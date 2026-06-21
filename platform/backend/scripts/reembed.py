"""Re-embed all stored FAQ and document chunks with the current embedding backend.

Required after switching EMBEDDING_BACKEND (e.g. GloVe-100 -> custom_hybrid-128),
because cosine search needs query and stored vectors in the same space. The DB
columns are REAL[] (variable length), so no schema change is needed - only the
stored values are rewritten.

Run from platform/backend with the backend venv and a populated .env:
    ./.venv/bin/python scripts/reembed.py
"""

from __future__ import annotations

import sys
import time

sys.path.insert(0, ".")

from app.dependencies import get_supabase_client  # noqa: E402
from app.services.embedding_service import EmbeddingService  # noqa: E402
from app.services.faq_service import faq_text  # noqa: E402


def reembed_faq(supabase, emb: EmbeddingService) -> int:
    rows = (supabase.table("faq").select("*").execute().data) or []
    if not rows:
        return 0
    t0 = time.perf_counter()
    texts = [faq_text(r["question"], r["answer"]) for r in rows]
    vectors = emb.embed_batch(texts)
    for r, v in zip(rows, vectors):
        supabase.table("faq").update({"embedding": v.astype(float).tolist()}).eq("id", r["id"]).execute()
    print(f"FAQ indexing time: {(time.perf_counter()-t0)*1000:.0f} ms ({len(rows)} rows)")
    return len(rows)


def reembed_chunks(supabase, emb: EmbeddingService, batch: int = 256) -> int:
    rows = (supabase.table("document_chunks").select("*").execute().data) or []
    if not rows:
        return 0
    t0 = time.perf_counter()
    for i in range(0, len(rows), batch):
        part = rows[i:i + batch]
        vectors = emb.embed_batch([r["content"] for r in part])
        for r, v in zip(part, vectors):
            supabase.table("document_chunks").update(
                {"embedding": v.astype(float).tolist()}
            ).eq("id", r["id"]).execute()
    print(f"Document embedding time: {(time.perf_counter()-t0)*1000:.0f} ms ({len(rows)} chunks)")
    return len(rows)


def main() -> None:
    supabase = get_supabase_client()
    emb = EmbeddingService()
    print(f"Re-embedding with backend='{emb.backend}' dim={emb.dim}")
    n_faq = reembed_faq(supabase, emb)
    n_chunks = reembed_chunks(supabase, emb)
    print(f"Done. Re-embedded {n_faq} FAQ and {n_chunks} chunks.")


if __name__ == "__main__":
    main()
