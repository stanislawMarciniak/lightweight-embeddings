from __future__ import annotations

import io
import logging
import os
import re
import uuid
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
from fastapi import UploadFile
from PyPDF2 import PdfReader
from storage3.exceptions import StorageApiError
from supabase import Client

from app.config import get_settings
from app.services.embedding_service import EmbeddingService
from app.services.retrieval import ScoredCandidate
from app.services.semantic_cache import get_semantic_cache
from app.services.vector_store import get_vector_store, search
from app.utils.perf import RequestTimer
from app.utils.text_splitter import split_into_sentences

logger = logging.getLogger("perf")

TOP_DOC_CHUNKS = 3
NEIGHBOR_WINDOW = 3


def expand_chunk_neighbors(
    chunk_rows: List[Dict[str, Any]],
    document_id: int,
    chunk_index: int,
    window: int = NEIGHBOR_WINDOW,
) -> str:
    """Return text from chunk_index ± window within the same document."""
    start = max(0, chunk_index - window)
    end = chunk_index + window
    neighbors = [
        c
        for c in chunk_rows
        if c.get("document_id") == document_id
        and start <= int(c.get("chunk_index", 0)) <= end
    ]
    neighbors.sort(key=lambda c: int(c.get("chunk_index", 0)))
    return " ".join(c.get("content", "") for c in neighbors if c.get("content"))


def doc_contexts_from_hits(
    chunk_rows: List[Dict[str, Any]],
    results: List[ScoredCandidate],
    top_n: int = TOP_DOC_CHUNKS,
    window: int = NEIGHBOR_WINDOW,
) -> List[Tuple[str, float, Optional[int]]]:
    """Top document hits from hybrid search, each expanded ±window sentences."""
    seen: set[Tuple[Any, int]] = set()
    out: List[Tuple[str, float, Optional[int]]] = []
    for hit in results:
        if not isinstance(hit, ScoredCandidate) or hit.candidate.kind != "doc":
            continue
        row = hit.candidate.row
        doc_id = row.get("document_id")
        if doc_id is None:
            continue
        chunk_index = int(row.get("chunk_index", 0))
        key = (doc_id, chunk_index)
        if key in seen:
            continue
        seen.add(key)
        context = expand_chunk_neighbors(chunk_rows, int(doc_id), chunk_index, window)
        if context:
            out.append((context, hit.score, int(doc_id)))
        if len(out) >= top_n:
            break
    return out


class DocumentService:
    def __init__(self, supabase: Client, embedding_service: EmbeddingService) -> None:
        self.supabase = supabase
        self.embedding_service = embedding_service
        self.settings = get_settings()
        self.store = get_vector_store()

    def _extract_text_from_bytes(self, file_bytes: bytes, filename: str) -> str:
        ext = os.path.splitext(filename)[1].lower()

        if ext == ".txt":
            return file_bytes.decode("utf-8", errors="ignore")
        if ext == ".pdf":
            pdf_reader = PdfReader(io.BytesIO(file_bytes))
            texts: List[str] = []
            for page in pdf_reader.pages:
                page_text = page.extract_text() or ""
                texts.append(page_text)
            return "\n".join(texts)

        raise ValueError("Unsupported file type. Only .txt and .pdf are allowed.")

    def upload_document(self, user_id: str, file: UploadFile) -> Tuple[Dict[str, Any], int]:
        filename = file.filename or "document"
        file_bytes = file.file.read()
        return self.upload_document_bytes(user_id, file_bytes, filename)

    def _ensure_storage_bucket(self, bucket: str) -> None:
        """Create the storage bucket if it does not exist (idempotent)."""
        try:
            existing = [b.id for b in self.supabase.storage.list_buckets()]
            if bucket in existing:
                return
            self.supabase.storage.create_bucket(bucket, options={"public": False})
            logger.info("Created storage bucket %s", bucket)
        except StorageApiError as e:
            # Bucket may already exist (race) or we lack permission to create
            if "already exists" in (e.message or "").lower() or e.status == 409:
                return
            logger.warning("Could not ensure bucket %s: %s", bucket, e.message)
            raise

    @staticmethod
    def _safe_storage_key(user_id: str, filename: str) -> str:
        """Supabase Storage object keys only allow a limited character set; names with
        non-ASCII (e.g. em-dash), spaces or parentheses are rejected with HTTP 400.
        Sanitize to [A-Za-z0-9._-] and add a short unique prefix so re-uploading the
        same filename cannot collide (which would also be a 400 'Duplicate')."""
        name = os.path.basename(filename or "document")
        stem, ext = os.path.splitext(name)
        safe_stem = re.sub(r"[^A-Za-z0-9._-]+", "_", stem).strip("._") or "document"
        safe_ext = re.sub(r"[^A-Za-z0-9.]+", "", ext).lower()
        return f"{user_id}/{uuid.uuid4().hex[:8]}_{safe_stem}{safe_ext}"

    def upload_document_bytes(
        self, user_id: str, file_bytes: bytes, filename: str
    ) -> Tuple[Dict[str, Any], int]:
        timer = RequestTimer(label="Document upload")
        storage_path = self._safe_storage_key(user_id, filename)
        bucket = self.settings.SUPABASE_STORAGE_BUCKET
        with timer.stage("Storage upload"):
            self._ensure_storage_bucket(bucket)
            self.supabase.storage.from_(bucket).upload(storage_path, file_bytes)

        with timer.stage("Document record insert"):
            doc_resp = (
                self.supabase.table("documents")
                .insert(
                    {
                        "user_id": user_id,
                        "filename": filename,
                        "storage_path": storage_path,
                    }
                )
                .execute()
            )
        document = doc_resp.data[0]

        with timer.stage("Text extraction"):
            text = self._extract_text_from_bytes(file_bytes, filename)
            sentences = split_into_sentences(text)

        chunks: List[Dict[str, Any]] = []
        if sentences:
            with timer.stage("Chunk embedding"):
                embeddings = self.embedding_service.embed_batch(sentences)
            for idx, (sentence, emb) in enumerate(zip(sentences, embeddings)):
                chunks.append(
                    {
                        "document_id": document["id"],
                        "user_id": user_id,
                        "content": sentence,
                        "embedding": emb.astype(float).tolist(),
                        "chunk_index": idx,
                    }
                )

        if chunks:
            with timer.stage("Chunk database insert"):
                batch_size = 500
                for i in range(0, len(chunks), batch_size):
                    batch = chunks[i : i + batch_size]
                    self.supabase.table("document_chunks").insert(batch).execute()

        self.store.invalidate(user_id)
        get_semantic_cache().invalidate(user_id)
        timings = timer.finish()
        logger.info(
            "Document saved: %r (%d chunks) | total=%.1f ms "
            "(storage=%.1f, doc_db=%.1f, extract=%.1f, embed=%.1f, chunks_db=%.1f)",
            filename,
            len(chunks),
            timings["total"],
            timings.get("Storage upload", 0.0),
            timings.get("Document record insert", 0.0),
            timings.get("Text extraction", 0.0),
            timings.get("Chunk embedding", 0.0),
            timings.get("Chunk database insert", 0.0),
        )
        return document, len(chunks)

    def list_documents(self, user_id: str) -> List[Dict[str, Any]]:
        resp = (
            self.supabase.table("documents")
            .select("*")
            .eq("user_id", user_id)
            .order("created_at", desc=False)
            .execute()
        )
        return resp.data or []

    def delete_document(self, user_id: str, document_id: int) -> None:
        # Fetch document to get storage path
        doc_resp = (
            self.supabase.table("documents")
            .select("*")
            .eq("user_id", user_id)
            .eq("id", document_id)
            .execute()
        )
        docs = doc_resp.data or []
        if docs:
            doc = docs[0]
            storage_path = doc.get("storage_path")
            if storage_path:
                bucket = self.settings.SUPABASE_STORAGE_BUCKET
                self.supabase.storage.from_(bucket).remove([storage_path])

        # Delete chunks
        (
            self.supabase.table("document_chunks")
            .delete()
            .eq("user_id", user_id)
            .eq("document_id", document_id)
            .execute()
        )

        # Delete document
        (
            self.supabase.table("documents")
            .delete()
            .eq("user_id", user_id)
            .eq("id", document_id)
            .execute()
        )
        self.store.invalidate(user_id)
        get_semantic_cache().invalidate(user_id)

    def _get_all_chunks_for_user(self, user_id: str) -> List[Dict[str, Any]]:
        resp = (
            self.supabase.table("document_chunks")
            .select("*")
            .eq("user_id", user_id)
            .order("document_id", desc=False)
            .order("chunk_index", desc=False)
            .execute()
        )
        return resp.data or []

    def _get_chunks_for_document(self, document_id: int) -> List[Dict[str, Any]]:
        resp = (
            self.supabase.table("document_chunks")
            .select("*")
            .eq("document_id", document_id)
            .order("chunk_index", desc=False)
            .execute()
        )
        return resp.data or []

    def find_best_context(
        self, user_id: str, message_embedding: np.ndarray
    ) -> Optional[Tuple[str, float]]:
        cached = self.store.get_chunks(user_id, self.supabase)
        hits = search(cached.matrix, message_embedding, top_k=1)
        if not hits:
            return None
        idx, best_sim = hits[0]
        if best_sim < self.settings.DOC_SIMILARITY_THRESHOLD:
            return None

        best_chunk = cached.rows[idx]
        document_id = int(best_chunk["document_id"])
        index = int(best_chunk["chunk_index"])
        context = expand_chunk_neighbors(cached.rows, document_id, index, NEIGHBOR_WINDOW)
        return context, best_sim

