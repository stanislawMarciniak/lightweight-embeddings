from __future__ import annotations

import logging
from typing import Any, Dict, List, Optional, Tuple

from supabase import Client

from app.config import get_settings
from app.services.document_service import (
    DocumentService,
    NEIGHBOR_WINDOW,
    TOP_DOC_CHUNKS,
    doc_contexts_from_hits,
)
from app.services.embedding_service import EmbeddingService
from app.services.faq_service import FAQService
from app.services.openai_service import OpenAIService
from app.services.retrieval import hybrid_search
from app.services.semantic_cache import get_semantic_cache
from app.services.vector_store import get_vector_store
from app.utils.intent_splitter import split_intents
from app.utils.perf import RequestTimer

logger = logging.getLogger(__name__)


class ChatService:
    def __init__(
        self,
        supabase: Client,
        embedding_service: EmbeddingService,
        openai_service: OpenAIService,
    ) -> None:
        self.supabase = supabase
        self.embedding_service = embedding_service
        self.openai_service = openai_service
        self.faq_service = FAQService(supabase, embedding_service)
        self.document_service = DocumentService(supabase, embedding_service)
        self.settings = get_settings()
        self.cache = get_semantic_cache()
        self.store = get_vector_store()

    @staticmethod
    def _title_from(message: str, limit: int = 48) -> str:
        title = " ".join(message.strip().split())
        return (title[: limit - 1] + "…") if len(title) > limit else (title or "New chat")

    # --- conversations ---

    def create_conversation(self, user_id: str, title: Optional[str] = None) -> Dict[str, Any]:
        resp = (
            self.supabase.table("conversations")
            .insert({"user_id": user_id, "title": (title or "New chat").strip() or "New chat"})
            .execute()
        )
        return resp.data[0]

    def list_conversations(self, user_id: str) -> List[Dict[str, Any]]:
        resp = (
            self.supabase.table("conversations")
            .select("*")
            .eq("user_id", user_id)
            .order("updated_at", desc=True)
            .execute()
        )
        return resp.data or []

    def get_conversation(self, user_id: str, conversation_id: str) -> Optional[Dict[str, Any]]:
        resp = (
            self.supabase.table("conversations")
            .select("*")
            .eq("user_id", user_id)
            .eq("id", conversation_id)
            .execute()
        )
        data = resp.data or []
        return data[0] if data else None

    def rename_conversation(self, user_id: str, conversation_id: str, title: str) -> Optional[Dict[str, Any]]:
        resp = (
            self.supabase.table("conversations")
            .update({"title": title.strip()})
            .eq("user_id", user_id)
            .eq("id", conversation_id)
            .execute()
        )
        data = resp.data or []
        return data[0] if data else None

    def delete_conversation(self, user_id: str, conversation_id: str) -> None:
        # chats are removed via ON DELETE CASCADE; delete here is belt-and-suspenders.
        self.supabase.table("chats").delete().eq("user_id", user_id).eq(
            "conversation_id", conversation_id
        ).execute()
        self.supabase.table("conversations").delete().eq("user_id", user_id).eq(
            "id", conversation_id
        ).execute()

    def _touch_conversation(self, conversation_id: str) -> None:
        from datetime import datetime, timezone

        self.supabase.table("conversations").update(
            {"updated_at": datetime.now(timezone.utc).isoformat()}
        ).eq("id", conversation_id).execute()

    def _save_chat(
        self,
        user_id: str,
        message: str,
        response: str,
        conversation_id: str,
        response_source: Optional[str] = None,
        response_document_name: Optional[str] = None,
    ) -> Dict[str, Any]:
        row = {
            "user_id": user_id,
            "conversation_id": conversation_id,
            "message": message,
            "response": response,
        }
        if response_source is not None:
            row["response_source"] = response_source
        if response_document_name is not None:
            row["response_document_name"] = response_document_name
        resp = self.supabase.table("chats").insert(row).execute()
        return resp.data[0]

    def _document_filename(self, user_id: str, document_id: Optional[int]) -> str:
        if document_id is None:
            return "document"
        doc_resp = (
            self.supabase.table("documents")
            .select("filename")
            .eq("user_id", user_id)
            .eq("id", document_id)
            .limit(1)
            .execute()
        )
        rows = doc_resp.data or []
        if rows and rows[0].get("filename"):
            return str(rows[0]["filename"])
        return "document"

    def list_chats(self, user_id: str) -> List[Dict[str, Any]]:
        resp = (
            self.supabase.table("chats")
            .select("*")
            .eq("user_id", user_id)
            .order("created_at", desc=False)
            .execute()
        )
        return resp.data or []

    def list_messages(self, user_id: str, conversation_id: str) -> List[Dict[str, Any]]:
        resp = (
            self.supabase.table("chats")
            .select("*")
            .eq("user_id", user_id)
            .eq("conversation_id", conversation_id)
            .order("created_at", desc=False)
            .execute()
        )
        return resp.data or []

    NO_ANSWER = "Answer not found."
    RETRIEVAL_TOP_K = 20  # fetch enough candidates to pick top-N document chunks

    def _retrieve(self, user_id: str, intent: str, embedding, corpus, timer: RequestTimer):
        """Hybrid (BM25 + dense) retrieval for one intent, with semantic cache.
        Returns a list of ScoredCandidate (empty => out-of-domain)."""
        with timer.stage("Semantic cache lookup"):
            cached, _sim = self.cache.lookup(user_id, embedding, query=intent)
        if cached is not None:
            return cached

        with timer.stage("FAQ retrieval"):
            results = hybrid_search(corpus, intent, embedding, top_k=self.RETRIEVAL_TOP_K)
        with timer.stage("Reranking"):
            pass  # hybrid_search already fuses + ranks; kept for log visibility
        self.cache.store(user_id, intent, embedding, results)
        return results

    def process_message(
        self, user_id: str, message: str, conversation_id: Optional[str] = None
    ) -> Tuple[str, str, Optional[str], Optional[str]]:
        """Answer `message` within a conversation. Returns
        (answer, conversation_id, response_source, response_document_name).
        """
        timer = RequestTimer()
        logger.info("Request received")

        # 0) resolve the conversation (auto-create on first message)
        if not conversation_id:
            conv = self.create_conversation(user_id, self._title_from(message))
            conversation_id = str(conv["id"])

        # 1) structural multi-intent parsing
        intents = split_intents(message)

        # 2) batched embedding of all intents
        with timer.stage("Query embedding"):
            emb_matrix = self.embedding_service.embed_batch(intents)

        # 3) hybrid retrieval per intent over the unified FAQ+document corpus
        corpus = self.store.get_corpus(user_id, self.supabase)
        chunk_rows = self.store.get_chunks(user_id, self.supabase).rows
        per_intent: List[Tuple[str, List]] = []
        for i, intent in enumerate(intents):
            results = self._retrieve(user_id, intent, emb_matrix[i], corpus, timer)
            per_intent.append((intent, results))

        # 4) compose answer
        faq_answers: List[str] = []
        doc_contexts: List[Tuple[str, float, Optional[int]]] = []
        any_doc_or_ood = False

        for intent, results in per_intent:
            if not results:
                any_doc_or_ood = True
                continue
            best = results[0]
            if best.candidate.kind == "faq":
                faq_answers.append(best.candidate.answer)
            else:
                any_doc_or_ood = True
                doc_contexts.extend(
                    doc_contexts_from_hits(
                        chunk_rows,
                        results,
                        top_n=TOP_DOC_CHUNKS,
                        window=NEIGHBOR_WINDOW,
                    )
                )

        # 4a) every intent answered directly by FAQ -> merge answers, no LLM
        if faq_answers and not any_doc_or_ood:
            if len(faq_answers) == 1:
                answer = faq_answers[0]
                source = "faq"
            else:
                answer = " ".join(a.strip() for a in faq_answers if a.strip())
                source = "multiple_faq"
            return self._respond(user_id, message, answer, timer, conversation_id, source)

        # 4b) nothing matched at all -> explicit not-found message
        if not faq_answers and not doc_contexts:
            return self._respond(
                user_id, message, self.NO_ANSWER, timer, conversation_id, None
            )

        # 4c) RAG: ground the LLM in document context (+ any matched FAQ answers)
        doc_contexts.sort(key=lambda x: x[1], reverse=True)
        context_parts = [c for c, _, _ in doc_contexts] + list(faq_answers)
        context_for_prompt = "\n\n".join(p for p in context_parts if p) or "No relevant context is available."
        document_name = self._document_filename(user_id, doc_contexts[0][2]) if doc_contexts else None
        with timer.stage("LLM generation"):
            answer = self.openai_service.ask_with_context(context_for_prompt, message)
        return self._respond(
            user_id,
            message,
            answer,
            timer,
            conversation_id,
            "document" if doc_contexts else "faq",
            document_name,
        )

    def _respond(
        self,
        user_id: str,
        message: str,
        answer: str,
        timer: RequestTimer,
        conversation_id: str,
        response_source: Optional[str],
        response_document_name: Optional[str] = None,
    ) -> Tuple[str, str, Optional[str], Optional[str]]:
        self._save_chat(
            user_id,
            message,
            answer,
            conversation_id,
            response_source=response_source,
            response_document_name=response_document_name,
        )
        self._touch_conversation(conversation_id)
        timer.finish()
        return answer, conversation_id, response_source, response_document_name

