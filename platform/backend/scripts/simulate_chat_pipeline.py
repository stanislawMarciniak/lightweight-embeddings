#!/usr/bin/env python
"""Simulate all platform latency paths and print per-stage timings (milliseconds).

Runs without Supabase or a running HTTP server. Default: execute every scenario
below and print a summary table for thesis diagrams.

Cases
-----
1. **autocomplete** — JIT-embed partial keystrokes → FAQ question ranking
   (``GET /faq/autocomplete``, debounced in the UI)
2. **chat_faq_cold** — full message, cache miss, FAQ direct answer (no LLM)
3. **chat_faq_cached** — same message again, semantic cache hit
4. **chat_no_match** — out-of-domain query → "Answer not found."
5. **chat_rag** — document context + mock LLM generation
6. **chat_multi_intent** — compound question, two FAQ answers merged

Examples::

    cd platform/backend
    python scripts/simulate_chat_pipeline.py
    python scripts/simulate_chat_pipeline.py --case autocomplete --partial "When is th"
    python scripts/simulate_chat_pipeline.py --case chat_rag --mermaid
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Tuple

HERE = os.path.dirname(os.path.abspath(__file__))
BACKEND = os.path.dirname(HERE)
sys.path.insert(0, BACKEND)

import numpy as np

from app.config import get_settings
from app.services.embedding_service import EmbeddingService
from app.services.faq_service import faq_text
from app.services.document_service import (
    NEIGHBOR_WINDOW,
    TOP_DOC_CHUNKS,
    doc_contexts_from_hits,
)
from app.services.retrieval import build_corpus, hybrid_search
from app.services.semantic_cache import get_semantic_cache
from app.services.semantic_model import load_encoder, resolve_model_path
from app.utils.intent_splitter import split_intents
from app.utils.perf import RequestTimer
from app.utils.text_norm import word_tokens

DEFAULT_FAQ = [
    ("How many people are required per team?", "At least 5"),
    ("When is the contest?", "10.10.2026"),
    ("Where is the contest?", "Contest is in Warsaw"),
]

DEFAULT_DOC = (
    "The annual programming contest accepts teams of five or more members. "
    "Registration closes two weeks before the event date."
)

ALL_CASES = (
    "autocomplete",
    "chat_faq_cold",
    "chat_faq_cached",
    "chat_no_match",
    "chat_rag",
    "chat_multi_intent",
)


@dataclass
class CaseResult:
    name: str
    title: str
    input_label: str
    path: str
    outcome: str
    cache_hit: bool = False
    timings: Dict[str, float] = field(default_factory=dict)
    suggestions: List[str] = field(default_factory=list)


# ---------------------------------------------------------------------------
# FAQ store (in-memory; mirrors vector_store FAQ cache)
# ---------------------------------------------------------------------------


def _load_faq_pairs(path: Optional[str]) -> List[Tuple[str, str]]:
    if path and os.path.isfile(path):
        pairs: List[Tuple[str, str]] = []
        with open(path, encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                obj = json.loads(line)
                q = str(obj.get("question", obj.get("q", ""))).strip()
                a = str(obj.get("answer", obj.get("a", ""))).strip()
                if q and a:
                    pairs.append((q, a))
        if pairs:
            return pairs
    return list(DEFAULT_FAQ)


def _normalize_matrix(embeddings: np.ndarray) -> np.ndarray:
    mat = embeddings.astype(np.float32, copy=False)
    norms = np.linalg.norm(mat, axis=1, keepdims=True)
    return mat / np.clip(norms, 1e-12, None)


def build_faq_store(
    embedding_service: EmbeddingService,
    faq_pairs: List[Tuple[str, str]],
) -> Tuple[List[Dict[str, Any]], np.ndarray]:
    texts = [faq_text(q, a) for q, a in faq_pairs]
    mat = _normalize_matrix(embedding_service.embed_batch(texts))
    rows = [
        {"id": i, "question": q, "answer": a, "embedding": mat[i].astype(float).tolist()}
        for i, (q, a) in enumerate(faq_pairs)
    ]
    return rows, mat


def build_corpus_with_docs(
    faq_rows: List[Dict[str, Any]],
    faq_mat: np.ndarray,
    embedding_service: EmbeddingService,
    doc_sentences: Optional[List[str]] = None,
) -> Any:
    chunk_rows: List[Dict[str, Any]] = []
    chunk_mat = np.zeros((0, faq_mat.shape[1]), dtype=np.float32)
    if doc_sentences:
        chunk_mat = _normalize_matrix(embedding_service.embed_batch(doc_sentences))
        chunk_rows = [
            {
                "content": s,
                "embedding": chunk_mat[i].astype(float).tolist(),
                "document_id": 1,
                "chunk_index": i,
            }
            for i, s in enumerate(doc_sentences)
        ]
    return build_corpus(faq_rows, faq_mat, chunk_rows, chunk_mat)


# ---------------------------------------------------------------------------
# Autocomplete (JIT embedding while typing)
# ---------------------------------------------------------------------------


def _rank_autocomplete(
    partial: str,
    partial_embedding: np.ndarray,
    rows: List[Dict[str, Any]],
    mat: np.ndarray,
    limit: int = 5,
) -> List[Dict[str, Any]]:
    """Same ranking as ``FAQService.autocomplete`` (dense + lexical)."""
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
        {
            "id": str(rows[i].get("id")),
            "question": rows[i].get("question"),
            "score": round(0.5 * lex + 0.5 * d, 4),
        }
        for lex, d, i in scored[:limit]
    ]


def simulate_autocomplete(
    partial: str,
    embedding_service: EmbeddingService,
    faq_rows: List[Dict[str, Any]],
    faq_mat: np.ndarray,
    limit: int = 5,
) -> CaseResult:
    timer = RequestTimer()
    with timer.stage("JIT query embedding"):
        emb = embedding_service.get_sentence_embedding(partial)
    with timer.stage("FAQ similarity search"):
        suggestions = _rank_autocomplete(partial, emb, faq_rows, faq_mat, limit=limit)
    timings = timer.finish()
    top = [s["question"] for s in suggestions[:3]]
    return CaseResult(
        name="autocomplete",
        title="FAQ autocomplete (while typing)",
        input_label=partial,
        path="jit_embed + faq_rank",
        outcome=", ".join(top) if top else "(no suggestions)",
        timings=timings,
        suggestions=top,
    )


# ---------------------------------------------------------------------------
# Chat pipeline (POST /chat)
# ---------------------------------------------------------------------------


RETRIEVAL_TOP_K = 20


def _chunk_rows_from_corpus(corpus: Any) -> List[Dict[str, Any]]:
    return [c.row for c in corpus.candidates if c.kind == "doc"]


def _retrieve(
    user_id: str,
    intent: str,
    embedding: np.ndarray,
    corpus: Any,
    cache,
    timer: RequestTimer,
) -> Tuple[List[Any], bool]:
    with timer.stage("Semantic cache lookup"):
        cached, _sim = cache.lookup(user_id, embedding, query=intent)
    if cached is not None:
        return cached, True
    with timer.stage("FAQ retrieval"):
        results = hybrid_search(corpus, intent, embedding, top_k=RETRIEVAL_TOP_K)
    with timer.stage("Reranking"):
        pass
    cache.store(user_id, intent, embedding, results)
    return results, False


def simulate_chat(
    message: str,
    corpus: Any,
    embedding_service: EmbeddingService,
    *,
    user_id: str = "sim-user",
    mock_llm_ms: float = 0.0,
    force_rag: bool = False,
    case_name: str = "chat",
    title: str = "Chat request",
) -> CaseResult:
    timer = RequestTimer()
    cache = get_semantic_cache()
    cache_hit_any = False

    with timer.stage("Intent parsing"):
        intents = split_intents(message)
    with timer.stage("Query embedding"):
        emb_matrix = embedding_service.embed_batch(intents)

    per_intent: List[Tuple[str, List[Any]]] = []
    for i, intent in enumerate(intents):
        results, hit = _retrieve(user_id, intent, emb_matrix[i], corpus, cache, timer)
        cache_hit_any = cache_hit_any or hit
        per_intent.append((intent, results))

    chunk_rows = _chunk_rows_from_corpus(corpus)
    faq_answers: List[Tuple[str, str]] = []
    doc_contexts: List[Tuple[str, float]] = []
    any_doc_or_ood = force_rag

    for intent, results in per_intent:
        if not results:
            any_doc_or_ood = True
            continue
        best = results[0]
        if best.candidate.kind == "faq" and not force_rag:
            faq_answers.append((intent, best.candidate.answer))
        else:
            any_doc_or_ood = True
            for context, score, _ in doc_contexts_from_hits(
                chunk_rows, results, top_n=TOP_DOC_CHUNKS, window=NEIGHBOR_WINDOW
            ):
                doc_contexts.append((context, score))

    if faq_answers and not any_doc_or_ood:
        answer = faq_answers[0][1] if len(faq_answers) == 1 else " ".join(
            a.strip() for _, a in faq_answers if a.strip()
        )
        path = "faq_direct"
    elif not faq_answers and not doc_contexts:
        answer = "Answer not found."
        path = "no_match"
    else:
        doc_contexts.sort(key=lambda x: x[1], reverse=True)
        context_parts = [c for c, _ in doc_contexts] + [a for _, a in faq_answers]
        context_for_prompt = "\n\n".join(p for p in context_parts if p)
        with timer.stage("LLM generation"):
            if mock_llm_ms > 0:
                time.sleep(mock_llm_ms / 1000.0)
                answer = (
                    f"[mock LLM, {mock_llm_ms:.0f} ms] "
                    f"{context_for_prompt[:100]}..."
                )
            else:
                answer = f"[dry-run RAG] {context_for_prompt[:120]}..."
        path = "rag"

    timings = timer.finish()
    return CaseResult(
        name=case_name,
        title=title,
        input_label=message,
        path=path,
        outcome=answer[:120] + ("..." if len(answer) > 120 else ""),
        cache_hit=cache_hit_any,
        timings=timings,
    )


# ---------------------------------------------------------------------------
# Reporting
# ---------------------------------------------------------------------------


def _chat_grouped(timings: Dict[str, float]) -> Dict[str, float]:
    return {
        "Embedding": timings.get("Query embedding", 0.0),
        "Cache": timings.get("Semantic cache lookup", 0.0),
        "Retrieval": timings.get("FAQ retrieval", 0.0) + timings.get("Reranking", 0.0),
        "Generation": timings.get("LLM generation", 0.0),
        "Other": timings.get("Intent parsing", 0.0),
    }


def print_case(result: CaseResult, index: int) -> None:
    t = result.timings
    total = t.get("total", 0.0)
    print()
    print("=" * 72)
    print(f"[{index}] {result.title}  ({result.name})")
    print("=" * 72)
    print(f"Input     : {result.input_label!r}")
    print(f"Path      : {result.path}")
    if result.name.startswith("chat"):
        print(f"Cache hit : {result.cache_hit}")
    if result.suggestions:
        print(f"Top FAQ   : {result.suggestions}")
    print(f"Outcome   : {result.outcome}")
    print()
    print("Stage timings (ms):")
    for key, val in t.items():
        if key == "total":
            continue
        pct = (val / total * 100.0) if total > 0 else 0.0
        print(f"  {key:28} {val:8.1f}  ({pct:5.1f}%)")
    print(f"  {'TOTAL':28} {total:8.1f}")

    if result.name == "autocomplete":
        g = {
            "JIT embedding": t.get("JIT query embedding", 0.0),
            "FAQ search": t.get("FAQ similarity search", 0.0),
        }
    else:
        g = _chat_grouped(t)
    print()
    print("Grouped:")
    for key, val in g.items():
        pct = (val / total * 100.0) if total > 0 else 0.0
        print(f"  {key:14} {val:8.1f}  ({pct:5.1f}%)")


def print_summary_table(results: List[CaseResult]) -> None:
    print()
    print("=" * 72)
    print("SUMMARY — total request time (ms)")
    print("=" * 72)
    print(f"{'Case':<22} {'Path':<14} {'Cache':<6} {'Total ms':>10}")
    print("-" * 72)
    for r in results:
        cache = "yes" if r.cache_hit else "no" if r.name.startswith("chat") else "n/a"
        total = r.timings.get("total", 0.0)
        print(f"{r.name:<22} {r.path:<14} {cache:<6} {total:10.1f}")
    print()


def print_mermaid_chat(result: CaseResult) -> None:
    if not result.name.startswith("chat"):
        return
    t = result.timings
    g = _chat_grouped(t)
    total = t.get("total", 0.0)
    print()
    print(f"%% Mermaid — {result.name}")
    print("sequenceDiagram")
    print("    participant Client")
    print("    participant API as FastAPI")
    print("    participant Chat as ChatService")
    print("    participant Emb as EmbeddingService")
    print("    participant Cache as SemanticCache")
    print("    participant Ret as HybridRetrieval")
    print("    participant LLM as OpenAI")
    print("    Client->>API: POST /chat")
    print("    API->>Chat: process_message()")
    print(f"    Chat->>Chat: intent parsing ({t.get('Intent parsing', 0):.1f} ms)")
    print(f"    Chat->>Emb: embed_batch() ({g['Embedding']:.1f} ms)")
    print(f"    Chat->>Cache: lookup() ({g['Cache']:.1f} ms)")
    if result.cache_hit:
        print("    Cache-->>Chat: HIT")
    else:
        print(f"    Chat->>Ret: hybrid_search() ({g['Retrieval']:.1f} ms)")
    if result.path == "rag":
        print(f"    Chat->>LLM: ask_with_context() ({g['Generation']:.1f} ms)")
    else:
        print("    Note over Chat: no LLM (FAQ direct or no match)")
    print("    API-->>Client: JSON")
    print(f"    Note over Client,API: Total {total:.1f} ms")


def print_mermaid_autocomplete(result: CaseResult) -> None:
    t = result.timings
    total = t.get("total", 0.0)
    print()
    print("%% Mermaid — autocomplete")
    print("sequenceDiagram")
    print("    participant UI as Chat textarea")
    print("    participant API as FastAPI")
    print("    participant Emb as EmbeddingService")
    print("    participant FAQ as FAQ store")
    print("    UI->>UI: debounce 120ms")
    print("    UI->>API: GET /faq/autocomplete?q=partial")
    print(f"    API->>Emb: get_sentence_embedding() ({t.get('JIT query embedding', 0):.1f} ms)")
    print(f"    API->>FAQ: dense + lexical rank ({t.get('FAQ similarity search', 0):.1f} ms)")
    print("    FAQ-->>API: top-k questions")
    print("    API-->>UI: suggestions")
    print(f"    Note over UI,API: Total {total:.1f} ms per keystroke (after debounce)")


def run_all_cases(
    embedding_service: EmbeddingService,
    faq_rows: List[Dict[str, Any]],
    faq_mat: np.ndarray,
    corpus: Any,
    *,
    partial: str,
    faq_message: str,
    mock_llm_ms: float,
    mermaid: bool,
) -> List[CaseResult]:
    cache = get_semantic_cache()
    cache.invalidate("sim-user")
    results: List[CaseResult] = []

    # 1) Autocomplete — JIT embed while typing
    results.append(
        simulate_autocomplete(partial, embedding_service, faq_rows, faq_mat)
    )

    # 2) Chat FAQ — cold (cache miss)
    cache.invalidate("sim-user")
    results.append(
        simulate_chat(
            faq_message,
            corpus,
            embedding_service,
            case_name="chat_faq_cold",
            title="Chat — FAQ direct (cache miss)",
        )
    )

    # 3) Chat FAQ — warm (cache hit, same message)
    results.append(
        simulate_chat(
            faq_message,
            corpus,
            embedding_service,
            case_name="chat_faq_cached",
            title="Chat — FAQ direct (cache hit)",
        )
    )

    # 4) Out-of-domain
    cache.invalidate("sim-user")
    results.append(
        simulate_chat(
            "Where is France?",
            corpus,
            embedding_service,
            case_name="chat_no_match",
            title="Chat — no match",
        )
    )

    # 5) RAG with mock LLM
    cache.invalidate("sim-user")
    rag_corpus = corpus  # includes doc chunk from setup
    results.append(
        simulate_chat(
            "How many team members are required?",
            rag_corpus,
            embedding_service,
            mock_llm_ms=mock_llm_ms,
            force_rag=True,
            case_name="chat_rag",
            title="Chat — RAG + LLM",
        )
    )

    # 6) Multi-intent — two explicit sub-questions (both FAQ hits, no LLM)
    cache.invalidate("sim-user")
    results.append(
        simulate_chat(
            "When is the contest? Where is the contest?",
            corpus,
            embedding_service,
            case_name="chat_multi_intent",
            title="Chat — multi-intent (two FAQ answers)",
        )
    )

    for i, r in enumerate(results, 1):
        print_case(r, i)

    print_summary_table(results)

    if mermaid:
        for r in results:
            if r.name == "autocomplete":
                print_mermaid_autocomplete(r)
            elif r.name.startswith("chat"):
                print_mermaid_chat(r)

    return results


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Simulate all platform paths and print stage timings (ms)."
    )
    parser.add_argument(
        "--case",
        choices=[*ALL_CASES, "all"],
        default="all",
        help="Which scenario to run (default: all)",
    )
    parser.add_argument(
        "--partial",
        default="When is th",
        help="Partial query for autocomplete case",
    )
    parser.add_argument(
        "--faq-message",
        default="When is the contest?",
        help="Full message for FAQ chat cases",
    )
    parser.add_argument(
        "--faq-file",
        default=os.path.join(BACKEND, "sample_faq_pairs.jsonl"),
        help="JSONL FAQ corpus",
    )
    parser.add_argument(
        "--mock-llm-ms",
        type=float,
        default=380.0,
        help="Simulated OpenAI latency for RAG case (ms)",
    )
    parser.add_argument(
        "--mermaid",
        action="store_true",
        help="Print Mermaid sequence diagrams",
    )
    args = parser.parse_args()

    settings = get_settings()
    print("Loading semantic encoder...")
    t0 = time.perf_counter()
    load_encoder(resolve_model_path(settings))
    print(f"Encoder ready in {(time.perf_counter() - t0) * 1000:.1f} ms (one-time startup)\n")

    embedding_service = EmbeddingService()
    faq_pairs = _load_faq_pairs(args.faq_file)

    t0 = time.perf_counter()
    faq_rows, faq_mat = build_faq_store(embedding_service, faq_pairs)
    corpus = build_corpus_with_docs(
        faq_rows, faq_mat, embedding_service, doc_sentences=[DEFAULT_DOC]
    )
    store_ms = (time.perf_counter() - t0) * 1000.0
    print(f"FAQ store + corpus build: {store_ms:.1f} ms ({len(faq_rows)} FAQ, {corpus.size} candidates)")
    print("  (one-time; not included in per-request totals)\n")

    if args.case == "all":
        run_all_cases(
            embedding_service,
            faq_rows,
            faq_mat,
            corpus,
            partial=args.partial,
            faq_message=args.faq_message,
            mock_llm_ms=args.mock_llm_ms,
            mermaid=args.mermaid,
        )
        return

    cache = get_semantic_cache()
    cache.invalidate("sim-user")
    result: CaseResult

    if args.case == "autocomplete":
        result = simulate_autocomplete(args.partial, embedding_service, faq_rows, faq_mat)
    elif args.case == "chat_faq_cold":
        result = simulate_chat(
            args.faq_message, corpus, embedding_service,
            case_name="chat_faq_cold", title="Chat — FAQ direct (cache miss)",
        )
    elif args.case == "chat_faq_cached":
        simulate_chat(
            args.faq_message, corpus, embedding_service,
            case_name="chat_faq_cold", title="(warm-up for cache)",
        )
        result = simulate_chat(
            args.faq_message, corpus, embedding_service,
            case_name="chat_faq_cached", title="Chat — FAQ direct (cache hit)",
        )
    elif args.case == "chat_no_match":
        result = simulate_chat(
            "Where is France?", corpus, embedding_service,
            case_name="chat_no_match", title="Chat — no match",
        )
    elif args.case == "chat_rag":
        result = simulate_chat(
            "How many team members are required?",
            corpus,
            embedding_service,
            mock_llm_ms=args.mock_llm_ms,
            force_rag=True,
            case_name="chat_rag",
            title="Chat — RAG + LLM",
        )
    elif args.case == "chat_multi_intent":
        result = simulate_chat(
            "When is the contest? Where is the contest?",
            corpus,
            embedding_service,
            case_name="chat_multi_intent",
            title="Chat — multi-intent",
        )
    else:
        raise SystemExit(f"Unknown case: {args.case}")

    print_case(result, 1)
    print_summary_table([result])
    if args.mermaid:
        if result.name == "autocomplete":
            print_mermaid_autocomplete(result)
        else:
            print_mermaid_chat(result)


if __name__ == "__main__":
    main()
