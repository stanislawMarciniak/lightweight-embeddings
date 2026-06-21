from __future__ import annotations

import json
import logging
import time
from typing import Annotated, Any, List

from fastapi import APIRouter, Depends, File, HTTPException, Query, UploadFile, status
from supabase import Client

from app.dependencies import get_supabase_client, get_current_user, get_embedding_service
from app.models.schemas import FAQCreate, FAQImportResult, FAQItem, MessageResponse
from app.services.embedding_service import EmbeddingService
from app.services.faq_service import FAQService
from app.utils.perf import timed


logger = logging.getLogger("perf")
router = APIRouter(prefix="/faq", tags=["faq"])


# Page context keys -> richer text for embedding (improves matching to FAQ).
# Unknown keys fall back to the de-slugified key itself.
_PAGE_CONTEXT_EXPANSIONS = {
    "payments": "payments billing invoices subscription refunds credit card",
    "account_settings": "account settings profile password email security login",
    "orders": "orders shipping delivery tracking returns",
    "support": "support help contact troubleshooting issues",
}


def _expand_context(page_context: str) -> str:
    key = page_context.strip().lower()
    return _PAGE_CONTEXT_EXPANSIONS.get(key, key.replace("_", " ").replace("-", " "))


def get_faq_service(
    supabase: Annotated[Client, Depends(get_supabase_client)],
    embedding_service: Annotated[EmbeddingService, Depends(get_embedding_service)],
) -> FAQService:
    return FAQService(supabase, embedding_service)


@router.post(
    "",
    response_model=FAQItem,
    status_code=status.HTTP_201_CREATED,
)
async def create_faq(
    payload: FAQCreate,
    current_user: Annotated[dict, Depends(get_current_user)],
    faq_service: Annotated[FAQService, Depends(get_faq_service)],
):
    t0 = time.perf_counter()
    faq = faq_service.create_faq(current_user["id"], payload)
    logger.info(
        "POST /faq request total: %.1f ms (user submit → saved in database)",
        (time.perf_counter() - t0) * 1000.0,
    )
    return faq


@router.get("", response_model=List[FAQItem])
async def list_faq(
    current_user: Annotated[dict, Depends(get_current_user)],
    faq_service: Annotated[FAQService, Depends(get_faq_service)],
):
    faqs = faq_service.list_faqs(current_user["id"])
    return faqs


def _normalize_import(raw: Any) -> List[dict]:
    """Accept a JSON list of FAQ objects or {"faqs": [...]}. Tolerates several key
    aliases (question/q, answer/a). Returns only valid {question, answer} items."""
    if isinstance(raw, dict):
        raw = raw.get("faqs") or raw.get("items") or raw.get("data") or []
    if not isinstance(raw, list):
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="JSON must be a list of {question, answer} objects (or {\"faqs\": [...]}).",
        )
    items: List[dict] = []
    for obj in raw:
        if not isinstance(obj, dict):
            continue
        q = (obj.get("question") or obj.get("q") or "").strip()
        a = (obj.get("answer") or obj.get("a") or "").strip()
        if q and a:
            items.append({"question": q, "answer": a})
    return items


@router.post("/import", response_model=FAQImportResult)
async def import_faq(
    current_user: Annotated[dict, Depends(get_current_user)],
    faq_service: Annotated[FAQService, Depends(get_faq_service)],
    file: UploadFile = File(...),
):
    """Bulk-add FAQ from an uploaded .json file (list of {question, answer})."""
    content = await file.read()
    try:
        raw = json.loads(content.decode("utf-8"))
    except (json.JSONDecodeError, UnicodeDecodeError) as exc:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=f"Invalid JSON file: {exc}",
        ) from exc

    items = _normalize_import(raw)
    if not items:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="No valid FAQ entries found. Each entry needs a non-empty question and answer.",
        )
    t0 = time.perf_counter()
    created = faq_service.bulk_create(current_user["id"], items)
    logger.info(
        "POST /faq/import request total: %.1f ms (%d rows, user submit → saved in database)",
        (time.perf_counter() - t0) * 1000.0,
        len(created),
    )
    total_in_file = len(raw if isinstance(raw, list) else raw.get("faqs", []) if isinstance(raw, dict) else [])
    return FAQImportResult(created=len(created), skipped=max(0, total_in_file - len(items)), items=created)


@router.get("/autocomplete")
async def faq_autocomplete(
    current_user: Annotated[dict, Depends(get_current_user)],
    faq_service: Annotated[FAQService, Depends(get_faq_service)],
    embedding_service: Annotated[EmbeddingService, Depends(get_embedding_service)],
    q: str = Query("", description="Partial query / keystrokes"),
    limit: int = Query(5, ge=1, le=10),
):
    """Semantic autocomplete: predict FAQ questions from partial keystrokes.
    Ultra-low latency (model <0.5 ms) so it is safe to call on each (debounced) keypress."""
    q = q.strip()
    if not q:
        return {"q": q, "suggestions": []}
    with timed("Autocomplete"):
        emb = embedding_service.get_sentence_embedding(q)
        suggestions = faq_service.autocomplete(current_user["id"], q, emb, limit=limit)
    return {"q": q, "suggestions": suggestions}


@router.get("/suggestions")
async def faq_suggestions(
    current_user: Annotated[dict, Depends(get_current_user)],
    faq_service: Annotated[FAQService, Depends(get_faq_service)],
    embedding_service: Annotated[EmbeddingService, Depends(get_embedding_service)],
    page_context: str = Query(..., min_length=1),
    limit: int = Query(3, ge=1, le=10),
):
    """Proactive FAQ: given the page the user is on, suggest the most relevant FAQ
    before they ask. Embeds the (expanded) page context and returns top-k FAQ."""
    context_text = _expand_context(page_context)
    with timed("FAQ suggestions"):
        emb = embedding_service.get_sentence_embedding(context_text)
        suggestions = faq_service.suggest(current_user["id"], emb, top_k=limit)
    return {"page_context": page_context, "suggestions": suggestions}


@router.delete(
    "/{faq_id}",
    response_model=MessageResponse,
)
async def delete_faq(
    faq_id: str,
    current_user: Annotated[dict, Depends(get_current_user)],
    faq_service: Annotated[FAQService, Depends(get_faq_service)],
):
    # Ensure FAQ belongs to user; delete is already filtered by user_id,
    # but we can perform a simple existence check if desired.
    faq_service.delete_faq(current_user["id"], faq_id)
    return MessageResponse(detail="FAQ deleted")

