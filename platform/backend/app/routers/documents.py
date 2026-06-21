from __future__ import annotations

import logging
import time
from typing import Annotated, List

from fastapi import APIRouter, Depends, File, HTTPException, UploadFile, status
from starlette.concurrency import run_in_threadpool
from storage3.exceptions import StorageApiError
from supabase import Client

from app.config import get_settings
from app.dependencies import get_supabase_client, get_current_user, get_embedding_service
from app.models.schemas import DocumentItem, DocumentUploadResponse, MessageResponse
from app.services.document_service import DocumentService
from app.services.embedding_service import EmbeddingService


logger = logging.getLogger("perf")
router = APIRouter(prefix="/documents", tags=["documents"])


def get_document_service(
    supabase: Annotated[Client, Depends(get_supabase_client)],
    embedding_service: Annotated[EmbeddingService, Depends(get_embedding_service)],
) -> DocumentService:
    return DocumentService(supabase, embedding_service)


@router.post(
    "/upload",
    response_model=DocumentUploadResponse,
    status_code=status.HTTP_201_CREATED,
)
async def upload_document(
    file: UploadFile = File(...),
    current_user: Annotated[dict, Depends(get_current_user)] = None,
    document_service: Annotated[DocumentService, Depends(get_document_service)] = None,
):
    if file.content_type not in ("text/plain", "application/pdf"):
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Only .txt and .pdf files are supported.",
        )

    # Read file in request context; pass bytes to thread (UploadFile is not safe across threads)
    file_bytes = await file.read()
    filename = file.filename or "document"

    t0 = time.perf_counter()
    try:
        document, chunks_created = await run_in_threadpool(
            document_service.upload_document_bytes,
            current_user["id"],
            file_bytes,
            filename,
        )
    except StorageApiError as e:
        bucket = get_settings().SUPABASE_STORAGE_BUCKET
        if e.status == 404 or (e.message and "not found" in e.message.lower()):
            raise HTTPException(
                status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
                detail=(
                    f"Storage bucket '{bucket}' not found. "
                    "Create it in Supabase Dashboard → Storage → New bucket."
                ),
            ) from e
        raise HTTPException(
            status_code=status.HTTP_502_BAD_GATEWAY,
            detail=e.message or "Storage error",
        ) from e

    logger.info(
        "POST /documents/upload request total: %.1f ms "
        "(%r, %d chunks, user submit → saved in database)",
        (time.perf_counter() - t0) * 1000.0,
        filename,
        chunks_created,
    )
    return DocumentUploadResponse(document=document, chunks_created=chunks_created)


@router.get("", response_model=List[DocumentItem])
async def list_documents(
    current_user: Annotated[dict, Depends(get_current_user)],
    document_service: Annotated[DocumentService, Depends(get_document_service)],
):
    docs = await run_in_threadpool(
        document_service.list_documents, current_user["id"]
    )
    return docs


@router.delete(
    "/{document_id}",
    response_model=MessageResponse,
)
async def delete_document(
    document_id: int,
    current_user: Annotated[dict, Depends(get_current_user)],
    document_service: Annotated[DocumentService, Depends(get_document_service)],
):
    await run_in_threadpool(
        document_service.delete_document, current_user["id"], document_id
    )
    return MessageResponse(detail="Document deleted")

