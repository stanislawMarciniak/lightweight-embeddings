from __future__ import annotations

import logging
from typing import Annotated

from fastapi import Depends, HTTPException, Header, status
from supabase import Client, create_client

from app.config import get_settings
from app.services.embedding_service import EmbeddingService


logger = logging.getLogger(__name__)


def get_supabase_client() -> Client:
    settings = get_settings()
    # Supabase client expects plain str; Settings.SUPABASE_URL is AnyHttpUrl
    url = str(settings.SUPABASE_URL)
    return create_client(url, settings.SUPABASE_KEY)


async def get_bearer_token(
    authorization: Annotated[str | None, Header(alias="Authorization")] = None,
) -> str:
    if not authorization:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Missing Authorization header",
        )
    parts = authorization.split()
    if len(parts) != 2 or parts[0].lower() != "bearer":
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Invalid Authorization header format",
        )
    return parts[1]


async def get_current_user(
    token: Annotated[str, Depends(get_bearer_token)],
    supabase: Annotated[Client, Depends(get_supabase_client)],
) -> dict:
    """
    Validate Supabase JWT and return user dict.
    """
    try:
        resp = supabase.auth.get_user(token)
    except Exception as exc:  # noqa: BLE001
        logger.exception("Error validating Supabase token")
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Invalid authentication credentials",
        ) from exc

    user = getattr(resp, "user", None) or getattr(resp, "data", None)
    if not user or not getattr(user, "id", None):
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Invalid authentication credentials",
        )

    # Normalize to dict with id field
    return {"id": str(user.id)}


def get_embedding_service() -> EmbeddingService:
    return EmbeddingService()


def get_chat_service(
    supabase: Annotated[Client, Depends(get_supabase_client)],
    embedding_service: Annotated[EmbeddingService, Depends(get_embedding_service)],
):
    # Lazy imports avoid loading chat/openai modules (and their deps) at import time.
    from app.services.chat_service import ChatService
    from app.services.openai_service import OpenAIService

    return ChatService(supabase, embedding_service, OpenAIService())

