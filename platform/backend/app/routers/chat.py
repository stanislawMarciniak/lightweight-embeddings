from __future__ import annotations

from typing import Annotated, List

from fastapi import APIRouter, Depends, status
from starlette.concurrency import run_in_threadpool

from app.dependencies import get_current_user, get_chat_service
from app.models.schemas import ChatItem, ChatRequest, ChatResponse
from app.services.chat_service import ChatService


router = APIRouter(prefix="", tags=["chat"])


@router.post(
    "/chat",
    response_model=ChatResponse,
    status_code=status.HTTP_200_OK,
)
async def create_chat(
    payload: ChatRequest,
    current_user: Annotated[dict, Depends(get_current_user)],
    chat_service: Annotated[ChatService, Depends(get_chat_service)],
):
    answer, conversation_id, response_source, response_document_name = await run_in_threadpool(
        chat_service.process_message,
        current_user["id"],
        payload.message,
        payload.conversation_id,
    )
    return ChatResponse(
        message=payload.message,
        response=answer,
        conversation_id=conversation_id,
        response_source=response_source,
        response_document_name=response_document_name,
    )


@router.get(
    "/chats",
    response_model=List[ChatItem],
)
async def list_chats(
    current_user: Annotated[dict, Depends(get_current_user)],
    chat_service: Annotated[ChatService, Depends(get_chat_service)],
):
    chats = await run_in_threadpool(
        chat_service.list_chats, current_user["id"]
    )
    return chats

