from __future__ import annotations

from typing import Annotated, List

from fastapi import APIRouter, Depends, HTTPException, status
from starlette.concurrency import run_in_threadpool

from app.dependencies import get_current_user, get_chat_service
from app.models.schemas import (
    ChatItem,
    ConversationCreate,
    ConversationItem,
    ConversationUpdate,
    MessageResponse,
)
from app.services.chat_service import ChatService

router = APIRouter(prefix="/conversations", tags=["conversations"])


@router.get("", response_model=List[ConversationItem])
async def list_conversations(
    current_user: Annotated[dict, Depends(get_current_user)],
    chat_service: Annotated[ChatService, Depends(get_chat_service)],
):
    return await run_in_threadpool(chat_service.list_conversations, current_user["id"])


@router.post("", response_model=ConversationItem, status_code=status.HTTP_201_CREATED)
async def create_conversation(
    payload: ConversationCreate,
    current_user: Annotated[dict, Depends(get_current_user)],
    chat_service: Annotated[ChatService, Depends(get_chat_service)],
):
    return await run_in_threadpool(
        chat_service.create_conversation, current_user["id"], payload.title
    )


@router.patch("/{conversation_id}", response_model=ConversationItem)
async def rename_conversation(
    conversation_id: str,
    payload: ConversationUpdate,
    current_user: Annotated[dict, Depends(get_current_user)],
    chat_service: Annotated[ChatService, Depends(get_chat_service)],
):
    updated = await run_in_threadpool(
        chat_service.rename_conversation, current_user["id"], conversation_id, payload.title
    )
    if updated is None:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="Conversation not found")
    return updated


@router.delete("/{conversation_id}", response_model=MessageResponse)
async def delete_conversation(
    conversation_id: str,
    current_user: Annotated[dict, Depends(get_current_user)],
    chat_service: Annotated[ChatService, Depends(get_chat_service)],
):
    await run_in_threadpool(chat_service.delete_conversation, current_user["id"], conversation_id)
    return MessageResponse(detail="Conversation deleted")


@router.get("/{conversation_id}/messages", response_model=List[ChatItem])
async def conversation_messages(
    conversation_id: str,
    current_user: Annotated[dict, Depends(get_current_user)],
    chat_service: Annotated[ChatService, Depends(get_chat_service)],
):
    return await run_in_threadpool(
        chat_service.list_messages, current_user["id"], conversation_id
    )
