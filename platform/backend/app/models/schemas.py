from __future__ import annotations

from datetime import datetime
from typing import List, Optional

from pydantic import BaseModel, Field, field_validator


#
# Chat
#


class ChatRequest(BaseModel):
    message: str = Field(..., min_length=1)
    conversation_id: Optional[str] = None


class ChatItem(BaseModel):
    id: str
    user_id: str
    conversation_id: Optional[str] = None
    message: str
    response: str
    response_source: Optional[str] = None
    response_document_name: Optional[str] = None
    created_at: datetime

    @field_validator("id", "conversation_id", mode="before")
    @classmethod
    def id_to_str(cls, v: object) -> Optional[str]:
        return str(v) if v is not None else None


class ChatResponse(BaseModel):
    message: str
    response: str
    conversation_id: str
    response_source: Optional[str] = None
    response_document_name: Optional[str] = None


#
# Conversations (one user can have many independent chats)
#


class ConversationCreate(BaseModel):
    title: Optional[str] = Field(default=None, max_length=200)


class ConversationUpdate(BaseModel):
    title: str = Field(..., min_length=1, max_length=200)


class ConversationItem(BaseModel):
    id: str
    user_id: str
    title: str
    created_at: datetime
    updated_at: Optional[datetime] = None

    @field_validator("id", mode="before")
    @classmethod
    def id_to_str(cls, v: object) -> str:
        return str(v) if v is not None else ""


#
# FAQ
#


class FAQCreate(BaseModel):
    question: str = Field(..., min_length=1)
    answer: str = Field(..., min_length=1)


class FAQItem(BaseModel):
    id: str
    user_id: str
    question: str
    answer: str
    embedding: List[float]
    created_at: datetime

    @field_validator("id", mode="before")
    @classmethod
    def id_to_str(cls, v: object) -> str:
        return str(v) if v is not None else ""


class FAQImportResult(BaseModel):
    created: int
    skipped: int
    items: List[FAQItem] = Field(default_factory=list)


#
# Documents
#


class DocumentItem(BaseModel):
    id: int
    user_id: str
    filename: str
    storage_path: str
    created_at: datetime

    @field_validator("id", mode="before")
    @classmethod
    def id_to_int(cls, v: object) -> int:
        if v is None:
            return 0
        return int(v)


class DocumentUploadResponse(BaseModel):
    document: DocumentItem
    chunks_created: int


#
# Generic
#


class MessageResponse(BaseModel):
    detail: str

