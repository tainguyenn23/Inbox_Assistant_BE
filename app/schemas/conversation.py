from datetime import datetime
from typing import Any, Literal
from uuid import UUID

from pydantic import BaseModel, Field

ConversationChannel = Literal[
    "web",
    "zalo",
    "facebook",
    "shopee",
    "tiktok",
    "manual",
]
ConversationStatus = Literal["open", "closed", "needs_human"]
MessageRole = Literal["user", "assistant", "system"]


class ConversationOut(BaseModel):
    id: UUID
    shop_id: UUID
    customer_id: str
    channel: ConversationChannel
    status: ConversationStatus
    metadata: dict[str, Any] = Field(default_factory=dict)
    created_at: datetime
    updated_at: datetime


class MessageOut(BaseModel):
    id: UUID
    shop_id: UUID
    conversation_id: UUID
    role: MessageRole
    content: str
    metadata: dict[str, Any] = Field(default_factory=dict)
    created_at: datetime
