from uuid import UUID

from pydantic import BaseModel, Field, field_validator

from app.schemas.conversation import ConversationChannel
from app.schemas.intent import IntentName
from app.schemas.reply import ProductCard


class ChatRequest(BaseModel):
    shop_id: UUID
    conversation_id: UUID | None = None
    customer_id: str = Field(..., min_length=1, max_length=200)
    channel: ConversationChannel = "web"
    message: str = Field(..., min_length=1, max_length=10_000)

    @field_validator("customer_id")
    @classmethod
    def normalize_customer_id(cls, value: str) -> str:
        normalized = value.strip()
        if not normalized:
            raise ValueError("customer_id must not be empty")
        return normalized


class ChatResponse(BaseModel):
    conversation_id: UUID
    reply: str
    intent: IntentName
    needs_human: bool
    confidence: float = Field(..., ge=0.0, le=1.0)
    products: list[ProductCard] = Field(default_factory=list)
