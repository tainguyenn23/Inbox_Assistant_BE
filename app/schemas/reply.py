from decimal import Decimal
from uuid import UUID

from pydantic import BaseModel, Field, HttpUrl, field_validator

from app.schemas.product import StockStatus


class ReplyResult(BaseModel):
    reply: str = Field(..., min_length=1, max_length=4000)
    used_product_ids: list[UUID] = Field(default_factory=list)

    @field_validator("reply")
    @classmethod
    def normalize_reply(cls, value: str) -> str:
        normalized = value.strip()
        if not normalized:
            raise ValueError("reply must not be empty")
        return normalized

    @field_validator("used_product_ids")
    @classmethod
    def deduplicate_product_ids(cls, value: list[UUID]) -> list[UUID]:
        return list(dict.fromkeys(value))


class ProductVariantPreview(BaseModel):
    attributes: dict[str, str] = Field(default_factory=dict)
    price: Decimal | None = None
    stock_status: StockStatus
    stock_quantity: int | None = Field(default=None, ge=0)


class ProductCard(BaseModel):
    id: UUID
    name: str
    image_url: HttpUrl | None = None
    url: HttpUrl | None = None
    currency: str
    min_price: Decimal | None = None
    max_price: Decimal | None = None
    availability: StockStatus
    variants_preview: list[ProductVariantPreview] = Field(default_factory=list)
    reason: str
    score: float = Field(..., ge=0.0, le=1.0)


class GroundedReplyOutput(BaseModel):
    result: ReplyResult
    product_cards: list[ProductCard] = Field(default_factory=list)
