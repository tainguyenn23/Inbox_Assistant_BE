from datetime import datetime
from typing import Any
from uuid import UUID

from pydantic import BaseModel, Field, HttpUrl, model_validator


class ClickRequest(BaseModel):
    shop_id: UUID
    conversation_id: UUID | None = None
    product_id: UUID | None = None
    url: HttpUrl


class ClickEventOut(BaseModel):
    id: UUID
    shop_id: UUID
    conversation_id: UUID | None = None
    product_id: UUID | None = None
    url: HttpUrl
    metadata: dict[str, Any] = Field(default_factory=dict)
    created_at: datetime


class TopQuery(BaseModel):
    query: str
    count: int = Field(..., ge=1)


class TopProductMetric(BaseModel):
    product_id: UUID
    name: str
    count: int = Field(..., ge=1)


class AnalyticsSummary(BaseModel):
    shop_id: UUID
    days: int = Field(..., ge=1, le=365)
    period_start: datetime
    period_end: datetime
    conversations: int = Field(default=0, ge=0)
    messages: int = Field(default=0, ge=0)
    top_queries: list[TopQuery] = Field(default_factory=list)
    top_recommended: list[TopProductMetric] = Field(default_factory=list)
    top_clicked: list[TopProductMetric] = Field(default_factory=list)

    @model_validator(mode="after")
    def validate_period(self) -> "AnalyticsSummary":
        if self.period_start >= self.period_end:
            raise ValueError("period_start must be before period_end")
        return self
