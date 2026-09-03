from dataclasses import dataclass
from typing import Literal
from uuid import UUID

from pydantic import BaseModel, Field, field_validator

@dataclass(frozen=True)
class EmbeddingRecord:
    chunk_index: int
    content: str
    content_hash: str
    embedding: list[float]
    model: str
    purpose: str
    metadata: dict[str, object]


@dataclass(frozen=True)
class EmbeddingRefreshResult:
    product_id: UUID
    action: str
    error: str | None = None
    error_code: str | None = None


class ReindexRequest(BaseModel):
    shop_id: UUID
    force: bool = False
    product_ids: list[UUID] | None = Field(
        default=None,
        min_length=1,
        max_length=1000,
    )

    @field_validator("product_ids")
    @classmethod
    def deduplicate_product_ids(
        cls,
        value: list[UUID] | None,
    ) -> list[UUID] | None:
        if value is None:
            return None
        return list(dict.fromkeys(value))


class ReindexFailure(BaseModel):
    product_id: UUID
    code: str


class ReindexResponse(BaseModel):
    shop_id: UUID
    force: bool
    scanned: int = Field(..., ge=0)
    reused: int = Field(..., ge=0)
    reindexed: int = Field(..., ge=0)
    failed: int = Field(..., ge=0)
    cleaned_embeddings: int = Field(default=0, ge=0)
    failed_products: list[ReindexFailure] = Field(default_factory=list)
    model: str
    dimension: int = Field(..., gt=0)
    purpose: Literal["product_retrieval"] = "product_retrieval"
    format_version: str
    normalization_version: str
