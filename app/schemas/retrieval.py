from decimal import Decimal
from typing import Literal
from uuid import UUID

from pydantic import (
    BaseModel,
    ConfigDict,
    Field,
    computed_field,
    field_validator,
    model_validator,
)

from app.schemas.product import ProductOut

RetrievalSource = Literal["metadata", "keyword", "semantic"]
_RETRIEVAL_SOURCE_ORDER: dict[RetrievalSource, int] = {
    "metadata": 0,
    "keyword": 1,
    "semantic": 2,
}


class SemanticSearchHit(BaseModel):
    """Minimal result returned by the vector-search repository."""

    product_id: UUID
    similarity: float = Field(..., ge=-1.0, le=1.0)


class KeywordSearchHit(BaseModel):
    """Product identifier and normalized literal-keyword match score."""

    product_id: UUID
    score: float = Field(..., ge=0.0, le=1.0)


class RankingScoreComponents(BaseModel):
    """Auditable components used to produce a candidate's final score."""

    metadata: float
    keyword: float
    vector: float
    url_bonus: float
    unavailable_penalty: float
    unknown_stock_penalty: float
    raw_score: float
    final_score: float = Field(..., ge=0.0, le=1.0)


class ProductCandidate(BaseModel):
    """One hydrated, tenant-safe product collected by retrieval paths."""

    model_config = ConfigDict(extra="forbid", validate_assignment=True)

    product_id: UUID
    shop_id: UUID
    product_data: ProductOut
    metadata_match: bool = False
    keyword_score: float = Field(default=0.0, ge=0.0, le=1.0)
    vector_similarity: float = Field(default=0.0, ge=0.0, le=1.0)
    hybrid_score: float = Field(default=0.0, ge=0.0, le=1.0)
    score_components: RankingScoreComponents | None = None
    retrieval_sources: list[RetrievalSource] = Field(default_factory=list)

    @field_validator("retrieval_sources")
    @classmethod
    def normalize_retrieval_sources(
        cls,
        value: list[RetrievalSource],
    ) -> list[RetrievalSource]:
        return sorted(
            set(value),
            key=_RETRIEVAL_SOURCE_ORDER.__getitem__,
        )

    @model_validator(mode="after")
    def validate_hydrated_product(self) -> "ProductCandidate":
        if self.product_data.id != self.product_id:
            raise ValueError("product_data.id must match product_id")
        if self.product_data.shop_id != self.shop_id:
            raise ValueError("product_data.shop_id must match shop_id")
        if not self.product_data.variants:
            raise ValueError("product_data must include current variants")
        for variant in self.product_data.variants:
            if variant.product_id != self.product_id:
                raise ValueError("variant.product_id must match product_id")
            if variant.shop_id != self.shop_id:
                raise ValueError("variant.shop_id must match shop_id")
        return self

    @computed_field
    @property
    def has_available_variant(self) -> bool:
        return any(
            variant.stock_status == "in_stock"
            or (
                variant.stock_quantity is not None
                and variant.stock_quantity > 0
            )
            for variant in self.product_data.variants
        )

    @computed_field
    @property
    def min_price(self) -> Decimal | None:
        return self.product_data.min_price

    @computed_field
    @property
    def max_price(self) -> Decimal | None:
        return self.product_data.max_price
