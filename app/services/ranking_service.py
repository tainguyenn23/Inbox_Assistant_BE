from pydantic import BaseModel, ConfigDict, Field, model_validator

from app.core.config import settings
from app.schemas.retrieval import ProductCandidate, RankingScoreComponents


class RankingConfig(BaseModel):
    """Validated, configurable weights and business adjustments."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    metadata_weight: float = Field(0.30, ge=0.0, le=1.0)
    keyword_weight: float = Field(0.30, ge=0.0, le=1.0)
    vector_weight: float = Field(0.40, ge=0.0, le=1.0)
    url_bonus: float = Field(0.05, ge=0.0, le=1.0)
    unavailable_penalty: float = Field(0.20, ge=0.0, le=1.0)
    unknown_stock_penalty: float = Field(0.0, ge=0.0, le=1.0)
    top_n: int = Field(5, ge=1, le=100)

    @model_validator(mode="after")
    def validate_weight_total(self) -> "RankingConfig":
        total = self.metadata_weight + self.keyword_weight + self.vector_weight
        if abs(total - 1.0) > 1e-9:
            raise ValueError("ranking weights must sum to 1.0")
        return self

    @classmethod
    def from_settings(cls) -> "RankingConfig":
        return cls(
            metadata_weight=settings.ranking_metadata_weight,
            keyword_weight=settings.ranking_keyword_weight,
            vector_weight=settings.ranking_vector_weight,
            url_bonus=settings.ranking_url_bonus,
            unavailable_penalty=settings.ranking_unavailable_penalty,
            unknown_stock_penalty=settings.ranking_unknown_stock_penalty,
            top_n=settings.ranking_top_n,
        )


def _all_stock_unknown(candidate: ProductCandidate) -> bool:
    return all(
        variant.stock_status == "unknown" and variant.stock_quantity is None
        for variant in candidate.product_data.variants
    )


def _score_candidate(
    candidate: ProductCandidate,
    config: RankingConfig,
) -> ProductCandidate:
    metadata = config.metadata_weight * float(candidate.metadata_match)
    keyword = config.keyword_weight * candidate.keyword_score
    vector = config.vector_weight * candidate.vector_similarity
    has_current_url = bool(
        candidate.product_data.affiliate_url
        or candidate.product_data.product_url
    )
    url_bonus = config.url_bonus if has_current_url else 0.0
    unavailable_penalty = (
        0.0
        if candidate.has_available_variant
        else -config.unavailable_penalty
    )
    unknown_stock_penalty = (
        -config.unknown_stock_penalty
        if _all_stock_unknown(candidate)
        else 0.0
    )
    raw_score = (
        metadata
        + keyword
        + vector
        + url_bonus
        + unavailable_penalty
        + unknown_stock_penalty
    )
    final_score = min(1.0, max(0.0, raw_score))
    components = RankingScoreComponents(
        metadata=metadata,
        keyword=keyword,
        vector=vector,
        url_bonus=url_bonus,
        unavailable_penalty=unavailable_penalty,
        unknown_stock_penalty=unknown_stock_penalty,
        raw_score=raw_score,
        final_score=final_score,
    )
    return candidate.model_copy(
        update={
            "hybrid_score": final_score,
            "score_components": components,
        }
    )


def rank_products(
    candidates: list[ProductCandidate],
    *,
    config: RankingConfig | None = None,
    top_n: int | None = None,
) -> list[ProductCandidate]:
    """Score copies of candidates and return a stable, deterministic top N."""

    resolved_config = config or RankingConfig.from_settings()
    resolved_top_n = resolved_config.top_n if top_n is None else top_n
    if not 1 <= resolved_top_n <= 100:
        raise ValueError("top_n must be between 1 and 100")

    scored = [
        (index, _score_candidate(candidate, resolved_config))
        for index, candidate in enumerate(candidates)
    ]
    scored.sort(
        key=lambda item: (
            -item[1].hybrid_score,
            item[0],
            str(item[1].product_id),
        )
    )
    return [candidate for _, candidate in scored[:resolved_top_n]]
