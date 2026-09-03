import asyncio
import logging
from dataclasses import dataclass, field
from time import perf_counter
from typing import cast
from uuid import UUID

import asyncpg

from app.core.config import settings
from app.llm.base import LLMProvider
from app.rag.constants import EMBEDDING_PURPOSE
from app.rag.text_normalizer import normalize_embedding_text
from app.repositories import embedding_repository, product_repository
from app.schemas.intent import IntentResult
from app.schemas.retrieval import (
    KeywordSearchHit,
    ProductCandidate,
    RetrievalSource,
    SemanticSearchHit,
)

logger = logging.getLogger(__name__)

_PRODUCT_INTENTS = {"product_recommendation", "product_info"}
_RRF_K = 60
_MAX_KEYWORDS = 20
_MAX_KEYWORD_LENGTH = 100


@dataclass
class _MergedEvidence:
    product_id: UUID
    metadata_match: bool = False
    keyword_score: float = 0.0
    vector_similarity: float = 0.0
    retrieval_sources: set[RetrievalSource] = field(default_factory=set)
    reciprocal_rank_score: float = 0.0


def _normalize_keywords(keywords: list[str]) -> list[str]:
    normalized: list[str] = []
    seen: set[str] = set()
    for keyword in keywords:
        try:
            value = normalize_embedding_text(keyword)[:_MAX_KEYWORD_LENGTH]
        except (TypeError, ValueError):
            continue
        identity = value.casefold()
        if identity in seen:
            continue
        seen.add(identity)
        normalized.append(value)
        if len(normalized) == _MAX_KEYWORDS:
            break
    return normalized


def _has_metadata_filters(intent: IntentResult, available_only: bool) -> bool:
    return any(
        (
            intent.min_price is not None,
            intent.max_price is not None,
            bool(intent.category),
            bool(intent.color),
            bool(intent.size),
            bool(intent.attribute_filters),
            available_only,
        )
    )


async def _metadata_path(
    pool: asyncpg.Pool,
    shop_id: UUID,
    intent: IntentResult,
    available_only: bool,
    limit: int,
) -> list[UUID]:
    if not _has_metadata_filters(intent, available_only):
        return []
    async with pool.acquire() as conn:
        return await product_repository.metadata_filter_product_ids(
            conn,
            shop_id,
            min_price=intent.min_price,
            max_price=intent.max_price,
            category=intent.category,
            color=intent.color,
            size=intent.size,
            attribute_filters=intent.attribute_filters,
            available_only=available_only,
            limit=limit,
        )


async def _keyword_path(
    pool: asyncpg.Pool,
    shop_id: UUID,
    keywords: list[str],
    limit: int,
) -> list[KeywordSearchHit]:
    if not keywords:
        return []
    async with pool.acquire() as conn:
        return await product_repository.keyword_search_hits(
            conn,
            shop_id,
            keywords,
            limit=limit,
            include_variant_attributes=True,
        )


async def _semantic_path(
    pool: asyncpg.Pool,
    shop_id: UUID,
    normalized_message: str,
    provider: LLMProvider,
    limit: int,
) -> list[SemanticSearchHit]:
    query_vector = await provider.embed_query(normalized_message)
    async with pool.acquire() as conn:
        return await embedding_repository.semantic_search(
            conn,
            shop_id,
            query_vector,
            provider.embedding_model_name,
            EMBEDDING_PURPOSE,
            limit,
        )


def _path_value_or_empty(
    source: RetrievalSource,
    value: list[UUID] | list[KeywordSearchHit] | list[SemanticSearchHit] | BaseException,
) -> list[UUID] | list[KeywordSearchHit] | list[SemanticSearchHit]:
    if isinstance(value, asyncio.CancelledError):
        raise value
    if isinstance(value, BaseException):
        logger.warning(
            "Product retrieval path failed",
            extra={
                "retrieval_source": source,
                "error_type": type(value).__name__,
            },
        )
        return []
    return value


def _merge_hits(
    metadata_ids: list[UUID],
    keyword_hits: list[KeywordSearchHit],
    semantic_hits: list[SemanticSearchHit],
    limit: int,
) -> list[_MergedEvidence]:
    merged: dict[UUID, _MergedEvidence] = {}

    def evidence(product_id: UUID) -> _MergedEvidence:
        return merged.setdefault(product_id, _MergedEvidence(product_id=product_id))

    for rank, product_id in enumerate(metadata_ids, start=1):
        item = evidence(product_id)
        item.metadata_match = True
        item.retrieval_sources.add("metadata")
        item.reciprocal_rank_score += 1.0 / (_RRF_K + rank)

    for rank, hit in enumerate(keyword_hits, start=1):
        item = evidence(hit.product_id)
        item.keyword_score = max(item.keyword_score, hit.score)
        item.retrieval_sources.add("keyword")
        item.reciprocal_rank_score += 1.0 / (_RRF_K + rank)

    for rank, hit in enumerate(semantic_hits, start=1):
        item = evidence(hit.product_id)
        item.vector_similarity = max(
            item.vector_similarity,
            min(1.0, max(0.0, hit.similarity)),
        )
        item.retrieval_sources.add("semantic")
        item.reciprocal_rank_score += 1.0 / (_RRF_K + rank)

    return sorted(
        merged.values(),
        key=lambda item: (-item.reciprocal_rank_score, str(item.product_id)),
    )[:limit]


async def retrieve_products(
    pool: asyncpg.Pool,
    shop_id: UUID,
    user_message: str,
    intent: IntentResult,
    provider: LLMProvider,
    *,
    candidate_limit: int | None = None,
    available_only: bool = False,
) -> list[ProductCandidate]:
    """Run independent retrieval paths and hydrate current SQL product facts."""

    started_at = perf_counter()
    if intent.intent not in _PRODUCT_INTENTS:
        return []
    try:
        normalized_message = normalize_embedding_text(user_message)
    except (TypeError, ValueError):
        return []

    resolved_limit = (
        settings.retrieval_candidate_limit
        if candidate_limit is None
        else candidate_limit
    )
    if not 1 <= resolved_limit <= 100:
        raise ValueError("candidate_limit must be between 1 and 100")
    keywords = _normalize_keywords(intent.keywords)

    metadata_result, keyword_result, semantic_result = await asyncio.gather(
        _metadata_path(pool, shop_id, intent, available_only, resolved_limit),
        _keyword_path(pool, shop_id, keywords, resolved_limit),
        _semantic_path(
            pool,
            shop_id,
            normalized_message,
            provider,
            resolved_limit,
        ),
        return_exceptions=True,
    )

    metadata_ids = cast(
        list[UUID],
        _path_value_or_empty("metadata", metadata_result),
    )
    keyword_hits = cast(
        list[KeywordSearchHit],
        _path_value_or_empty("keyword", keyword_result),
    )
    semantic_hits = cast(
        list[SemanticSearchHit],
        _path_value_or_empty("semantic", semantic_result),
    )
    selected = _merge_hits(
        metadata_ids,
        keyword_hits,
        semantic_hits,
        resolved_limit,
    )
    retrieval_counts = {
        "metadata": len(metadata_ids),
        "keyword": len(keyword_hits),
        "semantic": len(semantic_hits),
        "merged": len(selected),
    }
    if not selected:
        logger.info(
            "rag_retrieval_completed",
            extra={
                "shop_id": str(shop_id),
                "retrieval_counts": retrieval_counts,
                "candidate_count": 0,
                "duration_ms": round((perf_counter() - started_at) * 1000, 2),
            },
        )
        return []

    evidence_by_id = {item.product_id: item for item in selected}
    async with pool.acquire() as conn:
        products = await product_repository.get_active_products_by_ids(
            conn,
            shop_id,
            [item.product_id for item in selected],
        )

    candidates: list[ProductCandidate] = []
    for product in products:
        item = evidence_by_id.get(product.id)
        if item is None or not product.variants:
            continue
        candidates.append(
            ProductCandidate(
                product_id=product.id,
                shop_id=shop_id,
                product_data=product,
                metadata_match=item.metadata_match,
                keyword_score=item.keyword_score,
                vector_similarity=item.vector_similarity,
                retrieval_sources=list(item.retrieval_sources),
            )
        )
    logger.info(
        "rag_retrieval_completed",
        extra={
            "shop_id": str(shop_id),
            "retrieval_counts": retrieval_counts,
            "candidate_count": len(candidates),
            "duration_ms": round((perf_counter() - started_at) * 1000, 2),
        },
    )
    return candidates
