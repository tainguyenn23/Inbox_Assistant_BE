"""Safe incremental and forced product embedding reindex orchestration."""

from uuid import UUID

import asyncpg
from fastapi import status

from app.core.errors import SmartSalesException
from app.core.observability import bind_request_shop_id
from app.llm.base import LLMProvider
from app.rag.constants import (
    EMBEDDING_FORMAT_VERSION,
    EMBEDDING_PURPOSE,
    NORMALIZATION_VERSION,
)
from app.repositories import embedding_repository, product_repository, shop_repository
from app.schemas.embedding import (
    ReindexFailure,
    ReindexRequest,
    ReindexResponse,
)
from app.services.embedding_service import batch_refresh_product_embeddings


class ReindexShopNotFoundException(SmartSalesException):
    def __init__(self) -> None:
        super().__init__(
            code="SHOP_NOT_FOUND",
            message="Shop not found.",
            status_code=status.HTTP_404_NOT_FOUND,
        )


class ReindexProductSelectionException(SmartSalesException):
    """Do not reveal whether a missing selection belongs to another shop."""

    def __init__(self) -> None:
        super().__init__(
            code="REINDEX_PRODUCTS_NOT_FOUND",
            message="One or more selected active products were not found for this shop.",
            status_code=status.HTTP_404_NOT_FOUND,
        )


async def reindex_product_embeddings(
    conn: asyncpg.Connection,
    request: ReindexRequest,
    provider: LLMProvider,
) -> ReindexResponse:
    bind_request_shop_id(request.shop_id)
    if await shop_repository.get_shop_by_id(conn, request.shop_id) is None:
        raise ReindexShopNotFoundException()

    selected_ids = await product_repository.list_active_product_ids(
        conn,
        request.shop_id,
        request.product_ids,
    )
    if request.product_ids is not None and len(selected_ids) != len(
        request.product_ids
    ):
        raise ReindexProductSelectionException()

    results = await batch_refresh_product_embeddings(
        conn,
        request.shop_id,
        selected_ids,
        provider,
        force=request.force,
        active_only=True,
    )

    # Cleanup never touches active products. Failed active rebuilds retain their
    # previous rows; deleted products are removed by FK cascade.
    async with conn.transaction():
        cleaned_embeddings = (
            await embedding_repository.delete_inactive_product_embeddings(
                conn,
                request.shop_id,
            )
        )

    failures = [
        ReindexFailure(
            product_id=result.product_id,
            code=result.error_code or "EMBEDDING_REFRESH_FAILED",
        )
        for result in results
        if result.action == "failed"
    ]
    return ReindexResponse(
        shop_id=request.shop_id,
        force=request.force,
        scanned=len(selected_ids),
        reused=sum(result.action == "skipped" for result in results),
        reindexed=sum(result.action == "embedded" for result in results),
        failed=len(failures),
        cleaned_embeddings=cleaned_embeddings,
        failed_products=failures,
        model=provider.embedding_model_name,
        dimension=provider.embedding_dimension,
        purpose=EMBEDDING_PURPOSE,
        format_version=EMBEDDING_FORMAT_VERSION,
        normalization_version=NORMALIZATION_VERSION,
    )
