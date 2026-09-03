import logging
from time import perf_counter
from uuid import UUID

import asyncpg
from fastapi import status

from app.core.errors import SmartSalesException
from app.core.observability import bind_request_shop_id
from app.llm.base import LLMProvider
from app.repositories import product_repository, shop_repository
from app.schemas.product import ImportError, ProductCreate, ProductImportResult
from app.services.embedding_service import refresh_product_embedding


logger = logging.getLogger(__name__)


class ImportShopNotFoundException(SmartSalesException):
    def __init__(self, shop_id: UUID) -> None:
        super().__init__(
            code="SHOP_NOT_FOUND",
            message=f"Shop not found: {shop_id}",
            status_code=status.HTTP_404_NOT_FOUND,
        )


async def import_products(
    conn: asyncpg.Connection,
    shop_id: UUID,
    products: list[ProductCreate],
    llm_provider: LLMProvider,
) -> ProductImportResult:
    """Persist normalized products, then refresh embeddings after each commit."""

    bind_request_shop_id(shop_id)
    started_at = perf_counter()
    if await shop_repository.get_shop_by_id(conn, shop_id) is None:
        raise ImportShopNotFoundException(shop_id)

    result = ProductImportResult(failed=0)
    for row_number, product in enumerate(products, start=1):
        try:
            async with conn.transaction():
                saved, was_created = (
                    await product_repository.upsert_product_by_external_identity(
                        conn,
                        shop_id,
                        product,
                    )
                )
        except Exception as exc:  # noqa: BLE001
            result.failed += 1
            result.errors.append(
                ImportError(
                    row_number=row_number,
                    code="PRODUCT_PERSIST_FAILED",
                    reason=str(exc),
                )
            )
            continue

        if was_created:
            result.created += 1
        else:
            result.updated += 1

        refresh = await refresh_product_embedding(
            conn,
            shop_id,
            saved.id,
            llm_provider,
        )
        if refresh.action == "embedded":
            result.embedded += 1
        elif refresh.action == "skipped":
            result.embedding_skipped += 1
        else:
            result.failed += 1
            result.errors.append(
                ImportError(
                    row_number=row_number,
                    code="EMBEDDING_FAILED",
                    reason=refresh.error or "Embedding refresh failed",
                )
            )
    logger.info(
        "product_import_completed",
        extra={
            "shop_id": str(shop_id),
            "import_counts": {
                "created": result.created,
                "updated": result.updated,
                "failed": result.failed,
                "embedded": result.embedded,
                "skipped": result.embedding_skipped,
            },
            "duration_ms": round((perf_counter() - started_at) * 1000, 2),
        },
    )
    return result
