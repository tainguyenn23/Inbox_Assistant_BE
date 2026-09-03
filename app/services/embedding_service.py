import asyncio
import hashlib
import logging
import math
from time import perf_counter
from uuid import UUID

import asyncpg

from app.core.config import settings
from app.core.errors import EmbeddingDimensionException
from app.core.observability import bind_request_shop_id
from app.llm.base import LLMProvider, LLMProviderError
from app.rag.constants import (
    CHUNK_TYPE,
    DEFAULT_CHUNK_INDEX,
    EMBEDDING_DIMENSION,
    EMBEDDING_FORMAT_VERSION,
    EMBEDDING_PURPOSE,
    NORMALIZATION_VERSION,
)
from app.rag.product_content_builder import build_product_embedding_content
from app.rag.text_normalizer import normalize_embedding_text
from app.repositories import embedding_repository, product_repository
from app.schemas.embedding import EmbeddingRecord, EmbeddingRefreshResult
from app.schemas.product import ProductOut


logger = logging.getLogger(__name__)


class _ProductNotActiveError(RuntimeError):
    pass


def _refresh_error_code(exc: Exception) -> str:
    if isinstance(exc, _ProductNotActiveError):
        return "PRODUCT_NOT_ACTIVE"
    if isinstance(exc, EmbeddingDimensionException):
        return "EMBEDDING_DIMENSION_MISMATCH"
    if isinstance(exc, LLMProviderError):
        return "PROVIDER_ERROR"
    if isinstance(exc, (TimeoutError, ConnectionError)):
        return "PROVIDER_UNAVAILABLE"
    return "EMBEDDING_REFRESH_FAILED"


def _log_embedding_action(
    *,
    shop_id: UUID | None,
    product_id: UUID,
    action: str,
    provider: LLMProvider,
    content_hash: str = "",
    error_type: str | None = None,
) -> None:
    logger.info(
        "embedding_refresh_completed",
        extra={
            "shop_id": str(shop_id) if shop_id is not None else None,
            "product_id": str(product_id),
            "action": action,
            "model": provider.embedding_model_name,
            "dimension": provider.embedding_dimension,
            "format_version": EMBEDDING_FORMAT_VERSION,
            "content_hash_prefix": content_hash[:12] or None,
            "error_type": error_type,
        },
    )


def _content_and_hash(product: ProductOut) -> tuple[str, str]:
    content = normalize_embedding_text(build_product_embedding_content(product))
    digest = hashlib.sha256(content.encode("utf-8")).hexdigest()
    return content, digest


def _metadata() -> dict[str, object]:
    return {
        "format_version": EMBEDDING_FORMAT_VERSION,
        "normalization_version": NORMALIZATION_VERSION,
        "dimension": EMBEDDING_DIMENSION,
        "chunk_type": CHUNK_TYPE,
    }


def _validate_vector(vector: list[float], expected_dimension: int) -> None:
    if len(vector) != expected_dimension:
        raise EmbeddingDimensionException(expected_dimension, len(vector))
    if not all(math.isfinite(value) for value in vector):
        raise LLMProviderError("Embedding vector contains NaN or infinity")
    if math.sqrt(sum(value * value for value in vector)) <= 0:
        raise LLMProviderError("Embedding vector must not be zero")


def _can_reuse(
    existing: EmbeddingRecord | None,
    content_hash: str,
    provider: LLMProvider,
) -> bool:
    if existing is None:
        return False
    metadata = existing.metadata
    return (
        existing.content_hash == content_hash
        and existing.model == provider.embedding_model_name
        and len(existing.embedding) == provider.embedding_dimension
        and existing.purpose == EMBEDDING_PURPOSE
        and metadata.get("dimension") == provider.embedding_dimension
        and metadata.get("format_version") == EMBEDDING_FORMAT_VERSION
        and metadata.get("normalization_version") == NORMALIZATION_VERSION
        and metadata.get("chunk_type") == CHUNK_TYPE
    )


async def create_product_embedding(
    product: ProductOut,
    llm_provider: LLMProvider,
) -> EmbeddingRecord:
    content, content_hash = _content_and_hash(product)
    vector = await llm_provider.embed_document(content)
    _validate_vector(vector, settings.embedding_dimension)
    _log_embedding_action(
        shop_id=product.shop_id,
        product_id=product.id,
        action="embedded",
        provider=llm_provider,
        content_hash=content_hash,
    )
    return EmbeddingRecord(
        chunk_index=DEFAULT_CHUNK_INDEX,
        content=content,
        content_hash=content_hash,
        embedding=vector,
        model=llm_provider.embedding_model_name,
        purpose=EMBEDDING_PURPOSE,
        metadata=_metadata(),
    )


async def refresh_product_embedding(
    conn: asyncpg.Connection,
    shop_id: UUID,
    product_id: UUID,
    llm_provider: LLMProvider,
    *,
    force: bool = False,
    active_only: bool = False,
    _db_lock: asyncio.Lock | None = None,
) -> EmbeddingRefreshResult:
    bind_request_shop_id(shop_id)
    content_hash = ""
    async def load_state() -> tuple[
        ProductOut | None,
        str,
        str,
        EmbeddingRecord | None,
    ]:
        product = await product_repository.get_product_with_variants(
            conn, shop_id, product_id
        )
        if product is None:
            return None, "", "", None

        content, content_hash = _content_and_hash(product)
        existing = await embedding_repository.get_embedding(
            conn, shop_id, product_id, DEFAULT_CHUNK_INDEX
        )
        return product, content, content_hash, existing

    async def persist_embedding(
        content: str,
        content_hash: str,
        vector: list[float],
    ) -> None:
        # Chunk 0 and stale-chunk cleanup must commit or roll back together.
        async with conn.transaction():
            if active_only:
                current_status = await conn.fetchval(
                    """
                    SELECT status
                    FROM products
                    WHERE shop_id = $1 AND id = $2
                    FOR KEY SHARE
                    """,
                    shop_id,
                    product_id,
                )
                if current_status != "active":
                    raise _ProductNotActiveError(
                        "Product is no longer active"
                    )
            await embedding_repository.upsert_embedding(
                conn,
                shop_id,
                product_id,
                content,
                vector,
                llm_provider.embedding_model_name,
                chunk_index=DEFAULT_CHUNK_INDEX,
                content_hash=content_hash,
                purpose=EMBEDDING_PURPOSE,
                metadata=_metadata(),
            )
            await embedding_repository.delete_stale_chunks(
                conn,
                shop_id,
                product_id,
                [DEFAULT_CHUNK_INDEX],
            )

    try:
        if _db_lock is None:
            product, content, content_hash, existing = await load_state()
        else:
            async with _db_lock:
                product, content, content_hash, existing = await load_state()

        if product is None:
            _log_embedding_action(
                shop_id=shop_id,
                product_id=product_id,
                action="failed",
                provider=llm_provider,
                error_type="ProductNotFound",
            )
            return EmbeddingRefreshResult(
                product_id=product_id,
                action="failed",
                error="Product not found",
                error_code="PRODUCT_NOT_FOUND",
            )

        if active_only and product.status != "active":
            _log_embedding_action(
                shop_id=shop_id,
                product_id=product_id,
                action="failed",
                provider=llm_provider,
                content_hash=content_hash,
                error_type="ProductNotActive",
            )
            return EmbeddingRefreshResult(
                product_id=product_id,
                action="failed",
                error="Product is not active",
                error_code="PRODUCT_NOT_ACTIVE",
            )

        if not force and _can_reuse(existing, content_hash, llm_provider):
            _log_embedding_action(
                shop_id=shop_id,
                product_id=product_id,
                action="skipped",
                provider=llm_provider,
                content_hash=content_hash,
            )
            return EmbeddingRefreshResult(
                product_id=product_id,
                action="skipped",
            )

        vector = await llm_provider.embed_document(content)
        _validate_vector(vector, llm_provider.embedding_dimension)

        if _db_lock is None:
            await persist_embedding(content, content_hash, vector)
        else:
            async with _db_lock:
                await persist_embedding(content, content_hash, vector)
    except Exception as exc:  # noqa: BLE001
        _log_embedding_action(
            shop_id=shop_id,
            product_id=product_id,
            action="failed",
            provider=llm_provider,
            content_hash=content_hash,
            error_type=type(exc).__name__,
        )
        return EmbeddingRefreshResult(
            product_id=product_id,
            action="failed",
            error=str(exc),
            error_code=_refresh_error_code(exc),
        )
    _log_embedding_action(
        shop_id=shop_id,
        product_id=product_id,
        action="embedded",
        provider=llm_provider,
        content_hash=content_hash,
    )
    return EmbeddingRefreshResult(product_id=product_id, action="embedded")


async def batch_refresh_product_embeddings(
    conn: asyncpg.Connection,
    shop_id: UUID,
    product_ids: list[UUID],
    llm_provider: LLMProvider,
    *,
    force: bool = False,
    active_only: bool = False,
    concurrency: int | None = None,
) -> list[EmbeddingRefreshResult]:
    bind_request_shop_id(shop_id)
    started_at = perf_counter()
    resolved_concurrency = (
        settings.embedding_concurrency
        if concurrency is None
        else concurrency
    )
    if resolved_concurrency < 1:
        raise ValueError("concurrency must be greater than or equal to one")

    limiter = asyncio.Semaphore(resolved_concurrency)
    # asyncpg.Connection only permits one active operation. Serialize DB
    # phases while allowing independent provider calls to run concurrently.
    db_lock = asyncio.Lock()
    progress_lock = asyncio.Lock()
    completed = 0
    reused = 0
    rebuilt = 0
    failed = 0

    async def refresh(product_id: UUID) -> EmbeddingRefreshResult:
        nonlocal completed, reused, rebuilt, failed
        async with limiter:
            try:
                result = await refresh_product_embedding(
                    conn,
                    shop_id,
                    product_id,
                    llm_provider,
                    force=force,
                    active_only=active_only,
                    _db_lock=db_lock,
                )
            except Exception as exc:  # noqa: BLE001
                # Defensive boundary: one unexpected product failure must not
                # cancel sibling refreshes in asyncio.gather().
                result = EmbeddingRefreshResult(
                    product_id=product_id,
                    action="failed",
                    error=str(exc),
                    error_code="EMBEDDING_REFRESH_FAILED",
                )
            async with progress_lock:
                completed += 1
                reused += int(result.action == "skipped")
                rebuilt += int(result.action == "embedded")
                failed += int(result.action == "failed")
                if (
                    completed == len(product_ids)
                    or completed % settings.reindex_progress_interval == 0
                ):
                    logger.info(
                        "reindex_progress",
                        extra={
                            "shop_id": str(shop_id),
                            "force": force,
                            "scanned": len(product_ids),
                            "completed": completed,
                            "reused": reused,
                            "rebuilt": rebuilt,
                            "failed": failed,
                        },
                    )
            return result

    results = await asyncio.gather(
        *(refresh(product_id) for product_id in product_ids)
    )
    logger.info(
        "reindex_completed",
        extra={
            "shop_id": str(shop_id),
            "force": force,
            "scanned": len(product_ids),
            "reused": reused,
            "rebuilt": rebuilt,
            "failed": failed,
            "model": llm_provider.embedding_model_name,
            "duration_ms": round((perf_counter() - started_at) * 1000, 2),
        },
    )
    return results


async def batch_create_embeddings(
    products: list[ProductOut],
    llm_provider: LLMProvider,
    semaphore: asyncio.Semaphore | None = None,
) -> list[EmbeddingRecord]:
    """Compatibility helper for callers that already loaded products."""
    limiter = semaphore or asyncio.Semaphore(settings.embedding_concurrency)

    async def create(product: ProductOut) -> EmbeddingRecord:
        async with limiter:
            return await create_product_embedding(product, llm_provider)

    return await asyncio.gather(*(create(product) for product in products))
