import hashlib
import json
import math
from collections.abc import Iterable
from uuid import UUID

import asyncpg
from app.rag.constants import (
    CHUNK_TYPE,
    EMBEDDING_DIMENSION,
    EMBEDDING_FORMAT_VERSION,
    EMBEDDING_PURPOSE,
    NORMALIZATION_VERSION,
)
from app.schemas.embedding import EmbeddingRecord
from app.schemas.retrieval import SemanticSearchHit


def _validate_vector(embedding: list[float], *, field_name: str) -> None:
    if len(embedding) != EMBEDDING_DIMENSION:
        raise ValueError(
            f"{field_name} must contain exactly {EMBEDDING_DIMENSION} values"
        )
    if not all(math.isfinite(float(value)) for value in embedding):
        raise ValueError(f"{field_name} must contain only finite numbers")
    if math.sqrt(sum(float(value) ** 2 for value in embedding)) <= 0:
        raise ValueError(f"{field_name} must not be a zero vector")


def _embedding_to_vector_literal(embedding: list[float]) -> str:
    return "[" + ",".join(str(float(value)) for value in embedding) + "]"


def _embedding_from_db(value: object) -> list[float]:
    if isinstance(value, str):
        raw_values = value.strip("[]")
        return [float(item) for item in raw_values.split(",")] if raw_values else []
    if isinstance(value, Iterable):
        return [float(item) for item in value]
    raise TypeError(f"Unsupported embedding value type: {type(value).__name__}")


def _jsonb_to_dict(value: object) -> dict[str, object]:
    if value is None:
        return {}
    if isinstance(value, dict):
        return value
    if isinstance(value, str):
        parsed = json.loads(value)
        if isinstance(parsed, dict):
            return parsed
    raise TypeError("Embedding metadata must be a JSON object")


async def upsert_embedding(
    conn: asyncpg.Connection,
    shop_id: UUID,
    product_id: UUID,
    content: str,
    embedding: list[float],
    model: str,
    *,
    chunk_index: int = 0,
    content_hash: str | None = None,
    purpose: str = EMBEDDING_PURPOSE,
    metadata: dict[str, object] | None = None,
) -> None:
    if chunk_index < 0:
        raise ValueError("chunk_index must be greater than or equal to zero")
    if not content.strip():
        raise ValueError("content must not be empty")
    if not model.strip():
        raise ValueError("model must not be empty")
    if not purpose.strip():
        raise ValueError("purpose must not be empty")
    _validate_vector(embedding, field_name="embedding")

    resolved_hash = content_hash or hashlib.sha256(content.encode("utf-8")).hexdigest()
    resolved_metadata = {
        **(metadata or {}),
        "format_version": EMBEDDING_FORMAT_VERSION,
        "normalization_version": NORMALIZATION_VERSION,
        "dimension": EMBEDDING_DIMENSION,
        "chunk_type": CHUNK_TYPE,
    }
    await conn.execute(
        """
        INSERT INTO product_embeddings (
            shop_id, product_id, chunk_index, content, content_hash,
            embedding, embedding_model, embedding_purpose, metadata
        )
        VALUES ($1, $2, $3, $4, $5, $6::vector, $7, $8, $9::jsonb)
        ON CONFLICT (product_id, chunk_index)
        DO UPDATE SET
            shop_id = EXCLUDED.shop_id,
            content = EXCLUDED.content,
            content_hash = EXCLUDED.content_hash,
            embedding = EXCLUDED.embedding,
            embedding_model = EXCLUDED.embedding_model,
            embedding_purpose = EXCLUDED.embedding_purpose,
            metadata = EXCLUDED.metadata,
            updated_at = NOW()
        """,
        shop_id,
        product_id,
        chunk_index,
        content,
        resolved_hash,
        _embedding_to_vector_literal(embedding),
        model,
        purpose,
        json.dumps(resolved_metadata, ensure_ascii=False, sort_keys=True),
    )


async def semantic_search(
    conn: asyncpg.Connection,
    shop_id: UUID,
    query_embedding: list[float],
    embedding_model: str,
    purpose: str = EMBEDDING_PURPOSE,
    limit: int = 20,
) -> list[SemanticSearchHit]:
    if not embedding_model.strip():
        raise ValueError("embedding_model must not be empty")
    if not purpose.strip():
        raise ValueError("purpose must not be empty")
    if limit <= 0:
        raise ValueError("limit must be greater than zero")
    _validate_vector(query_embedding, field_name="query_embedding")

    rows = await conn.fetch(
        """
        SELECT
            pe.product_id,
            1 - (pe.embedding <=> $2::vector) AS similarity
        FROM product_embeddings AS pe
        JOIN products AS p
          ON p.id = pe.product_id AND p.shop_id = pe.shop_id
        WHERE pe.shop_id = $1
          AND p.status = 'active'
          AND pe.embedding_model = $3
          AND pe.embedding_purpose = $4
        ORDER BY pe.embedding <=> $2::vector
        LIMIT $5
        """,
        shop_id,
        _embedding_to_vector_literal(query_embedding),
        embedding_model,
        purpose,
        limit,
    )
    return [
        SemanticSearchHit(
            product_id=row["product_id"],
            similarity=float(row["similarity"]),
        )
        for row in rows
    ]


async def get_embedding(
    conn: asyncpg.Connection,
    shop_id: UUID,
    product_id: UUID,
    chunk_index: int = 0,
) -> EmbeddingRecord | None:
    row = await conn.fetchrow(
        """
        SELECT
            chunk_index, content, content_hash, embedding, embedding_model,
            embedding_purpose, metadata
        FROM product_embeddings
        WHERE shop_id = $1 AND product_id = $2 AND chunk_index = $3
        """,
        shop_id,
        product_id,
        chunk_index,
    )
    if row is None:
        return None
    return EmbeddingRecord(
        chunk_index=row["chunk_index"],
        content=row["content"],
        content_hash=row["content_hash"],
        embedding=_embedding_from_db(row["embedding"]),
        model=row["embedding_model"],
        purpose=row["embedding_purpose"],
        metadata=_jsonb_to_dict(row["metadata"]),
    )


get_embedding_by_product_id = get_embedding


async def delete_stale_chunks(
    conn: asyncpg.Connection,
    shop_id: UUID,
    product_id: UUID,
    valid_chunk_indexes: list[int],
) -> int:
    if any(chunk_index < 0 for chunk_index in valid_chunk_indexes):
        raise ValueError("valid_chunk_indexes must not contain negative values")
    result = await conn.execute(
        """
        DELETE FROM product_embeddings
        WHERE shop_id = $1
          AND product_id = $2
          AND NOT (chunk_index = ANY($3::integer[]))
        """,
        shop_id,
        product_id,
        valid_chunk_indexes,
    )
    return int(result.rsplit(" ", 1)[-1])


async def delete_embeddings_by_shop(
    conn: asyncpg.Connection,
    shop_id: UUID,
) -> int:
    result = await conn.execute(
        "DELETE FROM product_embeddings WHERE shop_id = $1",
        shop_id,
    )
    return int(result.rsplit(" ", 1)[-1])


async def delete_inactive_product_embeddings(
    conn: asyncpg.Connection,
    shop_id: UUID,
) -> int:
    """Delete only embeddings whose current product is not retrieval-eligible.

    Deleted products are already handled by the product embedding FK cascade.
    """
    result = await conn.execute(
        """
        DELETE FROM product_embeddings AS pe
        USING products AS p
        WHERE pe.shop_id = $1
          AND p.shop_id = $1
          AND p.id = pe.product_id
          AND p.status <> 'active'
        """,
        shop_id,
    )
    return int(result.rsplit(" ", 1)[-1])
