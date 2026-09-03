import json
from dataclasses import dataclass
from typing import Any
from uuid import UUID

import asyncpg

from app.schemas.analytics import ClickEventOut


@dataclass(frozen=True)
class ProductClickTarget:
    product_id: UUID
    affiliate_url: str | None
    product_url: str | None

    @property
    def current_url(self) -> str | None:
        return self.affiliate_url or self.product_url


def _jsonb_to_dict(value: Any) -> dict[str, Any]:
    if value is None:
        return {}
    if isinstance(value, dict):
        return value
    if isinstance(value, str):
        parsed = json.loads(value)
        if isinstance(parsed, dict):
            return parsed
    raise TypeError("Click metadata must be a JSON object")


async def conversation_belongs_to_shop(
    conn: asyncpg.Connection,
    shop_id: UUID,
    conversation_id: UUID,
) -> bool:
    return bool(
        await conn.fetchval(
            """
            SELECT TRUE
            FROM conversations
            WHERE shop_id = $1 AND id = $2
            FOR KEY SHARE
            """,
            shop_id,
            conversation_id,
        )
    )


async def get_product_click_target(
    conn: asyncpg.Connection,
    shop_id: UUID,
    product_id: UUID,
) -> ProductClickTarget | None:
    row = await conn.fetchrow(
        """
        SELECT id, affiliate_url, product_url
        FROM products
        WHERE shop_id = $1 AND id = $2
        FOR KEY SHARE
        """,
        shop_id,
        product_id,
    )
    if row is None:
        return None
    return ProductClickTarget(
        product_id=row["id"],
        affiliate_url=row["affiliate_url"],
        product_url=row["product_url"],
    )


async def create_click_event(
    conn: asyncpg.Connection,
    shop_id: UUID,
    url: str,
    *,
    conversation_id: UUID | None = None,
    product_id: UUID | None = None,
    metadata: dict[str, Any] | None = None,
) -> ClickEventOut:
    row = await conn.fetchrow(
        """
        INSERT INTO click_events (
            shop_id, conversation_id, product_id, url, metadata, created_at
        )
        VALUES ($1, $2, $3, $4, $5::jsonb, clock_timestamp())
        RETURNING *
        """,
        shop_id,
        conversation_id,
        product_id,
        url,
        json.dumps(metadata or {}, ensure_ascii=False, sort_keys=True),
    )
    if row is None:
        raise RuntimeError("Failed to create click event")
    return ClickEventOut(
        id=row["id"],
        shop_id=row["shop_id"],
        conversation_id=row["conversation_id"],
        product_id=row["product_id"],
        url=row["url"],
        metadata=_jsonb_to_dict(row["metadata"]),
        created_at=row["created_at"],
    )
