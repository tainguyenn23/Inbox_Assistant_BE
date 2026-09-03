from datetime import datetime
from uuid import UUID

import asyncpg

from app.schemas.analytics import TopProductMetric


async def get_activity_counts(
    conn: asyncpg.Connection,
    shop_id: UUID,
    period_start: datetime,
    period_end: datetime,
) -> tuple[int, int]:
    row = await conn.fetchrow(
        """
        SELECT
            (
                SELECT COUNT(*)
                FROM conversations
                WHERE shop_id = $1
                  AND created_at >= $2
                  AND created_at < $3
            ) AS conversations,
            (
                SELECT COUNT(*)
                FROM messages
                WHERE shop_id = $1
                  AND created_at >= $2
                  AND created_at < $3
            ) AS messages
        """,
        shop_id,
        period_start,
        period_end,
    )
    if row is None:
        return 0, 0
    return int(row["conversations"]), int(row["messages"])


async def get_top_query_counts(
    conn: asyncpg.Connection,
    shop_id: UUID,
    period_start: datetime,
    period_end: datetime,
    *,
    limit: int,
) -> list[tuple[str, int]]:
    rows = await conn.fetch(
        """
        SELECT content, COUNT(*) AS query_count
        FROM messages
        WHERE shop_id = $1
          AND role = 'user'
          AND created_at >= $2
          AND created_at < $3
        GROUP BY content
        ORDER BY query_count DESC, content ASC
        LIMIT $4
        """,
        shop_id,
        period_start,
        period_end,
        limit,
    )
    return [(row["content"], int(row["query_count"])) for row in rows]


async def get_top_recommended(
    conn: asyncpg.Connection,
    shop_id: UUID,
    period_start: datetime,
    period_end: datetime,
    *,
    limit: int,
) -> list[TopProductMetric]:
    rows = await conn.fetch(
        """
        SELECT p.id AS product_id, p.name, COUNT(*) AS metric_count
        FROM recommendations AS r
        JOIN products AS p
          ON p.id = r.product_id AND p.shop_id = r.shop_id
        WHERE r.shop_id = $1
          AND r.created_at >= $2
          AND r.created_at < $3
        GROUP BY p.id, p.name
        ORDER BY metric_count DESC, p.name ASC, p.id ASC
        LIMIT $4
        """,
        shop_id,
        period_start,
        period_end,
        limit,
    )
    return [
        TopProductMetric(
            product_id=row["product_id"],
            name=row["name"],
            count=int(row["metric_count"]),
        )
        for row in rows
    ]


async def get_top_clicked(
    conn: asyncpg.Connection,
    shop_id: UUID,
    period_start: datetime,
    period_end: datetime,
    *,
    limit: int,
) -> list[TopProductMetric]:
    rows = await conn.fetch(
        """
        SELECT p.id AS product_id, p.name, COUNT(*) AS metric_count
        FROM click_events AS ce
        JOIN products AS p
          ON p.id = ce.product_id AND p.shop_id = ce.shop_id
        WHERE ce.shop_id = $1
          AND ce.created_at >= $2
          AND ce.created_at < $3
        GROUP BY p.id, p.name
        ORDER BY metric_count DESC, p.name ASC, p.id ASC
        LIMIT $4
        """,
        shop_id,
        period_start,
        period_end,
        limit,
    )
    return [
        TopProductMetric(
            product_id=row["product_id"],
            name=row["name"],
            count=int(row["metric_count"]),
        )
        for row in rows
    ]
