"""Shop-scoped aggregate analytics with privacy-safe query snippets."""

from collections import defaultdict
from datetime import UTC, datetime, timedelta
import re
from uuid import UUID

import asyncpg

from app.core.config import settings
from app.core.errors import SmartSalesException
from app.core.observability import bind_request_shop_id
from app.repositories import analytics_repository, shop_repository
from app.schemas.analytics import AnalyticsSummary, TopQuery


_EMAIL_PATTERN = re.compile(r"\b[A-Z0-9._%+-]+@[A-Z0-9.-]+\.[A-Z]{2,}\b", re.I)
_PHONE_PATTERN = re.compile(r"(?<!\w)(?:\+?84|0)[\d .-]{8,14}(?!\w)")
_UUID_PATTERN = re.compile(
    r"\b[0-9a-f]{8}-[0-9a-f]{4}-[1-5][0-9a-f]{3}-"
    r"[89ab][0-9a-f]{3}-[0-9a-f]{12}\b",
    re.I,
)


class AnalyticsShopNotFoundException(SmartSalesException):
    def __init__(self) -> None:
        super().__init__(
            status_code=404,
            code="SHOP_NOT_FOUND",
            message="Shop not found.",
        )


def _query_snippet(value: str, max_length: int) -> str:
    snippet = " ".join(value.split())
    snippet = _EMAIL_PATTERN.sub("[email]", snippet)
    snippet = _PHONE_PATTERN.sub("[phone]", snippet)
    snippet = _UUID_PATTERN.sub("[id]", snippet)
    return snippet[:max_length].rstrip()


def _sanitize_and_merge_queries(
    rows: list[tuple[str, int]],
    *,
    limit: int,
    max_length: int,
) -> list[TopQuery]:
    counts: dict[str, int] = defaultdict(int)
    labels: dict[str, str] = {}
    for raw_query, count in rows:
        snippet = _query_snippet(raw_query, max_length)
        if not snippet:
            continue
        key = snippet.casefold()
        counts[key] += count
        labels.setdefault(key, snippet)

    ranked = sorted(
        counts,
        key=lambda key: (-counts[key], labels[key].casefold(), labels[key]),
    )
    return [
        TopQuery(query=labels[key], count=counts[key])
        for key in ranked[:limit]
    ]


async def get_analytics_summary(
    conn: asyncpg.Connection,
    *,
    shop_id: UUID,
    days: int | None = None,
    limit: int | None = None,
    period_end: datetime | None = None,
) -> AnalyticsSummary:
    bind_request_shop_id(shop_id)
    resolved_days = settings.analytics_default_days if days is None else days
    resolved_limit = settings.analytics_top_limit if limit is None else limit
    if not 1 <= resolved_days <= 365:
        raise ValueError("days must be between 1 and 365")
    if not 1 <= resolved_limit <= 50:
        raise ValueError("limit must be between 1 and 50")

    shop = await shop_repository.get_shop_by_id(conn, shop_id)
    if shop is None:
        raise AnalyticsShopNotFoundException()

    end = period_end or datetime.now(UTC)
    if end.tzinfo is None:
        end = end.replace(tzinfo=UTC)
    start = end - timedelta(days=resolved_days)

    conversations, messages = await analytics_repository.get_activity_counts(
        conn,
        shop_id=shop_id,
        period_start=start,
        period_end=end,
    )
    raw_queries = await analytics_repository.get_top_query_counts(
        conn,
        shop_id=shop_id,
        period_start=start,
        period_end=end,
        # Fetch extra rows because redaction can merge otherwise distinct queries.
        limit=min(resolved_limit * 20, 1000),
    )
    recommended_rows = await analytics_repository.get_top_recommended(
        conn,
        shop_id=shop_id,
        period_start=start,
        period_end=end,
        limit=resolved_limit,
    )
    clicked_rows = await analytics_repository.get_top_clicked(
        conn,
        shop_id=shop_id,
        period_start=start,
        period_end=end,
        limit=resolved_limit,
    )

    return AnalyticsSummary(
        shop_id=shop_id,
        days=resolved_days,
        period_start=start,
        period_end=end,
        conversations=conversations,
        messages=messages,
        top_queries=_sanitize_and_merge_queries(
            raw_queries,
            limit=resolved_limit,
            max_length=settings.analytics_query_snippet_length,
        ),
        top_recommended=recommended_rows,
        top_clicked=clicked_rows,
    )
