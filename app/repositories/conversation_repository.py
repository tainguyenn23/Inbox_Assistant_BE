import json
from typing import Any
from uuid import UUID

import asyncpg

from app.schemas.conversation import (
    ConversationChannel,
    ConversationOut,
    ConversationStatus,
    MessageOut,
    MessageRole,
)
from app.schemas.retrieval import ProductCandidate


def _jsonb_to_dict(value: Any) -> dict[str, Any]:
    if value is None:
        return {}
    if isinstance(value, dict):
        return value
    if isinstance(value, str):
        parsed = json.loads(value)
        if isinstance(parsed, dict):
            return parsed
    raise TypeError("JSONB value must be an object")


def _conversation_from_row(row: asyncpg.Record) -> ConversationOut:
    return ConversationOut(
        id=row["id"],
        shop_id=row["shop_id"],
        customer_id=row["customer_id"],
        channel=row["channel"],
        status=row["status"],
        metadata=_jsonb_to_dict(row["metadata"]),
        created_at=row["created_at"],
        updated_at=row["updated_at"],
    )


def _message_from_row(row: asyncpg.Record) -> MessageOut:
    return MessageOut(
        id=row["id"],
        shop_id=row["shop_id"],
        conversation_id=row["conversation_id"],
        role=row["role"],
        content=row["content"],
        metadata=_jsonb_to_dict(row["metadata"]),
        created_at=row["created_at"],
    )


async def create_conversation(
    conn: asyncpg.Connection,
    shop_id: UUID,
    customer_id: str,
    *,
    channel: ConversationChannel = "web",
    metadata: dict[str, Any] | None = None,
) -> ConversationOut:
    normalized_customer_id = customer_id.strip()
    if not normalized_customer_id:
        raise ValueError("customer_id must not be empty")
    row = await conn.fetchrow(
        """
        INSERT INTO conversations (
            shop_id, customer_id, channel, metadata
        )
        VALUES ($1, $2, $3, $4::jsonb)
        RETURNING *
        """,
        shop_id,
        normalized_customer_id,
        channel,
        json.dumps(metadata or {}, ensure_ascii=False, sort_keys=True),
    )
    if row is None:
        raise RuntimeError("Failed to create conversation")
    return _conversation_from_row(row)


async def get_conversation(
    conn: asyncpg.Connection,
    shop_id: UUID,
    conversation_id: UUID,
    *,
    customer_id: str | None = None,
) -> ConversationOut | None:
    normalized_customer_id = customer_id.strip() if customer_id else None
    row = await conn.fetchrow(
        """
        SELECT *
        FROM conversations
        WHERE shop_id = $1
          AND id = $2
          AND ($3::text IS NULL OR customer_id = $3)
        """,
        shop_id,
        conversation_id,
        normalized_customer_id,
    )
    return _conversation_from_row(row) if row is not None else None


async def get_or_create_conversation(
    conn: asyncpg.Connection,
    shop_id: UUID,
    conversation_id: UUID | None,
    customer_id: str,
    *,
    channel: ConversationChannel = "web",
    metadata: dict[str, Any] | None = None,
) -> ConversationOut | None:
    if conversation_id is not None:
        return await get_conversation(
            conn,
            shop_id,
            conversation_id,
            customer_id=customer_id,
        )
    return await create_conversation(
        conn,
        shop_id,
        customer_id,
        channel=channel,
        metadata=metadata,
    )


async def add_message(
    conn: asyncpg.Connection,
    shop_id: UUID,
    conversation_id: UUID,
    role: MessageRole,
    content: str,
    *,
    metadata: dict[str, Any] | None = None,
) -> MessageOut | None:
    normalized_content = content.strip()
    if not normalized_content:
        raise ValueError("message content must not be empty")
    row = await conn.fetchrow(
        """
        INSERT INTO messages (
            shop_id, conversation_id, role, content, metadata, created_at
        )
        SELECT c.shop_id, c.id, $3, $4, $5::jsonb, clock_timestamp()
        FROM conversations AS c
        WHERE c.shop_id = $1 AND c.id = $2
        RETURNING *
        """,
        shop_id,
        conversation_id,
        role,
        normalized_content,
        json.dumps(metadata or {}, ensure_ascii=False, sort_keys=True),
    )
    return _message_from_row(row) if row is not None else None


async def get_conversation_history(
    conn: asyncpg.Connection,
    shop_id: UUID,
    conversation_id: UUID,
    *,
    limit: int = 20,
) -> list[MessageOut]:
    if not 1 <= limit <= 100:
        raise ValueError("history limit must be between 1 and 100")
    rows = await conn.fetch(
        """
        SELECT m.*
        FROM messages AS m
        JOIN conversations AS c
          ON c.id = m.conversation_id AND c.shop_id = m.shop_id
        WHERE m.shop_id = $1
          AND m.conversation_id = $2
          AND c.shop_id = $1
        ORDER BY m.created_at DESC, m.id DESC
        LIMIT $3
        """,
        shop_id,
        conversation_id,
        limit,
    )
    return [_message_from_row(row) for row in reversed(rows)]


async def set_conversation_status(
    conn: asyncpg.Connection,
    shop_id: UUID,
    conversation_id: UUID,
    status: ConversationStatus,
) -> bool:
    result = await conn.execute(
        """
        UPDATE conversations
        SET status = $3
        WHERE shop_id = $1 AND id = $2
        """,
        shop_id,
        conversation_id,
        status,
    )
    return result == "UPDATE 1"


def _recommendation_reason(candidate: ProductCandidate) -> str:
    sources = ", ".join(candidate.retrieval_sources) or "product data"
    return f"Khớp theo {sources}"


async def save_recommendations(
    conn: asyncpg.Connection,
    shop_id: UUID,
    conversation_id: UUID,
    message_id: UUID,
    candidates: list[ProductCandidate],
) -> int:
    valid_message = await conn.fetchval(
        """
        SELECT EXISTS (
            SELECT 1
            FROM messages AS m
            JOIN conversations AS c
              ON c.id = m.conversation_id AND c.shop_id = m.shop_id
            WHERE m.shop_id = $1
              AND m.conversation_id = $2
              AND m.id = $3
              AND c.shop_id = $1
        )
        """,
        shop_id,
        conversation_id,
        message_id,
    )
    if not valid_message:
        return 0

    saved = 0
    async with conn.transaction():
        await conn.execute(
            """
            DELETE FROM recommendations
            WHERE shop_id = $1
              AND conversation_id = $2
              AND message_id = $3
            """,
            shop_id,
            conversation_id,
            message_id,
        )
        for candidate in candidates:
            if candidate.shop_id != shop_id:
                continue
            rank = saved + 1
            score_metadata = (
                candidate.score_components.model_dump(mode="json")
                if candidate.score_components is not None
                else {}
            )
            row = await conn.fetchrow(
                """
                INSERT INTO recommendations (
                    shop_id, conversation_id, message_id, product_id,
                    reason, rank, score, metadata
                )
                SELECT
                    p.shop_id, $2, $3, p.id, $5, $6, $7, $8::jsonb
                FROM products AS p
                WHERE p.shop_id = $1
                  AND p.id = $4
                  AND p.status = 'active'
                ON CONFLICT (message_id, product_id)
                DO UPDATE SET
                    reason = EXCLUDED.reason,
                    rank = EXCLUDED.rank,
                    score = EXCLUDED.score,
                    metadata = EXCLUDED.metadata
                RETURNING id
                """,
                shop_id,
                conversation_id,
                message_id,
                candidate.product_id,
                _recommendation_reason(candidate),
                rank,
                candidate.hybrid_score,
                json.dumps(score_metadata, ensure_ascii=False, sort_keys=True),
            )
            if row is not None:
                saved += 1
    return saved
