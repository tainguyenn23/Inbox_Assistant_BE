from uuid import UUID

import asyncpg
from fastapi import status

from ..core.errors import SmartSalesException
from ..schemas.shop import ShopCreate, ShopOut


class ShopSlugAlreadyExistsException(SmartSalesException):
    def __init__(self, slug: str) -> None:
        super().__init__(
            code="SHOP_SLUG_ALREADY_EXISTS",
            message=f"Shop slug already exists: {slug}",
            status_code=status.HTTP_409_CONFLICT,
        )


def _shop_out_from_row(row: asyncpg.Record) -> ShopOut:
    return ShopOut(
        id=row["id"],
        name=row["name"],
        slug=row["slug"],
        owner_email=row["owner_email"],
        created_at=row["created_at"],
    )


async def create_shop(conn: asyncpg.Connection, shop: ShopCreate) -> ShopOut:
    owner_email = str(shop.owner_email) if shop.owner_email is not None else None

    try:
        async with conn.transaction():
            row = await conn.fetchrow(
                """
                INSERT INTO shops (name, slug, owner_email)
                VALUES ($1, $2, $3)
                RETURNING id, name, slug, owner_email, created_at
                """,
                shop.name,
                shop.slug,
                owner_email,
            )
    except asyncpg.UniqueViolationError as exc:
        constraint_name = getattr(exc, "constraint_name", None)
        if constraint_name == "shops_slug_key":
            raise ShopSlugAlreadyExistsException(shop.slug) from exc
        raise

    if row is None:
        raise RuntimeError("Failed to create shop")

    return _shop_out_from_row(row)


async def get_shop_by_slug(conn: asyncpg.Connection, slug: str) -> ShopOut | None:
    row = await conn.fetchrow(
        """
        SELECT id, name, slug, owner_email, created_at
        FROM shops
        WHERE slug = $1
        """,
        slug,
    )

    if row is None:
        return None

    return _shop_out_from_row(row)


async def get_shop_by_id(conn: asyncpg.Connection, shop_id: UUID) -> ShopOut | None:
    row = await conn.fetchrow(
        """
        SELECT id, name, slug, owner_email, created_at
        FROM shops
        WHERE id = $1
        """,
        shop_id,
    )

    if row is None:
        return None

    return _shop_out_from_row(row)
