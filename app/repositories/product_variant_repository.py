import json
from collections import defaultdict
from typing import Any
from uuid import UUID, uuid4

import asyncpg
from app.schemas.product import ProductVariantCreate, ProductVariantOut


def _jsonb_to_dict(value: Any) -> dict[str, Any]:
    if value is None:
        return {}
    if isinstance(value, dict):
        return value
    parsed = json.loads(value)
    if not isinstance(parsed, dict):
        raise TypeError("Variant JSONB value must be an object")
    return parsed


def _url_to_str(value: Any) -> str | None:
    return str(value) if value is not None else None


def variant_out_from_row(row: asyncpg.Record) -> ProductVariantOut:
    return ProductVariantOut(
        id=row["id"],
        shop_id=row["shop_id"],
        product_id=row["product_id"],
        external_variant_id=row["external_variant_id"],
        sku=row["sku"],
        name=row["name"],
        attributes=_jsonb_to_dict(row["attributes"]),
        price=row["price"],
        original_price=row["original_price"],
        stock_quantity=row["stock_quantity"],
        stock_status=row["stock_status"],
        image_url=row["image_url"],
        metadata=_jsonb_to_dict(row["metadata"]),
        source_updated_at=row["source_updated_at"],
        created_at=row["created_at"],
        updated_at=row["updated_at"],
    )


async def _insert_one(
    conn: asyncpg.Connection,
    shop_id: UUID,
    product_id: UUID,
    variant: ProductVariantCreate,
) -> UUID:
    variant_id = uuid4()
    await conn.execute(
        """
        INSERT INTO product_variants (
            id, shop_id, product_id, external_variant_id, sku, name,
            attributes, price, original_price, stock_quantity, stock_status,
            image_url, metadata, source_updated_at
        )
        VALUES (
            $1, $2, $3, $4, $5, $6, $7::jsonb, $8, $9, $10, $11,
            $12, $13::jsonb, $14
        )
        """,
        variant_id,
        shop_id,
        product_id,
        variant.external_variant_id,
        variant.sku,
        variant.name,
        json.dumps(variant.attributes, ensure_ascii=False, sort_keys=True),
        variant.price,
        variant.original_price,
        variant.stock_quantity,
        variant.stock_status,
        _url_to_str(variant.image_url),
        json.dumps(variant.metadata, ensure_ascii=False, sort_keys=True),
        variant.source_updated_at,
    )
    return variant_id


async def create_many(
    conn: asyncpg.Connection,
    shop_id: UUID,
    product_id: UUID,
    variants: list[ProductVariantCreate],
) -> list[ProductVariantOut]:
    if not variants:
        raise ValueError("A product must have at least one variant")
    for variant in variants:
        await _insert_one(conn, shop_id, product_id, variant)
    grouped = await get_variants_by_product_ids(conn, shop_id, [product_id])
    return grouped[product_id]


async def replace_manual_variants(
    conn: asyncpg.Connection,
    shop_id: UUID,
    product_id: UUID,
    variants: list[ProductVariantCreate],
) -> list[ProductVariantOut]:
    if not variants:
        raise ValueError("A product must have at least one variant")
    await conn.execute(
        """
        DELETE FROM product_variants
        WHERE shop_id = $1 AND product_id = $2
        """,
        shop_id,
        product_id,
    )
    return await create_many(conn, shop_id, product_id, variants)


async def upsert_external_variants(
    conn: asyncpg.Connection,
    shop_id: UUID,
    product_id: UUID,
    variants: list[ProductVariantCreate],
) -> list[ProductVariantOut]:
    if not variants:
        raise ValueError("A product must have at least one variant")

    current_ids: list[UUID] = []
    for variant in variants:
        row = await conn.fetchrow(
            """
            SELECT id
            FROM product_variants
            WHERE shop_id = $1
              AND product_id = $2
              AND (
                ($3::text IS NOT NULL AND external_variant_id = $3)
                OR (
                    $3::text IS NULL
                    AND $4::text IS NOT NULL
                    AND sku = $4
                )
              )
            LIMIT 1
            """,
            shop_id,
            product_id,
            variant.external_variant_id,
            variant.sku,
        )
        if row is None:
            current_ids.append(await _insert_one(conn, shop_id, product_id, variant))
            continue

        variant_id = row["id"]
        current_ids.append(variant_id)
        await conn.execute(
            """
            UPDATE product_variants
            SET external_variant_id = $4,
                sku = $5,
                name = $6,
                attributes = $7::jsonb,
                price = $8,
                original_price = $9,
                stock_quantity = $10,
                stock_status = $11,
                image_url = $12,
                metadata = $13::jsonb,
                source_updated_at = $14
            WHERE shop_id = $1 AND product_id = $2 AND id = $3
            """,
            shop_id,
            product_id,
            variant_id,
            variant.external_variant_id,
            variant.sku,
            variant.name,
            json.dumps(variant.attributes, ensure_ascii=False, sort_keys=True),
            variant.price,
            variant.original_price,
            variant.stock_quantity,
            variant.stock_status,
            _url_to_str(variant.image_url),
            json.dumps(variant.metadata, ensure_ascii=False, sort_keys=True),
            variant.source_updated_at,
        )

    await conn.execute(
        """
        DELETE FROM product_variants
        WHERE shop_id = $1
          AND product_id = $2
          AND NOT (id = ANY($3::uuid[]))
        """,
        shop_id,
        product_id,
        current_ids,
    )
    grouped = await get_variants_by_product_ids(conn, shop_id, [product_id])
    return grouped[product_id]


async def get_variants_by_product_ids(
    conn: asyncpg.Connection,
    shop_id: UUID,
    product_ids: list[UUID],
) -> dict[UUID, list[ProductVariantOut]]:
    grouped: dict[UUID, list[ProductVariantOut]] = defaultdict(list)
    if not product_ids:
        return grouped
    rows = await conn.fetch(
        """
        SELECT *
        FROM product_variants
        WHERE shop_id = $1 AND product_id = ANY($2::uuid[])
        ORDER BY product_id,
                 COALESCE(external_variant_id, ''),
                 COALESCE(sku, ''),
                 id
        """,
        shop_id,
        product_ids,
    )
    for row in rows:
        grouped[row["product_id"]].append(variant_out_from_row(row))
    return grouped
