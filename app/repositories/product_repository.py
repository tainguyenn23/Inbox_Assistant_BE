import json
from typing import Any
from uuid import UUID

import asyncpg
from app.repositories import product_variant_repository
from app.schemas.product import ProductCreate, ProductOut, ProductVariantOut
from app.schemas.retrieval import KeywordSearchHit

PRODUCT_COLUMNS = """
    p.id,
    p.shop_id,
    p.source,
    p.external_product_id,
    p.external_shop_id,
    p.name,
    p.description,
    p.category,
    p.tags,
    p.currency,
    p.status,
    p.image_url,
    p.product_url,
    p.affiliate_url,
    p.metadata,
    p.source_updated_at,
    p.created_at,
    p.updated_at
"""


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


def _url_to_str(value: Any) -> str | None:
    return str(value) if value is not None else None


def _product_out_from_row(
    row: asyncpg.Record,
    variants: list[ProductVariantOut] | None = None,
) -> ProductOut:
    return ProductOut(
        id=row["id"],
        shop_id=row["shop_id"],
        source=row["source"],
        external_product_id=row["external_product_id"],
        external_shop_id=row["external_shop_id"],
        name=row["name"],
        description=row["description"],
        category=row["category"],
        tags=list(row["tags"]),
        currency=row["currency"],
        status=row["status"],
        image_url=row["image_url"],
        product_url=row["product_url"],
        affiliate_url=row["affiliate_url"],
        metadata=_jsonb_to_dict(row["metadata"]),
        source_updated_at=row["source_updated_at"],
        variants=variants or [],
        created_at=row["created_at"],
        updated_at=row["updated_at"],
    )


async def _insert_product_row(
    conn: asyncpg.Connection,
    shop_id: UUID,
    product: ProductCreate,
) -> asyncpg.Record:
    row = await conn.fetchrow(
        f"""
        INSERT INTO products (
            shop_id, source, external_product_id, external_shop_id,
            name, description, category, tags, currency, status,
            image_url, product_url, affiliate_url, metadata, source_updated_at
        )
        VALUES (
            $1, $2, $3, $4, $5, $6, $7, $8, $9, $10,
            $11, $12, $13, $14::jsonb, $15
        )
        ON CONFLICT (
            shop_id,
            source,
            (COALESCE(external_shop_id, '')),
            external_product_id
        ) WHERE external_product_id IS NOT NULL
        DO UPDATE SET
            name = EXCLUDED.name,
            description = EXCLUDED.description,
            category = EXCLUDED.category,
            tags = EXCLUDED.tags,
            currency = EXCLUDED.currency,
            status = EXCLUDED.status,
            image_url = EXCLUDED.image_url,
            product_url = EXCLUDED.product_url,
            affiliate_url = EXCLUDED.affiliate_url,
            metadata = EXCLUDED.metadata,
            source_updated_at = EXCLUDED.source_updated_at
        RETURNING {PRODUCT_COLUMNS.replace('p.', '')}
        """,
        shop_id,
        product.source,
        product.external_product_id,
        product.external_shop_id,
        product.name,
        product.description,
        product.category,
        product.tags,
        product.currency,
        product.status,
        _url_to_str(product.image_url),
        _url_to_str(product.product_url),
        _url_to_str(product.affiliate_url),
        json.dumps(product.metadata, ensure_ascii=False),
        product.source_updated_at,
    )
    if row is None:
        raise RuntimeError("Failed to create product")
    return row


async def _create_product_without_transaction(
    conn: asyncpg.Connection,
    shop_id: UUID,
    product: ProductCreate,
) -> ProductOut:
    row = await _insert_product_row(conn, shop_id, product)
    if product.source == "manual":
        variants = await product_variant_repository.replace_manual_variants(
            conn, shop_id, row["id"], product.variants
        )
    else:
        variants = await product_variant_repository.upsert_external_variants(
            conn, shop_id, row["id"], product.variants
        )
    return _product_out_from_row(row, variants)


async def create_product(
    conn: asyncpg.Connection,
    shop_id: UUID,
    product: ProductCreate,
) -> ProductOut:
    async with conn.transaction():
        return await _create_product_without_transaction(conn, shop_id, product)


async def find_by_external_identity(
    conn: asyncpg.Connection,
    shop_id: UUID,
    source: str,
    external_shop_id: str | None,
    external_product_id: str,
) -> ProductOut | None:
    row = await conn.fetchrow(
        f"""
        SELECT {PRODUCT_COLUMNS}
        FROM products AS p
        WHERE p.shop_id = $1
          AND p.source = $2
          AND COALESCE(p.external_shop_id, '') = COALESCE($3::text, '')
          AND p.external_product_id = $4
        """,
        shop_id,
        source,
        external_shop_id,
        external_product_id,
    )
    if row is None:
        return None
    products = await _products_from_rows(conn, shop_id, [row])
    return products[0]


async def upsert_product_by_external_identity(
    conn: asyncpg.Connection,
    shop_id: UUID,
    product: ProductCreate,
) -> tuple[ProductOut, bool]:
    existing = None
    if product.external_product_id is not None:
        existing = await find_by_external_identity(
            conn,
            shop_id,
            product.source,
            product.external_shop_id,
            product.external_product_id,
        )
    saved = await _create_product_without_transaction(conn, shop_id, product)
    return saved, existing is None


async def update_product(
    conn: asyncpg.Connection,
    shop_id: UUID,
    product_id: UUID,
    product: ProductCreate,
) -> ProductOut | None:
    existing = await get_product_by_id(conn, shop_id, product_id)
    if existing is None:
        return None
    async with conn.transaction():
        await conn.execute(
            """
            UPDATE products
            SET name=$3, description=$4, category=$5, tags=$6, currency=$7,
                status=$8, image_url=$9, product_url=$10, affiliate_url=$11,
                metadata=$12::jsonb, source_updated_at=$13
            WHERE shop_id=$1 AND id=$2
            """,
            shop_id,
            product_id,
            product.name,
            product.description,
            product.category,
            product.tags,
            product.currency,
            product.status,
            _url_to_str(product.image_url),
            _url_to_str(product.product_url),
            _url_to_str(product.affiliate_url),
            json.dumps(product.metadata, ensure_ascii=False, sort_keys=True),
            product.source_updated_at,
        )
        await product_variant_repository.replace_manual_variants(
            conn, shop_id, product_id, product.variants
        )
    return await get_product_by_id(conn, shop_id, product_id)


async def batch_create_products(
    conn: asyncpg.Connection,
    shop_id: UUID,
    products: list[ProductCreate],
) -> list[ProductOut]:
    if not products:
        return []
    # The caller normally owns the import transaction; this savepoint also makes
    # the repository safe when used independently.
    async with conn.transaction():
        return [
            await _create_product_without_transaction(conn, shop_id, product)
            for product in products
        ]


async def _fetch_variants_by_product_ids(
    conn: asyncpg.Connection,
    shop_id: UUID,
    product_ids: list[UUID],
) -> dict[UUID, list[ProductVariantOut]]:
    return await product_variant_repository.get_variants_by_product_ids(
        conn, shop_id, product_ids
    )


async def _products_from_rows(
    conn: asyncpg.Connection,
    shop_id: UUID,
    rows: list[asyncpg.Record],
) -> list[ProductOut]:
    variants = await _fetch_variants_by_product_ids(
        conn, shop_id, [row["id"] for row in rows]
    )
    return [_product_out_from_row(row, variants[row["id"]]) for row in rows]


async def get_products(
    conn: asyncpg.Connection,
    shop_id: UUID,
    page: int,
    limit: int,
    category: str | None,
    search: str | None,
) -> tuple[list[ProductOut], int]:
    offset = (page - 1) * limit
    category_filter = category.strip() if category and category.strip() else None
    search_filter = search.strip() if search and search.strip() else None
    where = """
        p.shop_id = $1
        AND ($2::text IS NULL OR p.category = $2)
        AND (
            $3::text IS NULL OR p.name ILIKE '%' || $3 || '%'
            OR p.description ILIKE '%' || $3 || '%'
            OR EXISTS (
                SELECT 1 FROM unnest(p.tags) AS tag
                WHERE tag ILIKE '%' || $3 || '%'
            )
        )
    """
    rows = await conn.fetch(
        f"""
        SELECT {PRODUCT_COLUMNS}
        FROM products AS p
        WHERE {where}
        ORDER BY p.created_at DESC, p.id DESC
        LIMIT $4 OFFSET $5
        """,
        shop_id,
        category_filter,
        search_filter,
        limit,
        offset,
    )
    total = await conn.fetchval(
        f"SELECT COUNT(*) FROM products AS p WHERE {where}",
        shop_id,
        category_filter,
        search_filter,
    )
    return await _products_from_rows(conn, shop_id, list(rows)), int(total or 0)


async def get_product_by_id(
    conn: asyncpg.Connection,
    shop_id: UUID,
    product_id: UUID,
) -> ProductOut | None:
    row = await conn.fetchrow(
        f"""
        SELECT {PRODUCT_COLUMNS}
        FROM products AS p
        WHERE p.shop_id = $1 AND p.id = $2
        """,
        shop_id,
        product_id,
    )
    if row is None:
        return None
    products = await _products_from_rows(conn, shop_id, [row])
    return products[0]


async def get_products_by_ids(
    conn: asyncpg.Connection,
    shop_id: UUID,
    ids: list[UUID],
) -> list[ProductOut]:
    if not ids:
        return []
    rows = await conn.fetch(
        f"""
        SELECT {PRODUCT_COLUMNS}
        FROM products AS p
        WHERE p.shop_id = $1 AND p.id = ANY($2::uuid[])
        ORDER BY array_position($2::uuid[], p.id)
        """,
        shop_id,
        ids,
    )
    return await _products_from_rows(conn, shop_id, list(rows))


async def get_active_products_by_ids(
    conn: asyncpg.Connection,
    shop_id: UUID,
    ids: list[UUID],
) -> list[ProductOut]:
    """Hydrate active products and all current variants in two batch queries."""

    if not ids:
        return []
    rows = await conn.fetch(
        f"""
        SELECT {PRODUCT_COLUMNS}
        FROM products AS p
        WHERE p.shop_id = $1
          AND p.status = 'active'
          AND p.id = ANY($2::uuid[])
        ORDER BY array_position($2::uuid[], p.id)
        """,
        shop_id,
        ids,
    )
    return await _products_from_rows(conn, shop_id, list(rows))


async def list_active_product_ids(
    conn: asyncpg.Connection,
    shop_id: UUID,
    product_ids: list[UUID] | None = None,
) -> list[UUID]:
    """Select a deterministic, tenant-safe active reindex scope."""
    if product_ids is None:
        rows = await conn.fetch(
            """
            SELECT id
            FROM products
            WHERE shop_id = $1 AND status = 'active'
            ORDER BY created_at, id
            """,
            shop_id,
        )
    else:
        if not product_ids:
            return []
        rows = await conn.fetch(
            """
            SELECT id
            FROM products
            WHERE shop_id = $1
              AND status = 'active'
              AND id = ANY($2::uuid[])
            ORDER BY array_position($2::uuid[], id)
            """,
            shop_id,
            product_ids,
        )
    return [row["id"] for row in rows]


get_product_with_variants = get_product_by_id
get_products_with_variants = get_products_by_ids
list_products = get_products


async def keyword_search(
    conn: asyncpg.Connection,
    shop_id: UUID,
    keywords: list[str],
    limit: int = 20,
) -> list[ProductOut]:
    hits = await keyword_search_hits(conn, shop_id, keywords, limit=limit)
    return await get_active_products_by_ids(
        conn,
        shop_id,
        [hit.product_id for hit in hits],
    )


async def keyword_search_hits(
    conn: asyncpg.Connection,
    shop_id: UUID,
    keywords: list[str],
    *,
    limit: int = 30,
    include_variant_attributes: bool = True,
) -> list[KeywordSearchHit]:
    """Search stable product text and optionally current variant attributes."""

    normalized = list(
        dict.fromkeys(keyword.strip() for keyword in keywords if keyword.strip())
    )
    if not normalized:
        return []
    if limit <= 0:
        raise ValueError("limit must be greater than zero")

    rows = await conn.fetch(
        """
        SELECT p.id AS product_id, matches.score
        FROM products AS p
        CROSS JOIN LATERAL (
            SELECT
                COUNT(*)::double precision
                / cardinality($2::text[]) AS score
            FROM unnest($2::text[]) AS requested(keyword)
            WHERE strpos(lower(COALESCE(p.name, '')), lower(requested.keyword)) > 0
               OR strpos(lower(COALESCE(p.description, '')), lower(requested.keyword)) > 0
               OR strpos(lower(COALESCE(p.category, '')), lower(requested.keyword)) > 0
               OR EXISTS (
                    SELECT 1
                    FROM unnest(p.tags) AS tag
                    WHERE strpos(lower(tag), lower(requested.keyword)) > 0
               )
               OR ($3::boolean AND EXISTS (
                    SELECT 1
                    FROM product_variants AS v,
                         LATERAL jsonb_each_text(v.attributes) AS attr(key, value)
                    WHERE v.shop_id = p.shop_id
                      AND v.product_id = p.id
                      AND (
                          strpos(lower(attr.key), lower(requested.keyword)) > 0
                          OR strpos(lower(attr.value), lower(requested.keyword)) > 0
                      )
               ))
        ) AS matches
        WHERE p.shop_id = $1
          AND p.status = 'active'
          AND matches.score > 0
        ORDER BY matches.score DESC, p.id
        LIMIT $4
        """,
        shop_id,
        normalized,
        include_variant_attributes,
        limit,
    )
    return [
        KeywordSearchHit(
            product_id=row["product_id"],
            score=float(row["score"]),
        )
        for row in rows
    ]


async def metadata_filter_product_ids(
    conn: asyncpg.Connection,
    shop_id: UUID,
    *,
    min_price: int | None = None,
    max_price: int | None = None,
    category: str | None = None,
    color: str | None = None,
    size: str | None = None,
    attribute_filters: dict[str, str] | None = None,
    available_only: bool = False,
    limit: int = 30,
) -> list[UUID]:
    """Return distinct product IDs whose same variant satisfies all filters."""

    if limit <= 0:
        raise ValueError("limit must be greater than zero")
    category_filter = category.strip() if category and category.strip() else None
    color_filter = color.strip() if color and color.strip() else None
    size_filter = size.strip() if size and size.strip() else None
    normalized_attributes = {
        key.strip(): value.strip()
        for key, value in (attribute_filters or {}).items()
        if key.strip() and value.strip()
    }

    rows = await conn.fetch(
        """
        SELECT DISTINCT p.id
        FROM products AS p
        JOIN product_variants AS v
          ON v.shop_id = p.shop_id AND v.product_id = p.id
        WHERE p.shop_id = $1
          AND p.status = 'active'
          AND ($2::text IS NULL OR lower(p.category) = lower($2))
          AND ($3::numeric IS NULL OR v.price >= $3)
          AND ($4::numeric IS NULL OR v.price <= $4)
          AND (
              NOT $5::boolean
              OR v.stock_status = 'in_stock'
              OR COALESCE(v.stock_quantity, 0) > 0
          )
          AND ($6::text IS NULL OR EXISTS (
              SELECT 1
              FROM jsonb_each_text(v.attributes) AS attr(key, value)
              WHERE lower(attr.key) IN ('color', 'colour', 'màu', 'màu sắc')
                AND lower(attr.value) = lower($6)
          ))
          AND ($7::text IS NULL OR EXISTS (
              SELECT 1
              FROM jsonb_each_text(v.attributes) AS attr(key, value)
              WHERE lower(attr.key) IN ('size', 'kích thước')
                AND lower(attr.value) = lower($7)
          ))
          AND NOT EXISTS (
              SELECT 1
              FROM jsonb_each_text($8::jsonb) AS requested(key, value)
              WHERE NOT EXISTS (
                  SELECT 1
                  FROM jsonb_each_text(v.attributes) AS actual(key, value)
                  WHERE lower(actual.key) = lower(requested.key)
                    AND lower(actual.value) = lower(requested.value)
              )
          )
        ORDER BY p.id
        LIMIT $9
        """,
        shop_id,
        category_filter,
        min_price,
        max_price,
        available_only,
        color_filter,
        size_filter,
        json.dumps(normalized_attributes, ensure_ascii=False, sort_keys=True),
        limit,
    )
    return [row["id"] for row in rows]


async def metadata_filter_search(
    conn: asyncpg.Connection,
    shop_id: UUID,
    max_price: int | None,
    category: str | None,
    color: str | None,
    size: str | None,
    limit: int = 30,
    *,
    min_price: int | None = None,
    attribute_filters: dict[str, str] | None = None,
    available_only: bool = False,
) -> list[ProductOut]:
    product_ids = await metadata_filter_product_ids(
        conn,
        shop_id,
        min_price=min_price,
        max_price=max_price,
        category=category,
        color=color,
        size=size,
        attribute_filters=attribute_filters,
        available_only=available_only,
        limit=limit,
    )
    return await get_active_products_by_ids(conn, shop_id, product_ids)
