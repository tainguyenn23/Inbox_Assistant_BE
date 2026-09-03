import math
import re
from collections.abc import Mapping, Sequence
from typing import Any
from uuid import UUID

from app.rag.constants import EMBEDDING_FORMAT_VERSION
from app.rag.text_normalizer import normalize_embedding_text
from app.schemas.product import ProductOut, ProductVariantOut

_EXCLUDED_METADATA_TOKENS = frozenset(
    {
        "affiliate",
        "availability",
        "created",
        "href",
        "inventory",
        "link",
        "price",
        "stock",
        "thumbnail",
        "timestamp",
        "updated",
        "uri",
        "url",
        "uuid",
    }
)
_EXCLUDED_COMPACT_METADATA_KEYS = frozenset(
    {
        "affiliateurl",
        "createdat",
        "externalproductid",
        "externalshopid",
        "externalvariantid",
        "id",
        "imageurl",
        "originalprice",
        "productid",
        "producturl",
        "saleprice",
        "shopid",
        "sourceupdatedat",
        "stockquantity",
        "stockstatus",
        "updatedat",
        "variantid",
    }
)
_URL_PATTERN = re.compile(r"(?:https?://|www\.)[^\s,;]+", re.IGNORECASE)
_UUID_PATTERN = re.compile(
    r"\b[0-9a-f]{8}-[0-9a-f]{4}-[1-5][0-9a-f]{3}-"
    r"[89ab][0-9a-f]{3}-[0-9a-f]{12}\b",
    re.IGNORECASE,
)


def _strip_for_retrieval(value: str) -> str:
    without_urls = _URL_PATTERN.sub("", value)
    return _UUID_PATTERN.sub("", without_urls)


def _normalize_inline_text(value: str) -> str:
    sanitized = _strip_for_retrieval(value).replace("\r", " ").replace("\n", " ")
    if not sanitized.strip():
        return ""
    return normalize_embedding_text(sanitized)


def _metadata_key_is_retrieval_safe(key: str) -> bool:
    normalized = _normalize_inline_text(key)
    if not normalized:
        return False

    # Split snake/kebab/space keys and common camelCase metadata keys.
    expanded = re.sub(r"(?<=[a-z0-9])(?=[A-Z])", "_", normalized)
    tokens = {
        token
        for token in re.split(r"[\W_]+", expanded.casefold())
        if token
    }
    compact = re.sub(r"[\W_]+", "", expanded.casefold())
    return not (
        tokens & _EXCLUDED_METADATA_TOKENS
        or compact in _EXCLUDED_COMPACT_METADATA_KEYS
    )


def _is_uuid_text(value: str) -> bool:
    try:
        UUID(value)
    except (ValueError, AttributeError):
        return False
    return True


def _render_metadata_value(value: Any) -> str | None:
    if value is None or isinstance(value, UUID):
        return None

    if isinstance(value, bool):
        return "true" if value else "false"

    if isinstance(value, int):
        return str(value)

    if isinstance(value, float):
        return str(value) if math.isfinite(value) else None

    if isinstance(value, str):
        normalized = _normalize_inline_text(value)
        if (
            not normalized
            or normalized.casefold() in {"none", "null"}
            or _is_uuid_text(normalized)
        ):
            return None
        return normalized

    if isinstance(value, Mapping):
        rendered_items: list[tuple[str, str]] = []
        for raw_key, raw_value in value.items():
            key = _normalize_inline_text(str(raw_key))
            if not _metadata_key_is_retrieval_safe(key):
                continue
            rendered_value = _render_metadata_value(raw_value)
            if rendered_value is not None:
                rendered_items.append((key, rendered_value))

        rendered_items.sort(key=lambda item: (item[0].casefold(), item[0]))
        if not rendered_items:
            return None
        return "; ".join(f"{key}={item}" for key, item in rendered_items)

    if isinstance(value, Sequence) and not isinstance(value, (str, bytes)):
        rendered_values = {
            rendered
            for item in value
            if (rendered := _render_metadata_value(item)) is not None
        }
        if not rendered_values:
            return None
        return ", ".join(
            sorted(rendered_values, key=lambda item: (item.casefold(), item))
        )

    # Product metadata comes from JSONB. Unsupported runtime objects must not
    # leak unstable repr() output into content hashes.
    return None


def _metadata_lines(prefix: str, metadata: Mapping[str, Any]) -> list[str]:
    rendered_items: list[tuple[str, str]] = []
    for raw_key, raw_value in metadata.items():
        key = _normalize_inline_text(str(raw_key))
        if not _metadata_key_is_retrieval_safe(key):
            continue
        rendered_value = _render_metadata_value(raw_value)
        if rendered_value is not None:
            rendered_items.append((key, rendered_value))

    rendered_items.sort(key=lambda item: (item[0].casefold(), item[0]))
    return [f"{prefix}.{key}: {value}" for key, value in rendered_items]


def _variant_sort_key(variant: ProductVariantOut) -> tuple[str, str, str]:
    return (
        variant.external_variant_id or "",
        variant.sku or "",
        str(variant.id),
    )


def _validate_loaded_variants(product: ProductOut) -> None:
    if not product.variants:
        raise ValueError(
            "Product embedding content requires at least one loaded variant"
        )

    variant_ids: set[UUID] = set()
    for variant in product.variants:
        if (
            variant.shop_id != product.shop_id
            or variant.product_id != product.id
        ):
            raise ValueError(
                "Product embedding content received a variant from another product or shop"
            )
        if variant.id in variant_ids:
            raise ValueError(
                "Product embedding content received duplicate variants"
            )
        variant_ids.add(variant.id)


def build_product_embedding_content(product: ProductOut) -> str:
    """Build deterministic product-v1 content from a DB-loaded product.

    The caller must provide the product with its complete variant collection.
    Dynamic facts (price, stock and URLs) are intentionally excluded because
    SQL remains their source of truth.
    """
    
    _validate_loaded_variants(product)

    normalized_tags = {
        normalized
        for tag in product.tags
        if (normalized := _normalize_inline_text(tag))
    }
    sorted_tags = sorted(
        normalized_tags,
        key=lambda item: (item.casefold(), item),
    )

    lines = [
        f"format: {EMBEDDING_FORMAT_VERSION}",
        f"name: {_strip_for_retrieval(product.name)}",
        f"description: {_strip_for_retrieval(product.description or '')}",
        f"category: {_strip_for_retrieval(product.category or '')}",
        f"tags: {', '.join(sorted_tags)}",
    ]
    lines.extend(_metadata_lines("product_metadata", product.metadata))

    for index, variant in enumerate(
        sorted(product.variants, key=_variant_sort_key),
        start=1,
    ):
        lines.extend(
            [
                (
                    f"variant_{index}_name: "
                    f"{_strip_for_retrieval(variant.name or '')}"
                ),
                (
                    f"variant_{index}_sku: "
                    f"{_strip_for_retrieval(variant.sku or '')}"
                ),
            ]
        )
        lines.extend(
            _metadata_lines(
                f"variant_{index}_attribute",
                variant.attributes,
            )
        )
        lines.extend(
            _metadata_lines(
                f"variant_{index}_metadata",
                variant.metadata,
            )
        )

    return "\n".join(lines)
