import json
from datetime import date, datetime
from decimal import Decimal
from typing import Any
from uuid import UUID

from app.schemas.product import ProductOut

_MAX_METADATA_KEYS = 20
_MAX_METADATA_LIST_ITEMS = 20
_MAX_METADATA_DEPTH = 3


def _json_safe(value: Any, *, depth: int = 0) -> Any:
    if value is None or isinstance(value, (str, int, float, bool)):
        return value
    if isinstance(value, Decimal):
        return str(value)
    if isinstance(value, UUID):
        return str(value)
    if isinstance(value, (date, datetime)):
        return value.isoformat()
    if depth >= _MAX_METADATA_DEPTH:
        return str(value)
    if isinstance(value, dict):
        items = sorted(value.items(), key=lambda item: str(item[0]).casefold())
        return {
            str(key): _json_safe(item, depth=depth + 1)
            for key, item in items[:_MAX_METADATA_KEYS]
        }
    if isinstance(value, (list, tuple)):
        return [
            _json_safe(item, depth=depth + 1)
            for item in value[:_MAX_METADATA_LIST_ITEMS]
        ]
    return str(value)


class ProductContextBuilder:
    """Build a bounded JSON context exclusively from current SQL products."""

    def __init__(
        self,
        *,
        max_products: int = 5,
        max_variants_per_product: int = 10,
        max_description_chars: int = 2000,
    ) -> None:
        if max_products <= 0 or max_variants_per_product <= 0:
            raise ValueError("product and variant limits must be greater than zero")
        if max_description_chars <= 0:
            raise ValueError("description limit must be greater than zero")
        self.max_products = max_products
        self.max_variants_per_product = max_variants_per_product
        self.max_description_chars = max_description_chars

    def build(self, products: list[ProductOut]) -> list[dict[str, Any]]:
        context: list[dict[str, Any]] = []
        for product in products[: self.max_products]:
            variants = [
                {
                    "attributes": dict(variant.attributes),
                    "price": (
                        str(variant.price) if variant.price is not None else None
                    ),
                    "stock_status": variant.stock_status,
                    "stock_quantity": variant.stock_quantity,
                    "metadata": _json_safe(variant.metadata),
                }
                for variant in product.variants[: self.max_variants_per_product]
            ]
            context.append(
                {
                    "id": str(product.id),
                    "name": product.name,
                    "description": (
                        product.description[: self.max_description_chars]
                        if product.description
                        else None
                    ),
                    "category": product.category,
                    "currency": product.currency,
                    "current_url": str(
                        product.affiliate_url or product.product_url
                    )
                    if (product.affiliate_url or product.product_url)
                    else None,
                    "metadata": _json_safe(product.metadata),
                    "variants": variants,
                }
            )
        return context

    def serialize(self, products: list[ProductOut]) -> str:
        return json.dumps(
            self.build(products),
            ensure_ascii=False,
            sort_keys=True,
            separators=(",", ":"),
        )
