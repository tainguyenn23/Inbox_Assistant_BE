import re
import unicodedata
from collections.abc import Sequence
from datetime import datetime
from decimal import Decimal
from typing import Any, Literal
from uuid import UUID

from pydantic import (
    BaseModel,
    ConfigDict,
    Field,
    HttpUrl,
    computed_field,
    field_validator,
    model_validator,
)

from app.rag.text_normalizer import normalize_embedding_text

ProductStatus = Literal["active", "inactive", "draft", "archived"]
StockStatus = Literal["in_stock", "out_of_stock", "preorder", "unknown"]
_PRODUCT_SOURCE_PATTERN = re.compile(r"^[a-z0-9][a-z0-9_-]*$")


def _normalize_nfc_and_strip(value: str) -> str:
    return unicodedata.normalize("NFC", value).strip()


def canonicalize_product_source(value: str) -> str:
    """Return the stable machine identifier used by import and persistence."""
    normalized = normalize_embedding_text(value).casefold()
    if not _PRODUCT_SOURCE_PATTERN.fullmatch(normalized):
        raise ValueError(
            "source must contain only lowercase letters, numbers, '_' or '-'"
        )
    return normalized


def _aggregate_availability(
    variants: Sequence["ProductVariantBase"],
) -> StockStatus:
    if any(
        variant.stock_status == "in_stock"
        or (variant.stock_quantity is not None and variant.stock_quantity > 0)
        for variant in variants
    ):
        return "in_stock"
    if variants and all(variant.stock_status == "out_of_stock" for variant in variants):
        return "out_of_stock"
    if any(variant.stock_status == "preorder" for variant in variants):
        return "preorder"
    return "unknown"


class ProductVariantBase(BaseModel):
    external_variant_id: str | None = None
    sku: str | None = None
    name: str | None = None
    attributes: dict[str, str] = Field(default_factory=dict)
    price: Decimal | None = Field(
        default=None,
        ge=0,
        max_digits=14,
        decimal_places=2,
    )
    original_price: Decimal | None = Field(
        default=None,
        ge=0,
        max_digits=14,
        decimal_places=2,
    )
    stock_quantity: int | None = Field(default=None, ge=0)
    stock_status: StockStatus = "unknown"
    image_url: HttpUrl | None = None
    metadata: dict[str, Any] = Field(default_factory=dict)
    source_updated_at: datetime | None = None

    @field_validator("external_variant_id", "sku", "name")
    @classmethod
    def normalize_optional_text(cls, value: str | None) -> str | None:
        if value is None:
            return None
        if not value.strip():
            return None
        normalized = normalize_embedding_text(value)
        return normalized or None

    @field_validator("attributes")
    @classmethod
    def normalize_attributes(cls, value: dict[str, str]) -> dict[str, str]:
        normalized: dict[str, str] = {}
        for raw_key, raw_value in value.items():
            key = _normalize_nfc_and_strip(raw_key)
            if not key:
                raise ValueError("attribute names must not be empty")
            normalized[key] = _normalize_nfc_and_strip(raw_value)
        return dict(
            sorted(normalized.items(), key=lambda pair: (pair[0].casefold(), pair[0]))
        )

    @model_validator(mode="after")
    def validate_price_and_stock(self) -> "ProductVariantBase":
        if (
            self.price is not None
            and self.original_price is not None
            and self.original_price < self.price
        ):
            raise ValueError("original_price must be greater than or equal to price")
        if self.stock_quantity == 0 and self.stock_status == "in_stock":
            raise ValueError("stock_status cannot be in_stock when stock_quantity is 0")
        if (
            self.stock_quantity is not None
            and self.stock_quantity > 0
            and self.stock_status == "out_of_stock"
        ):
            raise ValueError("stock_status cannot be out_of_stock when stock_quantity is positive")
        return self


class ProductVariantCreate(ProductVariantBase):
    pass


class ProductVariantOut(ProductVariantBase):
    model_config = ConfigDict(from_attributes=True)

    id: UUID
    shop_id: UUID
    product_id: UUID
    created_at: datetime
    updated_at: datetime


class ProductBase(BaseModel):
    source: str = "manual"
    external_product_id: str | None = None
    external_shop_id: str | None = None
    name: str = Field(..., max_length=500)
    description: str | None = None
    category: str | None = None
    tags: list[str] = Field(default_factory=list)
    currency: str = Field("VND", pattern=r"^[A-Z]{3}$")
    status: ProductStatus = "active"
    image_url: HttpUrl | None = None
    product_url: HttpUrl | None = None
    affiliate_url: HttpUrl | None = None
    metadata: dict[str, Any] = Field(default_factory=dict)
    source_updated_at: datetime | None = None

    @field_validator("name")
    @classmethod
    def normalize_required_text(cls, value: str) -> str:
        normalized = normalize_embedding_text(value)
        if not normalized:
            raise ValueError("must not be empty")
        return normalized

    @field_validator("source")
    @classmethod
    def normalize_source(cls, value: str) -> str:
        return canonicalize_product_source(value)

    @field_validator(
        "external_product_id",
        "external_shop_id",
        "description",
        "category",
    )
    @classmethod
    def normalize_optional_text(cls, value: str | None) -> str | None:
        if value is None:
            return None
        if not value.strip():
            return None
        normalized = normalize_embedding_text(value)
        return normalized or None

    @field_validator("tags")
    @classmethod
    def normalize_tags(cls, value: list[str]) -> list[str]:
        normalized = {
            unicodedata.normalize("NFC", normalized_item.lower())
            for item in value
            if (normalized_item := _normalize_nfc_and_strip(item))
        }
        return sorted(normalized)


class ProductCreate(ProductBase):
    variants: list[ProductVariantCreate]

    @field_validator("variants")
    @classmethod
    def require_variant(cls, value: list[ProductVariantCreate]) -> list[ProductVariantCreate]:
        if not value:
            raise ValueError("at least one variant is required")
        return value


class ProductCreateRequest(ProductBase):
    """Create endpoint input; only a manual product may omit its variants."""

    variants: list[ProductVariantCreate] | None = None

    @model_validator(mode="after")
    def validate_variants(self) -> "ProductCreateRequest":
        if self.variants == []:
            raise ValueError("at least one variant is required")
        if self.variants is None and "variants" in self.model_fields_set:
            raise ValueError("variants must not be null")
        if self.variants is None and self.source != "manual":
            raise ValueError("variants are required for non-manual products")
        return self


class ProductOut(ProductBase):
    model_config = ConfigDict(from_attributes=True)

    id: UUID
    shop_id: UUID
    variants: list[ProductVariantOut] = Field(default_factory=list)
    created_at: datetime
    updated_at: datetime

    @computed_field
    @property
    def min_price(self) -> Decimal | None:
        prices = [variant.price for variant in self.variants if variant.price is not None]
        return min(prices) if prices else None

    @computed_field
    @property
    def max_price(self) -> Decimal | None:
        prices = [variant.price for variant in self.variants if variant.price is not None]
        return max(prices) if prices else None

    @computed_field
    @property
    def availability(self) -> StockStatus:
        return _aggregate_availability(self.variants)


class ProductListItemOut(ProductBase):
    model_config = ConfigDict(from_attributes=True)

    id: UUID
    shop_id: UUID
    min_price: Decimal | None
    max_price: Decimal | None
    availability: StockStatus
    created_at: datetime
    updated_at: datetime

    @classmethod
    def from_product(cls, product: ProductOut) -> "ProductListItemOut":
        return cls.model_validate(product)


class ImportError(BaseModel):
    row_number: int = Field(..., ge=1)
    code: str = "IMPORT_ERROR"
    reason: str


class ProductImportRow(ProductCreate):
    row_number: int = Field(..., ge=1)


class ProductImportResult(BaseModel):
    created: int = Field(default=0, ge=0)
    updated: int = Field(default=0, ge=0)
    embedded: int = Field(default=0, ge=0)
    embedding_skipped: int = Field(default=0, ge=0)
    failed: int = Field(..., ge=0)
    errors: list[ImportError] = Field(default_factory=list)

    @computed_field
    @property
    def imported(self) -> int:
        """Backward-compatible total processed products."""
        return self.created + self.updated
