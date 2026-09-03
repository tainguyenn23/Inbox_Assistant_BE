from app.schemas.product import ProductCreate, ProductVariantCreate


class NormalizedVariant(ProductVariantCreate):
    """Canonical variant model shared by JSON, CSV and XLSX adapters."""


class NormalizedProduct(ProductCreate):
    """Canonical product model consumed by the import orchestration service."""

