from collections.abc import Sequence
from typing import Annotated, Any
from uuid import UUID

import asyncpg
from fastapi import APIRouter, Depends, File, Form, Query, UploadFile, status
from pydantic import BaseModel, Field

from app.api.deps import get_db, get_llm_provider
from app.core.errors import SmartSalesException
from app.core.observability import bind_request_shop_id
from app.llm.base import LLMProvider
from app.repositories import product_repository
from app.schemas.common import BaseResponse
from app.schemas.product import (
    ImportError,
    ProductCreate,
    ProductCreateRequest,
    ProductImportResult,
    ProductListItemOut,
    ProductOut,
    ProductVariantCreate,
)
from app.importers import parse_csv_import, parse_json_import, parse_xlsx_import
from app.services.import_service import import_products

router = APIRouter(prefix="/products", tags=["products"])


class ProductJsonImportRequest(BaseModel):
    # One tenant owns every product in this import request.
    shop_id: UUID
    # Keep rows raw so the parser can collect product-level validation errors.
    products: list[Any] = Field(default_factory=list)


class ProductListResponse(BaseModel):
    items: list[ProductListItemOut]
    total: int = Field(..., ge=0)
    page: int = Field(..., ge=1)
    limit: int = Field(..., ge=1)


class ProductNotFoundException(SmartSalesException):
    def __init__(self, product_id: UUID) -> None:
        super().__init__(
            code="PRODUCT_NOT_FOUND",
            message=f"Product not found: {product_id}",
            status_code=status.HTTP_404_NOT_FOUND,
        )


class InvalidProductImportRequestException(SmartSalesException):
    def __init__(self, message: str) -> None:
        super().__init__(
            code="INVALID_PRODUCT_IMPORT_REQUEST",
            message=message,
            status_code=status.HTTP_422_UNPROCESSABLE_CONTENT,
        )


async def _run_product_import(
    conn: asyncpg.Connection,
    shop_id: UUID,
    products: Sequence[ProductCreate],
    parse_errors: list[ImportError],
    llm_provider: LLMProvider,
) -> BaseResponse[ProductImportResult]:
    bind_request_shop_id(shop_id)
    # Always enter the orchestration service, even with zero valid products,
    # so shop validation cannot be bypassed by an empty or fully invalid input.
    import_result = await import_products(
        conn=conn,
        shop_id=shop_id,
        products=list(products),
        llm_provider=llm_provider,
    )

    result = ProductImportResult(
        created=import_result.created,
        updated=import_result.updated,
        embedded=import_result.embedded,
        embedding_skipped=import_result.embedding_skipped,
        failed=len(parse_errors) + import_result.failed,
        errors=[*parse_errors, *import_result.errors],
    )

    return BaseResponse(data=result, error=None)


@router.post(
    "",
    response_model=BaseResponse[ProductOut],
    status_code=status.HTTP_201_CREATED,
)
async def create_product(
    product: ProductCreateRequest,
    shop_id: UUID,
    conn: asyncpg.Connection = Depends(get_db),
) -> BaseResponse[ProductOut]:
    bind_request_shop_id(shop_id)
    variants = product.variants
    if variants is None:
        # A manual product without visible options is still represented by one
        # explicit sellable variant. Unknown price/stock remain unknown.
        variants = [ProductVariantCreate(name="Default")]
    product_create = ProductCreate.model_validate(
        {**product.model_dump(exclude={"variants"}), "variants": variants}
    )
    created_product = await product_repository.create_product(
        conn, shop_id, product_create
    )
    return BaseResponse(data=created_product, error=None)


@router.post(
    "/import/json",
    response_model=BaseResponse[ProductImportResult],
    status_code=status.HTTP_201_CREATED,
)
async def import_products_json(
    body: ProductJsonImportRequest,
    conn: asyncpg.Connection = Depends(get_db),
    llm_provider: LLMProvider = Depends(get_llm_provider),
) -> BaseResponse[ProductImportResult]:
    products, parse_errors = parse_json_import(body.products)
    return await _run_product_import(
        conn=conn,
        shop_id=body.shop_id,
        products=products,
        parse_errors=parse_errors,
        llm_provider=llm_provider,
    )


@router.post(
    "/import/file",
    response_model=BaseResponse[ProductImportResult],
    status_code=status.HTTP_201_CREATED,
)
async def import_products_file(
    shop_id: Annotated[UUID, Form()],
    file: Annotated[UploadFile, File()],
    conn: asyncpg.Connection = Depends(get_db),
    llm_provider: LLMProvider = Depends(get_llm_provider),
) -> BaseResponse[ProductImportResult]:
    filename = (file.filename or "").casefold()
    if not filename.endswith((".csv", ".xlsx")):
        raise InvalidProductImportRequestException(
            "File must use .csv or .xlsx extension"
        )

    try:
        file_content = await file.read()
    finally:
        await file.close()

    if filename.endswith(".xlsx"):
        products, parse_errors = parse_xlsx_import(file_content)
    else:
        products, parse_errors = parse_csv_import(file_content)

    return await _run_product_import(
        conn=conn,
        shop_id=shop_id,
        products=products,
        parse_errors=parse_errors,
        llm_provider=llm_provider,
    )


@router.get("", response_model=BaseResponse[ProductListResponse])
async def get_products(
    shop_id: UUID,
    page: int = Query(1, ge=1),
    limit: int = Query(20, ge=1, le=100),
    category: str | None = None,
    search: str | None = None,
    conn: asyncpg.Connection = Depends(get_db),
) -> BaseResponse[ProductListResponse]:
    bind_request_shop_id(shop_id)
    products, total = await product_repository.get_products(
        conn,
        shop_id,
        page,
        limit,
        category,
        search,
    )

    return BaseResponse(
        data=ProductListResponse(
            items=[ProductListItemOut.from_product(product) for product in products],
            total=total,
            page=page,
            limit=limit,
        ),
        error=None,
    )


@router.get("/{product_id}", response_model=BaseResponse[ProductOut])
async def get_product_by_id(
    product_id: UUID,
    shop_id: UUID,
    conn: asyncpg.Connection = Depends(get_db),
) -> BaseResponse[ProductOut]:
    bind_request_shop_id(shop_id)
    product = await product_repository.get_product_by_id(conn, shop_id, product_id)
    if product is None:
        raise ProductNotFoundException(product_id)

    return BaseResponse(data=product, error=None)
