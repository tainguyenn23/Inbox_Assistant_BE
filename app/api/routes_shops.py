import asyncpg
from fastapi import APIRouter, Depends, status

from app.api.deps import get_db
from app.core.errors import SmartSalesException
from app.core.observability import bind_request_shop_id
from app.repositories import shop_repository
from app.schemas.common import BaseResponse
from app.schemas.shop import ShopCreate, ShopOut

router = APIRouter(prefix="/shops", tags=["shops"])
db_conn_dep = Depends(get_db)

class ShopNotFoundException(SmartSalesException):
    def __init__(self, slug: str) -> None:
        super().__init__(
            code="SHOP_NOT_FOUND",
            message=f"Shop not found: {slug}",
            status_code=status.HTTP_404_NOT_FOUND,
        )


@router.post(
    "",
    response_model=BaseResponse[ShopOut],
    status_code=status.HTTP_201_CREATED,
)
async def create_shop(
    shop: ShopCreate,
    conn: asyncpg.Connection = db_conn_dep,
) -> BaseResponse[ShopOut]:
    created_shop = await shop_repository.create_shop(conn, shop)
    bind_request_shop_id(created_shop.id)
    return BaseResponse(data=created_shop, error=None)


@router.get("/{slug}", response_model=BaseResponse[ShopOut])
async def get_shop_by_slug(
    slug: str,
    conn: asyncpg.Connection = db_conn_dep,
) -> BaseResponse[ShopOut]:
    shop = await shop_repository.get_shop_by_slug(conn, slug)
    if shop is None:
        raise ShopNotFoundException(slug)

    bind_request_shop_id(shop.id)
    return BaseResponse(data=shop, error=None)
