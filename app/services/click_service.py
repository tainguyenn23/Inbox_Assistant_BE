"""Click tracking rules and tenant-safe validation."""

from urllib.parse import urlsplit, urlunsplit

import asyncpg

from app.core.errors import SmartSalesException
from app.core.observability import bind_request_shop_id
from app.repositories import click_repository, shop_repository
from app.schemas.analytics import ClickEventOut, ClickRequest


class ClickShopNotFoundException(SmartSalesException):
    def __init__(self) -> None:
        super().__init__(
            status_code=404,
            code="SHOP_NOT_FOUND",
            message="Shop not found.",
        )


class ClickContextNotFoundException(SmartSalesException):
    """Use one response for missing and cross-shop references to avoid leaking IDs."""

    def __init__(self) -> None:
        super().__init__(
            status_code=404,
            code="CLICK_CONTEXT_NOT_FOUND",
            message="Click context not found for this shop.",
        )


class ClickUrlMismatchException(SmartSalesException):
    def __init__(self) -> None:
        super().__init__(
            status_code=422,
            code="CLICK_URL_MISMATCH",
            message="Click URL does not match the product's current URL.",
        )


class ProductClickUrlUnavailableException(SmartSalesException):
    def __init__(self) -> None:
        super().__init__(
            status_code=422,
            code="PRODUCT_CLICK_URL_UNAVAILABLE",
            message="The product has no current click URL.",
        )


def _comparable_url(value: str) -> str:
    """Normalize only URL syntax that does not change the destination."""
    parts = urlsplit(value.strip())
    path = parts.path.rstrip("/") or "/"
    return urlunsplit(
        (parts.scheme.lower(), parts.netloc.lower(), path, parts.query, parts.fragment)
    )


async def record_click(
    conn: asyncpg.Connection,
    request: ClickRequest,
) -> ClickEventOut:
    bind_request_shop_id(request.shop_id)
    async with conn.transaction():
        shop = await shop_repository.get_shop_by_id(conn, request.shop_id)
        if shop is None:
            raise ClickShopNotFoundException()

        if request.conversation_id is not None:
            conversation_exists = (
                await click_repository.conversation_belongs_to_shop(
                    conn,
                    shop_id=request.shop_id,
                    conversation_id=request.conversation_id,
                )
            )
            if not conversation_exists:
                raise ClickContextNotFoundException()

        click_url = str(request.url)
        metadata: dict[str, str] = {"url_source": "client"}

        if request.product_id is not None:
            target = await click_repository.get_product_click_target(
                conn,
                shop_id=request.shop_id,
                product_id=request.product_id,
            )
            if target is None:
                raise ClickContextNotFoundException()

            current_url = target.current_url
            if current_url is None:
                raise ProductClickUrlUnavailableException()
            if _comparable_url(click_url) != _comparable_url(current_url):
                raise ClickUrlMismatchException()

            # Always persist the trusted, current database value.
            click_url = current_url
            metadata = {"url_source": "current_product"}

        return await click_repository.create_click_event(
            conn,
            shop_id=request.shop_id,
            conversation_id=request.conversation_id,
            product_id=request.product_id,
            url=click_url,
            metadata=metadata,
        )
