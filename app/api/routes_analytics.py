"""Click tracking and shop analytics endpoints."""

from typing import Annotated
from uuid import UUID

import asyncpg
from fastapi import APIRouter, Depends, Query, status

from app.api.deps import get_db
from app.schemas.analytics import AnalyticsSummary, ClickEventOut, ClickRequest
from app.schemas.common import BaseResponse
from app.services import analytics_service, click_service


router = APIRouter()


@router.post(
    "/click",
    response_model=BaseResponse[ClickEventOut],
    status_code=status.HTTP_201_CREATED,
    tags=["clicks"],
)
async def create_click(
    payload: ClickRequest,
    conn: asyncpg.Connection = Depends(get_db),
) -> BaseResponse[ClickEventOut]:
    event = await click_service.record_click(conn, payload)
    return BaseResponse(data=event, error=None)


@router.get(
    "/analytics",
    response_model=BaseResponse[AnalyticsSummary],
    tags=["analytics"],
)
async def get_analytics(
    shop_id: UUID,
    days: Annotated[int | None, Query(ge=1, le=365)] = None,
    limit: Annotated[int | None, Query(ge=1, le=50)] = None,
    conn: asyncpg.Connection = Depends(get_db),
) -> BaseResponse[AnalyticsSummary]:
    summary = await analytics_service.get_analytics_summary(
        conn,
        shop_id=shop_id,
        days=days,
        limit=limit,
    )
    return BaseResponse(data=summary, error=None)
