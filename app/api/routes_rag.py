"""RAG maintenance endpoints."""

import asyncpg
from fastapi import APIRouter, Depends

from app.api.deps import get_db, get_llm_provider
from app.llm.base import LLMProvider
from app.schemas.common import BaseResponse
from app.schemas.embedding import ReindexRequest, ReindexResponse
from app.services.reindex_service import reindex_product_embeddings


router = APIRouter(prefix="/rag", tags=["rag"])


@router.post("/reindex", response_model=BaseResponse[ReindexResponse])
async def reindex(
    request: ReindexRequest,
    conn: asyncpg.Connection = Depends(get_db),
    provider: LLMProvider = Depends(get_llm_provider),
) -> BaseResponse[ReindexResponse]:
    result = await reindex_product_embeddings(conn, request, provider)
    return BaseResponse(data=result, error=None)
