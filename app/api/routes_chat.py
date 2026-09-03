import asyncpg
from fastapi import APIRouter, Depends

from app.api.deps import get_db_pool, get_llm_provider
from app.llm.base import LLMProvider
from app.schemas.chat import ChatRequest, ChatResponse
from app.schemas.common import BaseResponse
from app.services.chat_service import orchestrate_chat

router = APIRouter(prefix="/chat", tags=["chat"])


@router.post("", response_model=BaseResponse[ChatResponse])
async def chat(
    request: ChatRequest,
    pool: asyncpg.Pool = Depends(get_db_pool),
    provider: LLMProvider = Depends(get_llm_provider),
) -> BaseResponse[ChatResponse]:
    result = await orchestrate_chat(pool, request, provider)
    return BaseResponse(data=result, error=None)
