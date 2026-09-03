from collections.abc import AsyncGenerator

import asyncpg
from fastapi import Depends

from app.core.config import Settings, settings
from app.core.database import (
    get_database_pool as core_get_database_pool,
    get_db as core_get_db,
)
from app.llm.base import LLMProvider
from app.llm.factory import get_llm_provider as create_llm_provider

_llm_provider: LLMProvider | None = None

def get_settings() -> Settings:
    return settings


async def get_db() -> AsyncGenerator[asyncpg.Connection, None]:
    async for db in core_get_db():
        yield db


async def get_db_pool() -> asyncpg.Pool:
    return await core_get_database_pool()


def get_llm_provider(
    app_settings: Settings = Depends(get_settings),
) -> LLMProvider:
    global _llm_provider
    if _llm_provider is None:
        _llm_provider = create_llm_provider(app_settings)
    return _llm_provider


async def close_llm_provider() -> None:
    global _llm_provider
    provider = _llm_provider
    _llm_provider = None
    close = getattr(provider, "aclose", None)
    if close is not None:
        await close()
