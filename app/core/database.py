import asyncio
from collections.abc import AsyncGenerator
from urllib.parse import urlsplit

import asyncpg

from app.core.config import settings

_database_pool: asyncpg.Pool | None = None
_database_pool_lock = asyncio.Lock()


def get_database_url() -> str:
    return settings.database_url


def ensure_test_database_is_safe(
    test_database_url: str,
    production_database_url: str,
) -> None:
    """Refuse a test URL that targets the configured production database."""
    if not test_database_url:
        raise RuntimeError("TEST_DATABASE_URL is required for database tests")
    if not production_database_url:
        raise RuntimeError("DATABASE_URL is required to validate TEST_DATABASE_URL")

    if _database_target(test_database_url) == _database_target(production_database_url):
        raise RuntimeError(
            "TEST_DATABASE_URL must target a different database host or database "
            "name than DATABASE_URL"
        )


def _database_target(database_url: str) -> tuple[str, str]:
    parsed = urlsplit(database_url)
    host = (parsed.hostname or "").lower()
    database_name = parsed.path.strip("/")
    if not host or not database_name:
        raise RuntimeError("Database URL must include both host and database name")
    return host, database_name


async def verify_database_connection() -> bool:
    try:
        connection = await asyncpg.connect(dsn=get_database_url(), timeout=5)
        try:
            result = await connection.fetchval("SELECT 1")
            return result == 1
        finally:
            await connection.close()
    except (asyncpg.PostgresError, OSError, TimeoutError):
        return False


async def get_database_pool() -> asyncpg.Pool:
    """Return the shared pool used by request and concurrent retrieval paths."""

    global _database_pool
    if _database_pool is not None and not _database_pool.is_closing():
        return _database_pool

    async with _database_pool_lock:
        if _database_pool is None or _database_pool.is_closing():
            _database_pool = await asyncpg.create_pool(
                dsn=get_database_url(),
                min_size=1,
                max_size=10,
                timeout=5,
                statement_cache_size=0,
            )
    return _database_pool


async def close_database_pool() -> None:
    global _database_pool
    pool = _database_pool
    _database_pool = None
    if pool is not None:
        await pool.close()


async def get_db() -> AsyncGenerator[asyncpg.Connection, None]:
    pool = await get_database_pool()
    async with pool.acquire() as connection:
        yield connection
