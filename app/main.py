from contextlib import asynccontextmanager

from fastapi import APIRouter, FastAPI, Response
from fastapi.exceptions import RequestValidationError
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from starlette.exceptions import HTTPException as StarletteHTTPException

from app.api.deps import close_llm_provider
from app.api.routes_analytics import router as analytics_router
from app.api.routes_chat import router as chat_router
from app.api.routes_products import router as products_router
from app.api.routes_rag import router as rag_router
from app.api.routes_shops import router as shops_router
from app.core.config import settings
from app.core.database import close_database_pool, verify_database_connection
from app.core.errors import (
    SmartSalesException,
    http_exception_handler,
    llm_provider_exception_handler,
    smartsales_exception_handler,
    unexpected_exception_handler,
    validation_exception_handler,
)
from app.core.logging import configure_logging
from app.core.middleware import RequestContextMiddleware
from app.llm.base import LLMProviderError
from app.schemas.common import BaseResponse


class HealthResponse(BaseModel):
    status: str
    env: str
    db_reachable: bool
class DatabaseHealthResponse(BaseModel):
    status: str
    reachable: bool
    query: str


@asynccontextmanager
async def lifespan(_: FastAPI):
    try:
        yield
    finally:
        try:
            await close_llm_provider()
        finally:
            await close_database_pool()


configure_logging()
app = FastAPI(lifespan=lifespan)

app.add_middleware(
    CORSMiddleware,
    allow_origins=settings.cors_origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
    expose_headers=["X-Request-ID"],
)
app.add_middleware(RequestContextMiddleware)

app.add_exception_handler(SmartSalesException, smartsales_exception_handler)
app.add_exception_handler(RequestValidationError, validation_exception_handler)
app.add_exception_handler(StarletteHTTPException, http_exception_handler)
app.add_exception_handler(LLMProviderError, llm_provider_exception_handler)
app.add_exception_handler(Exception, unexpected_exception_handler)
api_router = APIRouter(prefix="/api/v1")
api_router.include_router(chat_router)
api_router.include_router(analytics_router)
api_router.include_router(products_router)
api_router.include_router(rag_router)
api_router.include_router(shops_router)


@api_router.get("/health", response_model=BaseResponse[HealthResponse])
async def health_check(response: Response) -> BaseResponse[HealthResponse]:
    db_reachable = await is_database_reachable()
    if not db_reachable:
        response.status_code = 503

    return BaseResponse(
        data=HealthResponse(
            status="ok" if db_reachable else "degraded",
            env=settings.app_env,
            db_reachable=db_reachable,
        ),
        error=None,
    )


app.include_router(api_router)


@app.get("/health/db", response_model=BaseResponse[DatabaseHealthResponse])
async def database_health_check(response: Response) -> BaseResponse[DatabaseHealthResponse]:
    db_reachable = await is_database_reachable()
    if not db_reachable:
        response.status_code = 503

    return BaseResponse(
        data=DatabaseHealthResponse(
            status="ok" if db_reachable else "unreachable",
            reachable=db_reachable,
            query="SELECT 1",
        ),
        error=None,
    )


async def is_database_reachable() -> bool:
    try:
        return await verify_database_connection()
    except Exception:  # noqa: BLE001
        return False

# uvicorn app.main:app --reload
