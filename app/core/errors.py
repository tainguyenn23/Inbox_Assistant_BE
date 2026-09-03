import logging

from fastapi import Request, status
from fastapi.exceptions import RequestValidationError
from fastapi.responses import JSONResponse
from starlette.exceptions import HTTPException as StarletteHTTPException

from app.llm.base import LLMProviderError


logger = logging.getLogger("app.errors")


def _error_response(status_code: int, code: str, message: str) -> JSONResponse:
    return JSONResponse(
        status_code=status_code,
        content={
            "data": None,
            "error": {
                "code": code,
                "message": message,
            },
        },
    )


def _log_error(status_code: int, error_type: str) -> None:
    level = logging.ERROR if status_code >= 500 else logging.WARNING
    logger.log(
        level,
        "request_failed",
        extra={"status": status_code, "error_type": error_type},
    )


class SmartSalesException(Exception):
    def __init__(
        self,
        code: str,
        message: str,
        status_code: int = status.HTTP_400_BAD_REQUEST,
    ) -> None:
        self.code = code
        self.message = message
        self.status_code = status_code
        super().__init__(message)


class EmbeddingDimensionException(SmartSalesException):
    def __init__(self, expected: int, actual: int) -> None:
        super().__init__(
            code="EMBEDDING_DIMENSION_MISMATCH",
            message=f"Embedding dimension mismatch: expected {expected}, got {actual}",
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
        )


async def smartsales_exception_handler(
    request: Request,
    exc: Exception,
) -> JSONResponse:
    if not isinstance(exc, SmartSalesException):
        return await unexpected_exception_handler(request, exc)

    _log_error(exc.status_code, type(exc).__name__)
    return _error_response(exc.status_code, exc.code, exc.message)


async def validation_exception_handler(
    request: Request,
    exc: Exception,
) -> JSONResponse:
    if not isinstance(exc, RequestValidationError):
        return await unexpected_exception_handler(request, exc)
    _log_error(status.HTTP_422_UNPROCESSABLE_CONTENT, type(exc).__name__)
    return _error_response(
        status.HTTP_422_UNPROCESSABLE_CONTENT,
        "VALIDATION_ERROR",
        "Request validation failed.",
    )


async def http_exception_handler(
    request: Request,
    exc: Exception,
) -> JSONResponse:
    if not isinstance(exc, StarletteHTTPException):
        return await unexpected_exception_handler(request, exc)
    status_code = exc.status_code
    if status_code == status.HTTP_404_NOT_FOUND:
        code, message = "NOT_FOUND", "Resource not found."
    elif status_code == status.HTTP_429_TOO_MANY_REQUESTS:
        code, message = "RATE_LIMITED", "Too many requests."
    else:
        code, message = "HTTP_ERROR", "Request could not be completed."
    _log_error(status_code, type(exc).__name__)
    return _error_response(status_code, code, message)


async def llm_provider_exception_handler(
    request: Request,
    exc: Exception,
) -> JSONResponse:
    if not isinstance(exc, LLMProviderError):
        return await unexpected_exception_handler(request, exc)
    status_code = (
        status.HTTP_429_TOO_MANY_REQUESTS
        if exc.status_code == status.HTTP_429_TOO_MANY_REQUESTS
        else status.HTTP_503_SERVICE_UNAVAILABLE
    )
    code = (
        "LLM_RATE_LIMITED"
        if status_code == status.HTTP_429_TOO_MANY_REQUESTS
        else "LLM_PROVIDER_UNAVAILABLE"
    )
    message = (
        "LLM provider rate limit exceeded."
        if status_code == status.HTTP_429_TOO_MANY_REQUESTS
        else "LLM provider is temporarily unavailable."
    )
    _log_error(status_code, type(exc).__name__)
    return _error_response(status_code, code, message)


async def unexpected_exception_handler(
    request: Request,
    exc: Exception,
) -> JSONResponse:
    _log_error(status.HTTP_500_INTERNAL_SERVER_ERROR, type(exc).__name__)
    return _error_response(
        status.HTTP_500_INTERNAL_SERVER_ERROR,
        "INTERNAL_SERVER_ERROR",
        "Internal server error.",
    )
