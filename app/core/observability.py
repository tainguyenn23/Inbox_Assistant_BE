"""Request-scoped observability helpers with a strict safe-field policy."""

from __future__ import annotations

from contextvars import ContextVar
from dataclasses import dataclass
from uuid import UUID


@dataclass
class RequestLogContext:
    request_id: str
    shop_id: str | None = None


_request_context: ContextVar[RequestLogContext | None] = ContextVar(
    "request_log_context",
    default=None,
)


def set_request_context(request_id: str):
    """Install context for one request; the caller must reset the token."""
    return _request_context.set(RequestLogContext(request_id=request_id))


def reset_request_context(token: object) -> None:
    _request_context.reset(token)  # type: ignore[arg-type]


def bind_request_shop_id(shop_id: UUID | str) -> None:
    """Attach a tenant ID without leaking context outside an active request."""
    context = _request_context.get()
    if context is not None:
        context.shop_id = str(shop_id)


def current_request_id() -> str | None:
    context = _request_context.get()
    return context.request_id if context is not None else None


def current_shop_id() -> str | None:
    context = _request_context.get()
    return context.shop_id if context is not None else None

