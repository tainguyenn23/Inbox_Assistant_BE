"""Pure ASGI request tracing middleware."""

from __future__ import annotations

import logging
import re
from time import perf_counter
from typing import Any
from uuid import uuid4

from app.core.observability import (
    current_shop_id,
    reset_request_context,
    set_request_context,
)


logger = logging.getLogger("app.http")
_REQUEST_ID_PATTERN = re.compile(r"^[A-Za-z0-9._:-]{1,128}$")


def _request_id_from_headers(headers: list[tuple[bytes, bytes]]) -> str:
    for name, value in headers:
        if name.lower() != b"x-request-id":
            continue
        candidate = value.decode("latin-1").strip()
        if _REQUEST_ID_PATTERN.fullmatch(candidate):
            return candidate
        break
    return str(uuid4())


class RequestContextMiddleware:
    def __init__(self, app: Any) -> None:
        self.app = app

    async def __call__(self, scope: dict[str, Any], receive: Any, send: Any) -> None:
        if scope.get("type") != "http":
            await self.app(scope, receive, send)
            return

        request_id = _request_id_from_headers(scope.get("headers", []))
        token = set_request_context(request_id)
        started_at = perf_counter()
        status_code = 500

        async def send_with_request_id(message: dict[str, Any]) -> None:
            nonlocal status_code
            if message.get("type") == "http.response.start":
                status_code = int(message.get("status", 500))
                headers = [
                    (name, value)
                    for name, value in message.get("headers", [])
                    if name.lower() != b"x-request-id"
                ]
                headers.append((b"x-request-id", request_id.encode("ascii")))
                message["headers"] = headers
            await send(message)

        try:
            await self.app(scope, receive, send_with_request_id)
        finally:
            route = scope.get("route")
            route_path = getattr(route, "path", None) or scope.get("path", "")
            logger.info(
                "http_request_completed",
                extra={
                    "request_id": request_id,
                    "shop_id": current_shop_id(),
                    "route": route_path,
                    "method": scope.get("method"),
                    "status": status_code,
                    "duration_ms": round((perf_counter() - started_at) * 1000, 2),
                },
            )
            reset_request_context(token)

