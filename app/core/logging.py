from datetime import UTC, datetime
import json
import logging
from typing import Any

from app.core.config import settings
from app.core.observability import current_request_id, current_shop_id


_STANDARD_RECORD_FIELDS = set(logging.makeLogRecord({}).__dict__)
_SAFE_STRUCTURED_FIELDS = {
    "request_id",
    "shop_id",
    "route",
    "method",
    "status",
    "duration_ms",
    "conversation_id",
    "intent",
    "product_count",
    "candidate_count",
    "timings_ms",
    "provider",
    "model",
    "operation",
    "latency_ms",
    "attempt",
    "http_status",
    "error_type",
    "retrieval_source",
    "retrieval_counts",
    "top_scores",
    "intent_duration_ms",
    "retrieval_duration_ms",
    "rank_duration_ms",
    "reply_duration_ms",
    "product_id",
    "action",
    "dimension",
    "format_version",
    "content_hash_prefix",
    "import_counts",
    "created",
    "updated",
    "failed",
    "embedded",
    "skipped",
    "force",
    "scanned",
    "completed",
    "reused",
    "rebuilt",
}


def _json_default(value: Any) -> str:
    return str(value)


class StructuredJsonFormatter(logging.Formatter):
    """Serialize only explicitly approved fields to prevent accidental leaks."""

    def format(self, record: logging.LogRecord) -> str:
        payload: dict[str, Any] = {
            "timestamp": datetime.now(UTC).isoformat(),
            "level": record.levelname,
            "logger": record.name,
            "event": record.getMessage(),
        }
        request_id = getattr(record, "request_id", None) or current_request_id()
        shop_id = getattr(record, "shop_id", None) or current_shop_id()
        payload["request_id"] = request_id
        payload["shop_id"] = shop_id

        for key, value in record.__dict__.items():
            if key in _STANDARD_RECORD_FIELDS or key not in _SAFE_STRUCTURED_FIELDS:
                continue
            if value is not None:
                payload[key] = value
        return json.dumps(
            payload,
            ensure_ascii=False,
            separators=(",", ":"),
            default=_json_default,
        )


def configure_logging() -> None:
    root = logging.getLogger()
    root.setLevel(settings.log_level.upper())
    if not root.handlers:
        root.addHandler(logging.StreamHandler())
    formatter = StructuredJsonFormatter()
    for handler in root.handlers:
        handler.setFormatter(formatter)
