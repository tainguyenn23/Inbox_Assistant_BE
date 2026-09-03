import json
import logging
import re
from decimal import Decimal, InvalidOperation
from functools import lru_cache
from pathlib import Path
from typing import Any

from app.llm.base import LLMProvider
from app.rag.text_normalizer import normalize_embedding_text
from app.schemas.intent import IntentResult

logger = logging.getLogger(__name__)

PROMPT_TEMPLATE_PATH = (
    Path(__file__).resolve().parents[1] / "prompts" / "intent_extraction.md"
)
MAX_USER_MESSAGE_LENGTH = 500

_CODE_FENCE_PATTERN = re.compile(
    r"^\s*```(?:json)?\s*(.*?)\s*```\s*$",
    re.IGNORECASE | re.DOTALL,
)
_PRICE_AMOUNT_PATTERN = re.compile(
    r"(?P<number>\d+(?:[.,]\d+)?)\s*"
    r"(?P<unit>k|nghìn|ngàn|nghin|ngan|triệu|trieu|tr)\b",
    re.IGNORECASE,
)
_PRICE_RANGE_SEPARATOR_PATTERN = re.compile(
    r"^\s*(?:đến|den|tới|toi|–|—|-)\s*$",
    re.IGNORECASE,
)
_MIN_PRICE_PREFIX_PATTERN = re.compile(
    r"(?:từ|tu|trên|tren|ít nhất|it nhat|tối thiểu|toi thieu)\s*$",
    re.IGNORECASE,
)
_MAX_PRICE_PREFIX_PATTERN = re.compile(
    r"(?:dưới|duoi|không quá|khong qua|tối đa|toi da)\s*$",
    re.IGNORECASE,
)
_MIN_PRICE_SUFFIX_PATTERN = re.compile(
    r"^\s*(?:trở lên|tro len|hoặc hơn|hoac hon)\b",
    re.IGNORECASE,
)
_MAX_PRICE_SUFFIX_PATTERN = re.compile(
    r"^\s*(?:trở xuống|tro xuong|hoặc thấp hơn|hoac thap hon)\b",
    re.IGNORECASE,
)


def normalize_message(message: str) -> str:
    """Normalize once, then enforce the online intent input limit."""
    normalized = normalize_embedding_text(message)
    return normalized[:MAX_USER_MESSAGE_LENGTH]


@lru_cache(maxsize=1)
def _system_instruction() -> str:
    return PROMPT_TEMPLATE_PATH.read_text(encoding="utf-8").strip()


def _parse_json_response(response: str) -> dict[str, Any]:
    if not isinstance(response, str):
        raise TypeError("Intent response must be a string")

    text = response.strip()
    fenced = _CODE_FENCE_PATTERN.fullmatch(text)
    if fenced is not None:
        text = fenced.group(1).strip()

    payload = json.loads(text)
    if not isinstance(payload, dict):
        raise TypeError("Intent response must be a JSON object")
    return payload


def _price_value(number: str, unit: str) -> int:
    try:
        numeric_value = Decimal(number.replace(",", "."))
    except InvalidOperation as exc:
        raise ValueError("Invalid price amount") from exc

    normalized_unit = unit.casefold()
    multiplier = (
        1_000
        if normalized_unit in {"k", "nghìn", "ngàn", "nghin", "ngan"}
        else 1_000_000
    )
    return int(numeric_value * multiplier)


def _extract_price_range(message: str) -> tuple[int | None, int | None]:
    mentions = [
        (
            match.start(),
            match.end(),
            _price_value(match.group("number"), match.group("unit")),
        )
        for match in _PRICE_AMOUNT_PATTERN.finditer(message)
    ]
    if not mentions:
        return None, None

    for first, second in zip(mentions, mentions[1:]):
        between = message[first[1] : second[0]]
        if _PRICE_RANGE_SEPARATOR_PATTERN.fullmatch(between):
            return first[2], second[2]

    start, end, value = mentions[0]
    prefix = message[max(0, start - 40) : start]
    suffix = message[end : end + 40]
    if _MAX_PRICE_PREFIX_PATTERN.search(prefix) or _MAX_PRICE_SUFFIX_PATTERN.search(
        suffix
    ):
        return None, value
    if _MIN_PRICE_PREFIX_PATTERN.search(prefix) or _MIN_PRICE_SUFFIX_PATTERN.search(
        suffix
    ):
        return value, None
    return None, None


def _apply_domain_rules(result: IntentResult, message: str) -> IntentResult:
    min_price, max_price = _extract_price_range(message)
    updates: dict[str, Any] = {}
    if min_price is not None:
        updates["min_price"] = min_price
    if max_price is not None:
        updates["max_price"] = max_price
    if result.intent == "policy_question":
        updates["needs_human"] = True

    if not updates:
        return result
    return IntentResult.model_validate({**result.model_dump(), **updates})


def _fallback_intent(normalized_message: str) -> IntentResult:
    return IntentResult(
        intent="product_recommendation",
        keywords=[normalized_message],
        confidence=0.3,
    )


def _invalid_message_intent() -> IntentResult:
    return IntentResult(
        intent="out_of_scope",
        keywords=[],
        confidence=0.0,
    )


async def extract_intent(
    message: str,
    llm_provider: LLMProvider,
) -> IntentResult:
    """Extract a validated intent without letting provider failures escape."""
    try:
        normalized_message = normalize_message(message)
    except (TypeError, ValueError):
        return _invalid_message_intent()

    try:
        response = await llm_provider.chat_structured(
            normalized_message,
            system=_system_instruction(),
            response_schema=IntentResult.model_json_schema(),
        )
        result = IntentResult.model_validate(_parse_json_response(response))
        return _apply_domain_rules(result, normalized_message)
    except Exception as error:  # noqa: BLE001
        # Do not log user text or raw provider output.
        logger.warning(
            "Intent extraction failed; using fallback (%s)",
            type(error).__name__,
        )
        return _fallback_intent(normalized_message)
