import json
import logging
import re
import unicodedata
from decimal import Decimal
from functools import lru_cache
from pathlib import Path
from uuid import UUID

import asyncpg

from app.core.config import settings
from app.llm.base import LLMProvider
from app.rag.product_context_builder import ProductContextBuilder
from app.repositories import product_repository
from app.schemas.product import ProductOut
from app.schemas.reply import (
    GroundedReplyOutput,
    ProductCard,
    ProductVariantPreview,
    ReplyResult,
)
from app.schemas.retrieval import ProductCandidate
from app.services.intent_service import normalize_message

logger = logging.getLogger(__name__)

PROMPT_TEMPLATE_PATH = Path(__file__).resolve().parents[1] / "prompts" / "sales_reply.md"
_NO_PRODUCT_REPLY = (
    "Hiện shop chưa có sản phẩm phù hợp với yêu cầu này. "
    "Bạn có thể cho biết thêm nhu cầu để shop hỗ trợ tốt hơn nhé."
)
_CODE_FENCE_PATTERN = re.compile(
    r"^\s*```(?:json)?\s*(.*?)\s*```\s*$",
    re.IGNORECASE | re.DOTALL,
)
_UUID_PATTERN = re.compile(
    r"\b[0-9a-fA-F]{8}-[0-9a-fA-F]{4}-[1-5][0-9a-fA-F]{3}-"
    r"[89abAB][0-9a-fA-F]{3}-[0-9a-fA-F]{12}\b"
)
_URL_PATTERN = re.compile(r"https?://[^\s<>\]\[{}\"']+", re.IGNORECASE)
_MONEY_PATTERN = re.compile(
    r"(?P<number>\d+(?:[.,]\d+)?)\s*"
    r"(?P<unit>k|nghìn|ngàn|triệu|tr|vnd|đ|₫)(?!\w)",
    re.IGNORECASE,
)
_STOCK_CLAIM_PATTERN = re.compile(
    r"\b(còn hàng|sẵn hàng|có sẵn|in[ -]?stock)\b",
    re.IGNORECASE,
)
_ATTRIBUTE_ALIAS_GROUPS = (
    ("color", "colour", "màu", "màu sắc"),
    ("size", "kích thước"),
    ("dung lượng", "capacity"),
    ("phiên bản", "version"),
    ("hương vị", "flavor", "flavour"),
    ("khối lượng", "trọng lượng", "weight"),
)


@lru_cache(maxsize=1)
def _system_instruction() -> str:
    return PROMPT_TEMPLATE_PATH.read_text(encoding="utf-8").strip()


def _parse_reply(response: str) -> ReplyResult:
    if not isinstance(response, str):
        raise TypeError("Reply response must be a string")
    text = response.strip()
    fenced = _CODE_FENCE_PATTERN.fullmatch(text)
    if fenced is not None:
        text = fenced.group(1).strip()
    payload = json.loads(text)
    if not isinstance(payload, dict):
        raise TypeError("Reply response must be a JSON object")
    return ReplyResult.model_validate(payload)


def _normalized_claim_text(value: str) -> str:
    return unicodedata.normalize("NFC", value).casefold()


def _money_value(number: str, unit: str) -> Decimal:
    value = Decimal(number.replace(",", "."))
    normalized_unit = unit.casefold()
    if normalized_unit == "k" or normalized_unit in {"nghìn", "ngàn"}:
        return value * 1_000
    if normalized_unit in {"triệu", "tr"}:
        return value * 1_000_000
    return value


def _validate_attribute_claims(reply: str, products: list[ProductOut]) -> None:
    normalized_reply = _normalized_claim_text(reply)
    attributes = [
        (
            _normalized_claim_text(key),
            _normalized_claim_text(value),
        )
        for product in products
        for variant in product.variants
        for key, value in variant.attributes.items()
    ]
    for aliases in _ATTRIBUTE_ALIAS_GROUPS:
        normalized_aliases = tuple(_normalized_claim_text(alias) for alias in aliases)
        allowed_values = {
            value
            for key, value in attributes
            if key in normalized_aliases
        }
        for alias in normalized_aliases:
            for match in re.finditer(rf"\b{re.escape(alias)}\b", normalized_reply):
                claim_window = normalized_reply[match.end() : match.end() + 60]
                if not allowed_values or not any(
                    value in claim_window for value in allowed_values
                ):
                    raise ValueError("Reply contains an unsupported attribute claim")


def _variant_is_available(variant: object) -> bool:
    stock_status = getattr(variant, "stock_status", None)
    stock_quantity = getattr(variant, "stock_quantity", None)
    return stock_status == "in_stock" or (
        stock_quantity is not None and stock_quantity > 0
    )


def _validate_stock_claims(reply: str, products: list[ProductOut]) -> None:
    for sentence in re.split(r"[.!?\n]+", reply):
        if not _STOCK_CLAIM_PATTERN.search(sentence):
            continue
        normalized_sentence = _normalized_claim_text(sentence)
        named_products = [
            product
            for product in products
            if _normalized_claim_text(product.name) in normalized_sentence
        ]
        relevant_products = named_products or products
        mentioned_values = {
            _normalized_claim_text(value)
            for product in relevant_products
            for variant in product.variants
            for value in variant.attributes.values()
            if _normalized_claim_text(value) in normalized_sentence
        }
        claim_is_supported = any(
            _variant_is_available(variant)
            and all(
                value
                in {
                    _normalized_claim_text(item)
                    for item in variant.attributes.values()
                }
                for value in mentioned_values
            )
            for product in relevant_products
            for variant in product.variants
        )
        if not claim_is_supported:
            raise ValueError("Reply contains an unsupported stock claim")


def _validate_price_claims(reply: str, products: list[ProductOut]) -> None:
    allowed_prices = {
        variant.price
        for product in products
        for variant in product.variants
        if variant.price is not None
    }
    for match in _MONEY_PATTERN.finditer(reply):
        if _money_value(match.group("number"), match.group("unit")) not in allowed_prices:
            raise ValueError("Reply contains an unsupported price claim")


def _validate_reply_grounding(
    result: ReplyResult,
    products: list[ProductOut],
) -> ReplyResult:
    allowed_ids = {product.id for product in products}
    if not result.used_product_ids:
        raise ValueError("Reply must reference at least one provided product")
    if not set(result.used_product_ids).issubset(allowed_ids):
        raise ValueError("Reply referenced a product outside the grounded context")

    used_ids = set(result.used_product_ids)
    used_products = [product for product in products if product.id in used_ids]

    mentioned_ids = {UUID(value) for value in _UUID_PATTERN.findall(result.reply)}
    if not mentioned_ids.issubset(allowed_ids):
        raise ValueError("Reply text contains an unknown product ID")

    allowed_urls = {
        str(url).rstrip("/")
        for product in products
        for url in (product.affiliate_url, product.product_url)
        if url is not None
    }
    mentioned_urls = {
        value.rstrip(".,;:!?)/")
        for value in _URL_PATTERN.findall(result.reply)
    }
    if not mentioned_urls.issubset(allowed_urls):
        raise ValueError("Reply text contains an unknown product URL")
    _validate_attribute_claims(result.reply, used_products)
    _validate_stock_claims(result.reply, used_products)
    _validate_price_claims(result.reply, used_products)
    return result


def _fallback_reply(products: list[ProductOut]) -> ReplyResult:
    if not products:
        return ReplyResult(reply=_NO_PRODUCT_REPLY, used_product_ids=[])
    names = ", ".join(product.name for product in products)
    return ReplyResult(
        reply=(
            f"Dạ, shop có {len(products)} sản phẩm phù hợp: {names}. "
            "Giá và tình trạng hiện tại được hiển thị trong các sản phẩm bên dưới."
        ),
        used_product_ids=[product.id for product in products],
    )


def _candidate_reason(candidate: ProductCandidate) -> str:
    sources = ", ".join(candidate.retrieval_sources) or "product data"
    return f"Khớp theo {sources}"


def _build_product_cards(
    products: list[ProductOut],
    candidate_by_id: dict[UUID, ProductCandidate],
    used_product_ids: list[UUID],
    max_variant_previews: int,
) -> list[ProductCard]:
    used_ids = set(used_product_ids)
    cards: list[ProductCard] = []
    for product in products:
        candidate = candidate_by_id.get(product.id)
        if candidate is None or product.id not in used_ids:
            continue
        cards.append(
            ProductCard(
                id=product.id,
                name=product.name,
                image_url=product.image_url,
                url=product.affiliate_url or product.product_url,
                currency=product.currency,
                min_price=product.min_price,
                max_price=product.max_price,
                availability=product.availability,
                variants_preview=[
                    ProductVariantPreview(
                        attributes=variant.attributes,
                        price=variant.price,
                        stock_status=variant.stock_status,
                        stock_quantity=variant.stock_quantity,
                    )
                    for variant in product.variants[:max_variant_previews]
                ],
                reason=_candidate_reason(candidate),
                score=candidate.hybrid_score,
            )
        )
    return cards


async def generate_grounded_reply(
    conn: asyncpg.Connection,
    shop_id: UUID,
    user_message: str,
    ranked_candidates: list[ProductCandidate],
    provider: LLMProvider,
    *,
    context_builder: ProductContextBuilder | None = None,
) -> GroundedReplyOutput:
    """Reload current SQL facts, then generate and validate a grounded reply."""

    normalized_message = normalize_message(user_message)
    candidate_by_id = {
        candidate.product_id: candidate
        for candidate in ranked_candidates[: settings.grounding_max_products]
        if candidate.shop_id == shop_id
    }
    candidate_ids = list(candidate_by_id)
    fresh_products = await product_repository.get_active_products_by_ids(
        conn,
        shop_id,
        candidate_ids,
    )
    fresh_products = [product for product in fresh_products if product.variants]
    if not fresh_products:
        result = _fallback_reply([])
        return GroundedReplyOutput(result=result, product_cards=[])

    builder = context_builder or ProductContextBuilder(
        max_products=settings.grounding_max_products,
        max_variants_per_product=settings.grounding_max_variants_per_product,
        max_description_chars=settings.grounding_max_description_chars,
    )
    user_prompt = (
        "USER_MESSAGE_JSON:\n"
        + json.dumps(
            {"message": normalized_message},
            ensure_ascii=False,
            sort_keys=True,
            separators=(",", ":"),
        )
        + "\nPRODUCT_CONTEXT_JSON:\n"
        + builder.serialize(fresh_products)
    )

    try:
        raw_reply = await provider.chat_structured(
            user_prompt,
            system=_system_instruction(),
            response_schema=ReplyResult.model_json_schema(),
        )
        result = _validate_reply_grounding(
            _parse_reply(raw_reply),
            fresh_products,
        )
    except Exception as error:  # noqa: BLE001
        logger.warning(
            "Grounded reply generation failed; using SQL fallback (%s)",
            type(error).__name__,
        )
        result = _fallback_reply(fresh_products)

    used_ids = set(result.used_product_ids)
    ordered_used_ids = [
        product.id for product in fresh_products if product.id in used_ids
    ]
    grounded_result = ReplyResult(
        reply=result.reply,
        used_product_ids=ordered_used_ids,
    )
    cards = _build_product_cards(
        fresh_products,
        candidate_by_id,
        ordered_used_ids,
        settings.grounding_max_variant_previews,
    )
    return GroundedReplyOutput(result=grounded_result, product_cards=cards)
