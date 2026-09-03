import logging
from time import perf_counter
from uuid import UUID

import asyncpg
from fastapi import status

from app.core.errors import SmartSalesException
from app.core.observability import bind_request_shop_id
from app.llm.base import LLMProvider
from app.repositories import conversation_repository, shop_repository
from app.schemas.chat import ChatRequest, ChatResponse
from app.schemas.intent import IntentResult
from app.schemas.reply import GroundedReplyOutput, ReplyResult
from app.schemas.retrieval import ProductCandidate
from app.services.grounded_reply_service import generate_grounded_reply
from app.services.intent_service import extract_intent, normalize_message
from app.services.ranking_service import rank_products
from app.services.retrieval_service import retrieve_products

logger = logging.getLogger(__name__)

_GREETING_REPLY = "Xin chào! Bạn đang tìm sản phẩm nào để shop hỗ trợ nhé?"
_POLICY_REPLY = (
    "Hiện dữ liệu sản phẩm không có thông tin chính sách này. "
    "Shop sẽ chuyển câu hỏi cho nhân viên hỗ trợ."
)
_OUT_OF_SCOPE_REPLY = (
    "Mình chỉ có thể hỗ trợ tìm hiểu và lựa chọn sản phẩm của shop."
)


class ChatShopNotFoundException(SmartSalesException):
    def __init__(self, shop_id: UUID) -> None:
        super().__init__(
            code="SHOP_NOT_FOUND",
            message=f"Shop not found: {shop_id}",
            status_code=status.HTTP_404_NOT_FOUND,
        )


class ChatConversationNotFoundException(SmartSalesException):
    def __init__(self) -> None:
        super().__init__(
            code="CONVERSATION_NOT_FOUND",
            message="Conversation was not found for this shop and customer",
            status_code=status.HTTP_404_NOT_FOUND,
        )


class InvalidChatMessageException(SmartSalesException):
    def __init__(self) -> None:
        super().__init__(
            code="INVALID_CHAT_MESSAGE",
            message="Message must contain non-whitespace text",
            status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
        )


def _fast_reply(intent: IntentResult) -> GroundedReplyOutput | None:
    if intent.intent == "greeting":
        return GroundedReplyOutput(
            result=ReplyResult(reply=_GREETING_REPLY, used_product_ids=[]),
            product_cards=[],
        )
    if intent.intent == "policy_question":
        return GroundedReplyOutput(
            result=ReplyResult(reply=_POLICY_REPLY, used_product_ids=[]),
            product_cards=[],
        )
    if intent.intent == "out_of_scope":
        return GroundedReplyOutput(
            result=ReplyResult(reply=_OUT_OF_SCOPE_REPLY, used_product_ids=[]),
            product_cards=[],
        )
    return None


async def _start_chat(
    pool: asyncpg.Pool,
    request: ChatRequest,
):
    async with pool.acquire() as conn:
        if await shop_repository.get_shop_by_id(conn, request.shop_id) is None:
            raise ChatShopNotFoundException(request.shop_id)
        try:
            normalized_message = normalize_message(request.message)
        except (TypeError, ValueError) as exc:
            raise InvalidChatMessageException() from exc
        async with conn.transaction():
            conversation = await conversation_repository.get_or_create_conversation(
                conn,
                request.shop_id,
                request.conversation_id,
                request.customer_id,
                channel=request.channel,
            )
            if conversation is None:
                raise ChatConversationNotFoundException()
            user_message = await conversation_repository.add_message(
                conn,
                request.shop_id,
                conversation.id,
                "user",
                normalized_message,
            )
            if user_message is None:
                raise ChatConversationNotFoundException()
    return conversation, normalized_message


async def _product_reply(
    pool: asyncpg.Pool,
    request: ChatRequest,
    normalized_message: str,
    intent: IntentResult,
    provider: LLMProvider,
) -> tuple[
    GroundedReplyOutput,
    list[ProductCandidate],
    dict[str, float],
    dict[str, int],
    list[float],
]:
    telemetry: dict[str, float] = {}
    stage_started = perf_counter()
    candidates = await retrieve_products(
        pool,
        request.shop_id,
        normalized_message,
        intent,
        provider,
    )
    telemetry["retrieval_ms"] = (perf_counter() - stage_started) * 1000
    retrieval_counts = {
        source: sum(
            source in candidate.retrieval_sources for candidate in candidates
        )
        for source in ("metadata", "keyword", "semantic")
    }
    retrieval_counts["total"] = len(candidates)

    stage_started = perf_counter()
    ranked = rank_products(candidates)
    telemetry["rank_ms"] = (perf_counter() - stage_started) * 1000
    top_scores = [round(item.hybrid_score, 6) for item in ranked]

    stage_started = perf_counter()
    async with pool.acquire() as conn:
        output = await generate_grounded_reply(
            conn,
            request.shop_id,
            normalized_message,
            ranked,
            provider,
        )
    telemetry["reply_ms"] = (perf_counter() - stage_started) * 1000
    used_ids = set(output.result.used_product_ids)
    used_candidates = [
        candidate for candidate in ranked if candidate.product_id in used_ids
    ]
    return output, used_candidates, telemetry, retrieval_counts, top_scores


async def _persist_chat_result(
    pool: asyncpg.Pool,
    shop_id: UUID,
    conversation_id: UUID,
    intent: IntentResult,
    output: GroundedReplyOutput,
    recommended: list[ProductCandidate],
) -> None:
    message_metadata = {
        "intent": intent.intent,
        "needs_human": intent.needs_human,
        "confidence": intent.confidence,
        "used_product_ids": [str(value) for value in output.result.used_product_ids],
    }
    async with pool.acquire() as conn:
        async with conn.transaction():
            assistant_message = await conversation_repository.add_message(
                conn,
                shop_id,
                conversation_id,
                "assistant",
                output.result.reply,
                metadata=message_metadata,
            )
            if assistant_message is None:
                raise ChatConversationNotFoundException()
            if intent.needs_human:
                updated = await conversation_repository.set_conversation_status(
                    conn,
                    shop_id,
                    conversation_id,
                    "needs_human",
                )
                if not updated:
                    raise ChatConversationNotFoundException()
            if recommended:
                await conversation_repository.save_recommendations(
                    conn,
                    shop_id,
                    conversation_id,
                    assistant_message.id,
                    recommended,
                )


async def orchestrate_chat(
    pool: asyncpg.Pool,
    request: ChatRequest,
    provider: LLMProvider,
) -> ChatResponse:
    """Run the Milestone 12 chat flow without exposing ungrounded product facts."""

    bind_request_shop_id(request.shop_id)
    started_at = perf_counter()
    timings: dict[str, float] = {}
    stage_started = perf_counter()
    conversation, normalized_message = await _start_chat(pool, request)
    timings["conversation_ms"] = (perf_counter() - stage_started) * 1000

    stage_started = perf_counter()
    intent = await extract_intent(normalized_message, provider)
    timings["intent_ms"] = (perf_counter() - stage_started) * 1000

    output = _fast_reply(intent)
    recommended: list[ProductCandidate] = []
    retrieval_counts = {"metadata": 0, "keyword": 0, "semantic": 0, "total": 0}
    top_scores: list[float] = []
    rag_timings = {"retrieval_ms": 0.0, "rank_ms": 0.0, "reply_ms": 0.0}
    if output is None:
        stage_started = perf_counter()
        (
            output,
            recommended,
            rag_timings,
            retrieval_counts,
            top_scores,
        ) = await _product_reply(
            pool,
            request,
            normalized_message,
            intent,
            provider,
        )
        timings["retrieval_ranking_reply_ms"] = (
            perf_counter() - stage_started
        ) * 1000

    logger.info(
        "rag_pipeline_completed",
        extra={
            "shop_id": str(request.shop_id),
            "intent": intent.intent,
            "intent_duration_ms": round(timings["intent_ms"], 2),
            "retrieval_duration_ms": round(rag_timings["retrieval_ms"], 2),
            "retrieval_counts": retrieval_counts,
            "rank_duration_ms": round(rag_timings["rank_ms"], 2),
            "reply_duration_ms": round(rag_timings["reply_ms"], 2),
            "top_scores": top_scores,
        },
    )

    stage_started = perf_counter()
    await _persist_chat_result(
        pool,
        request.shop_id,
        conversation.id,
        intent,
        output,
        recommended,
    )
    timings["persistence_ms"] = (perf_counter() - stage_started) * 1000
    timings["total_ms"] = (perf_counter() - started_at) * 1000
    logger.info(
        "Chat orchestration completed",
        extra={
            "shop_id": str(request.shop_id),
            "conversation_id": str(conversation.id),
            "intent": intent.intent,
            "product_count": len(output.product_cards),
            "timings_ms": {key: round(value, 2) for key, value in timings.items()},
        },
    )
    return ChatResponse(
        conversation_id=conversation.id,
        reply=output.result.reply,
        intent=intent.intent,
        needs_human=intent.needs_human,
        confidence=intent.confidence,
        products=output.product_cards,
    )
