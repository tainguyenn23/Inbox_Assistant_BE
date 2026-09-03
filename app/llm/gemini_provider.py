from __future__ import annotations

import asyncio
import logging
import math
from collections.abc import Awaitable, Callable
from time import perf_counter
from typing import TYPE_CHECKING, Any

import httpx

from app.llm.base import LLMProvider, LLMProviderError
from app.rag.constants import DOCUMENT_PREFIX, QUERY_PREFIX
from app.rag.text_normalizer import normalize_embedding_text

if TYPE_CHECKING:
    from app.core.config import Settings


logger = logging.getLogger(__name__)


class GeminiProvider(LLMProvider):
    _BASE_URL = "https://generativelanguage.googleapis.com/v1beta"

    # Giữ model ID dạng ngắn trong nội bộ.
    _DEFAULT_EMBEDDING_MODEL = "gemini-embedding-2"
    _DEFAULT_CHAT_MODEL = "gemini-3.5-flash"

    _DEFAULT_MAX_TOKENS = 1000
    _DEFAULT_TIMEOUT_SECONDS = 30.0
    _DEFAULT_CONNECT_TIMEOUT_SECONDS = 5.0
    _DEFAULT_READ_TIMEOUT_SECONDS = 20.0
    _DEFAULT_WRITE_TIMEOUT_SECONDS = 10.0
    _DEFAULT_POOL_TIMEOUT_SECONDS = 5.0
    _RETRYABLE_STATUS_CODES = frozenset({
        408,
        429,
        500,
        502,
        503,
        504,
    })

    def __init__(
        self,
        settings: Settings,
        *,
        transport: httpx.AsyncBaseTransport | None = None,
        sleep: Callable[[float], Awaitable[None]] = asyncio.sleep,
    ) -> None:
        api_key = settings.gemini_api_key.strip()
        if not api_key:
            raise LLMProviderError(
                "GEMINI_API_KEY is required for GeminiProvider"
            )

        self._api_key = api_key

        self._embedding_dimension = int(settings.embedding_dimension)
        if not 128 <= self._embedding_dimension <= 3072:
            raise LLMProviderError(
                "Gemini embedding dimension must be between "
                "128 and 3072"
            )

        configured_embedding_model = (
            settings.embedding_model.strip()
            or self._DEFAULT_EMBEDDING_MODEL
        )
        self._embedding_model = self._normalize_model_id(
            configured_embedding_model
        )

        configured_chat_model = str(
            getattr(settings, "gemini_chat_model", "") or ""
        ).strip()

        self._model_name = self._normalize_model_id(
            configured_chat_model or self._DEFAULT_CHAT_MODEL
        )

        self._max_tokens = int(
            getattr(
                settings,
                "gemini_max_tokens",
                self._DEFAULT_MAX_TOKENS,
            )
            or self._DEFAULT_MAX_TOKENS
        )

        if self._max_tokens <= 0:
            raise LLMProviderError(
                "GEMINI_MAX_TOKENS must be greater than zero"
            )

        self._max_retries = int(getattr(settings, "gemini_max_retries", 2))
        if self._max_retries < 0:
            raise LLMProviderError(
                "GEMINI_MAX_RETRIES must be greater than or equal to zero"
            )

        self._timeout_seconds = float(
            getattr(
                settings,
                "gemini_timeout_seconds",
                self._DEFAULT_TIMEOUT_SECONDS,
            )
        )
        if self._timeout_seconds <= 0:
            raise LLMProviderError(
                "GEMINI_TIMEOUT_SECONDS must be greater than zero"
            )

        self._http_timeout = httpx.Timeout(
            connect=self._read_positive_timeout(
                settings,
                "gemini_connect_timeout_seconds",
                self._DEFAULT_CONNECT_TIMEOUT_SECONDS,
            ),
            read=self._read_positive_timeout(
                settings,
                "gemini_read_timeout_seconds",
                self._DEFAULT_READ_TIMEOUT_SECONDS,
            ),
            write=self._read_positive_timeout(
                settings,
                "gemini_write_timeout_seconds",
                self._DEFAULT_WRITE_TIMEOUT_SECONDS,
            ),
            pool=self._read_positive_timeout(
                settings,
                "gemini_pool_timeout_seconds",
                self._DEFAULT_POOL_TIMEOUT_SECONDS,
            ),
        )
        self._operation_timeouts = {
            "document_embedding": self._read_optional_timeout(
                settings,
                "gemini_document_timeout_seconds",
            ),
            "query_embedding": self._read_optional_timeout(
                settings,
                "gemini_query_timeout_seconds",
            ),
            "chat": self._read_optional_timeout(
                settings,
                "gemini_chat_timeout_seconds",
            ),
        }
        self._sleep = sleep
        self._client = httpx.AsyncClient(
            base_url=self._BASE_URL,
            timeout=self._http_timeout,
            headers={
                "Content-Type": "application/json",
                "x-goog-api-key": self._api_key,
            },
            transport=transport,
        )

    @property
    def model_name(self) -> str:
        return self._model_name

    @property
    def embedding_dimension(self) -> int:
        return self._embedding_dimension

    @property
    def embedding_model_name(self) -> str:
        return self._embedding_model

    async def embed_query(self, query: str) -> list[float]:
        """
        Tạo vector cho câu hỏi của khách hàng hoặc truy vấn tìm kiếm.
        """
        query = self._normalize_embedding_input(
            query,
            field_name="query",
        )

        prepared_text = f"{QUERY_PREFIX}{query}"

        return await self._embed_prepared_text(prepared_text)

    async def embed_document(
        self,
        text: str,
    ) -> list[float]:
        """
        Tạo vector cho dữ liệu được lưu vào vector database:
        sản phẩm, FAQ, chính sách, hướng dẫn hoặc đoạn hội thoại,...
        """
        normalized_text = self._normalize_embedding_input(
            text,
            field_name="document text",
        )

        prepared_text = f"{DOCUMENT_PREFIX}{normalized_text}"

        return await self._embed_prepared_text(
            prepared_text,
            operation="document_embedding",
        )

    async def _embed_prepared_text(
        self,
        prepared_text: str,
        operation: str = "query_embedding",
    ) -> list[float]:
        url = (
            f"/models/{self._embedding_model}:embedContent"
        )

        payload: dict[str, Any] = {
            "model": f"models/{self._embedding_model}",
            "content": {
                "parts": [
                    {
                        "text": prepared_text,
                    }
                ]
            },
            "embedContentConfig": {
                "outputDimensionality": self._embedding_dimension,
            },
        }

        data = await self._post_json(
            url=url,
            payload=payload,
            operation=operation,
        )

        embedding_object = data.get("embedding")
        embedding = (
            embedding_object.get("values")
            if isinstance(embedding_object, dict)
            else None
        )

        if not isinstance(embedding, list) or not embedding:
            raise LLMProviderError(
                "Gemini embedding API response is missing values"
            )

        try:
            vector = [float(value) for value in embedding]
        except (TypeError, ValueError) as exc:
            raise LLMProviderError(
                "Gemini embedding API returned invalid "
                "numeric values"
            ) from exc

        if len(vector) != self._embedding_dimension:
            raise LLMProviderError(
                "Gemini embedding dimension mismatch: "
                f"expected {self._embedding_dimension}, "
                f"got {len(vector)}"
            )

        if not all(math.isfinite(value) for value in vector):
            raise LLMProviderError(
                "Gemini embedding contains NaN or infinity"
            )

        vector_norm = math.sqrt(
            sum(value * value for value in vector)
        )

        if vector_norm <= 0:
            raise LLMProviderError(
                "Gemini embedding returned a zero vector"
            )

        return vector

    async def chat(
        self,
        prompt: str,
        system: str = "",
    ) -> str:
        return await self._chat(
            prompt,
            system=system,
            response_schema=None,
        )

    async def chat_structured(
        self,
        prompt: str,
        system: str,
        response_schema: dict[str, Any],
    ) -> str:
        if not response_schema:
            raise LLMProviderError("Structured response schema must not be empty")
        return await self._chat(
            prompt,
            system=system,
            response_schema=response_schema,
        )

    async def _chat(
        self,
        prompt: str,
        *,
        system: str,
        response_schema: dict[str, Any] | None,
    ) -> str:
        prompt = self._validate_text(
            prompt,
            field_name="prompt",
        )

        url = (
            f"/models/{self._model_name}:generateContent"
        )

        payload: dict[str, Any] = {
            "contents": [
                {
                    "role": "user",
                    "parts": [
                        {
                            "text": prompt,
                        }
                    ],
                }
            ],
            "generationConfig": {
                "maxOutputTokens": self._max_tokens,
            },
        }

        if response_schema is not None:
            payload["generationConfig"].update(
                {
                    "responseMimeType": "application/json",
                    "responseJsonSchema": response_schema,
                }
            )

        if system and system.strip():
            payload["systemInstruction"] = {
                "parts": [
                    {
                        "text": system.strip(),
                    }
                ]
            }

        data = await self._post_json(
            url=url,
            payload=payload,
            operation="chat",
        )

        prompt_feedback = data.get("promptFeedback")
        if prompt_feedback is not None and not isinstance(prompt_feedback, dict):
            raise LLMProviderError(
                "Gemini chat API returned invalid promptFeedback"
            )

        if isinstance(prompt_feedback, dict):
            block_reason = prompt_feedback.get("blockReason")
            if block_reason:
                raise LLMProviderError(
                    "Gemini chat request was blocked: "
                    f"{block_reason}"
                )

        candidates = data.get("candidates")

        if not isinstance(candidates, list) or not candidates:
            raise LLMProviderError(
                "Gemini chat API response is missing candidates"
            )

        candidate = candidates[0]
        if not isinstance(candidate, dict):
            raise LLMProviderError(
                "Gemini chat API returned an invalid candidate"
            )

        finish_reason = candidate.get("finishReason")

        content = candidate.get("content")
        if not isinstance(content, dict):
            raise LLMProviderError(
                "Gemini chat API response is missing candidate content"
            )

        parts = content.get("parts")

        if not isinstance(parts, list) or not parts:
            raise LLMProviderError(
                "Gemini chat API response is missing "
                "content parts"
            )

        text = "".join(
            part.get("text", "")
            for part in parts
            if isinstance(part, dict)
            and isinstance(part.get("text"), str)
        ).strip()

        if finish_reason not in (None, "STOP"):
            finish_message = candidate.get("finishMessage")
            suffix = (
                f" ({finish_message})"
                if isinstance(finish_message, str) and finish_message.strip()
                else ""
            )
            raise LLMProviderError(
                "Gemini chat generation did not finish normally: "
                f"{finish_reason}{suffix}"
            )

        if not text:
            raise LLMProviderError(
                "Gemini chat API response is empty"
            )

        return text

    async def _post_json(
        self,
        url: str,
        payload: dict[str, Any],
        operation: str,
    ) -> dict[str, Any]:
        last_error: Exception | None = None

        for attempt in range(self._max_retries + 1):
            attempt_started = perf_counter()
            attempt_number = attempt + 1
            try:
                timeout_seconds = (
                    self._operation_timeouts.get(operation)
                    or self._timeout_seconds
                )
                async with asyncio.timeout(timeout_seconds):
                    response = await self._client.post(
                        url,
                        json=payload,
                    )
            except (
                TimeoutError,
                httpx.TimeoutException,
                httpx.NetworkError,
            ) as exc:
                last_error = exc
                will_retry = attempt < self._max_retries
                self._log_attempt(
                    operation=operation,
                    attempt=attempt_number,
                    status="retrying" if will_retry else "failed",
                    latency_ms=(perf_counter() - attempt_started) * 1000,
                    error_type=type(exc).__name__,
                )

                if will_retry:
                    await self._sleep(float(2**attempt))
                    continue

                break

            if response.status_code in self._RETRYABLE_STATUS_CODES:
                will_retry = attempt < self._max_retries
                self._log_attempt(
                    operation=operation,
                    attempt=attempt_number,
                    status="retrying" if will_retry else "failed",
                    latency_ms=(perf_counter() - attempt_started) * 1000,
                    http_status=response.status_code,
                )
                if will_retry:
                    delay = self._retry_delay(
                        response=response,
                        attempt=attempt,
                    )
                    await self._sleep(delay)
                    continue

                error_detail = self._safe_error_detail(response)

                raise LLMProviderError(
                    f"Gemini {operation} API returned HTTP "
                    f"{response.status_code}: {error_detail}",
                    status_code=(
                        429 if response.status_code == 429 else 503
                    ),
                )

            try:
                response.raise_for_status()
            except httpx.HTTPStatusError as exc:
                self._log_attempt(
                    operation=operation,
                    attempt=attempt_number,
                    status="failed",
                    latency_ms=(perf_counter() - attempt_started) * 1000,
                    http_status=exc.response.status_code,
                    error_type=type(exc).__name__,
                )
                error_detail = self._safe_error_detail(exc.response)
                raise LLMProviderError(
                    f"Gemini {operation} API returned HTTP "
                    f"{exc.response.status_code}: "
                    f"{error_detail}",
                    status_code=503,
                ) from exc

            try:
                data = response.json()
            except ValueError as exc:
                self._log_attempt(
                    operation=operation,
                    attempt=attempt_number,
                    status="failed",
                    latency_ms=(perf_counter() - attempt_started) * 1000,
                    http_status=response.status_code,
                    error_type=type(exc).__name__,
                )
                raise LLMProviderError(
                    f"Gemini {operation} API response "
                    "is not valid JSON"
                ) from exc

            if not isinstance(data, dict):
                self._log_attempt(
                    operation=operation,
                    attempt=attempt_number,
                    status="failed",
                    latency_ms=(perf_counter() - attempt_started) * 1000,
                    http_status=response.status_code,
                    error_type="InvalidResponseType",
                )
                raise LLMProviderError(
                    f"Gemini {operation} API response "
                    "must be a JSON object"
                )

            self._log_attempt(
                operation=operation,
                attempt=attempt_number,
                status="success",
                latency_ms=(perf_counter() - attempt_started) * 1000,
                http_status=response.status_code,
            )
            return data

        raise LLMProviderError(
            f"Gemini {operation} API request failed after "
            f"{self._max_retries + 1} attempts",
            status_code=503,
        ) from last_error

    def _log_attempt(
        self,
        *,
        operation: str,
        attempt: int,
        status: str,
        latency_ms: float,
        http_status: int | None = None,
        error_type: str | None = None,
    ) -> None:
        model = (
            self._model_name
            if operation == "chat"
            else self._embedding_model
        )
        level = logging.INFO if status == "success" else logging.WARNING
        logger.log(
            level,
            "llm_request_attempt",
            extra={
                "provider": "gemini",
                "model": model,
                "operation": operation,
                "latency_ms": round(latency_ms, 2),
                "attempt": attempt,
                "status": status,
                "http_status": http_status,
                "error_type": error_type,
            },
        )

    def _safe_error_detail(self, response: httpx.Response) -> str:
        return response.text[:1000].replace(self._api_key, "[REDACTED]")

    @staticmethod
    def _retry_delay(
        response: httpx.Response,
        attempt: int,
    ) -> float:
        retry_after = response.headers.get("Retry-After")

        if retry_after:
            try:
                return min(
                    max(float(retry_after), 0.0),
                    30.0,
                )
            except ValueError:
                pass

        return float(min(2**attempt, 8))

    @staticmethod
    def _normalize_model_id(model: str) -> str:
        model_id = model.strip().removeprefix("models/")

        if not model_id:
            raise LLMProviderError(
                "Gemini model name must not be empty"
            )

        if "/" in model_id:
            raise LLMProviderError(
                f"Invalid Gemini model ID: {model}"
            )

        return model_id

    @staticmethod
    def _read_positive_timeout(
        settings: Settings,
        name: str,
        default: float,
    ) -> float:
        value = float(getattr(settings, name, default))
        if value <= 0:
            raise LLMProviderError(
                f"{name.upper()} must be greater than zero"
            )
        return value

    @staticmethod
    def _read_optional_timeout(
        settings: Settings,
        name: str,
    ) -> float | None:
        raw_value = getattr(settings, name, None)
        if raw_value is None:
            return None
        value = float(raw_value)
        if value <= 0:
            raise LLMProviderError(
                f"{name.upper()} must be greater than zero"
            )
        return value

    @staticmethod
    def _validate_text(
        text: str,
        *,
        field_name: str,
    ) -> str:
        if not isinstance(text, str):
            raise LLMProviderError(
                f"Gemini {field_name} must be a string"
            )

        normalized = text.strip()

        if not normalized:
            raise LLMProviderError(
                f"Gemini {field_name} must not be empty"
            )

        return normalized

    @staticmethod
    def _normalize_embedding_input(
        text: str,
        *,
        field_name: str,
    ) -> str:
        try:
            return normalize_embedding_text(text)
        except (TypeError, ValueError) as exc:
            raise LLMProviderError(
                f"Gemini {field_name} must be a non-empty string"
            ) from exc

    async def aclose(self) -> None:
        await self._client.aclose()
