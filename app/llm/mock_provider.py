from typing import Any

from app.llm.base import LLMProvider


class MockLLMProvider(LLMProvider):
    def __init__(
        self,
        embedding_dimension: int = 768,
        embedding_value: float = 0.1,
        chat_response: str = "{}",
        model_name: str = "mock-llm",
        embedding_model_name: str = "gemini-embedding-2",
    ) -> None:
        self._embedding_dimension = embedding_dimension
        self._embedding_value = embedding_value
        self._chat_response = chat_response
        self._model_name = model_name
        self._embedding_model_name = embedding_model_name
        self.document_calls = 0
        self.query_calls = 0
        self.chat_calls = 0
        self.last_document: str | None = None
        self.last_query: str | None = None
        self.last_chat_prompt: str | None = None
        self.last_chat_system: str | None = None
        self.last_response_schema: dict[str, Any] | None = None

    @property
    def model_name(self) -> str:
        return self._model_name

    @property
    def embedding_dimension(self) -> int:
        return self._embedding_dimension

    @property
    def embedding_model_name(self) -> str:
        return self._embedding_model_name

    async def embed_document(self, text: str) -> list[float]:
        self.document_calls += 1
        self.last_document = text
        return [self._embedding_value] * self._embedding_dimension

    async def embed_query(self, query: str) -> list[float]:
        self.query_calls += 1
        self.last_query = query
        return [self._embedding_value] * self._embedding_dimension

    async def chat(self, prompt: str, system: str = "") -> str:
        self.chat_calls += 1
        self.last_chat_prompt = prompt
        self.last_chat_system = system
        return self._chat_response

    async def chat_structured(
        self,
        prompt: str,
        system: str,
        response_schema: dict[str, Any],
    ) -> str:
        self.last_response_schema = response_schema
        return await self.chat(prompt, system)

    def set_chat_response(self, response: str) -> None:
        self._chat_response = response
