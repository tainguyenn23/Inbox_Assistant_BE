from abc import ABC, abstractmethod
from typing import Any


class LLMProviderError(Exception):
    """Raised when an LLM provider cannot complete a request."""

    def __init__(self, message: str, *, status_code: int | None = None) -> None:
        self.status_code = status_code
        super().__init__(message)


class LLMProvider(ABC):
    @property
    @abstractmethod
    def model_name(self) -> str:
        """Return the chat model name used by this provider."""
        raise NotImplementedError

    @property
    @abstractmethod
    def embedding_model_name(self) -> str:
        """Return the short embedding model ID."""
        raise NotImplementedError

    @property
    @abstractmethod
    def embedding_dimension(self) -> int:
        """Return the expected embedding vector dimension."""
        raise NotImplementedError
    @abstractmethod
    async def embed_document(self, text: str) -> list[float]:
        """Create an embedding vector for input text."""
        raise NotImplementedError
    @abstractmethod
    async def embed_query(self, query: str) -> list[float]:
        "Query an embedding vector for input text."
        raise NotImplementedError
    @abstractmethod
    async def chat(self, prompt: str, system: str = "") -> str:
        """Generate a chat response from a prompt and optional system message."""
        raise NotImplementedError

    async def chat_structured(
        self,
        prompt: str,
        system: str,
        response_schema: dict[str, Any],
    ) -> str:
        """Generate JSON, falling back to ordinary chat when unsupported."""
        return await self.chat(prompt, system)
