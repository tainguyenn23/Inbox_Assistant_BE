from app.llm.base import LLMProvider


class OpenAIProvider(LLMProvider):
    def __init__(
        self,
        model_name: str = "openai-stub",
        embedding_dimension: int = 1536,
        embedding_model_name: str = "openai-embedding-stub",
    ) -> None:
        self._model_name = model_name
        self._embedding_dimension = embedding_dimension
        self._embedding_model_name = embedding_model_name

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
        raise NotImplementedError(
            "OpenAIProvider.embed_document is not implemented yet. "
            "Full OpenAI support is planned for Phase 2."
        )

    async def embed_query(self, query: str) -> list[float]:
        raise NotImplementedError(
            "OpenAIProvider.embed_query is not implemented yet. "
            "Full OpenAI support is planned for Phase 2."
        )

    async def chat(self, prompt: str, system: str = "") -> str:
        raise NotImplementedError(
            "OpenAIProvider.chat is not implemented yet. "
            "Full OpenAI support is planned for Phase 2."
        )
