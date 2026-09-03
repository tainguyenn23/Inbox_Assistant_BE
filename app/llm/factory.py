from typing import TYPE_CHECKING

from app.llm.base import LLMProvider
from app.llm.gemini_provider import GeminiProvider
from app.llm.openai_provider import OpenAIProvider

if TYPE_CHECKING:
    from app.core.config import Settings


def get_llm_provider(settings: "Settings") -> LLMProvider:
    provider_name = settings.llm_provider.strip().lower()

    if provider_name == "openai":
        return OpenAIProvider(embedding_dimension=settings.embedding_dimension)

    return GeminiProvider(settings)
