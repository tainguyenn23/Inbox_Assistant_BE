from typing import Annotated

from pydantic import Field, field_validator, model_validator
from pydantic_settings import BaseSettings, NoDecode, SettingsConfigDict


class Settings(BaseSettings):
    supabase_url: str = ""
    supabase_service_key: str
    database_url: str
    llm_provider: str = "gemini"
    gemini_api_key: str = ""
    openai_api_key: str = ""
    gemini_chat_model: str = "gemini-3.5-flash"
    gemini_max_tokens: int = Field(1000, gt=0)
    gemini_timeout_seconds: float = Field(30.0, gt=0)
    gemini_connect_timeout_seconds: float = Field(5.0, gt=0)
    gemini_read_timeout_seconds: float = Field(20.0, gt=0)
    gemini_write_timeout_seconds: float = Field(10.0, gt=0)
    gemini_pool_timeout_seconds: float = Field(5.0, gt=0)
    gemini_document_timeout_seconds: float | None = Field(None, gt=0)
    gemini_query_timeout_seconds: float | None = Field(None, gt=0)
    gemini_chat_timeout_seconds: float | None = Field(None, gt=0)
    gemini_max_retries: int = Field(2, ge=0, le=10)
    embedding_dimension: int = 768
    embedding_model: str = "gemini-embedding-2"
    embedding_concurrency: int = Field(5, ge=1, le=50)
    reindex_progress_interval: int = Field(25, ge=1, le=1000)
    retrieval_candidate_limit: int = Field(30, ge=1, le=100)
    ranking_metadata_weight: float = Field(0.30, ge=0.0, le=1.0)
    ranking_keyword_weight: float = Field(0.30, ge=0.0, le=1.0)
    ranking_vector_weight: float = Field(0.40, ge=0.0, le=1.0)
    ranking_url_bonus: float = Field(0.05, ge=0.0, le=1.0)
    ranking_unavailable_penalty: float = Field(0.20, ge=0.0, le=1.0)
    ranking_unknown_stock_penalty: float = Field(0.0, ge=0.0, le=1.0)
    ranking_top_n: int = Field(5, ge=1, le=100)
    grounding_max_products: int = Field(5, ge=1, le=20)
    grounding_max_variants_per_product: int = Field(10, ge=1, le=100)
    grounding_max_variant_previews: int = Field(3, ge=1, le=20)
    grounding_max_description_chars: int = Field(2000, ge=100, le=10_000)
    analytics_default_days: int = Field(30, ge=1, le=365)
    analytics_top_limit: int = Field(10, ge=1, le=50)
    analytics_query_snippet_length: int = Field(120, ge=20, le=500)
    max_import_rows: int = Field(200, ge=1, le=10_000)
    app_env: str = "development"
    cors_origins: Annotated[list[str], NoDecode]
    log_level: str = "INFO"
    next_public_api_base_url: str = "http://localhost:8000/api/v1"

    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        extra="allow",
    )

    @field_validator("cors_origins", mode="before")
    @classmethod
    def parse_cors_origins(cls, value: str | list[str]) -> list[str]:
        if isinstance(value, str):
            return [origin.strip() for origin in value.split(",") if origin.strip()]
        return value

    @field_validator("embedding_model", "gemini_chat_model", mode="before")
    @classmethod
    def normalize_model_id(cls, value: str) -> str:
        model_id = str(value).strip().removeprefix("models/")
        if not model_id or "/" in model_id:
            raise ValueError("model ID must be a short model name without models/ prefix")
        return model_id

    @field_validator("embedding_dimension")
    @classmethod
    def validate_embedding_dimension(cls, value: int) -> int:
        if value != 768:
            raise ValueError("Milestone 0-7 requires EMBEDDING_DIMENSION=768")
        return value

    @model_validator(mode="after")
    def validate_ranking_weights(self) -> "Settings":
        total = (
            self.ranking_metadata_weight
            + self.ranking_keyword_weight
            + self.ranking_vector_weight
        )
        if abs(total - 1.0) > 1e-9:
            raise ValueError("ranking weights must sum to 1.0")
        return self


settings = Settings() # type: ignore
