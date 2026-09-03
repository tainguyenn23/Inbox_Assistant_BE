from typing import Literal

from pydantic import BaseModel, ConfigDict, Field, model_validator

IntentName = Literal[
    "product_recommendation",
    "product_info",
    "policy_question",
    "greeting",
    "out_of_scope",
]


class IntentResult(BaseModel):
    """Validated, provider-independent intent used by product retrieval."""

    model_config = ConfigDict(extra="forbid")

    intent: IntentName
    keywords: list[str] = Field(default_factory=list)
    category: str | None = None
    min_price: int | None = Field(default=None, ge=0)
    max_price: int | None = Field(default=None, ge=0)
    color: str | None = None
    size: str | None = None
    use_case: str | None = None
    attribute_filters: dict[str, str] = Field(default_factory=dict)
    needs_human: bool = False
    confidence: float = Field(default=0.0, ge=0.0, le=1.0)

    @model_validator(mode="after")
    def validate_price_range(self) -> "IntentResult":
        if (
            self.min_price is not None
            and self.max_price is not None
            and self.min_price > self.max_price
        ):
            raise ValueError("min_price must be less than or equal to max_price")
        return self
