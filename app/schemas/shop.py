from datetime import datetime
from uuid import UUID

from pydantic import BaseModel, ConfigDict, EmailStr, Field


class ShopCreate(BaseModel):
    name: str
    slug: str = Field(..., pattern=r"^[a-z0-9-]+$")
    owner_email: EmailStr | None = None


class ShopOut(BaseModel):
    model_config = ConfigDict(from_attributes=True)

    id: UUID
    name: str
    slug: str
    owner_email: EmailStr | None
    created_at: datetime
