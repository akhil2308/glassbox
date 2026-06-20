"""Request bodies the API accepts. Response shapes live in `results.py`."""

from __future__ import annotations

from pydantic import BaseModel, Field

from app.core.config import settings


class PromptRequest(BaseModel):
    prompt: str = Field(min_length=1)
    model: str = settings.default_model


class AblationRequest(PromptRequest):
    component: str = "block"


class ModelInfo(BaseModel):
    name: str
    display_name: str
    gated: bool
    loaded: bool
