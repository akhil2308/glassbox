"""Request bodies the API accepts. Response shapes live in `results.py`."""

from __future__ import annotations

from pydantic import BaseModel, Field

from app.core.config import settings


class PromptRequest(BaseModel):
    prompt: str = Field(min_length=1)
    model: str = settings.default_model


class AblationRequest(PromptRequest):
    component: str = "block"


class ModelArch(BaseModel):
    norm: str  # "LayerNorm" / "RMSNorm"
    attention: str  # "MHA" / "GQA 4:1"
    vocab: int
    soft_cap: bool


class ModelInfo(BaseModel):
    name: str
    display_name: str
    gated: bool
    loaded: bool
    # Populated only for resident models — describing an unloaded model would force a load.
    arch: ModelArch | None = None
