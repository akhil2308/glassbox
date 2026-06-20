"""Shared router dependencies.

`get_manager` hands routers the resident-model cache (built once in the app lifespan).
`resolve_model` is the preamble every inference route runs: validate → load → cap length.
"""

from __future__ import annotations

from fastapi import HTTPException, Request

from app.core.config import settings
from app.core.manager import ModelManager
from app.core.models import REGISTRY
from app.schemas.requests import PromptRequest


def get_manager(request: Request) -> ModelManager:
    """The process-wide ModelManager, stashed on app.state during the lifespan."""
    return request.app.state.manager


def resolve_model(manager: ModelManager, req: PromptRequest):
    """Validate the model, load it, cap the prompt length; return the loaded bridge.

    Raises the API's standard HTTPExceptions:
      400 unknown model · 503 gated without HF_TOKEN · 422 prompt too long.
    """
    if req.model not in REGISTRY:
        raise HTTPException(
            status_code=400,
            detail=f"Unknown model {req.model!r}. Known: {sorted(REGISTRY)}",
        )
    try:
        model = manager.get(req.model)
    except PermissionError as exc:
        # Gated model requested without HF_TOKEN — actionable, no stack trace.
        raise HTTPException(status_code=503, detail=str(exc)) from exc

    n_tokens = model.to_tokens(req.prompt).shape[1]
    if n_tokens > settings.max_prompt_tokens:
        raise HTTPException(
            status_code=422,
            detail=f"Prompt is {n_tokens} tokens; max is {settings.max_prompt_tokens}.",
        )
    return model
