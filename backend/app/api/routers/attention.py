"""POST /attention — per-layer, per-head attention patterns."""

from __future__ import annotations

from fastapi import APIRouter, Depends

from app.api.deps import get_manager, resolve_model
from app.core.manager import ModelManager
from app.schemas.requests import PromptRequest
from app.schemas.results import AttentionResult
from app.services.attention import attention

router = APIRouter()


# Plain `def` (not async): FastAPI runs sync handlers in a threadpool, so the CPU-bound
# forward passes won't block the event loop.
@router.post("/attention", response_model=AttentionResult)
def run_attention(
    req: PromptRequest, manager: ModelManager = Depends(get_manager)
) -> AttentionResult:
    model = resolve_model(manager, req)
    return attention(model, req.prompt, model_name=req.model)
