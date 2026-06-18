"""POST /logit-lens — decode the residual stream at every layer."""

from __future__ import annotations

from fastapi import APIRouter, Depends

from glassbox.api.deps import get_manager, resolve_model
from glassbox.core.manager import ModelManager
from glassbox.schemas.requests import PromptRequest
from glassbox.schemas.results import LogitLensResult
from glassbox.services.logit_lens import logit_lens

router = APIRouter()


# Plain `def` (not async): FastAPI runs sync handlers in a threadpool, so the CPU-bound
# forward passes won't block the event loop.
@router.post("/logit-lens", response_model=LogitLensResult)
def run_logit_lens(
    req: PromptRequest, manager: ModelManager = Depends(get_manager)
) -> LogitLensResult:
    model = resolve_model(manager, req)
    return logit_lens(model, req.prompt, model_name=req.model)
