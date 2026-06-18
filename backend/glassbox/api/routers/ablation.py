"""POST /ablation — sweep layer ablations and measure the effect on the prediction."""

from __future__ import annotations

from fastapi import APIRouter, Depends, HTTPException

from glassbox.api.deps import get_manager, resolve_model
from glassbox.core.config import settings
from glassbox.core.manager import ModelManager
from glassbox.schemas.requests import AblationRequest
from glassbox.schemas.results import AblationResult
from glassbox.services.ablation import ablation

router = APIRouter()


# Plain `def` (not async): FastAPI runs sync handlers in a threadpool, so the CPU-bound
# forward passes won't block the event loop.
@router.post("/ablation", response_model=AblationResult)
def run_ablation(
    req: AblationRequest, manager: ModelManager = Depends(get_manager)
) -> AblationResult:
    if req.component not in settings.ablation_components:
        raise HTTPException(
            status_code=422,
            detail=(
                f"Unknown component {req.component!r}. Known: {list(settings.ablation_components)}"
            ),
        )
    model = resolve_model(manager, req)
    return ablation(model, req.prompt, component=req.component, model_name=req.model)
