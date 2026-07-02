"""Metadata routes: the model registry and a health probe. No inference here."""

from __future__ import annotations

from fastapi import APIRouter, Depends

from app.api.deps import get_manager
from app.core.manager import ModelManager
from app.core.models import REGISTRY
from app.schemas.requests import ModelInfo
from app.services.arch import describe_arch

router = APIRouter()


@router.get("/models", response_model=list[ModelInfo])
def list_models(manager: ModelManager = Depends(get_manager)):
    out = []
    for name, entry in REGISTRY.items():
        bridge = manager.peek(name)  # resident only — never triggers a load for metadata
        out.append(
            ModelInfo(
                name=name,
                display_name=entry["display"],
                gated=entry["gated"],
                loaded=bridge is not None,
                arch=describe_arch(bridge) if bridge is not None else None,
            )
        )
    return out


@router.get("/health")
def health(manager: ModelManager = Depends(get_manager)):
    return {
        "status": "ok",
        "device": manager.device,
        "loaded_models": manager.loaded_names(),
    }
