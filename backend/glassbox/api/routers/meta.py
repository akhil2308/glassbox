"""Metadata routes: the model registry and a health probe. No inference here."""

from __future__ import annotations

from fastapi import APIRouter, Depends

from glassbox.api.deps import get_manager
from glassbox.core.manager import ModelManager
from glassbox.core.models import REGISTRY
from glassbox.schemas.requests import ModelInfo

router = APIRouter()


@router.get("/models", response_model=list[ModelInfo])
def list_models(manager: ModelManager = Depends(get_manager)):
    loaded = set(manager.loaded_names())
    return [
        ModelInfo(
            name=name,
            display_name=entry["display"],
            gated=entry["gated"],
            loaded=name in loaded,
        )
        for name, entry in REGISTRY.items()
    ]


@router.get("/health")
def health(manager: ModelManager = Depends(get_manager)):
    return {
        "status": "ok",
        "device": manager.device,
        "loaded_models": manager.loaded_names(),
    }
