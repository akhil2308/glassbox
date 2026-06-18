"""FastAPI application — wiring only.

Builds the app, configures CORS, warms the reference model at startup, and mounts the routers.
No request handling, no numerics, no config literals live here — those are in `api/routers/`,
`services/`, and `core/config.py` respectively.

Run it:
    uv run uvicorn glassbox.main:app --reload
"""

from __future__ import annotations

from contextlib import asynccontextmanager

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from glassbox.api.routers import ablation, attention, logit_lens, meta
from glassbox.core.config import settings
from glassbox.core.manager import ModelManager


@asynccontextmanager
async def lifespan(app: FastAPI):
    # Build the manager and warm the always-on reference model once, at startup —
    # not per request.
    manager = ModelManager(device=settings.device)
    manager.warm(settings.default_model)
    app.state.manager = manager
    yield


app = FastAPI(title="GLASSBOX", lifespan=lifespan)
app.add_middleware(
    CORSMiddleware,
    allow_origins=settings.cors_origin_list,
    allow_methods=["*"],
    allow_headers=["*"],
)

app.include_router(meta.router)
app.include_router(logit_lens.router)
app.include_router(attention.router)
app.include_router(ablation.router)
