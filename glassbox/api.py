"""FastAPI backend — thin handlers over the logit lens.

Each handler does one job: validate → ask the ModelManager → serialize. No model cache lives
here (that's ModelManager); no numerics live here (that's logit_lens/build_result).

Run it:
    uv run uvicorn glassbox.api:app --reload
"""

from __future__ import annotations

import os
from contextlib import asynccontextmanager

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field

from glassbox.logit_lens import logit_lens
from glassbox.manager import ModelManager
from glassbox.models import REGISTRY
from glassbox.schemas import LogitLensResult

MAX_PROMPT_TOKENS = 128
DEFAULT_MODEL = "gpt2"

# CORS origins are config-driven; default to the Vite dev server.
CORS_ORIGINS = os.environ.get(
    "GLASSBOX_CORS_ORIGINS", "http://localhost:5173"
).split(",")


@asynccontextmanager
async def lifespan(app: FastAPI):
    # Build the manager and warm the always-on reference model once, at startup —
    # not per request.
    manager = ModelManager()
    manager.warm(DEFAULT_MODEL)
    app.state.manager = manager
    yield


app = FastAPI(title="GLASSBOX", lifespan=lifespan)
app.add_middleware(
    CORSMiddleware,
    allow_origins=CORS_ORIGINS,
    allow_methods=["*"],
    allow_headers=["*"],
)


class LogitLensRequest(BaseModel):
    prompt: str = Field(min_length=1)
    model: str = DEFAULT_MODEL


class ModelInfo(BaseModel):
    name: str
    display_name: str
    gated: bool
    loaded: bool


@app.get("/models", response_model=list[ModelInfo])
def list_models():
    loaded = set(app.state.manager.loaded_names())
    return [
        ModelInfo(
            name=name,
            display_name=entry["display"],
            gated=entry["gated"],
            loaded=name in loaded,
        )
        for name, entry in REGISTRY.items()
    ]


@app.get("/health")
def health():
    manager: ModelManager = app.state.manager
    return {
        "status": "ok",
        "device": manager.device,
        "loaded_models": manager.loaded_names(),
    }


# Plain `def` (not async): FastAPI runs sync handlers in a threadpool, so the CPU-bound
# forward pass won't block the event loop.
@app.post("/logit-lens", response_model=LogitLensResult)
def run_logit_lens(req: LogitLensRequest) -> LogitLensResult:
    if req.model not in REGISTRY:
        raise HTTPException(
            status_code=400,
            detail=f"Unknown model {req.model!r}. Known: {sorted(REGISTRY)}",
        )

    try:
        model = app.state.manager.get(req.model)
    except PermissionError as exc:
        # Gated model requested without HF_TOKEN — actionable, no stack trace.
        raise HTTPException(status_code=503, detail=str(exc)) from exc

    n_tokens = model.to_tokens(req.prompt).shape[1]
    if n_tokens > MAX_PROMPT_TOKENS:
        raise HTTPException(
            status_code=422,
            detail=f"Prompt is {n_tokens} tokens; max is {MAX_PROMPT_TOKENS}.",
        )

    return logit_lens(model, req.prompt, model_name=req.model)
