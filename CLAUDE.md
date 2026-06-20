# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## What this is

GLASSBOX decodes a transformer's forward pass and shows it: logit lens (residual stream decoded
at every layer), attention patterns (per-layer, per-head query/key), and layer ablation (zero a
component, measure what breaks). Backend is FastAPI + TransformerLens; frontend is React + Vite
+ D3, talking JSON over HTTP.

## Commands

From the repo root (Makefile is the entry point):

```bash
make setup       # uv sync (backend) + npm ci (frontend)
make dev         # backend (:8000) + frontend (:5173) together
make api         # backend only, with reload
make web         # frontend only
make test        # full backend suite — boots a real model, downloads GPT-2 (~500MB) first run
make test-fast   # backend tests with `-m "not slow"` — no model load
make lint        # ruff check (backend) + eslint (frontend)
make format      # ruff format + ruff check --fix (backend)
make fmt-check   # ruff format --check, no writes (used in CI)
```

Single backend test: `cd backend && uv run pytest tests/test_ablation.py::test_name -v`.
Frontend type-check + build: `cd frontend && npm run build` (runs `tsc -b` then `vite build`).
Standalone CLI demo (renders a heatmap to `backend/outputs/`): `cd backend && uv run python scripts/run_logit_lens.py`.

## Backend architecture (`backend/glassbox/`)

Strict layering, each module a single responsibility:

- `main.py` — FastAPI app construction, CORS, lifespan (warms `settings.default_model` once at
  startup), router mounting. No request handling or numerics here.
- `api/routers/` — one `APIRouter` per endpoint (`logit_lens.py`, `attention.py`, `ablation.py`,
  `meta.py`). `api/deps.py` has `get_manager` (pulls the `ModelManager` off `app.state`) and
  `resolve_model` (the shared preamble: validate model name → load → cap prompt length →
  standard HTTPExceptions: 400 unknown model, 503 gated without `HF_TOKEN`, 422 too long).
- `core/manager.py` — `ModelManager`: thread-safe, LRU-evicting cache of resident
  `TransformerBridge`s (default cap 2). Endpoint handlers never touch the cache dict directly,
  always `manager.get(name)`.
- `core/models.py` — `REGISTRY` (plain data: known checkpoints, gated flag, display name) and
  `load_model` (boots a `TransformerBridge`, calls `enable_compatibility_mode()`, sets eval).
  `pick_device()` defaults to CPU even on Apple Silicon — see Gotchas below.
- `core/config.py` — `Settings` (pydantic-settings), env-prefixed `GLASSBOX_*`, also reads a
  `backend/.env`. `HF_TOKEN` is deliberately *not* part of this object (it's a HF-convention env
  var read directly in `load_model`).
- `services/` — the actual numerics; the single boundary where tensors become JSON-serializable
  result objects. Each service module (`logit_lens.py`, `attention.py`, `ablation.py`,
  `tokens.py`) follows the same extract → compute (pure) → build-result shape.
- `schemas/` — `requests.py` (request bodies, e.g. `PromptRequest`, `AblationRequest`) and
  `results.py` (response shapes; the frontend's `api.ts` interfaces mirror these by hand — keep
  them in sync when changing either side).

## Frontend architecture (`frontend/src/`)

- `App.tsx` — single-page tab UI (`lens` / `attention` / `ablation`), one result slot per tab so
  switching tabs preserves the last render. `onRun` dispatches to the active tab's API call.
- `api.ts` — all backend calls; `ApiError` carries HTTP status + a friendly message (503 gets a
  "accept the license + set HF_TOKEN" hint appended). Types here are hand-mirrored from
  `backend/glassbox/schemas/results.py` — there's no codegen, so update both sides together.
- `components/` — one component per visual concern (`LogitLensGrid`/`LogitLensStream`/
  `LogitLensSection`, `AttentionView`, `AblationView`, plus `PromptBar`, `StatusBar`,
  `EmptyState`, `LoadingOverlay`).
- `theme.ts` — shared color/font tokens, referenced via inline `style` rather than Tailwind
  classes for anything color-related (Tailwind utility classes are still used for layout/spacing).

## Critical invariants — do not "simplify" these away

- **Logit lens decode must go through `model.ln_final(...) @ model.W_U + b_U`**, not
  `cache.apply_ln_to_stack`. The latter was measured unfaithful (~15 logits off on GPT-2);
  `ln_final` (a `NormalizationBridge` on `TransformerBridge`) reproduces real forward logits to
  ~1e-5 and is architecture-general (LayerNorm or RMSNorm). A faithfulness test in
  `backend/tests/` guards this — see `plans/GLASSBOX-weekend2-plan.md` for the spike that pinned
  it down.
- **CPU is the default device, including on Apple Silicon**, even though MPS is available.
  `pick_device()` in `core/models.py` picks CUDA if present, else CPU — never MPS automatically.
  TransformerLens's MPS backend can silently return incorrect numbers; this was confirmed by a
  measured residual-stream reconstruction error. Pass `device="mps"` / `GLASSBOX_DEVICE=mps`
  explicitly only once that's verified trustworthy again.
- **`transformer-lens` is pinned exactly (`==3.4.0`)**, not floored — `TransformerBridge`'s
  surface moves between releases and the decode path above is verified against this exact
  version.
- Gemma soft-caps final logits; the lens deliberately does not apply that cap when decoding
  intermediate layers, since the cap is monotonic and never changes token ranking.
- Only base/pretrained checkpoints (not `-it` instruction-tuned) belong in `REGISTRY` — the lens
  is about raw next-token mechanics.

## Configuration

`GLASSBOX_*` env vars (or `backend/.env`): `GLASSBOX_CORS_ORIGINS`, `GLASSBOX_DEFAULT_MODEL`,
`GLASSBOX_DEVICE` (`cpu`/`cuda`/`mps`). `HF_TOKEN` (unprefixed) is required to load gated models
(e.g. Gemma). Frontend reads `VITE_API_URL` for the backend base URL (default
`http://localhost:8000`).

## CI

`.github/workflows/ci.yml` runs two independent jobs: backend (`uv sync --extra dev`, ruff check,
ruff format --check, pytest — full suite, with HF cache for GPT-2) and frontend (`npm ci`,
eslint, `vite build`).
