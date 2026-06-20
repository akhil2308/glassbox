# GLASSBOX — Weekend 2 Execution Plan (multi-model logit lens + web UI)

> ## ✅ STATUS: COMPLETE — 2026-06-18, branch `feat/v2` (one commit per phase, nothing pushed)
>
> **Do not re-run Phases 0–3 — they are built, verified, and committed.** What's below is kept
> as the design record. Read this banner first; the body has one important correction baked in.
>
> | Phase | Commit | State |
> |---|---|---|
> | 0 — TransformerBridge core + decode split + faithfulness tests | `59003f7` | ✅ done, GPT-2 green |
> | 1 — pydantic schemas + all-positions | `41ed707` | ✅ done |
> | 2 — FastAPI backend + ModelManager | `e3e49a6` | ✅ done |
> | 3 — React/Vite/Tailwind UI | `970e94d` | ✅ done, screenshot-verified |
>
> **⚠️ KEYSTONE CORRECTION (the plan below was wrong here — the code is right):**
> Phase 0.3 says decode via `cache.apply_ln_to_stack(...)` and avoid `model.ln_final`.
> **That is backwards on TransformerLens 3.4.0.** `apply_ln_to_stack`'s `pos_slice` is silently
> ignored → decode off by ~15 logits → fails the faithfulness guardrail. The shipped code decodes
> via **`model.ln_final(stack) @ W_U + b_U`**, which on the bridge is a `NormalizationBridge`:
> faithful to ~2.5e-5 AND architecture-general (handles RMSNorm). **Do not "restore"
> `apply_ln_to_stack`.** See `tests/test_faithfulness.py` and the [[glassbox-gotchas]] memory.
>
> **What's still open (carried forward, NOT a redo):**
> - **Gemma-3-1B gate (Phase 0.5):** never ran — `HF_TOKEN` is unset. It's in the registry as
>   `loaded:false`; the API (503 path), UI (disabled option), and faithfulness test (skip-with-reason)
>   already handle it. To finish: accept the `google/gemma-3-1b-pt` license on HuggingFace,
>   `export HF_TOKEN`, then `uv run pytest tests/test_faithfulness.py` — the Gemma cases run
>   automatically. If green, the dropdown carries two models with no code change.
> - **Push / PR:** commits are local on `feat/v2` only.
>
> **What's next → Weekend 3** (see bottom "Out of scope"): attention-pattern views + layer
> ablation (this is where D3 enters). Build it on the same extract→decode→build→serve→render spine.
>
> **How to run what exists:** `uv run uvicorn glassbox.api:app` (warms GPT-2), then
> `cd web && npm run dev` → http://localhost:5173. `uv run pytest` for the faithfulness net;
> `uv run python scripts/run_logit_lens.py` for the CLI/heatmap.

**Audience:** a coding agent executing against the `akhil2308/glassbox` repo, `feat/v1` branch.
**Goal of this milestone:** make the logit lens **live** — wrap it in FastAPI and render it in a
minimal React page — while migrating the core to TransformerLens 3.0's `TransformerBridge` so the
decode is architecture-general. GPT-2-small is the always-on reference. Gemma-3-1B is validated as
a **separate gate that does not block shipping the UI** (see Phase 0.5).

**Locked decisions (do not relitigate):**
- Backend: **FastAPI**. No Rust.
- Frontend: **Vite + React + TypeScript + Tailwind**. No Next.js. Hand-rolled SVG/div grid for
  the logit-lens heatmap — **do not** add a charting library this milestone.
- Models: **GPT-2-small** (reference / smoke test, always-on) and **Gemma-3-1B** (first modern
  target, behind a non-blocking validation gate). Design for arbitrary models.
- Multi-model is built in **now**, via `TransformerBridge`, not deferred.

**Guiding principle:** the simplest thing that faithfully shows the mechanism. Reject any step
that adds infra the milestone doesn't require. **If a single architecture (Gemma) fights the
decode, ship the live UI on GPT-2 and pick Gemma up next session — do not let it hold the
milestone hostage.**

**Code style (applies throughout):**
- **Single Responsibility Principle.** Each function/module does one job along the data flow:
  extract → decode → serialize → serve → render. No function both runs the model *and* shapes
  output. Splits are specified per-phase below.
- Registry/config is **data**, kept separate from behavior (loaders, handlers).
- Pure functions where possible (decode takes a cache + tensor, returns a tensor — no I/O, no
  model run inside).

---

## Phase 0 — Multi-model core (the foundation; do this first and completely)

The existing `glassbox/logit_lens.py` uses the **deprecated** `HookedTransformer.from_pretrained`
and a GPT-2-specific decode (`model.ln_final → model.W_U → model.b_U`). Replace both with an
architecture-general path — **validated on GPT-2 only in this phase.** Gemma is Phase 0.5.

### 0.0 De-risk the new API first (10 minutes, before writing the keystone PR)
`TransformerBridge` is new and moving. In a throwaway notebook, confirm the exact surface this
plan depends on, on GPT-2: `boot_transformers`, `enable_compatibility_mode`, `run_with_cache`,
`accumulated_resid(..., return_labels=True)`, `apply_ln_to_stack`, and whether `model.b_U`
exists. If any signature differs, fix the plan before building on it.

### 0.1 Dependencies & environment
- Bump `pyproject.toml`: **pin** `transformer-lens==<exact version confirmed in 0.0>` (not a
  floating `>=` — the Bridge API is too new to float a floor). Add `torch`, `transformers`,
  `accelerate`, `pydantic>=2`, `fastapi`, `uvicorn[standard]`. Keep `matplotlib`/`numpy` (script
  still uses them).
- Add `HF_TOKEN` support: read from env. Gated models (Gemma) require it.
- **Prerequisite for the human, not the agent:** accept the Gemma license on HuggingFace and
  export `HF_TOKEN`. The agent should fail loudly with a clear message if a gated model is
  requested without a token — never silently fall back.

### 0.2 Model loading — `glassbox/models.py` (new)
Thin loader + **separate** registry. Keep the two apart (SRP: registry is data, loader is behavior).

```python
from transformer_lens.model_bridge import TransformerBridge

# Registry is plain config data — no behavior here.
REGISTRY = {
    "gpt2":       {"hf_name": "gpt2",                "gated": False, "display": "GPT-2 small"},
    "gemma-3-1b": {"hf_name": "google/gemma-3-1b-pt", "gated": True,  "display": "Gemma 3 1B"},
}  # confirm exact HF checkpoint ids at implementation time; use base/-pt, not -it.

def load_model(model_name: str, device: str | None = None):
    device = device or pick_device()          # keep existing CPU-default logic
    bridge = TransformerBridge.boot_transformers(REGISTRY[model_name]["hf_name"], device=device)
    bridge.enable_compatibility_mode()         # HookedTransformer-equivalent numerics
    bridge.eval()
    return bridge
```

- `enable_compatibility_mode()` is **required** — it folds LayerNorm / centers weights so the
  decode path below behaves uniformly across architectures.
- Keep `pick_device()` from the current code (CPU default; the MPS+Torch-2.8 correctness warning
  is real). Allow `device="mps"` opt-in but do not default to it.

### 0.3 Architecture-general decode — rewrite `glassbox/logit_lens.py` (split by SRP)
Replace the GPT-2-specific unembed with the cache's own normalization helper. **Split the old
monolith into three single-responsibility pieces along the data flow:**

```python
# 1. extract — runs the model, pulls per-layer residual snapshots. One job: get the stack.
@torch.no_grad()
def extract_resid_stack(model, tokens):
    logits, cache = model.run_with_cache(tokens)
    accum, labels = cache.accumulated_resid(layer=-1, incl_mid=False, return_labels=True)
    return logits, cache, accum, labels      # accum: [n_layers+1, batch, pos, d_model]

# 2. decode — pure. No model run, no I/O. normalize → unembed → softmax.
def decode_stack(model, cache, accum, position):
    stack  = accum[:, 0, position, :]                  # [n_layers+1, d_model]
    normed = cache.apply_ln_to_stack(stack, layer=-1)  # uniform across architectures
    b_U    = getattr(model, "b_U", 0)                  # many archs have no unembed bias
    layer_logits = normed @ model.W_U + b_U
    return layer_logits.softmax(dim=-1)                # [n_layers+1, d_vocab]

# 3. build_result — tensors → Pydantic objects (see Phase 1). One job: shape output.
```

- **Do not** call `model.ln_final` directly — `apply_ln_to_stack` is the portable equivalent and
  is what makes RMSNorm models work without special-casing.
- `b_U` may not exist on every architecture — default to `0`, don't assume GPT-2's bias.
- Ignore Gemma's final logit soft-cap for the lens: it's monotonic, so it doesn't change token
  ranking. Note this in a code comment so it isn't "fixed" later by mistake.

### 0.4 The correctness guardrail — in `tests/`, NOT inline (most important task in this phase)
Faithfulness is the invariant that protects the entire "any model" promise. It belongs in
`tests/test_faithfulness.py`, **not** as an inline `assert` inside `decode_stack` — an inline
assert runs on every production request and vanishes under `python -O`.

```python
# tests/test_faithfulness.py — parametrized over (model, prompt)
def test_top_layer_matches_forward(model, prompt):
    logits, cache, accum, _ = extract_resid_stack(model, model.to_tokens(prompt))
    probs = decode_stack(model, cache, accum, position=-1)
    final_top = int(logits[0, -1].argmax())
    lens_top  = int(probs[-1].argmax())
    assert lens_top == final_top, "decode is unfaithful for this architecture"

    # Stronger check for non-capped models (e.g. GPT-2): argmax can agree by luck on easy
    # prompts even when the decode is subtly wrong. Skip for soft-capped models (Gemma).
    if not model_is_soft_capped(model):
        assert torch.allclose(layer_logits_top, logits[0, -1], atol=1e-3)
```

- Argmax-equality is the **portable** check (correct for Gemma — its soft-cap is monotonic).
- `allclose` on the top-layer logits is the **stronger** check; apply it only to non-capped models.
- Run for **≥5 prompts each**.

### 0.5 Gemma validation — a non-blocking gate (decoupled from the web UI)
Gemma is the *first* RMSNorm target and the proof the decode is architecture-general. But it must
**not** block Phases 2–3. Treat it as a gate:
- Run the Phase 0.4 faithfulness tests against `gemma-3-1b`.
- **Green:** the "any model" promise is proven; the dropdown ships with two models.
- **Red / fighting you:** ship the live UI on GPT-2 only, leave Gemma in the registry marked
  `loaded: false`, and carry the fix into next session. The keystone PR (GPT-2) stays merged and
  `main` stays runnable either way.

### 0.6 Generalize position handling (needed for the web UI)
Add a `positions: list[int] | None = None` param. When `None`, decode **all** positions (so the
UI can show any token's journey). Return a result whose `layers` dimension is per-position.
Keep `position=-1` single-position behavior available for the script.

- **Define `answer_prob` precisely:** for a given position, it's the probability that *that
  position's own final-layer top token* receives at each layer. The "answer materializes upward"
  story reads cleanly only for the **last** position — document this so the frontend color
  semantics aren't muddy.
- **BOS:** `prepend_bos` differs by model. The BOS column's "journey" is meaningless — flag it in
  the result (e.g. `is_bos: bool` per position) so the UI can label or hide column 0.

### 0.7 Phase 0 acceptance criteria
- [ ] `load_model("gpt2")` boots via `TransformerBridge`; registry is separate data from the loader.
- [ ] `logit_lens.py` is split into `extract_resid_stack` / `decode_stack` / `build_result`
      (decode is pure: no model run, no I/O).
- [ ] `scripts/run_logit_lens.py` runs unchanged in spirit on GPT-2 (swap loader import).
- [ ] Faithfulness tests live in `tests/`, pass on GPT-2 for ≥5 prompts (argmax + allclose).
- [ ] Gemma faithfulness tests run; result (pass/fail) recorded — **not** a blocker for Phase 2+.
- [ ] No remaining reference to `HookedTransformer.from_pretrained` in the codebase.
- [ ] All-positions decode produces sane shapes; last position matches the single-position path;
      BOS position is flagged.

---

## Phase 1 — Serialization layer (`glassbox/schemas.py`, new)

Convert the dataclasses to **Pydantic v2** `BaseModel` so FastAPI serializes them for free.

- `LayerPrediction(layer, label, top_token, top_prob, answer_prob)`
- `TokenJourney(position, str_token, is_bos, layers: list[LayerPrediction])`
- `LogitLensResult(prompt, model_name, str_tokens, final_top_token, tokens: list[TokenJourney])`

`build_result` (from 0.3) is the only place tensors become these objects — keep it the single
boundary between numerics and JSON.

Acceptance:
- [ ] `result.model_dump()` returns clean JSON (floats, not tensors).
- [ ] A round-trip `LogitLensResult(**result.model_dump())` succeeds.

---

## Phase 2 — FastAPI backend (`glassbox/api.py` + `glassbox/manager.py`, new)

### 2.1 Model lifecycle — a `ModelManager`, separate from the handlers (SRP)
The `{model_name: bridge}` cache, lazy loading, and eviction are **one responsibility** — put
them in a `ModelManager` class in its own module. Endpoint handlers must not touch the cache dict
directly; they call `manager.get(name)`.

- Load the model **once** at startup, not per request (boot is multi-second; inference is
  sub-second). A `lifespan` context manager constructs the `ModelManager` and warms GPT-2.
- `ModelManager` owns: lazy load on first request for a model, an in-memory cache, and a cap of
  **1–2 resident models** with eviction to bound memory (Gemma-3-1B fp32 on CPU is a few GB).

### 2.2 Endpoints — thin handlers (validate → manager → serialize)
- `GET /models` → registry list with `{name, display_name, gated, loaded}` (reads `REGISTRY` +
  `manager` state).
- `POST /logit-lens` → body `{prompt: str, model: str = "gpt2"}` → `LogitLensResult` JSON.
  - Define the handler as a **plain `def`, not `async def`** — FastAPI runs sync handlers in a
    threadpool, so the CPU-bound forward pass won't block the event loop. Free correctness.
  - Validate `model` against the registry; **400** on unknown.
  - **Cap prompt length** (reject > N tokens, e.g. 128) so a giant prompt can't hang/OOM the
    single worker; **422** with a clear message.
  - **503** with an actionable message if a gated model is requested without `HF_TOKEN`.
- `GET /health` → `{status, device, loaded_models}`.

### 2.3 CORS
Allow `http://localhost:5173` (Vite dev default). Keep it config-driven.

Acceptance:
- [ ] `uvicorn glassbox.api:app` starts; GPT-2 loads in `lifespan`, not per request.
- [ ] Cache/eviction lives in `ModelManager`; handlers don't touch the cache dict.
- [ ] `curl POST /logit-lens` returns valid JSON for `gpt2` **well under a second after warm-up**
      (latency target scoped to GPT-2 only — Gemma-3-1B fp32 on CPU will be slower; that's fine).
- [ ] Over-length prompt → 422; unknown model → 400; gated model without token → 503 + actionable
      message (no stack trace).
- [ ] If Gemma passed 0.5, switching `model: "gemma-3-1b"` loads it lazily and returns a faithful
      result.

---

## Phase 3 — React frontend (`web/`, new — Vite scaffold)

### 3.1 Scaffold
- `npm create vite@latest web -- --template react-ts`, add Tailwind.
- One screen. No router. State via `useState`. No global state lib.

### 3.2 Components
- **PromptBar:** text input + model `<select>` (populated from `GET /models`, disabled options
  for `loaded: false` if Gemma didn't pass 0.5) + Run button. Use `onClick`/`onChange` handlers —
  **no `<form>` tags**.
- **LogitLensGrid:** the core view. Rows = layers `0..n_layers` (0 = embeddings, top = full model);
  columns = prompt tokens. Each cell shows the layer's top-predicted token (truncated) with
  background-color intensity driven by `answer_prob`. **Label or hide the BOS column** (`is_bos`).
  - Plain `<div>`/CSS grid or an SVG `<g>` grid. **No charting library.**
  - Hover tooltip: full token text + `top_prob` + `answer_prob`.
  - Clicking a column pins that token's full journey (uses the all-positions data from 0.6).
- **StatusBar:** active model, device, request latency.

### 3.3 Data flow
- `fetch` to the FastAPI base URL (env var, default `http://localhost:8000`).
- Loading + error states: 422 (prompt too long), 400 (bad model), 503 (gated → show "accept
  license + set HF_TOKEN").

Acceptance:
- [ ] Typing a prompt, picking a model, clicking Run renders a live logit-lens grid.
- [ ] The "answer materializes upward" effect is visible on the last token (low → high
      `answer_prob` climbing rows).
- [ ] BOS column is labeled/hidden, not rendered as a meaningless journey.
- [ ] Error states surface clearly (422 / 400 / 503).
- [ ] Works for `gpt2` (and `gemma-3-1b` iff 0.5 passed).

---

## Out of scope this milestone (do NOT build yet)
- Attention-pattern views and layer ablation → Weekend 3 (this is where D3 enters).
- SAEs / feature browser → stretch.
- Auth, persistence, deployment → not now. Local-first dev tool.
- nnsight/remote execution → only when a target model won't fit locally. Gemma-3-1B fits.

## Suggested commit/PR sequence
1. `chore: spike TransformerBridge API surface on GPT-2` (Phase 0.0 — throwaway notebook, no merge).
2. `feat: migrate to TransformerBridge + architecture-general decode (GPT-2)` — split into
   extract/decode/build, faithfulness tests in `tests/`. **The keystone PR; must stand alone.**
3. `test: gemma-3-1b faithfulness gate` (Phase 0.5 — non-blocking; records pass/fail).
4. `feat: pydantic schemas + all-positions logit lens` (Phase 1 + 0.6).
5. `feat: FastAPI backend with ModelManager lifecycle` (Phase 2).
6. `feat: minimal React logit-lens UI` (Phase 3).

Each PR leaves `main` runnable. **PR 2 is the keystone**: it ships a live-ready GPT-2 decode with
a faithfulness net, and it does **not** depend on Gemma passing. If the Gemma gate (PR 3) is green,
the dropdown carries two models; if not, the UI still ships on GPT-2 and Gemma follows next session.
