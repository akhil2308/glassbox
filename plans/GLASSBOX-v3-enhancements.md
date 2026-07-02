# GLASSBOX — v3 enhancement plan (UI + capabilities)

Status: proposal, nothing built. Ranked by value-per-effort — do Tier 1 top-to-bottom, stop
whenever it stops being fun. Everything respects the existing invariants (ln_final decode path,
CPU default, TL pinned at 3.4.0, base checkpoints only).

Current state (what this plan builds on): logit lens with per-token journeys + simulate +
compare, per-layer/head attention (heatmap + arcs), per-layer ablation (block/attn/mlp),
2-model registry, arch badge, faithfulness test suite.

---

## Tier 1 — cheap, high value (hours each)

### 1. More models in the registry
The single biggest capability-per-line win: each model is ~5 lines of data in
`backend/app/core/models.py::REGISTRY`.
- Add: `gpt2-medium`, `pythia-160m`, `pythia-410m` (ungated, small, well-studied in the
  interp literature), and optionally `qwen2.5-0.5b`.
- Gate each addition on the existing faithfulness test passing for it (parametrize
  `tests/test_faithfulness.py` over the new names, mark `slow`).
- Makes Compare mode actually interesting (GPT-2 vs Pythia layer counts differ — the grids
  already render independently, so no work needed there).
- Risk: `TransformerBridge` support varies per architecture on TL 3.4.0 — verify each with the
  faithfulness test before committing; drop any that fail rather than special-casing.

### 2. Top-k alternatives per lens cell
Right now each cell shows only the top-1 token. The most common question when reading a lens
grid is "what else was it considering?"
- Backend: `LayerPrediction` gains `top_k: list[tuple[str, float]]` (k=5). One extra
  `topk()` in `build_result`; payload stays small.
- Frontend: hover tooltip (or click popover) on a grid cell listing the 5 candidates with
  probabilities. Mirror the type in `api.ts`.
- Files: `schemas/results.py`, `services/logit_lens.py`, `api.ts`, `LogitLensGrid.tsx`.

### 3. Shareable URLs (prompt + model + tab in query params)
- `useEffect` to read `?prompt=&model=&tab=` on mount, `history.replaceState` on run.
  ~20 lines in `App.tsx`, no router dependency (ladder rung 3: native platform feature).
- Turns every interesting finding into a pasteable link — big for a demo/portfolio project.

### 4. Attention head overview grid ("find the interesting head")
The current UI makes you flip through layer×head dropdowns blind (GPT-2 = 144 combinations).
Show all heads of one layer as small-multiple mini-heatmaps (no labels, ~60px each); click one
to open it in the existing detail view.
- Pure frontend — `AttentionResult.patterns` already contains every head. New small component
  inside `AttentionView.tsx`, reuse `weightStyle`.
- This is how people actually found induction heads; it's the missing discovery affordance.

### 5. Example prompts
A few curated prompts as clickable chips under the PromptBar, each chosen to show something
("The Eiffel Tower is in" → fact recall; "When Mary and John went to the store, John gave a
drink to" → the IOI induction classic; a repeated-sequence prompt for induction heads).
- Static array + buttons in `PromptBar.tsx`. Zero backend.

---

## Tier 2 — real features (a day-ish each)

### 6. Per-head ablation
Ablation currently zeroes a whole layer's attn/mlp/block. Zeroing a single head is the classic
"which head is load-bearing" experiment and pairs perfectly with #4 (spot a head → delete it).
- Backend: `AblationRequest` gains optional `head: int`; hook zeroes `z[:, :, head, :]` for the
  chosen layer instead of the whole component. Result shape unchanged (effects over layers when
  head is null, over heads of one layer when set).
- Frontend: when `component == "attn"`, a second selector for head; bar chart x-axis becomes
  heads. Reuse `BarChart` as-is.
- Files: `schemas/requests.py`, `services/ablation.py`, `routers/ablation.py`,
  `AblationView.tsx`, one new test.

### 7. Backend `/generate` endpoint (replace the n² simulate loop)
`runSimulate` currently re-runs the full lens once per token — O(n²) forward passes and
re-tokenization drift risk (already flagged with a `ponytail:` comment).
- One endpoint that generates k tokens greedily server-side and returns the lens result per
  step, streamed as NDJSON lines (plain `StreamingResponse`; no SSE/websocket dependency —
  `fetch` + `ReadableStream` reads it fine).
- Add `temperature`/`top_k` sampling params here if desired — this is also where sampling
  belongs, not in the lens.
- Keeps the existing UI animation code; only `runSimulate` in `api.ts` changes.
- Do this only when simulate on longer prompts actually feels slow — the lazy loop works today.

### 8. Direct logit attribution (per-layer contribution, not just accumulation)
The lens shows the *accumulated* residual; DLA shows what each layer *added* toward the final
answer — "which layer wrote this token in".
- Backend: `cache.decompose_resid` (or per-layer deltas of the existing `accumulated_resid`),
  dot with the answer token's unembed direction. New small service function + field on the lens
  result (`layer_contribution: float` per LayerPrediction) or a separate endpoint.
- Frontend: a bar strip under each token journey, or a toggle on the existing grid coloring
  (accumulated prob vs delta).
- The cheapest genuinely-new interpretability capability; reuses the whole extract pipeline.

### 9. Model-loading progress
First load of a model downloads ~0.5–2GB with no feedback beyond a spinner and an eventual
result. Add a `/models/{name}/load` POST that returns immediately plus a `loading` state in
`ModelManager` surfaced through `/models`; frontend polls (the `fetchModels` refresh already
exists) and shows "downloading weights…" in the StatusBar.
- Skip real byte-level progress (HF hub callbacks are fiddly) — a three-state
  `not_loaded / loading / loaded` badge is 90% of the UX for 10% of the work.

---

## Tier 3 — only if the project keeps going (multi-day)

### 10. Activation patching
The flagship next capability: run a clean and a corrupted prompt, copy one layer/position's
activation from clean into corrupted, measure recovery. This is *the* causal-tracing technique
(ROME-style) and would make GLASSBOX cover the full intro-to-mech-interp toolkit.
- Needs: two-prompt request schema with equal token lengths (validate!), a patching service
  using TL hooks, a results heatmap (layer × position, recovery %) — the LogitLensGrid rendering
  pattern reuses directly.
- Big, but it's one endpoint + one view, same shape as the existing three tabs. Design as
  tab #4.

### 11. Attention with value weighting / info-flow view
Raw attention weights overstate what actually moves information; `pattern × |v|` (value-weighted
attention) is more faithful. Toggle on the existing views. Moderate backend change (needs `v`
from the cache), payload doubles — do only after #4 proves people live in the attention tab.

### 12. Export as PNG / SVG
`svg → canvas → toBlob` for the D3 views; `html-to-image` would be a new dependency, so prefer
serializing the SVG directly (it already is one). Nice for sharing; not load-bearing.

---

## Explicitly skipped (and why)

- **Websockets** — NDJSON streaming over plain fetch covers generation; no new protocol.
- **State-management library / router** — App.tsx's useState + query params scale fine at 3–4
  tabs. Add a router only if pages (not tabs) appear.
- **OpenAPI-generated TS client** — the hand-mirrored types are 5 small interfaces; codegen is
  more machinery than the problem. Revisit past ~10 endpoints.
- **Instruction-tuned models, chat UI** — against the project's own invariant (base checkpoints
  only, raw next-token mechanics).
- **MPS device support** — stays opt-in until TL's MPS numerics are verified again (measured
  silently-wrong before).
- **Auth/multi-user/persistence** — it's a local research toy; YAGNI.
- **Mobile layout work** — the visualizations are dense grids; desktop-only is the honest scope.
  Keep the existing overflow-x scrolling as the fallback.

## Suggested order

1 → 3 → 5 (one afternoon, mostly data + tiny UI) → 2 → 4 (the two "make the lens/attention
tabs actually explorable" wins) → 6 → 8 → 7 → 9 → then decide if 10 (patching) is the next
weekend project.

Each backend item lands with: schema change mirrored in `api.ts`, one test (fast where
possible), and `make test-fast && make lint` green.
