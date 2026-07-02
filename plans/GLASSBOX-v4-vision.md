# GLASSBOX — v4 vision (after v3 ships)

Status: vision doc, nothing built. Assumes v3 landed: multi-model registry, top-k lens cells,
head overview grid, per-head ablation, streaming `/generate`, direct logit attribution, and
activation patching as tab #4.

v1–v3 answer "what is the model doing?" — v4's theme is the next three questions:
**"which parts do what?" (discovery), "what do they mean?" (features), and "can I change it?"
(intervention)**. Plus one wildcard nobody else's toy has: **time**.

Each item below is a standalone weekend-sized (or multi-weekend) project. They don't depend on
each other except where noted — pick by excitement, not order.

---

## A. Automated discovery — "stop hunting heads by hand"

### A1. Head fingerprinting / the Head Zoo
Auto-classify every head in a model by running diagnostic prompts and scoring the resulting
patterns against known signatures:
- **previous-token head**: attention mass on position i-1
- **induction head**: on a repeated random-token sequence `[A B ... A]`, attends from the second
  `A` to the token after the first `A`
- **duplicate-token head**: attends to earlier copies of the same token
- **positional/BOS-sink head**: mass parked on BOS
Backend: one endpoint that runs 3–4 fixed diagnostic prompts and returns a `[layer][head]`
score per signature (each detector is a ~10-line dot product over the existing attention
patterns — no new numerics infrastructure). Frontend: the v3 head-overview grid gains colored
badges; click a badge to see the diagnostic prompt that fired. This turns "flip through 144
heads" into "here are your 6 induction heads."

### A2. Circuit graph (path patching lite)
The marquee feature. For a given prompt + answer, build a directed graph: which heads' outputs
matter for the final logit, and which earlier heads feed *them* (one level of composition via
patching a head's input from its ablated counterfactual). Render as a D3 force/layered DAG —
layers bottom-to-top, heads as nodes, edge weight = patching effect. The IOI circuit,
rediscovered live in the browser.
- Cost: O(heads²) forward passes if done naively — cap to top-k heads from a first DLA pass
  (score all heads once, only patch edges among the top ~15). GPT-2-small on CPU stays in
  tens-of-seconds territory; stream progress over the v3 NDJSON channel.
- This is the hardest item in the doc and the most impressive. Do A1 first; it's the on-ramp.

## B. Features — "what does this direction *mean*?"

### B1. SAE feature lens (Neuronpedia-lite)
Load a pretrained sparse autoencoder for GPT-2-small (SAELens has published weights; no
training, inference only) and add a lens mode that shows, per token per layer, the top firing
*features* instead of top tokens — with feature labels pulled from Neuronpedia's public API
(cached to disk; keep working offline with the raw feature id if the API is unavailable).
- New dependency (`sae-lens`) — justified, this is genuinely new capability, not convenience.
- Scope guard: **one model, one layer's SAE** first (e.g., resid_post layer 8). Prove the tab,
  then widen. Feature dashboards, activation histograms, feature search: all later.
- This is the current frontier of the field in one tab; nothing else in the project says
  "I understand where interp is going" louder.

### B2. Neuron view (only if B1 stalls)
Top-k dataset examples that maximally activate a chosen MLP neuron, over a small bundled text
corpus (~1MB). Strictly worse science than SAE features (neurons are polysemantic — that's the
point of SAEs), so build it only as a fallback if SAELens fights the TL 3.4.0 pin.

## C. Intervention — "grab the steering wheel"

### C1. Steering vectors (ActAdd)
Compute a direction from a contrast pair ("I love this" − "I hate this" at layer L), then add
`α · direction` into the residual stream during `/generate` and watch the continuation change.
UI: two small prompt boxes for the pair, a layer picker, and an α slider; output rendered as
two side-by-side generations (α=0 vs α=chosen) reusing the compare-mode layout.
- Backend is small: one hook adding a cached vector, riding the v3 generate endpoint.
- Highest fun-per-line in this doc. Demo gold.

### C2. Live attention knockout during generation
While simulating, click a head (or an edge in the arc view) to zero it for the *rest of the
generation* and watch the text derail in real time. Mostly wiring: v3 per-head ablation hooks +
the generate stream + a bit of UI state. Do after C1 proves the intervention plumbing.

## D. Time — the wildcard

### D1. Lens over training time (Pythia checkpoints)
Pythia publishes revision checkpoints across training (step1k, step10k, ... step143k). Add a
"training step" slider to the lens and ablation tabs: same prompt, same view, scrubbing through
*the model learning*. Watch induction heads switch on mid-training; watch a fact get memorized.
- No new numerics at all — the entire feature is registry entries with an HF `revision` field
  plus the manager treating (model, revision) as the cache key. The bottleneck is disk/download
  (~10 checkpoints × 160M model ≈ manageable; cap the slider to 6–8 curated steps).
- Nobody's weekend project has this. Pairs beautifully with A1 (fingerprint per checkpoint →
  "induction emerges at step 20k" as a chart).

## E. Storytelling — make it teach

### E1. Guided tours
Scripted walkthroughs driving the existing UI: "Watch a fact assemble" (lens), "Find the
induction heads" (attention + A1), "Break the model" (ablation), "Rediscover IOI" (patching).
Implementation is a static array of steps `{tab, prompt, selection, caption}` and a next/prev
overlay — no new backend. GLASSBOX's real audience is people *learning* interp; this converts
the toolkit into a course. Cheapest item in the doc, arguably the highest total value.

### E2. Session notebook / shareable reports
Every run already produces JSON; record runs into a session log and export a self-contained
HTML report (results inlined, no live backend needed to view). Start with "copy result as
JSON" + "download report" — no persistence layer, no accounts, localStorage only.

## F. Infrastructure (only as pulled in by the above)

- **GPU/remote backend option**: `GLASSBOX_API_URL` already decouples the frontend; document
  running the backend on a CUDA box (e.g., a rented GPU) for 1–7B models. No code, just docs +
  registry entries — do it the day CPU latency actually hurts (A2, D1 are the likely triggers).
- **Result cache**: memoize (model, prompt, endpoint) → JSON with an LRU dict. Twenty lines,
  do it when tours (E1) re-run the same prompts constantly.
- **Payload diet for big models**: per-layer attention fetch instead of the full
  `[L][H][n][n]` blob — only when a >24-layer model actually lands in the registry.

---

## Explicitly skipped, even with unlimited imagination

- **Training/fine-tuning anything in-app** — GLASSBOX inspects models; D1 gets the "learning"
  story from free published checkpoints instead.
- **Chat interface / instruction-tuned models** — still against the core invariant.
- **Multi-user / hosted SaaS / accounts** — E2's static HTML export is the shareable artifact;
  hosting other people's compute is a different project.
- **Agentic "auto-interpret this model" LLM loops** — fun, but it outsources exactly the
  understanding the tool exists to build. Revisit only as a labeling aid inside B1.
- **3D visualizations** — the data is layers × positions × heads; 2D small-multiples beat a
  spinning cube every time.

## Suggested order

E1 (tours — cheap, compounds everything) → A1 (head zoo) → C1 (steering) → D1 (training-time
slider) → B1 (SAE lens) → A2 (circuit graph) → C2 → E2. F items only when pulled in.

Same ground rules as v3: every backend change mirrored in `api.ts`, faithfulness-style test
for any new numerics path, `make test-fast && make lint` green per item.
