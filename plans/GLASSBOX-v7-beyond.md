# GLASSBOX — v7: beyond the GPT forward pass (after v6 ships)

Status: vision doc. Assumes v6 landed: experiment specs, circuit CI, 7B/GPU path, corpus mode,
faithfulness dashboard.

Every version so far lenses one thing: a decoder-only text transformer predicting the next
token. v7's theme: **the assumptions become parameters**. New modalities, new architectures,
and the unit of analysis growing from one forward pass to one *reasoning trace*. This is the
speculative edge of the roadmap — expect half of it to be cut on contact with reality.

---

## A. New inputs — vision joins text

### A1. ViT mode (promoted from v5's waitlist)
Logit lens over a vision transformer: patches instead of tokens, class labels instead of
vocabulary, attention maps rendered *on the image*. The services layer generalizes (it's still
resid → norm → unembed); the views need an image-grid twin of the token chips.
- Gate: TransformerLens-family support for ViTs at whatever version the v6 circuit CI has
  certified. If the bridge can't load one, this whole section waits — don't hand-roll a
  second model backend.

### A2. Multimodal lens (CLIP-style or small VLM)
Watch image information enter the text stream: which layer does "the picture of a dog" become
retrievable as the token " dog"? Highest wow-factor in the doc, and strictly after A1 —
it's A1's machinery plus a text decoder.

## B. New architectures — is the lens even transformer-shaped?

### B1. State-space models (Mamba-class)
No attention, but there is still a residual stream — so the logit lens applies, ablation
applies, and the attention tab honestly greys out. Shipping "here's what transfers and what
doesn't" is itself an interpretability statement.
- Contingent on library support (TL's Mamba coverage, or a minimal adapter satisfying the
  small surface our services actually use: `run_with_cache`-ish, `ln_final`, `W_U`). Define
  that surface as a protocol first — it's the refactor A1 needs anyway, do it once.

## C. New unit of analysis — from token to thought

### C1. Reasoning-trace lens
Generate a chain-of-thought with a small reasoning-tuned model, then treat the *trace* as the
object: per-step lens snapshots, which earlier step each step attends to (step-level arc
diagram — the token arc view, aggregated), ablate a reasoning step and watch the answer change.
- Note the invariant collision: this needs an instruction/reasoning-tuned model. Resolution as
  in v5's diff tab — a dedicated tab whose *subject* is tuned-model behavior, while the base
  tabs keep their base-model rule.
- The step-ablation experiment ("delete step 3, does the conclusion survive?") reuses the
  ablation machinery on the prompt rather than the weights. Cheap and genuinely novel UI.

### C2. Circuit tracking across a trace
v5's attribution graphs, sampled at each reasoning step: does the model reuse one circuit per
step, or different ones per step type? Pure composition of existing pieces; expensive to run;
corpus mode (v6-B2) provides the aggregation.

---

## Skipped

- **Audio/video models, diffusion models** — different math, different project.
- **Interpretability of RLHF/reward models** — real research topic, but there's no lens story
  yet worth a tab; revisit when the field has one.
- **Supporting every architecture** — B1's protocol defines the club's membership rules;
  models that can't satisfy the minimal surface stay out rather than getting adapters built.

## Suggested order

B1's protocol refactor first (everything in this doc stands on it) → A1 → C1 → A2 → C2.
Cut ruthlessly: any section still blocked on upstream library support after a spike gets
parked, not hand-rolled.
