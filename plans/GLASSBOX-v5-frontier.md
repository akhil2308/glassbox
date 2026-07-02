# GLASSBOX — v5 frontier (after v4 ships)

Status: vision doc, nothing built. Assumes v4 landed: head zoo, circuit graph, SAE feature
lens, steering, training-time slider, guided tours, session report export.

v1–v3 inspect one forward pass. v4 discovers parts and intervenes on them. v5's theme:
**stop studying one model and start studying *models*** — universality across checkpoints,
diffing across fine-tunes, frontier attribution methods, and a lab where you can grow a model
from scratch and watch a circuit be born. Plus the door v4 deliberately kept shut, opened
carefully: an AI assistant *inside* the microscope.

These are multi-weekend projects. Independent unless noted; pick by excitement.

---

## A. The science of models — comparison as a first-class citizen

### A1. Circuit universality checker
v4's head zoo fingerprints one model. Run the same fingerprinting battery across *every*
registry model and render the money chart: induction score vs layer-depth-fraction, one line
per model. Do GPT-2, Pythia, and Gemma all put induction heads at ~the same relative depth?
That's a real open-science question ("universality") answerable with plumbing we already have.
- Backend: loop existing A1(v4) endpoint over models; cache aggressively (results are static
  per model — write them to a JSON file next to the registry, ship precomputed).
- Frontend: one comparative scatter/line view. Small.
- Cheapest item in this doc; genuinely publishable-blog-post output.

### A2. Model diffing (base vs fine-tune)
The registry invariant says base checkpoints only — keep it for the *lens*, but add a dedicated
**Diff tab** that loads a base/fine-tune pair (e.g., `pythia-160m` vs a chat-tuned sibling) and
shows where they diverge on the same prompt: per-layer KL divergence of lens distributions,
per-head attention pattern distance, DLA deltas. "What did fine-tuning actually change?" is the
question every practitioner has; almost no tool answers it visually.
- Reuses: lens pipeline ×2 and the compare-mode layout. The metrics themselves are new but
  tiny — KL between two lens distributions is a one-liner over data both pipelines already
  return.
- The invariant survives: the diff tab is explicitly *about* the base↔tuned relationship,
  not about pretending a chat model is a base model.

## B. Frontier methods — the 2024–25 toolkit

### B1. Attribution patching (gradient-based)
v4's circuit graph costs O(heads²) forward passes. Attribution patching approximates every
patch effect with **two forward passes + one backward** — the standard scaling trick in the
literature. Add it as the default engine behind the existing circuit-graph UI, with exact
patching kept as a "verify this edge" click-through.
- Backend: needs gradients through the bridge (drop `torch.no_grad()` on one path, one
  backward per metric). Numerics risk is real — gate with a test comparing attribution vs
  exact patching on GPT-2 for a fixed prompt (rank correlation, not exact match).
- Unblocks circuit graphs on models where O(heads²) is hopeless. Prerequisite for B2 at scale.

### B2. Attribution graphs via transcoders (the "Biology of an LLM" view)
The current frontier: replace MLPs with pretrained transcoders, then trace *feature-level*
causal graphs — which interpretable feature activates which, from embedding to logit. Open
weights exist for GPT-2; open-source reference implementations exist to crib the math from.
- This is v4's circuit graph upgraded from "heads as nodes" to "features as nodes" — the same
  D3 DAG view renders it; the work is the backend graph construction.
- Hardest, most impressive item in the doc. Only start after B1 works and v4's B1 (SAE lens)
  proved the feature-loading stack.

### B3. Feature steering (upgrade of v4 steering)
v4 steers with contrast-pair directions. v5 steers with *named SAE features*: search a feature
("Golden Gate Bridge"), drag a slider, watch generation obsess. Wiring is v4-C1's hook + v4-B1's
feature dictionary — mostly UI: feature search box on the steering tab.

## C. The nursery — train a model *inside* GLASSBOX

### C1. Grokking lab
v4 skipped training; v5 makes one careful exception: a 1-layer toy transformer on modular
addition — the classic grokking setup — trains in minutes on CPU. Train it live with a loss
curve, checkpoint every N steps, then turn the *entire existing toolkit* on the checkpoints:
watch the lens go from noise → memorization → the Fourier circuit crystallizing at grokking.
- Backend: a self-contained `services/toylab.py` (model is ~50 lines of plain PyTorch), a
  training endpoint streaming loss over the NDJSON channel, checkpoints in memory.
- The lens/attention/ablation views need a "toy vocab" mode (numbers, not BPE tokens) — the
  one real UI cost.
- Pedagogically this is the endgame: GLASSBOX doesn't just show you a trained model, it shows
  you *a circuit being born*. Pairs with a guided tour (v4-E1) as the capstone lesson.

## D. The copilot — v4's forbidden door, opened with a latch

### D1. Interp copilot (labeling aid only)
v4 skipped "agentic auto-interp" because it outsources understanding. The narrow version keeps
the human in the loop: a sidebar assistant that *labels what you're already looking at* — "this
head's pattern looks like previous-token attention (score 0.91)", drafts feature descriptions
from top-activating examples, and suggests the next diagnostic prompt. Never runs experiments
by itself; every suggestion is a button the user clicks.
- Backend: one endpoint proxying to the Claude API (key from env, feature stays hidden without
  it). Prompt = the JSON the UI already has. No agent framework, no loop — one call per ask.
- Strict scope latch: the copilot annotates views; it never drives them.

## E. Platform — only if GLASSBOX outgrows one person

### E1. Findings library
Every interesting result (a head, a feature, a circuit, a steering vector) gets a "save
finding" button → local JSONL + a gallery view with the v4 report-export as the share format.
No accounts, no server-side store; a findings file in the repo is the community mechanism
(PRs of findings — the registry pattern applied to knowledge).

### E2. Probe plugin interface
Third parties add a new analysis as one Python module: `run(model, prompt, params) -> result`
+ a result-type hint that picks one of the existing renderers (grid / bars / graph / text).
Do this only when a *second* person actually wants to add a probe — an interface with one
implementer is v1-invariant bait.

### E3. Vision transformer mode (ViT logit lens)
The lens generalizes to ViTs (patches instead of tokens, class-label unembed). A genuinely
different audience opens up. But it forks the tokenizer assumptions through every view —
gate it behind real demand, and prototype as a separate route reusing only the services layer.

---

## Explicitly skipped, still

- **Hosted multi-tenant SaaS** — still a different project; E1's findings-file keeps sharing
  free.
- **Full auto-interp agent loops** — D1's latch stays: annotation yes, autonomous
  experimentation no. If that changes it's a v6 debate, not scope creep.
- **Training real language models in-app** — C1's toy lab is the deliberate ceiling; anything
  bigger is a training platform, not a microscope.
- **Distributed/multi-GPU serving** — registry + remote CUDA box (v4-F) covers every model
  small enough to be worth lensing interactively.
- **3D, still.**

## Suggested order

A1 (universality — cheap, uses everything v4 built) → C1 (grokking lab — the capstone demo)
→ B1 (attribution patching — unblocks scale) → A2 (model diffing) → B3 (feature steering) →
D1 (copilot) → B2 (attribution graphs — the summit) → E items only when pulled.

Ground rules unchanged: schema changes mirrored in `api.ts`, a faithfulness-style test for
every new numerics path (B1's rank-correlation gate especially), `make test-fast && make lint`
green per item.
