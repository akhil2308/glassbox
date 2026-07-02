# GLASSBOX — v6: from toy to instrument (after v5 ships)

Status: vision doc. Assumes v5 landed: universality checker, model diffing, attribution
patching, grokking lab, copilot, findings library.

By v5 GLASSBOX can discover, explain, and intervene. v6's theme: **other people's work depends
on it**. A microscope becomes an instrument when results are reproducible, scriptable, and
trusted at scale. Fewer fireworks than v4/v5 — this is the version where the project grows up.

---

## A. Reproducibility — experiments as artifacts

### A1. Experiment specs
Every run (lens, patch, circuit, steering) is already a request body. Promote that: a saved
experiment is a small YAML/JSON file — model, revision, prompt(s), method, params, expected
result hash. `glassbox run experiment.yaml` executes it headless and diffs against the pinned
expectation. The UI gains "export as experiment" next to v4's report export.
- The backend services are already pure enough; the work is a thin CLI (`argparse`, no
  framework) and a result-hashing convention (round floats, hash the JSON).

### A2. Interp regression tests ("circuit CI")
The killer application of A1: pin a finding — "heads 5.1 and 5.5 are induction heads,
score > 0.8" — as an experiment file in a repo, run the suite in CI. When a new TransformerLens
version, a quantization change, or a new checkpoint breaks a circuit, a test goes red.
Nobody has "unit tests for model internals" as a product; GLASSBOX is one `pytest` plugin away.
- Ships as: `glassbox-check` (a pytest collection hook over `experiments/*.yaml`) + docs.
- This also finally lets us *unpin* transformer-lens safely: the circuit suite is the
  compatibility gate the version pin was standing in for.

## B. Scale — models people actually ask about

### B1. 7B-class support, GPU-first path
Promote v4's "remote CUDA box" doc note to a supported mode: quantized loading (whatever the
currently certified TL version supports — verify, don't assume; "certified" per A2, which
replaces the hard version pin with the circuit-CI gate), per-layer attention fetch (the v4-F payload
diet, now mandatory), result streaming everywhere, and honest UX for 30-second experiments
(progress, cancel, queue-of-one).
- Registry gains `llama-3.2-1b`/`gemma-2-2b`-class entries first; 7B only where the
  faithfulness test passes under quantization (test tolerance needs loosening — measure, then
  pick the bound; a quantized lens that's *qualitatively* faithful is the bar).

### B2. Batch/corpus mode
Run one probe over a prompt *dataset* (a few hundred lines of text) and aggregate: which heads
fire, which features appear, mean answer_prob per layer. Turns anecdotes ("on this prompt...")
into statistics ("across 500 prompts..."). Backend batches through the existing services;
frontend gets one aggregate view (distributions, not per-prompt grids).
- This is the prerequisite for taking any v5 finding seriously, and the natural companion to
  A2's regression suite (test on distributions, not single prompts).

## C. Trust — error bars on the microscope

### C1. Faithfulness dashboard
The project's soul is the measured-faithfulness invariant. Surface it: a diagnostics page
showing, per loaded model, the actual lens-vs-forward reconstruction error, attribution-vs-exact
patching rank correlation (v5-B1's gate), and SAE reconstruction loss (v4-B1). Green/amber/red.
When a user sees a surprising result, the first question is "can I trust the instrument?" —
answer it in-product.
- Mostly re-running existing tests as endpoints and caching the numbers.

---

## Skipped

- **Plugin marketplace / packaging ecosystem** — v5-E2's single-module interface is still
  waiting for its second implementer.
- **Distributed serving, job queues, k8s** — one GPU box and a queue-of-one covers the actual
  usage until proven otherwise.
- **Copilot autonomy** — v5-D1 deferred this debate to v6; the answer is no. The latch stays
  (annotation yes, autonomous experimentation no) — an instrument release is exactly the wrong
  place to make results harder to attribute to a human.
- **New interp methods** — v6 deliberately adds none. Consolidation release.

## Suggested order

A1 → A2 (the reproducibility spine — everything else hangs off it) → C1 → B1 → B2.
