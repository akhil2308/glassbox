# glassbox
See what a language model is thinking, layer by layer. Logit lens · attention patterns · layer ablation — built on TransformerLens + FastAPI + React.

## Status

- **Weekend 1 — logit lens (done).** Load GPT-2-small, decode the residual stream at every
  layer, and watch the next-token prediction get assembled as data flows upward.

## Quickstart

```bash
uv sync                                   # create venv, install deps (torch, transformer_lens)
uv run python scripts/run_logit_lens.py   # run the logit lens on 5 prompts -> outputs/logit_lens.png
```

First run downloads GPT-2-small (~500 MB). Runs on CPU by default — the MPS backend on
Apple Silicon (PyTorch 2.7.x) can return silently-incorrect numbers, and GPT-2-small is
small enough that CPU is plenty fast.

## Layout

- `glassbox/` — reusable core. `logit_lens.py` is the heart: run a prompt, get the
  per-layer top prediction. This is what the FastAPI backend will call in Weekend 2.
- `scripts/` — runnable entry points (CPU, prints + matplotlib).
- `outputs/` — generated figures.
