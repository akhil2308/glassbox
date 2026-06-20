# GLASSBOX — Simulate button (autoregressive generation animation)

> **STATUS: DONE** — 2026-06-20, branch `feat/v2`, not committed.

## What changed

The logit lens previously did one forward pass: given a prompt, predict the next token and stop
(`result.final_top_token`). This adds a **Simulate** button that repeats that call — feeding the
predicted token back into the prompt — to animate normal autoregressive generation, with the full
lens grid re-rendering per generated token.

No backend change. `/logit-lens` already returns the next token; "generation" is just a frontend
loop over the existing endpoint, one call per token. Each response is one animation frame.

## Files touched

- `frontend/src/api.ts` — added `runSimulate(prompt, model, maxTokens, onFrame, cancelled)`: loops
  `runLogitLens`, appends `final_top_token` to the prompt each step, stops at `maxTokens` or when
  `cancelled()` returns true.
- `frontend/src/App.tsx` — added `simulating`, `maxTokens`, `genFrom` state and a `cancelRef`;
  `onSimulate` runs the loop into the existing `lens` result slot, `onStop` flips the cancel ref.
- `frontend/src/components/PromptBar.tsx` — added a max-tokens number input and a Simulate/Stop
  toggle button next to Run.
- `frontend/src/components/LogitLensSection.tsx` — forces grid mode (not stream) while simulating,
  hides the mode toggle during a run, forwards `genFrom`.
- `frontend/src/components/LogitLensGrid.tsx` — accepts `genFrom`; columns at or past that position
  (the generated tokens) get an accent highlight.

## Known limitations (deliberate, not bugs)

- **Greedy only** — no sampling/temperature/top-k. `final_top_token` is already argmax.
- **No KV-cache reuse** — each step re-runs the full forward pass over the growing prompt via
  `/logit-lens`. Fine for short demo generations on CPU; a streaming backend `/generate` endpoint
  with cache reuse is the upgrade path if this is ever too slow.
- **No EOS auto-stop** — the max-tokens cap and the Stop button are the only stop conditions.
- **String-level re-tokenization** — each step appends `final_top_token` as a string and
  re-tokenizes the whole prompt, rather than appending a token id. Round-trips cleanly for GPT-2
  BPE in practice; flagged with a `ponytail:` comment in `api.ts` if it ever needs to change.

## Bug fixed during review

`PromptBar.tsx`'s Enter-key handler on the prompt input only checked `!busy`, not `!simulating` —
so pressing Enter mid-simulation could fire a second concurrent `runLogitLens` call racing against
the in-flight `runSimulate` loop's `setLens` calls. Fixed by checking the same `disabled` flag the
Run button already uses.

## Verification

`cd frontend && npm run build` (`tsc -b` + `vite build`) passes clean. Manual run: `make web`,
Logit Lens tab, type a prompt, set max tokens, click Simulate — grid grows one highlighted column
per token; Stop halts within one frame. No frontend test framework exists (CI is eslint + vite
build only), so this is verified by the manual run rather than an automated test.
