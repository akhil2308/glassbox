"""The logit lens: decode the residual stream into vocabulary predictions at every layer.

The core trick of this whole project. A transformer carries a per-token vector up through its
layers (the *residual stream*). Normally we only decode that vector into a next-token
prediction at the very top. The logit lens decodes it at *every* layer, so we can watch the
prediction get assembled as data flows upward.

Mechanically, split into three single-responsibility steps along the data flow:
  1. extract — run the model, pull the per-layer residual snapshots (the "stack").
  2. decode  — pure: normalize each layer's vector and unembed it into vocab probabilities.
  3. build   — shape the resulting tensors into result objects.

Why `model.ln_final` and not `cache.apply_ln_to_stack`: on `TransformerBridge`, `ln_final`
is a `NormalizationBridge` that reproduces the real forward logits to ~1e-5 *and* adapts to
the architecture (LayerNorm or RMSNorm). `apply_ln_to_stack` was measured to be unfaithful
(off by ~15 logits on GPT-2) — see plans/GLASSBOX-weekend2-plan.md. The faithfulness test in
tests/ is the guardrail that protects this invariant.
"""

from __future__ import annotations

import torch

from app.schemas.results import LayerPrediction, LogitLensResult, TokenJourney
from app.services.tokens import bos_flags


@torch.no_grad()
def extract_resid_stack(model, tokens):
    """Run the model and pull the accumulated residual stream after every layer.

    One job: get the stack. Returns `(logits, cache, accum, labels)` where
    `accum` is `[n_layers+1, batch, pos, d_model]` — the raw (un-normalized) residual
    snapshot after embeddings (index 0) through the final layer (index -1).
    """
    logits, cache = model.run_with_cache(tokens)
    accum, labels = cache.accumulated_resid(layer=-1, incl_mid=False, return_labels=True)
    return logits, cache, accum, labels


def decode_stack(model, accum, positions=None):
    """Pure decode: normalize → unembed → softmax. No model run, no I/O.

    `accum` is the `[n_layers+1, batch, pos, d_model]` stack from `extract_resid_stack`.
    `positions` selects which token positions to decode:
      * None         → decode *all* positions (so the UI can show any token's journey).
      * int / list   → decode just those.

    Returns probabilities of shape `[n_layers+1, n_pos, d_vocab]`.
    """
    stack = accum[:, 0, :, :]  # drop batch -> [n_layers+1, pos, d_model]
    if positions is not None:
        idx = [positions] if isinstance(positions, int) else positions
        stack = stack[:, idx, :]

    # ln_final is the model's own final normalization (a NormalizationBridge): faithful to the
    # real forward output and architecture-general. Applied per (layer, position).
    normed = model.ln_final(stack)

    # Some architectures have no unembed bias; default to 0 rather than assuming GPT-2's b_U.
    b_U = getattr(model, "b_U", 0)
    layer_logits = normed @ model.W_U + b_U

    # NOTE: Gemma soft-caps its final logits. We deliberately do *not* apply the cap here: it's
    # monotonic, so it never changes token ranking for the lens. Don't "fix" this later.
    return layer_logits.softmax(dim=-1)


def build_result(model, prompt, model_name, tokens, logits, accum, labels) -> LogitLensResult:
    """Shape decoded tensors into a `LogitLensResult`. The single numerics → objects boundary.

    Decodes *every* position so the UI can show any token's journey up the residual stream.
    `answer_prob` for a position tracks that position's own final-layer top token (the cleanest
    "answer materializes upward" reading is on the last position).
    """
    str_tokens = model.to_str_tokens(prompt)
    probs = decode_stack(model, accum, positions=None)  # [n_layers+1, n_pos, d_vocab]
    n_pos = probs.shape[1]

    final_top_id = int(logits[0, -1].argmax())
    final_top_token = model.to_single_str_token(final_top_id)

    is_bos = bos_flags(model, tokens)

    journeys: list[TokenJourney] = []
    for pos in range(n_pos):
        # Each position's own final-layer prediction is what its journey converges toward.
        pos_final_top_id = int(logits[0, pos].argmax())
        layers: list[LayerPrediction] = []
        for i, label in enumerate(labels):
            layer_probs = probs[i, pos]
            top_id = int(layer_probs.argmax())
            layers.append(
                LayerPrediction(
                    layer=i,
                    label=label,
                    top_token=model.to_single_str_token(top_id),
                    top_prob=float(layer_probs[top_id]),
                    answer_prob=float(layer_probs[pos_final_top_id]),
                )
            )
        journeys.append(
            TokenJourney(
                position=pos,
                str_token=str_tokens[pos],
                is_bos=is_bos[pos],
                layers=layers,
            )
        )

    return LogitLensResult(
        prompt=prompt,
        model_name=model_name,
        str_tokens=str_tokens,
        final_top_token=final_top_token,
        tokens=journeys,
    )


def logit_lens(model, prompt: str, model_name: str | None = None) -> LogitLensResult:
    """Convenience wrapper: extract → decode → build, for all positions."""
    model_name = model_name or getattr(model.cfg, "model_name", "model")
    tokens = model.to_tokens(prompt)
    logits, _cache, accum, labels = extract_resid_stack(model, tokens)
    return build_result(model, prompt, model_name, tokens, logits, accum, labels)
