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

from dataclasses import dataclass

import torch


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


@dataclass
class LayerPrediction:
    """The logit-lens readout at one layer, for the prediction position."""

    layer: int            # 0 == embeddings only; n_layers == full model output
    label: str            # human label, e.g. "0_pre", "final_post"
    top_token: str        # the model's top guess decoded here
    top_prob: float       # softmax probability of that guess
    # probability mass the *final* answer already has at this layer — lets us watch it sharpen
    answer_prob: float


@dataclass
class LogitLensResult:
    prompt: str
    str_tokens: list[str]          # the prompt split into tokens, as the model sees them
    final_top_token: str           # the model's actual top prediction (full forward pass)
    layers: list[LayerPrediction]  # one entry per layer (+ embeddings)


def build_result(model, prompt, logits, accum, labels, position) -> LogitLensResult:
    """Shape decoded tensors into a result object. The single numerics → objects boundary.

    (Phase 1 replaces this with pydantic schemas and an all-positions result; for now it
    keeps the single-position dataclass shape the weekend-1 script depends on.)
    """
    str_tokens = model.to_str_tokens(prompt)
    probs = decode_stack(model, accum, positions=position)[:, 0, :]  # [n_layers+1, d_vocab]

    final_top_id = int(logits[0, position].argmax())
    final_top_token = model.to_single_str_token(final_top_id)

    layers: list[LayerPrediction] = []
    for i, label in enumerate(labels):
        layer_probs = probs[i]
        top_id = int(layer_probs.argmax())
        layers.append(
            LayerPrediction(
                layer=i,
                label=label,
                top_token=model.to_single_str_token(top_id),
                top_prob=float(layer_probs[top_id]),
                answer_prob=float(layer_probs[final_top_id]),
            )
        )

    return LogitLensResult(
        prompt=prompt,
        str_tokens=str_tokens,
        final_top_token=final_top_token,
        layers=layers,
    )


def logit_lens(model, prompt: str, position: int = -1) -> LogitLensResult:
    """Convenience wrapper: extract → decode → build for a single position (default: last)."""
    tokens = model.to_tokens(prompt)
    logits, _cache, accum, labels = extract_resid_stack(model, tokens)
    return build_result(model, prompt, logits, accum, labels, position)
