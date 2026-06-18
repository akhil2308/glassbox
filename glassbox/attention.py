"""Attention patterns: which tokens look at which, per layer and head.

The second view of the data flow (after the logit lens). Where the residual stream is the
"highway", attention is how tokens pass information *to each other* along it. For each layer and
head we pull the `[query, key]` matrix of attention weights — row `q` is how token `q` distributes
its attention over the keys it can see.

Same single-responsibility split as the logit lens:
  1. extract — run the model, stack every layer's attention pattern. One job: get the tensor.
  2. build   — shape that tensor into an `AttentionResult`. The one numerics → objects boundary.
"""

from __future__ import annotations

import torch

from glassbox.schemas import AttentionResult
from glassbox.tokens import bos_flags

# Decimal places kept when serializing weights. The payload is n_layers·n_heads·q·k floats; the
# lens only ranks/visualizes these, so 4 dp is plenty and roughly halves the JSON size.
_ROUND = 4


@torch.no_grad()
def extract_attention(model, tokens):
    """Run the model and stack every layer's attention pattern.

    Returns a tensor of shape `[n_layers, n_heads, query_pos, key_pos]`. Each query row is a
    softmax over keys; the model is causal, so the strictly-upper-triangular half is 0.
    """
    _logits, cache = model.run_with_cache(tokens)
    return torch.stack(
        [cache["pattern", layer][0] for layer in range(model.cfg.n_layers)]
    )


def build_attention_result(model, prompt, model_name, tokens, patterns) -> AttentionResult:
    """Shape the attention tensor into an `AttentionResult` (the numerics → objects boundary)."""
    n_layers, n_heads = patterns.shape[0], patterns.shape[1]
    rounded = [
        [[[round(w, _ROUND) for w in key_row] for key_row in head] for head in layer]
        for layer in patterns.tolist()
    ]
    return AttentionResult(
        model_name=model_name,
        prompt=prompt,
        str_tokens=model.to_str_tokens(prompt),
        is_bos=bos_flags(model, tokens),
        n_layers=n_layers,
        n_heads=n_heads,
        patterns=rounded,
    )


def attention(model, prompt: str, model_name: str | None = None) -> AttentionResult:
    """Convenience wrapper: extract → build."""
    model_name = model_name or getattr(model.cfg, "model_name", "model")
    tokens = model.to_tokens(prompt)
    patterns = extract_attention(model, tokens)
    return build_attention_result(model, prompt, model_name, tokens, patterns)
