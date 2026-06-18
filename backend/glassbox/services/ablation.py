"""Layer ablation: what breaks if we delete a layer?

The third view of the data flow. A transformer block *reads* the residual stream and *writes* back
to it (`resid_post = resid_pre + attn_out + mlp_out`). So we can "delete" a block by zeroing its
writes — making it a pass-through — and watch how the final prediction shifts. Sweeping one layer
at a time over the whole model shows which layers are load-bearing for a given prediction.

Same single-responsibility split:
  1. extract (`ablate_sweep`) — run the model once per layer with that layer neutralized.
  2. build  (`build_ablation_result`) — shape the measurements into an `AblationResult`.

`component` chooses what gets zeroed: the whole block (attn + mlp), just attention, or just the MLP.
"""

from __future__ import annotations

import torch

from glassbox.schemas.results import AblationEffect, AblationResult

# What each component zeroes. Zeroing a block's writes to the residual stream makes it an identity
# map for that token's vector — i.e. deletes the layer's contribution.
_COMPONENTS: dict[str, list[str]] = {
    "block": ["attn_out", "mlp_out"],
    "attn": ["attn_out"],
    "mlp": ["mlp_out"],
}


def ablation_hooks(layer: int, component: str):
    """Forward hooks that neutralize `layer`'s chosen writes to the residual stream."""

    def zero(tensor, hook):
        return torch.zeros_like(tensor)

    return [(f"blocks.{layer}.hook_{name}", zero) for name in _COMPONENTS[component]]


@torch.no_grad()
def ablate_sweep(model, tokens, component: str = "block"):
    """Ablate each layer in turn; measure the effect on the last position's next-token prediction.

    Returns `(baseline_logits, per_layer_logits)`:
      * `baseline_logits` — last-position logits with nothing ablated.
      * `per_layer_logits[l]` — last-position logits with layer `l`'s `component` zeroed.
    Pure of I/O; runs the model `n_layers + 1` times (sub-second each on CPU for these models).
    """
    if component not in _COMPONENTS:
        raise ValueError(f"Unknown component {component!r}. Known: {sorted(_COMPONENTS)}")

    baseline_logits = model(tokens)[0, -1]
    per_layer_logits = [
        model.run_with_hooks(tokens, fwd_hooks=ablation_hooks(layer, component))[0, -1]
        for layer in range(model.cfg.n_layers)
    ]
    return baseline_logits, per_layer_logits


def build_ablation_result(
    model, prompt, model_name, tokens, component, baseline_logits, per_layer_logits
) -> AblationResult:
    """Shape the sweep's logits into an `AblationResult` (the numerics → objects boundary)."""
    answer_id = int(baseline_logits.argmax())
    baseline_probs = baseline_logits.softmax(dim=-1)

    effects: list[AblationEffect] = []
    for layer, logits in enumerate(per_layer_logits):
        probs = logits.softmax(dim=-1)
        top_id = int(logits.argmax())
        effects.append(
            AblationEffect(
                layer=layer,
                ablated_top_token=model.to_single_str_token(top_id),
                ablated_top_prob=float(probs[top_id]),
                answer_prob=float(probs[answer_id]),
                answer_kept=(top_id == answer_id),
            )
        )

    return AblationResult(
        model_name=model_name,
        prompt=prompt,
        str_tokens=model.to_str_tokens(prompt),
        component=component,
        baseline_top_token=model.to_single_str_token(answer_id),
        baseline_top_prob=float(baseline_probs[answer_id]),
        effects=effects,
    )


def ablation(
    model, prompt: str, component: str = "block", model_name: str | None = None
) -> AblationResult:
    """Convenience wrapper: extract → build."""
    model_name = model_name or getattr(model.cfg, "model_name", "model")
    tokens = model.to_tokens(prompt)
    baseline_logits, per_layer_logits = ablate_sweep(model, tokens, component)
    return build_ablation_result(
        model, prompt, model_name, tokens, component, baseline_logits, per_layer_logits
    )
