"""Guardrails for layer ablation.

Two invariants protect the "delete a layer" story:
  * the sweep's baseline is the model's *real* prediction (nothing secretly ablated), and
  * zeroing a block's writes truly neutralizes it — so ablating *every* block makes the final
    residual collapse back to the post-embedding residual. If that holds for all layers at once, a
    single-layer ablation is doing exactly what we claim.
"""

from __future__ import annotations

import os

import pytest
import torch

from app.core.models import load_model
from app.services.ablation import ablate_sweep, ablation_hooks, build_ablation_result
from tests.test_faithfulness import PROMPTS

# Every test here boots a real model.
pytestmark = pytest.mark.slow


@pytest.fixture(scope="session")
def gpt2():
    return load_model("gpt2", device="cpu")


@pytest.mark.parametrize("prompt", PROMPTS)
def test_baseline_is_honest(gpt2, prompt):
    model = gpt2
    tokens = model.to_tokens(prompt)
    baseline_logits, per_layer_logits = ablate_sweep(model, tokens, component="block")
    result = build_ablation_result(
        model, prompt, "gpt2", tokens, "block", baseline_logits, per_layer_logits
    )

    # The sweep's baseline must match a plain forward pass — nothing ablated.
    real_top = model.to_single_str_token(int(model(tokens)[0, -1].argmax()))
    assert result.baseline_top_token == real_top
    assert len(result.effects) == model.cfg.n_layers


def test_block_ablation_is_a_passthrough(gpt2):
    """Ablating every block at once collapses the final residual to the embeddings residual.

    `resid_post = resid_pre + attn_out + mlp_out`; zeroing both outputs makes each block identity,
    so the vector that reaches the unembed is exactly what came out of the embedding layer.
    """
    model = gpt2
    tokens = model.to_tokens("The capital of France is")

    _l, baseline_cache = model.run_with_cache(tokens)
    embeddings_resid = baseline_cache["resid_pre", 0]

    hooks = []
    for layer in range(model.cfg.n_layers):
        hooks += ablation_hooks(layer, "block")

    with model.hooks(fwd_hooks=hooks):
        _l2, ablated_cache = model.run_with_cache(tokens)
    final_resid = ablated_cache["resid_post", model.cfg.n_layers - 1]

    assert torch.allclose(final_resid, embeddings_resid, atol=1e-4)


# --- Gemma gate: the ablation sweep is architecture-general; confirm it holds on Gemma's deeper
# RMSNorm/GQA stack. Non-blocking — runs only with HF_TOKEN.
# ponytail: one prompt, baseline-honesty check; gpt2 above covers the passthrough invariant.
@pytest.mark.skipif(not os.environ.get("HF_TOKEN"), reason="Gemma gate: needs HF_TOKEN")
def test_gemma_baseline_is_honest():
    prompt = "The capital of France is"
    model = load_model("gemma-3-1b", device="cpu")
    tokens = model.to_tokens(prompt)
    baseline_logits, per_layer_logits = ablate_sweep(model, tokens, component="block")
    result = build_ablation_result(
        model, prompt, "gemma-3-1b", tokens, "block", baseline_logits, per_layer_logits
    )

    real_top = model.to_single_str_token(int(model(tokens)[0, -1].argmax()))
    assert result.baseline_top_token == real_top
    assert len(result.effects) == model.cfg.n_layers
