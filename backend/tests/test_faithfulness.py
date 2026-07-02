"""The correctness guardrail for the whole "any model" promise.

The logit lens is only meaningful if its top-layer decode reproduces the model's *real*
forward-pass output. If that fails, every lower-layer readout is suspect. This test pins the
invariant down so a future refactor (or a tempting "simplification" back to
`apply_ln_to_stack`) can't silently break it.

It lives in tests/, not as an inline `assert` in `decode_stack`: an inline assert would run on
every production request and vanish under `python -O`.

Two checks:
  * argmax-equality — the portable one. Correct even for soft-capped models (Gemma), whose cap
    is monotonic and so never changes the top token.
  * allclose on the top-layer logits — the stronger one. argmax can agree by luck on easy
    prompts even when the decode is subtly wrong, so we also require the reconstructed top-layer
    logits to match the real forward logits. Applied only to non-soft-capped models.
"""

from __future__ import annotations

import os

import pytest
import torch

from app.core.models import REGISTRY, load_model
from app.services.arch import describe_arch
from app.services.arch import soft_capped as model_is_soft_capped
from app.services.logit_lens import decode_stack, extract_resid_stack

PROMPTS = [
    "The capital of France is",
    "2 plus 2 equals",
    "Water is made of hydrogen and",
    "The opposite of hot is",
    "My name is",
]

# model_is_soft_capped lives in app.services.arch now (shared with the arch badge). It gates the
# stronger `allclose` check below to non-soft-capped models. NB: it must treat the -1.0 cfg value
# as "disabled" (cap > 0) — otherwise GPT-2 reads as soft-capped and this check silently no-ops.


@pytest.fixture(scope="session")
def gpt2():
    return load_model("gpt2", device="cpu")


@pytest.mark.slow
@pytest.mark.parametrize("prompt", PROMPTS)
def test_gpt2_top_layer_matches_forward(gpt2, prompt):
    model = gpt2
    tokens = model.to_tokens(prompt)
    logits, _cache, accum, _labels = extract_resid_stack(model, tokens)

    # Decode the last position; top layer (index -1) must reproduce the real forward output.
    probs = decode_stack(model, accum, positions=-1)  # [n_layers+1, 1, d_vocab]
    lens_top = int(probs[-1, 0].argmax())
    final_top = int(logits[0, -1].argmax())
    assert lens_top == final_top, f"decode unfaithful (argmax) for {prompt!r}"

    # Stronger check: reconstruct the top-layer *logits* and compare directly. GPT-2 is not
    # soft-capped, so the reconstruction should match the real logits closely.
    if not model_is_soft_capped(model):
        normed = model.ln_final(accum[:, 0, -1, :])
        b_U = getattr(model, "b_U", 0)
        top_layer_logits = (normed @ model.W_U + b_U)[-1]
        assert torch.allclose(top_layer_logits, logits[0, -1], atol=1e-3), (
            f"decode unfaithful (allclose) for {prompt!r}"
        )


# --- Gemma gate: non-blocking. Runs only if HF_TOKEN is present and the license is accepted.
gemma_reason = "Gemma gate deferred: needs HF_TOKEN + accepted license (non-blocking per plan)"


@pytest.mark.slow
@pytest.mark.skipif(not os.environ.get("HF_TOKEN"), reason=gemma_reason)
@pytest.mark.parametrize("prompt", PROMPTS)
def test_gemma_top_layer_matches_forward(prompt):
    model = load_model("gemma-3-1b", device="cpu")
    tokens = model.to_tokens(prompt)
    logits, _cache, accum, _labels = extract_resid_stack(model, tokens)

    probs = decode_stack(model, accum, positions=-1)
    lens_top = int(probs[-1, 0].argmax())
    final_top = int(logits[0, -1].argmax())
    # Argmax is the portable check — correct even though Gemma soft-caps its logits.
    assert lens_top == final_top, f"decode unfaithful (argmax) for {prompt!r}"


def test_registry_is_data_only():
    # The registry must stay plain config data (no callables) — loader behavior lives elsewhere.
    for _name, entry in REGISTRY.items():
        assert {"hf_name", "gated", "display"} <= entry.keys()
        assert not any(callable(v) for v in entry.values())


# --- Arch badge drift guard: describe_arch must reflect the real cfg, not stale hand-written data.
@pytest.mark.slow
def test_gpt2_arch_matches_cfg(gpt2):
    arch = describe_arch(gpt2)
    assert arch.norm == "LayerNorm"
    assert arch.attention == "MHA"  # GPT-2 is plain multi-head, no GQA
    assert arch.vocab == 50257
    assert arch.soft_cap is False  # the -1.0 cfg sentinel means "disabled", not "capped"


@pytest.mark.slow
@pytest.mark.skipif(not os.environ.get("HF_TOKEN"), reason=gemma_reason)
def test_gemma_arch_matches_cfg():
    arch = describe_arch(load_model("gemma-3-1b", device="cpu"))
    assert arch.norm == "RMSNorm"
    assert arch.attention.startswith("GQA")  # Gemma 3 uses grouped-query attention
    assert arch.vocab == 262144
