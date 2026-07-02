"""Guardrails for the attention-pattern extraction.

These pin down what the numbers *mean* so a refactor can't silently ship a pattern that isn't a
real attention distribution: each query row must be a probability distribution over the keys it is
allowed to see, and a causal model must never attend to future tokens.
"""

from __future__ import annotations

import os

import pytest
import torch

from app.core.models import load_model
from app.services.attention import extract_attention
from tests.test_faithfulness import PROMPTS

# Every test here boots a real model.
pytestmark = pytest.mark.slow


@pytest.fixture(scope="session")
def gpt2():
    return load_model("gpt2", device="cpu")


@pytest.mark.parametrize("prompt", PROMPTS)
def test_attention_shape_and_distribution(gpt2, prompt):
    model = gpt2
    tokens = model.to_tokens(prompt)
    patterns = extract_attention(model, tokens)

    n_tokens = tokens.shape[1]
    assert patterns.shape == (model.cfg.n_layers, model.cfg.n_heads, n_tokens, n_tokens)

    # Every query row is a softmax over keys -> sums to 1.
    row_sums = patterns.sum(dim=-1)
    assert torch.allclose(row_sums, torch.ones_like(row_sums), atol=1e-4), (
        f"attention rows are not distributions for {prompt!r}"
    )


@pytest.mark.parametrize("prompt", PROMPTS)
def test_attention_is_causal(gpt2, prompt):
    model = gpt2
    tokens = model.to_tokens(prompt)
    patterns = extract_attention(model, tokens)

    # A query at position q may not attend to keys at position > q.
    n_tokens = patterns.shape[-1]
    upper = torch.triu(torch.ones(n_tokens, n_tokens), diagonal=1).bool()
    assert patterns[..., upper].abs().max() == 0.0, (
        f"attention leaks to future tokens for {prompt!r}"
    )


# --- Gemma gate: the one genuinely-new arch shape. GQA expands to full query-head count, and the
# sliding-window (local) layers stay causal. Non-blocking — runs only with HF_TOKEN. One prompt:
# the gpt2 cases above already cover the distribution/causality matrix.
# ponytail: single prompt; this exists to catch arch breakage, not re-test the invariants.
@pytest.mark.skipif(not os.environ.get("HF_TOKEN"), reason="Gemma gate: needs HF_TOKEN")
def test_gemma_attention_shape_is_full_and_causal():
    model = load_model("gemma-3-1b", device="cpu")
    tokens = model.to_tokens("The capital of France is")
    patterns = extract_attention(model, tokens)

    n_tokens = tokens.shape[1]
    # GQA must be expanded to one row per *query* head, not per KV head.
    assert patterns.shape == (model.cfg.n_layers, model.cfg.n_heads, n_tokens, n_tokens)
    upper = torch.triu(torch.ones(n_tokens, n_tokens), diagonal=1).bool()
    assert patterns[..., upper].abs().max() == 0.0
