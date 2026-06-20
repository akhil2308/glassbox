"""Serialization schemas — the JSON shape the API and frontend speak.

Pydantic v2 models so FastAPI serializes them for free and so a round-trip
(`LogitLensResult(**result.model_dump())`) is guaranteed clean (floats, not tensors).

`build_result` in logit_lens.py is the single place tensors become these objects — the one
boundary between numerics and JSON.
"""

from __future__ import annotations

from pydantic import BaseModel


class LayerPrediction(BaseModel):
    """The logit-lens readout at one layer, for one token position."""

    layer: int  # 0 == embeddings only; n_layers == full model output
    label: str  # human label, e.g. "0_pre", "final_post"
    top_token: str  # the model's top guess decoded at this layer
    top_prob: float  # softmax probability of that guess
    # Probability mass that *this position's own final-layer top token* has at this layer —
    # lets us watch the answer sharpen as data flows up. Cleanest story on the last position.
    answer_prob: float


class TokenJourney(BaseModel):
    """One prompt token's trip up the residual stream: its readout at every layer."""

    position: int
    str_token: str
    is_bos: bool  # position 0's BOS column has a meaningless "journey" — UI hides/labels it
    layers: list[LayerPrediction]


class LogitLensResult(BaseModel):
    prompt: str
    model_name: str
    str_tokens: list[str]  # the prompt split into tokens, as the model sees them
    final_top_token: str  # the model's actual top prediction at the last position
    tokens: list[TokenJourney]  # one journey per prompt token


class AttentionResult(BaseModel):
    """Every layer's per-head attention pattern: who (query) looks at whom (key)."""

    model_name: str
    prompt: str
    str_tokens: list[str]
    is_bos: list[bool]  # position 0's BOS row/col is meaningless — UI dims it
    n_layers: int
    n_heads: int
    # patterns[layer][head][query][key] — each query row softmaxes to 1 over keys; the model is
    # causal, so key > query entries are 0. Weights are rounded to keep the JSON payload small.
    patterns: list[list[list[list[float]]]]


class AblationEffect(BaseModel):
    """What ablating one layer's contribution does to the last position's prediction."""

    layer: int
    ablated_top_token: str  # the model's new top guess once this layer is neutralized
    ablated_top_prob: float
    # Probability the *baseline* answer still gets once this layer is ablated. Low = the layer was
    # load-bearing for that answer.
    answer_prob: float
    answer_kept: bool  # did the top-1 prediction survive ablating this layer?


class AblationResult(BaseModel):
    model_name: str
    prompt: str
    str_tokens: list[str]
    component: str  # "block" | "attn" | "mlp" — what got zeroed per layer
    baseline_top_token: str  # the model's real top prediction, nothing ablated
    baseline_top_prob: float
    effects: list[AblationEffect]  # one per layer
