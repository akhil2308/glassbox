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

    layer: int            # 0 == embeddings only; n_layers == full model output
    label: str            # human label, e.g. "0_pre", "final_post"
    top_token: str        # the model's top guess decoded at this layer
    top_prob: float       # softmax probability of that guess
    # Probability mass that *this position's own final-layer top token* has at this layer —
    # lets us watch the answer sharpen as data flows up. Cleanest story on the last position.
    answer_prob: float


class TokenJourney(BaseModel):
    """One prompt token's trip up the residual stream: its readout at every layer."""

    position: int
    str_token: str
    is_bos: bool          # position 0's BOS column has a meaningless "journey" — UI hides/labels it
    layers: list[LayerPrediction]


class LogitLensResult(BaseModel):
    prompt: str
    model_name: str
    str_tokens: list[str]          # the prompt split into tokens, as the model sees them
    final_top_token: str           # the model's actual top prediction at the last position
    tokens: list[TokenJourney]     # one journey per prompt token
