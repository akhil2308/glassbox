"""The logit lens: decode the residual stream into vocabulary predictions at every layer.

The core trick of this whole project. A transformer carries a per-token vector up
through its layers (the *residual stream*). Normally we only decode that vector into a
next-token prediction at the very top. The logit lens decodes it at *every* layer, so we
can watch the prediction get assembled as data flows upward.

How it works, mechanically:
  1. Run the model and cache the residual stream after every layer.
  2. At each layer, take the vector for the position we care about (the last token).
  3. Apply the model's final layer-norm, then its unembedding matrix (W_U).
  4. That gives a logit distribution over the vocabulary — "what would the model predict
     if it had to commit right now?"
"""

from __future__ import annotations

from dataclasses import dataclass

import torch
from transformer_lens import HookedTransformer


def pick_device() -> str:
    """Pick a compute device.

    Note: TransformerLens warns that the MPS backend (Apple Silicon, PyTorch 2.7.x) can
    produce *silently incorrect* results, and we measured a real reconstruction error there.
    Correctness matters more than speed for interpretability, and GPT-2-small is tiny, so we
    default to CPU. Pass device="mps" explicitly to opt back in once it's trustworthy.
    """
    if torch.cuda.is_available():
        return "cuda"
    return "cpu"


def load_model(model_name: str = "gpt2", device: str | None = None) -> HookedTransformer:
    """Load an open model with its internals exposed. 'gpt2' is GPT-2-small (12 layers)."""
    device = device or pick_device()
    model = HookedTransformer.from_pretrained(model_name, device=device)
    model.eval()
    return model


@dataclass
class LayerPrediction:
    """The logit-lens readout at one layer, for the prediction position."""

    layer: int            # 0 == embeddings only; n_layers == full model output
    label: str            # human label, e.g. "embed", "blocks.5.hook_resid_post"
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


@torch.no_grad()
def logit_lens(
    model: HookedTransformer,
    prompt: str,
    position: int = -1,
) -> LogitLensResult:
    """Run `prompt` through `model` and decode the residual stream at every layer.

    `position` is which token's residual stream to decode (default: the last token, i.e.
    the one whose job is to predict what comes next).
    """
    tokens = model.to_tokens(prompt)
    str_tokens = model.to_str_tokens(prompt)

    logits, cache = model.run_with_cache(tokens)

    # accumulated_resid stacks the *raw* residual stream after each layer:
    # [n_layers+1, batch, pos, d_model]. We deliberately don't let it apply layer-norm —
    # instead we push each layer's vector through the model's own ln_final below, which is
    # the faithful logit lens (and reproduces the real output exactly at the top layer).
    accum_resid, labels = cache.accumulated_resid(
        layer=-1, incl_mid=False, return_labels=True
    )

    # Decode every layer's vector at our chosen position into vocabulary logits, using the
    # exact final layer-norm + unembedding (W_U, b_U) the model uses for its real output.
    resid_at_pos = accum_resid[:, 0, position, :]
    normed = model.ln_final(resid_at_pos)
    layer_logits = normed @ model.W_U + model.b_U
    layer_probs = layer_logits.softmax(dim=-1)

    # The model's actual final prediction — the thing the lower layers are converging toward.
    final_top_id = int(logits[0, position].argmax())
    final_top_token = model.to_single_str_token(final_top_id)

    layers: list[LayerPrediction] = []
    for i, label in enumerate(labels):
        probs = layer_probs[i]
        top_id = int(probs.argmax())
        layers.append(
            LayerPrediction(
                layer=i,
                label=label,
                top_token=model.to_single_str_token(top_id),
                top_prob=float(probs[top_id]),
                answer_prob=float(probs[final_top_id]),
            )
        )

    return LogitLensResult(
        prompt=prompt,
        str_tokens=str_tokens,
        final_top_token=final_top_token,
        layers=layers,
    )
