"""Model loading — a thin loader over TransformerLens's `TransformerBridge`.

Two things live here, kept deliberately apart (single responsibility):
  * `REGISTRY` — plain config *data*. Which checkpoints we know about, no behavior.
  * `load_model` — the *behavior* that boots one of them.

We use `TransformerBridge` (TransformerLens 3.x) rather than the deprecated
`HookedTransformer.from_pretrained`, because the Bridge gives a uniform, architecture
-general surface: its `ln_final` is a `NormalizationBridge` that adapts to LayerNorm or
RMSNorm models alike, which is what lets the logit-lens decode work on any model without
special-casing. See plans/GLASSBOX-weekend2-plan.md for the faithfulness spike that pinned
this down.
"""

from __future__ import annotations

import os

import torch
from transformer_lens.model_bridge import TransformerBridge

# --- Registry: data only, no behavior. -------------------------------------------------
# Use base / -pt checkpoints (not instruction-tuned -it): the lens is about raw next-token
# mechanics. Confirm exact HF ids when adding a model.
REGISTRY: dict[str, dict] = {
    "gpt2": {
        "hf_name": "gpt2",
        "gated": False,
        "display": "GPT-2 small",
    },
    "gemma-3-1b": {
        "hf_name": "google/gemma-3-1b-pt",
        "gated": True,
        "display": "Gemma 3 1B",
    },
}


def pick_device() -> str:
    """Pick a compute device.

    We default to CPU on Apple Silicon on purpose: TransformerLens warns that the MPS
    backend can produce *silently incorrect* results, and we measured a real residual-stream
    reconstruction error there. Correctness beats speed for interpretability, and the models
    here are small. Pass `device="mps"` explicitly to opt back in once it's trustworthy.
    """
    if torch.cuda.is_available():
        return "cuda"
    return "cpu"


def load_model(model_name: str, device: str | None = None) -> TransformerBridge:
    """Boot a registered model via `TransformerBridge`.

    `enable_compatibility_mode()` is required — it folds LayerNorm and centers weights so the
    downstream decode (`model.ln_final(...) @ model.W_U + b_U`) behaves uniformly across
    architectures and reproduces the real forward logits.
    """
    if model_name not in REGISTRY:
        raise KeyError(
            f"Unknown model {model_name!r}. Known models: {sorted(REGISTRY)}"
        )

    entry = REGISTRY[model_name]
    if entry["gated"] and not os.environ.get("HF_TOKEN"):
        # Fail loudly — never silently fall back to a different model.
        raise PermissionError(
            f"Model {model_name!r} ({entry['hf_name']}) is gated. Accept its license on "
            f"HuggingFace and export HF_TOKEN before loading it."
        )

    device = device or pick_device()
    bridge = TransformerBridge.boot_transformers(entry["hf_name"], device=device)
    bridge.enable_compatibility_mode()
    bridge.eval()
    return bridge
