"""GLASSBOX — see what a language model is thinking, layer by layer."""

from glassbox.logit_lens import (
    LayerPrediction,
    LogitLensResult,
    build_result,
    decode_stack,
    extract_resid_stack,
    logit_lens,
)
from glassbox.models import REGISTRY, load_model, pick_device

__all__ = [
    "LayerPrediction",
    "LogitLensResult",
    "REGISTRY",
    "build_result",
    "decode_stack",
    "extract_resid_stack",
    "load_model",
    "logit_lens",
    "pick_device",
]
