"""GLASSBOX — see what a language model is thinking, layer by layer."""

from glassbox.core.models import REGISTRY, load_model, pick_device
from glassbox.schemas.results import (
    AblationEffect,
    AblationResult,
    AttentionResult,
    LayerPrediction,
    LogitLensResult,
    TokenJourney,
)
from glassbox.services.ablation import ablate_sweep, ablation, build_ablation_result
from glassbox.services.attention import attention, build_attention_result, extract_attention
from glassbox.services.logit_lens import (
    build_result,
    decode_stack,
    extract_resid_stack,
    logit_lens,
)

__all__ = [
    "AblationEffect",
    "AblationResult",
    "AttentionResult",
    "LayerPrediction",
    "LogitLensResult",
    "REGISTRY",
    "TokenJourney",
    "ablate_sweep",
    "ablation",
    "attention",
    "build_ablation_result",
    "build_attention_result",
    "build_result",
    "decode_stack",
    "extract_attention",
    "extract_resid_stack",
    "load_model",
    "logit_lens",
    "pick_device",
]
