"""Describe a model's architecture from its `cfg` — the data behind the UI's arch badge.

Read at runtime from the loaded bridge rather than hand-written per model: no guessing, and it
self-corrects across checkpoints. The one subtlety is the soft-cap sentinel — see `soft_capped`.
"""

from __future__ import annotations

from app.schemas.requests import ModelArch

# cfg attrs that hold a final/attn logit soft-cap threshold, across TransformerLens versions.
_SOFT_CAP_ATTRS = ("output_logits_soft_cap", "final_logit_softcap", "attn_scores_soft_cap")


def soft_capped(model) -> bool:
    """True if the model actually soft-caps its logits.

    The cap is a positive threshold; TransformerLens uses -1.0 (and None) as the *disabled*
    sentinel. So `if val:` is wrong — -1.0 is truthy. Gate on `val > 0`. This is the predicate
    the faithfulness test relies on to decide whether to run its strong `allclose` check.
    """
    cfg = model.cfg
    for attr in _SOFT_CAP_ATTRS:
        val = getattr(cfg, attr, None)
        if val and val > 0:
            return True
    return False


def describe_arch(model) -> ModelArch:
    """Pull the badge-worthy architecture facts off a loaded model's cfg."""
    cfg = model.cfg
    norm = {"LN": "LayerNorm", "LNPre": "LayerNorm", "RMS": "RMSNorm", "RMSPre": "RMSNorm"}.get(
        getattr(cfg, "normalization_type", None), str(getattr(cfg, "normalization_type", "?"))
    )
    n_kv = getattr(cfg, "n_key_value_heads", None)
    # n_key_value_heads is None for plain multi-head attention; a smaller count means GQA.
    attention = "MHA" if not n_kv or n_kv >= cfg.n_heads else f"GQA {cfg.n_heads}:{n_kv}"
    return ModelArch(
        norm=norm,
        attention=attention,
        vocab=int(cfg.d_vocab),
        soft_cap=soft_capped(model),
    )
