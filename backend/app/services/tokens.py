"""Tiny token helpers shared across views.

One job: turn a model + token tensor into per-position metadata the UI needs. Kept apart so the
logit-lens and attention builders don't each re-derive the same BOS check.
"""

from __future__ import annotations


def bos_flags(model, tokens) -> list[bool]:
    """Per-position `is_bos` flags for the (single-batch) `tokens`.

    Only position 0 can be the beginning-of-sequence marker; its "journey"/attention row is
    meaningless, so the UI dims or labels it. Models differ in whether they `prepend_bos`, so we
    compare the actual token id against the tokenizer's `bos_token_id` rather than assuming.
    """
    bos_id = getattr(model.tokenizer, "bos_token_id", None)
    token_ids = tokens[0].tolist()
    return [pos == 0 and bos_id is not None and tid == bos_id for pos, tid in enumerate(token_ids)]
