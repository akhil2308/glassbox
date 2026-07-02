"""Model lifecycle — one responsibility: owning the resident-model cache.

Booting a bridge is multi-second; inference is sub-second. So we load once and keep models
resident, with a small cap and LRU eviction to bound memory (Gemma-3-1B fp32 on CPU is a few
GB). Endpoint handlers never touch the cache dict — they call `manager.get(name)`.
"""

from __future__ import annotations

import threading
from collections import OrderedDict

from app.core.models import REGISTRY, load_model, pick_device


class ModelManager:
    """Lazy-loading, LRU-evicting cache of `TransformerBridge`s."""

    def __init__(self, device: str | None = None, max_resident: int = 2):
        self.device = device or pick_device()
        self.max_resident = max_resident
        self._models: OrderedDict[str, object] = OrderedDict()
        self._lock = threading.Lock()

    def get(self, name: str):
        """Return the bridge for `name`, loading it on first use. LRU-evicts past the cap.

        Raises KeyError (unknown model) / PermissionError (gated, no token) from `load_model`.
        """
        if name not in REGISTRY:
            raise KeyError(name)

        with self._lock:
            if name in self._models:
                self._models.move_to_end(name)  # mark most-recently-used
                return self._models[name]

        # Load outside the lock — booting is slow and we don't want to block other models.
        model = load_model(name, device=self.device)

        with self._lock:
            self._models[name] = model
            self._models.move_to_end(name)
            while len(self._models) > self.max_resident:
                self._models.popitem(last=False)  # evict least-recently-used
            return self._models[name]

    def warm(self, name: str) -> None:
        """Preload a model (e.g. at startup) so the first request isn't slow."""
        self.get(name)

    def loaded_names(self) -> list[str]:
        with self._lock:
            return list(self._models.keys())

    def peek(self, name: str):
        """Return the resident bridge for `name`, or None — never loads. For cheap metadata."""
        with self._lock:
            return self._models.get(name)
