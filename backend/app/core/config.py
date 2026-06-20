"""Central configuration — one place for every tunable.

Replaces the module-level constants and ad-hoc `os.environ` reads that used to be scattered
across the API layer. Everything is overridable via `GLASSBOX_`-prefixed env vars (or a `.env`
file), so deployments don't have to edit code.

`HF_TOKEN` deliberately lives *outside* this object: it's a HuggingFace convention read directly
in `core.models.load_model`, not a GLASSBOX setting.
"""

from __future__ import annotations

from functools import cached_property

from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    model_config = SettingsConfigDict(env_prefix="GLASSBOX_", env_file=".env", extra="ignore")

    # Reject prompts longer than this many tokens (cheap guard against pathological payloads).
    max_prompt_tokens: int = 128
    # The model warmed at startup and used when a request omits `model`.
    default_model: str = "gpt2"
    # What `/ablation` will zero per layer. Anything else is a 422.
    ablation_components: tuple[str, ...] = ("block", "attn", "mlp")

    # Comma-separated browser origins allowed by CORS. Default is the Vite dev server.
    # Stored as a raw string (env parsing of list types expects JSON); split via `cors_origin_list`.
    cors_origins: str = "http://localhost:5173"

    # Force a compute device ("cpu" | "cuda" | "mps"). None lets `pick_device` choose — which
    # defaults to CPU on Apple Silicon on purpose (MPS can be silently incorrect for the lens).
    device: str | None = None

    @cached_property
    def cors_origin_list(self) -> list[str]:
        return [origin.strip() for origin in self.cors_origins.split(",") if origin.strip()]


settings = Settings()
