"""Model registry mapping short CLI keys to embedding classes and default checkpoints."""

from .base_embedding import BaseEmbedding
from .clip_embedding import CLIPEmbedding

MODELS: dict[str, tuple[type[BaseEmbedding], str]] = {
    "clip": (CLIPEmbedding, "openai/clip-vit-base-patch32"),
}


def resolve_model(model_key: str, checkpoint: str | None = None) -> tuple[type[BaseEmbedding], str]:
    """Return (embedder_class, checkpoint_name) for a registered model key."""
    if model_key not in MODELS:
        valid = ", ".join(sorted(MODELS))
        raise ValueError(f"Unknown model {model_key!r}; valid models: {valid}")
    embedder_cls, default_checkpoint = MODELS[model_key]
    return embedder_cls, checkpoint or default_checkpoint
