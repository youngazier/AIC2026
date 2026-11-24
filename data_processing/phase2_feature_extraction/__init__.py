"""Embedding models used by the feature-extraction phase."""

from .base_embedding import BaseEmbedding
from .clip_embedding import CLIPEmbedding
from .models import MODELS, resolve_model

__all__ = ["BaseEmbedding", "CLIPEmbedding", "MODELS", "resolve_model"]
