"""Embedding models used by the feature-extraction phase."""

from .base_embedding import BaseEmbedding
from .clip_embedding import CLIPEmbedding

__all__ = ["BaseEmbedding", "CLIPEmbedding"]
