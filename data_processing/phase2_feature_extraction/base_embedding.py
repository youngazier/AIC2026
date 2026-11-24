"""Common interface for multimodal embedding models."""

from abc import ABC, abstractmethod
from os import PathLike
from typing import TypeAlias

import numpy as np
from PIL import Image
import torch


ImageInput: TypeAlias = Image.Image | np.ndarray | str | PathLike[str]


class BaseEmbedding(ABC):
    """Abstract interface implemented by text/image embedding backends."""

    @abstractmethod
    def encode_text(self, text: str | list[str]) -> torch.Tensor:
        """Encode one string or a batch of strings into normalized vectors."""

    @abstractmethod
    def encode_image(self, image: ImageInput) -> torch.Tensor:
        """Encode one image into a normalized one-dimensional vector."""

    @abstractmethod
    def encode_images_batch(self, images: list[ImageInput]) -> torch.Tensor:
        """Encode a batch of images into normalized vectors."""
