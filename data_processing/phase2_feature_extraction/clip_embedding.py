"""CLIP ViT-B/32 embedding implementation."""

from os import PathLike
from pathlib import Path

import numpy as np
from PIL import Image
import torch
import torch.nn.functional as F
from transformers import CLIPModel, CLIPProcessor

from .base_embedding import BaseEmbedding, ImageInput


class CLIPEmbedding(BaseEmbedding):
    """Generate compatible text and image embeddings with Hugging Face CLIP.

    The model is downloaded and cached by Hugging Face on first use. Returned
    tensors live on CPU, which makes them safe to persist or pass to NumPy.
    """

    DEFAULT_MODEL = "openai/clip-vit-base-patch32"

    def __init__(self, model_name: str = DEFAULT_MODEL, device: str = "auto") -> None:
        self.device = self._resolve_device(device)
        self.processor = CLIPProcessor.from_pretrained(model_name)
        self.model = CLIPModel.from_pretrained(model_name).to(self.device).eval()

    @staticmethod
    def _resolve_device(device: str) -> torch.device:
        if device == "auto":
            device = "cuda" if torch.cuda.is_available() else "cpu"
        resolved = torch.device(device)
        if resolved.type == "cuda" and not torch.cuda.is_available():
            raise ValueError("CUDA was requested but is unavailable")
        return resolved

    @staticmethod
    def _load_image(image: ImageInput) -> Image.Image:
        if isinstance(image, (str, PathLike)):
            with Image.open(Path(image)) as opened:
                return opened.convert("RGB")
        if isinstance(image, Image.Image):
            return image.convert("RGB")
        if isinstance(image, np.ndarray):
            if image.ndim not in (2, 3):
                raise ValueError("An image array must have two or three dimensions")
            return Image.fromarray(image).convert("RGB")
        raise TypeError(f"Unsupported image type: {type(image).__name__}")

    @staticmethod
    def _normalize(features) -> torch.Tensor:
        # transformers 5 returns BaseModelOutputWithPooling here, while older
        # releases return the projected tensor directly.
        if not isinstance(features, torch.Tensor):
            features = getattr(features, "pooler_output", None)
        if not isinstance(features, torch.Tensor) or features.ndim != 2:
            raise TypeError("CLIP did not return a two-dimensional feature tensor")
        return F.normalize(features, p=2, dim=-1).detach().cpu()

    @torch.inference_mode()
    def encode_text(self, text: str | list[str]) -> torch.Tensor:
        if not isinstance(text, str) and not text:
            raise ValueError("At least one text input is required")
        inputs = self.processor(text=text, return_tensors="pt", padding=True)
        inputs = {name: value.to(self.device) for name, value in inputs.items()}
        features = self.model.get_text_features(**inputs)
        result = self._normalize(features)
        return result[0] if isinstance(text, str) else result

    @torch.inference_mode()
    def encode_image(self, image: ImageInput) -> torch.Tensor:
        return self.encode_images_batch([image])[0]

    @torch.inference_mode()
    def encode_images_batch(self, images: list[ImageInput]) -> torch.Tensor:
        if not images:
            raise ValueError("At least one image is required")
        loaded = [self._load_image(image) for image in images]
        inputs = self.processor(images=loaded, return_tensors="pt")
        inputs = {name: value.to(self.device) for name, value in inputs.items()}
        return self._normalize(self.model.get_image_features(**inputs))
