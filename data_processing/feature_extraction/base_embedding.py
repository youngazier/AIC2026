from abc import ABC, abstractmethod
import numpy as np
import torch

class BaseEmbedding(ABC):
    @abstractmethod
    def encode_text(self, text: str) -> np.ndarray:
        pass

    @abstractmethod
    def encode_image(self, image_path: str) -> np.ndarray:
        pass

    @abstractmethod
    def encode_images_batch(self, image_tensors: torch.Tensor) -> np.ndarray:
        pass
