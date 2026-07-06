import os
import sys

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "../../")))
from utils.config_loader import load_config
config = load_config()

# hf_cache = config["paths"].get("huggingface_cache_dir", "/tmp2/maitanha/vgu/huggingface_cache")
# os.environ["HF_HOME"] = hf_cache

import numpy as np
import open_clip
import torch
from PIL import Image
from data_processing.feature_extraction.base_embedding import BaseEmbedding

class SigLIP2Embedding(BaseEmbedding):
    def __init__(self, model_name="ViT-gopt-16-SigLIP2-384", pretrained="webli"):
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        print(f"[{self.__class__.__name__}] Initializing on {self.device}...")
        
        self.model, _, self.preprocess = open_clip.create_model_and_transforms(
            model_name,
            pretrained=pretrained,
            device=self.device,
        )
        self.model.eval()
        self.tokenizer = open_clip.get_tokenizer(model_name)
        
        print(f"[{self.__class__.__name__}] Ready.")

    def encode_image(self, image_path: str) -> np.ndarray:
        try:
            image = Image.open(image_path).convert("RGB")
            image = self.preprocess(image).unsqueeze(0).to(self.device)
            with torch.no_grad():
                image_feature = self.model.encode_image(image)
                image_feature /= image_feature.norm(dim=-1, keepdim=True)
            return image_feature.cpu().numpy().astype(np.float32)
        except Exception as e:
            print(f"Error encoding image {image_path}: {e}")
            return None

    def encode_text(self, text: str) -> np.ndarray:
        try:
            text_tokens = self.tokenizer([text]).to(self.device)
            with torch.no_grad():
                text_feature = self.model.encode_text(text_tokens)
                text_feature /= text_feature.norm(dim=-1, keepdim=True)
            return text_feature.cpu().numpy().astype(np.float32)
        except Exception as e:
            print(f"Error encoding text '{text}': {e}")
            return None

    def encode_images_batch(self, image_tensors: torch.Tensor) -> np.ndarray:
        try:
            with torch.no_grad():
                image_features = self.model.encode_image(image_tensors.to(self.device))
                image_features /= image_features.norm(dim=-1, keepdim=True)
            return image_features.cpu().numpy().astype(np.float32)
        except Exception as e:
            print(f"Error encoding image batch: {e}")
            return None

