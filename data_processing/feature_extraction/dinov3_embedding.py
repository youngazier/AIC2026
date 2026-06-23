import os
from tqdm import tqdm
import numpy as np
from PIL import Image, UnidentifiedImageError
from typing import List, Tuple, Set, Dict
from pathlib import Path
import torch
import faiss
from glob import glob
import sys
from transformers import AutoImageProcessor, AutoModel

project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), "../../"))
if project_root not in sys.path:
    sys.path.insert(0, project_root)

from base_embedding import BaseEmbedding

from utils.config_loader import load_config
config = load_config()

class DinoV3Embedding(BaseEmbedding):
    def __init__(self):
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

        model_name = config["models"]["dinov3"]["model_name"]
        self.processor = AutoImageProcessor.from_pretrained(model_name)
        self.model = AutoModel.from_pretrained(model_name).to(self.device).eval()
    
        print(f"[{self.__class__.__name__}] Ready.")

    def encode_text(self, text: str) -> np.ndarray:
        pass

    def encode_image(self, image_path: str) -> np.ndarray:
        try:
            image = Image.open(image_path).convert("RGB")
        except (UnidentifiedImageError, FileNotFoundError):
            print(f"[WARNING]: Could not open or identify image file: {image_path}")
            return None
        
        with torch.no_grad():
            inputs = self.processor(images=image, return_tensors="pt").to(self.device)
            outputs = self.model(**inputs)
            # DINOv3 embeddings are in last_hidden_state. Average pooling is a common way to get a single vector.
            image_feature = outputs.last_hidden_state.mean(dim=1).squeeze().cpu().numpy()
            image_feature /= np.linalg.norm(image_feature)
            
        return image_feature

    def encode_images_batch(self, image_tensors: torch.Tensor) -> np.ndarray:
        try:
            with torch.no_grad():
                outputs = self.model(image_tensors.to(self.device))
                image_features = outputs.last_hidden_state.mean(dim=1).cpu().numpy()
                image_features /= np.linalg.norm(image_features, axis=1, keepdims=True)
            return image_features
        except Exception as e:
            print(f"Error encoding image batch: {e}")
            return None