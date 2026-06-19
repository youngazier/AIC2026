import sys
import os
import numpy as np
from PIL import Image
from tqdm import tqdm
import argparse
import time
import torch
from torchvision import transforms
from transformers import XLMRobertaTokenizer
from timm import create_model 
from timm.data.constants import IMAGENET_INCEPTION_MEAN, IMAGENET_INCEPTION_STD


project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), "../../"))
if project_root not in sys.path:
    sys.path.insert(0, project_root)
    
from base_embedding import BaseEmbedding

from utils.config_loader import load_config
config = load_config()

# ==========================================
unilm_repo_path = os.path.join(config["paths"]["support_models_dir"], "unilm", "beit3")
cached_utils = sys.modules.pop('utils', None)
sys.path.insert(0, unilm_repo_path)

try:
    from modeling_finetune import beit3_large_patch16_384_retrieval
finally:
    # Rút đường dẫn Microsoft ra khỏi hệ thống
    if sys.path[0] == unilm_repo_path:
        sys.path.pop(0)
    elif unilm_repo_path in sys.path:
        sys.path.remove(unilm_repo_path)
        
    # Trả lại utils gốc của dự án
    if cached_utils is not None:
        sys.modules['utils'] = cached_utils
# ==========================================

class Beit3Embedding(BaseEmbedding):
    def __init__(self):
            self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
            
            # Load cấu hình từ config.yaml
            model_checkpoint = config["models"]["beit3"]["model_checkpoint"]
            tokenizer_checkpoint = config["models"]["beit3"]["tokenizer_checkpoint"]
            
            print(f"[{self.__class__.__name__}] Initializing on {self.device}...")
            
            # 1. Thiết lập Model
            checkpoint = torch.load(model_checkpoint, map_location="cpu")
            self.model = create_model('beit3_large_patch16_384_retrieval')
            self.model.load_state_dict(checkpoint['model'])
            self.model.to(self.device)
            self.model.eval()
            
            # 2. Thiết lập Tokenizer bằng cách tạo liên kết/copy sentencepiece.bpe.model để from_pretrained load chuẩn
            spm_dir = os.path.dirname(tokenizer_checkpoint)
            target_spm = os.path.join(spm_dir, "sentencepiece.bpe.model")
            if not os.path.exists(target_spm):
                try:
                    os.symlink("beit3.spm", target_spm)
                except Exception:
                    import shutil
                    shutil.copy(tokenizer_checkpoint, target_spm)
            
            self.tokenizer = XLMRobertaTokenizer.from_pretrained(spm_dir, local_files_only=True)
            
            # 3. Thiết lập Image Processor
            self.image_processor = transforms.Compose([
                transforms.Resize((384, 384), interpolation=transforms.InterpolationMode.BICUBIC),
                transforms.ToTensor(),
                transforms.Normalize(mean=IMAGENET_INCEPTION_MEAN, std=IMAGENET_INCEPTION_STD)
            ])
            print(f"[{self.__class__.__name__}] Ready.")

    def _get_text_segment(self, text: str, max_len: int = 64):
        tokens = self.tokenizer.tokenize(text)
        tokens = self.tokenizer.convert_tokens_to_ids(tokens)

        if len(tokens) > max_len - 2:
            tokens = tokens[:max_len - 2]

        tokens = (
            [self.tokenizer.bos_token_id]
            + tokens[:]
            + [self.tokenizer.eos_token_id]
        )
        num_tokens = len(tokens)
        padding_mask = [0] * num_tokens + [1] * (max_len - num_tokens)
        language_tokens = tokens + [self.tokenizer.pad_token_id] * (
            max_len - num_tokens
        )
        return torch.tensor([language_tokens]), torch.tensor([padding_mask]), num_tokens

    def encode_image(self, image_path: str) -> np.ndarray:
        try:
            raw_image = Image.open(image_path).convert("RGB")
            image_tensor = self.image_processor(raw_image).unsqueeze(0).to(self.device)
            
            with torch.no_grad():
                image_feature, _ = self.model(image=image_tensor, only_infer=True)
                image_feature /= image_feature.norm(dim=-1, keepdim=True)
                
            return image_feature.cpu().numpy().astype(np.float32)
        except Exception as e:
            print(f"Error encoding image {image_path}: {e}")
            return None

    def encode_text(self, text: str) -> np.ndarray:
        try:
            language_tokens, padding_mask, _ = self._get_text_segment(text)
            
            with torch.no_grad():
                _, text_feature = self.model(
                    text_description=language_tokens.to(self.device), 
                    padding_mask=padding_mask.to(self.device), 
                    only_infer=True
                )
                text_feature /= text_feature.norm(dim=-1, keepdim=True)
                
            return text_feature.cpu().numpy().astype(np.float32)
        except Exception as e:
            print(f"Error encoding text '{text}': {e}")
            return None

    def encode_images_batch(self, image_tensors: torch.Tensor) -> np.ndarray:
        try:
            with torch.no_grad():
                image_features, _ = self.model(image=image_tensors.to(self.device), only_infer=True)
                image_features /= image_features.norm(dim=-1, keepdim=True)
            return image_features.cpu().numpy().astype(np.float32)
        except Exception as e:
            print(f"Error encoding image batch: {e}")
            return None

    def compute_similarity(self, image_feature: np.ndarray, text_feature: np.ndarray) -> float:
        if image_feature is None or text_feature is None:
            return 0.0
            
        vector_a = image_feature.flatten()
        vector_b = text_feature.flatten()
        
        norm_a = np.linalg.norm(vector_a)
        norm_b = np.linalg.norm(vector_b)
        
        if norm_a == 0 or norm_b == 0:
            return 0.0
            
        return float(np.dot(vector_a, vector_b) / (norm_a * norm_b))