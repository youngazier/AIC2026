import os
import sys
import numpy as np
import faiss
import torch
import logging
from tqdm import tqdm
import argparse
from torch.utils.data import Dataset, DataLoader
from PIL import Image
from glob import glob

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "../../")))

from utils.load_all_video_keyframes_info import load_all_video_keyframes_info

from data_processing.feature_extraction.siglip2_embedding import SigLIP2Embedding
from data_processing.feature_extraction.beit3_embedding import Beit3Embedding
from data_processing.feature_extraction.dinov3_embedding import DinoV3Embedding

from utils.config_loader import load_config

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class KeyframeDataset(Dataset):
    def __init__(self, keyframe_paths, keyframe_names, preprocess_fn):
        self.keyframe_paths = keyframe_paths
        self.keyframe_names = keyframe_names
        self.preprocess_fn = preprocess_fn

    def __len__(self):
        return len(self.keyframe_paths)

    def __getitem__(self, idx):
        path = self.keyframe_paths[idx]
        name = self.keyframe_names[idx]
        try:
            image = Image.open(path).convert("RGB")
            tensor = self.preprocess_fn(image)
            return tensor, name
        except Exception as e:
            logger.error(f"Error loading image {path}: {e}")
            return None, name

def collate_fn(batch):
    batch = [item for item in batch if item[0] is not None]
    if not batch:
        return None
    tensors, names = zip(*batch)
    return torch.stack(tensors), names


def main():
    parser = argparse.ArgumentParser(description="Run feature extraction on keyframes")
    parser.add_argument("--model", type=str, default="siglip2", choices=["siglip2", "beit3", "dinov3"], help="Model to use for embedding")
    parser.add_argument("--batch_size", type=int, default=64, help="Batch size for extracting features")
    args = parser.parse_args()

    config = load_config()
    all_video, video_keyframe_dict = load_all_video_keyframes_info(config["paths"]["data_dir"])

    # Initialize model
    if args.model == "siglip2":
        extractor = SigLIP2Embedding()
    elif args.model == "beit3":
        extractor = Beit3Embedding()
    elif args.model == "dinov3":
        extractor = DinoV3Embedding()
    else:
        raise ValueError("Unsupported model")
    
    out_dir = os.path.join(config["paths"]["features_dir"], f"{args.model}")
    index_path = os.path.join(config["paths"]["index_dir"], f"{args.model}_keyframe_embedding.index")
    meta_path = os.path.join(config["paths"]["index_dir"], f"{args.model}_keyframe_metadata.npy")

    os.makedirs(out_dir, exist_ok=True)
    os.makedirs(config["paths"]["index_dir"], exist_ok=True)

    preprocess_fn = getattr(extractor, "preprocess", None) or getattr(extractor, "image_processor", None)

    # Biến lưu trữ số chiều vector động (Dynamic Dimension)
    feature_dim = None

    for v in tqdm(all_video, desc=f"Processing videos with {args.model}"):
        if v not in video_keyframe_dict or not video_keyframe_dict[v]:
            logger.warning(f"Video {v} has no keyframes in video_keyframe_dict. Skipping.")
            continue
            
        np_out = f"{out_dir}/{v}.npy"
        if os.path.exists(np_out):
            print(f"{np_out} exists, skip")
            continue
            
        # Tìm thư mục chứa keyframe thực tế của video v
        video_dir_candidates = glob(os.path.join(config["paths"]["keyframes_dir"], "*", v))
        video_dir = video_dir_candidates[0] if video_dir_candidates else os.path.join(config["paths"]["keyframes_dir"], v)

        keyframe_paths = []
        keyframe_names = []
        for k in video_keyframe_dict[v]:
            keyframe_path = os.path.join(video_dir, f"{k}.jpg")
            if os.path.exists(keyframe_path):
                keyframe_paths.append(keyframe_path)
                keyframe_names.append(k)
            else:
                logger.warning(f"Image not found: {keyframe_path}")
                
        keyframe_list = []
        if keyframe_paths:
            dataset = KeyframeDataset(keyframe_paths, keyframe_names, preprocess_fn)
            dataloader = DataLoader(
                dataset,
                batch_size=args.batch_size,
                shuffle=False,
                num_workers=4,
                collate_fn=collate_fn
            )
            
            for batch in dataloader:
                if batch is None:
                    continue
                image_tensors, _ = batch
                embeddings = extractor.encode_images_batch(image_tensors)
                if embeddings is not None:
                    if feature_dim is None:
                        feature_dim = embeddings.shape[-1]
                        logger.info(f"Dynamically inferred feature dimension: {feature_dim}")
                    keyframe_list.append(embeddings)
                    
        if keyframe_list:
            keyframe_array = np.vstack(keyframe_list)
            np.save(np_out, keyframe_array)
        else:
            # Xử lý trường hợp video không có keyframe nào hợp lệ
            if feature_dim is not None:
                keyframe_array = np.empty((0, feature_dim))
                np.save(np_out, keyframe_array)
            else:
                logger.warning(f"Video {v} is completely empty and global dimension is unknown yet. Skipping saving.")
                continue

    # Load features into dictionary
    embedding_dict = {}
    for v in all_video:
        if v not in video_keyframe_dict:
            continue
        clip_path = f"{out_dir}/{v}.npy"
        if not os.path.exists(clip_path):
            continue
            
        a = np.load(clip_path)
        # Bỏ qua các array rỗng để không bị lỗi lúc map dictionary
        if a.size == 0:
            continue
            
        embedding_dict[v] = {}
        
        # Tìm thư mục chứa keyframe thực tế của video v
        video_dir_candidates = glob(os.path.join(config["paths"]["keyframes_dir"], "*", v))
        video_dir = video_dir_candidates[0] if video_dir_candidates else os.path.join(config["paths"]["keyframes_dir"], v)
        
        valid_keyframes = []
        for k in video_keyframe_dict[v]:
            keyframe_path = os.path.join(video_dir, f"{k}.jpg")
            if os.path.exists(keyframe_path):
                valid_keyframes.append(k)
                
        for i, k in enumerate(valid_keyframes):
            if i < a.shape[0]:
                embedding_dict[v][k] = a[i]

    # Prepare data for FAISS index
    embedding_list = []
    embedding_info = []
    for v in all_video:
        if v not in video_keyframe_dict or v not in embedding_dict:
            continue
        for k in video_keyframe_dict[v]:
            if k in embedding_dict[v]:
                embedding_list.append(embedding_dict[v][k])
                embedding_info.append((v, k))
                
    if not embedding_list:
        logger.error("No embeddings extracted. Exiting.")
        return

    embedding_array = np.array(embedding_list, dtype=np.float32)
    info_array = np.array(embedding_info)

    # --- FAISS GPU INDEXING ---
    # Tự động lấy dimension từ chính mảng dữ liệu cuối cùng thay vì hardcode
    dimension = embedding_array.shape[1] 
    cpu_index = faiss.IndexFlatL2(dimension)
    
    if torch.cuda.is_available():
        logger.info(f"Configuring FAISS on GPU with dimension {dimension}...")
        res = faiss.StandardGpuResources()
        index = faiss.index_cpu_to_gpu(res, 0, cpu_index)
    else:
        logger.warning(f"GPU not available for FAISS, using CPU fallback with dimension {dimension}.")
        index = cpu_index

    logger.info("Adding vectors to FAISS index...")
    index.add(embedding_array)
    
    logger.info("Saving index to disk...")
    index_to_save = faiss.index_gpu_to_cpu(index) if torch.cuda.is_available() else index
    faiss.write_index(index_to_save, index_path)
    np.save(meta_path, info_array)
    logger.info("Done.")

if __name__ == "__main__":
    main()