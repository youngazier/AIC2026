import os
import sys
import json
import numpy as np
import faiss
import torch
import logging
from collections import defaultdict
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


def create_extractor(model_name):
    if model_name == "siglip2":
        return SigLIP2Embedding()
    if model_name == "beit3":
        return Beit3Embedding()
    if model_name == "dinov3":
        return DinoV3Embedding()
    raise ValueError("Unsupported model")


def build_flat_index_if_changed(embedding_array, infos, index_path, meta_path, label, use_gpu=False):
    """
    Build IndexFlatL2 + metadata song hành, nhưng BỎ QUA nếu metadata trên đĩa
    đã y hệt metadata mới — tức playlist/global đó không có video mới thì
    không ghi lại gì cả (yêu cầu cập nhật incremental).
    """
    info_array = np.array(infos)
    if os.path.exists(index_path) and os.path.exists(meta_path):
        old_info = np.load(meta_path, allow_pickle=True)
        if old_info.shape == info_array.shape and np.array_equal(old_info, info_array):
            logger.info(f"[{label}] {len(info_array)} vector, không có video mới — giữ nguyên index.")
            return False

    dimension = embedding_array.shape[1]
    index = faiss.IndexFlatL2(dimension)
    if use_gpu and torch.cuda.is_available():
        res = faiss.StandardGpuResources()
        gpu_index = faiss.index_cpu_to_gpu(res, 0, index)
        gpu_index.add(embedding_array)
        index = faiss.index_gpu_to_cpu(gpu_index)
    else:
        index.add(embedding_array)

    faiss.write_index(index, index_path)
    np.save(meta_path, info_array)
    logger.info(f"[{label}] Đã ghi {index.ntotal} vector (dim={dimension}) -> {os.path.basename(index_path)}")
    return True


def main():
    parser = argparse.ArgumentParser(description="Run feature extraction on keyframes")
    parser.add_argument("--model", type=str, default="siglip2", choices=["siglip2", "beit3", "dinov3"], help="Model to use for embedding")
    parser.add_argument("--batch_size", type=int, default=64, help="Batch size for extracting features")
    args = parser.parse_args()

    config = load_config()
    all_video, video_keyframe_dict = load_all_video_keyframes_info(config["paths"]["data_dir"])

    # Model được nạp LAZY: chỉ khởi tạo khi thực sự có video cần encode.
    # Nhờ đó chạy lại script lúc mọi video đã có .npy (chỉ để cập nhật index
    # playlist) sẽ không tốn thời gian nạp model lên GPU.
    extractor = None

    out_dir = os.path.join(config["paths"]["features_dir"], f"{args.model}")
    index_path = os.path.join(config["paths"]["index_dir"], f"{args.model}_keyframe_embedding.index")
    meta_path = os.path.join(config["paths"]["index_dir"], f"{args.model}_keyframe_metadata.npy")

    os.makedirs(out_dir, exist_ok=True)
    os.makedirs(config["paths"]["index_dir"], exist_ok=True)

    preprocess_fn = None

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
            if extractor is None:
                extractor = create_extractor(args.model)
                preprocess_fn = getattr(extractor, "preprocess", None) or getattr(extractor, "image_processor", None)
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

    # --- 1. GLOBAL INDEX (toàn bộ video) ---
    build_flat_index_if_changed(
        embedding_array, embedding_info, index_path, meta_path,
        label=f"{args.model}/global", use_gpu=True,
    )

    # --- 2. PLAYLIST INDEXES (chỉ build lại playlist có video mới) ---
    vp_path = os.path.join(config["paths"]["data_dir"], "video_to_playlist.json")
    if not os.path.exists(vp_path):
        logger.warning(
            f"Không tìm thấy {vp_path} — bỏ qua bước index theo playlist. "
            f"Chạy data_processing/build_playlist_mapping.py trước rồi chạy lại script này."
        )
        return
    with open(vp_path, encoding="utf-8") as f:
        video_playlist = json.load(f)

    playlist_dir = os.path.join(config["paths"]["index_dir"], "playlist")
    os.makedirs(playlist_dir, exist_ok=True)

    # Gom video theo playlist, giữ nguyên thứ tự all_video để index ổn định
    playlist_videos = defaultdict(list)
    unmapped = []
    for v in all_video:
        if v not in embedding_dict:
            continue
        playlist = video_playlist.get(v)
        if playlist:
            playlist_videos[playlist].append(v)
        else:
            unmapped.append(v)
    if unmapped:
        logger.warning(
            f"{len(unmapped)} video chưa có trong video_to_playlist.json — "
            f"chỉ nằm trong index global. VD: {unmapped[:5]}"
        )

    for playlist in sorted(playlist_videos):
        vectors, infos = [], []
        for v in playlist_videos[playlist]:
            for k in video_keyframe_dict[v]:
                if k in embedding_dict[v]:
                    vectors.append(embedding_dict[v][k])
                    infos.append((v, k))
        if not vectors:
            continue
        build_flat_index_if_changed(
            np.array(vectors, dtype=np.float32),
            infos,
            os.path.join(playlist_dir, f"{args.model}_{playlist}.index"),
            os.path.join(playlist_dir, f"{args.model}_{playlist}_keyframe_metadata.npy"),
            label=f"{args.model}/{playlist}",
        )

    logger.info("Done.")

if __name__ == "__main__":
    main()