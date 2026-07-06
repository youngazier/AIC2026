import os
import sys
import json
import numpy as np
import faiss
import torch
from glob import glob

# Append project root to sys.path
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if project_root not in sys.path:
    sys.path.append(project_root)

from utils.config_loader import load_config

# Set Hugging Face cache TRƯỚC khi import model.
# KHÔNG bật HF_HUB_OFFLINE: repo timm/ViT-gopt-16-SigLIP2-384 không có config.json,
# chế độ offline làm transformers ném OSError thay vì bỏ qua file thiếu đó.
config = load_config()
hf_cache = config["paths"].get("huggingface_cache_dir", "/tmp2/maitanha/vgu/huggingface_cache")
os.environ.setdefault("HF_HOME", hf_cache)

from data_processing.feature_extraction.siglip2_embedding import SigLIP2Embedding
from data_processing.feature_extraction.beit3_embedding import Beit3Embedding
from data_processing.feature_extraction.dinov3_embedding import DinoV3Embedding

def test_model_embeddings(config, model_name):
    print(f"\n--- Testing Embeddings for Model: {model_name} ---")
    out_dir = os.path.join(config["paths"]["features_dir"], model_name)
    
    if not os.path.exists(out_dir):
        print(f"Features directory '{out_dir}' does not exist. Skipping model check.")
        return False
        
    npy_files = glob(os.path.join(out_dir, "*.npy"))
    if not npy_files:
        print(f"No .npy files found in {out_dir}. Skipping model check.")
        return False
        
    total_npy = len(npy_files)
    successful = 0
    mismatches = []
    
    for npy_path in sorted(npy_files):
        video_id = os.path.basename(npy_path)[:-4]
        try:
            a = np.load(npy_path)
        except Exception as e:
            mismatches.append(f"Video {video_id}: Failed to load .npy file: {e}")
            continue
        
        # Resolve keyframe dir dynamically under keyframes_dir
        video_dir_candidates = glob(os.path.join(config["paths"]["keyframes_dir"], "*", video_id))
        video_dir = video_dir_candidates[0] if video_dir_candidates else os.path.join(config["paths"]["keyframes_dir"], video_id)
        
        if not os.path.exists(video_dir):
            image_count = 0
        else:
            image_count = len(glob(os.path.join(video_dir, "*.jpg")))
            
        if a.shape[0] != image_count:
            mismatches.append(f"Video {video_id}: Vector count {a.shape[0]} != Image count {image_count}")
        else:
            successful += 1
            
    print(f"Verified .npy files: {successful}/{total_npy}")
    if mismatches:
        print("Mismatches found:")
        for m in mismatches[:10]:
            print(f" - {m}")
        if len(mismatches) > 10:
            print(f" ... and {len(mismatches) - 10} more.")
        return False
    else:
        print("All local .npy file shapes match keyframe image counts perfectly!")
        return True

def test_playlist_indexes(config, model_name):
    """
    Kiểm tra tầng index playlist (data/index/playlist/) của một model:
      1. Mỗi file {model}_{playlist}.index có metadata song hành và ntotal khớp nhau.
      2. Mọi video trong metadata playlist đúng là thuộc playlist đó
         (theo data/video_to_playlist.json).
      3. Mọi cặp (video, keyframe) của playlist đều có mặt trong metadata global.
      4. Tổng vector các playlist = số vector global thuộc video đã được map playlist
         (phần chênh lệch đúng bằng các video chưa có trong video_to_playlist.json).
    Chỉ đọc file, KHÔNG nạp model — chạy rất nhanh.
    """
    print(f"\n--- Testing Playlist Indexes for Model: {model_name} ---")
    index_dir = config["paths"]["index_dir"]
    playlist_dir = os.path.join(index_dir, "playlist")

    if not os.path.isdir(playlist_dir):
        print(f"Playlist index directory '{playlist_dir}' does not exist. Skipping.")
        return False

    vp_path = os.path.join(config["paths"]["data_dir"], "video_to_playlist.json")
    if not os.path.exists(vp_path):
        print(f"{vp_path} not found. Run data_processing/build_playlist_mapping.py first. Skipping.")
        return False
    with open(vp_path, encoding="utf-8") as f:
        video_playlist = json.load(f)

    global_meta_path = os.path.join(index_dir, f"{model_name}_keyframe_metadata.npy")
    if not os.path.exists(global_meta_path):
        print(f"Global metadata not found at {global_meta_path}. Skipping.")
        return False
    global_meta = np.load(global_meta_path, allow_pickle=True)
    global_pairs = {(v, k) for v, k in global_meta}

    prefix = f"{model_name}_"
    index_files = sorted(glob(os.path.join(playlist_dir, f"{prefix}*.index")))
    if not index_files:
        print(f"No playlist index found for model '{model_name}' in {playlist_dir}.")
        return False

    errors = []
    playlist_vector_total = 0
    for idx_path in index_files:
        playlist = os.path.basename(idx_path)[len(prefix):-len(".index")]
        meta_path = os.path.join(playlist_dir, f"{prefix}{playlist}_keyframe_metadata.npy")
        if not os.path.exists(meta_path):
            errors.append(f"[{playlist}] Missing metadata file: {os.path.basename(meta_path)}")
            continue

        try:
            index = faiss.read_index(idx_path)
            meta = np.load(meta_path, allow_pickle=True)
        except Exception as e:
            errors.append(f"[{playlist}] Failed to load index/metadata: {e}")
            continue

        if index.ntotal != len(meta):
            errors.append(f"[{playlist}] Index has {index.ntotal} vectors but metadata has {len(meta)} entries")

        videos_in_meta = {v for v, _ in meta}
        wrong_videos = sorted(v for v in videos_in_meta if video_playlist.get(v) != playlist)
        if wrong_videos:
            errors.append(f"[{playlist}] {len(wrong_videos)} video(s) not mapped to this playlist. E.g: {wrong_videos[:5]}")

        missing_in_global = [(v, k) for v, k in meta if (v, k) not in global_pairs]
        if missing_in_global:
            errors.append(f"[{playlist}] {len(missing_in_global)} pair(s) absent from global metadata. E.g: {missing_in_global[:5]}")

        playlist_vector_total += len(meta)
        print(f" [{playlist}] {index.ntotal} vectors, {len(videos_in_meta)} videos")

    mapped_global_count = sum(1 for v, _ in global_meta if video_playlist.get(v))
    unmapped_videos = sorted({v for v, _ in global_meta if not video_playlist.get(v)})
    if playlist_vector_total != mapped_global_count:
        errors.append(
            f"Coverage mismatch: playlist indexes hold {playlist_vector_total} vectors "
            f"but global index has {mapped_global_count} vectors belonging to mapped videos"
        )
    if unmapped_videos:
        print(f" Note: {len(unmapped_videos)} video(s) only exist in the global index "
              f"(not in video_to_playlist.json). E.g: {unmapped_videos[:5]}")

    if errors:
        print("Playlist index errors found:")
        for e in errors:
            print(f" - {e}")
        return False
    print(f"All {len(index_files)} playlist index(es) are consistent with the global index!")
    return True


def encode_query(extractor, query):
    """Encode text -> vector float32 shape (1, dim) sẵn sàng đưa vào FAISS."""
    query_vector = extractor.encode_text(query)
    if query_vector is None:
        return None
    if len(query_vector.shape) == 1:
        query_vector = query_vector.reshape(1, -1)
    elif len(query_vector.shape) == 3:
        query_vector = query_vector.squeeze(0)
    return query_vector.astype(np.float32)


def test_faiss_retrieval(config, model_name, query_texts):
    print(f"\n--- Testing FAISS Index and Retrieval for Model: {model_name} ---")
    index_dir = config["paths"]["index_dir"]
    
    if model_name == "siglip2":
        idx_path = os.path.join(index_dir, "siglip2_keyframe_embedding.index")
        meta_path = os.path.join(index_dir, "siglip2_keyframe_metadata.npy")
        try:
            extractor = SigLIP2Embedding()
        except Exception as e:
            print(f"Failed to load SigLIP2 model: {e}")
            return
    elif model_name == "beit3":
        idx_path = os.path.join(index_dir, "beit3_keyframe_embedding.index")
        meta_path = os.path.join(index_dir, "beit3_keyframe_metadata.npy")
        try:
            extractor = Beit3Embedding()
        except Exception as e:
            print(f"Failed to load BEiT-3 model: {e}")
            return
    elif model_name == "dinov3":
        idx_path = os.path.join(index_dir, "dinov3_keyframe_embedding.index")
        meta_path = os.path.join(index_dir, "dinov3_keyframe_metadata.npy")
        try:
            extractor = DinoV3Embedding()
        except Exception as e:
            print(f"Failed to load DINOv3 model: {e}")
            return
    else:
        print(f"Unsupported model name: {model_name}")
        return
            
    if not os.path.exists(idx_path) or not os.path.exists(meta_path):
        print(f"FAISS index or metadata files do not exist at {index_dir}. Skipping retrieval test.")
        return
        
    # Load index and metadata
    try:
        index = faiss.read_index(idx_path)
        metadata = np.load(meta_path, allow_pickle=True)
    except Exception as e:
        print(f"Failed to load FAISS index or metadata: {e}")
        return
    
    print(f"FAISS Index size (total vectors): {index.ntotal}")
    print(f"Metadata size (total entries): {len(metadata)}")
    
    if index.ntotal != len(metadata):
        print(f"ERROR: Index size ({index.ntotal}) does not match metadata size ({len(metadata)})!")
        return
    else:
        print("Index size and metadata count match perfectly!")
        
    print("Testing sample query retrieval:")
    for query in query_texts:
        print(f"\nQuery: '{query}'")
        query_vector = encode_query(extractor, query)
        if query_vector is None:
            print(" - Error encoding query text.")
            continue

        # Search top-5 results
        top_k = 5
        distances, indices = index.search(query_vector, top_k)

        print(f" Top-{top_k} Results:")
        for idx, (dist, meta_idx) in enumerate(zip(distances[0], indices[0])):
            if meta_idx == -1:
                print(f"  {idx+1}. No match (empty slot)")
                continue
            video_id, keyframe_id = metadata[meta_idx]
            print(f"  {idx+1}. Video: {video_id} | Keyframe: {keyframe_id} | Score (L2 Distance): {dist:.4f}")

    # --- Playlist retrieval: search từng index playlist, kết quả PHẢI nằm gọn trong playlist đó ---
    playlist_dir = os.path.join(index_dir, "playlist")
    vp_path = os.path.join(config["paths"]["data_dir"], "video_to_playlist.json")
    if not os.path.isdir(playlist_dir) or not os.path.exists(vp_path):
        print("\nPlaylist dir or video_to_playlist.json missing — skipping playlist retrieval test.")
        return
    with open(vp_path, encoding="utf-8") as f:
        video_playlist = json.load(f)

    query = query_texts[0]
    query_vector = encode_query(extractor, query)
    if query_vector is None:
        print(" - Error encoding query text for playlist retrieval.")
        return

    prefix = f"{model_name}_"
    print(f"\nTesting playlist retrieval with query: '{query}'")
    for idx_path in sorted(glob(os.path.join(playlist_dir, f"{prefix}*.index"))):
        playlist = os.path.basename(idx_path)[len(prefix):-len(".index")]
        meta_path = os.path.join(playlist_dir, f"{prefix}{playlist}_keyframe_metadata.npy")
        if not os.path.exists(meta_path):
            print(f" [{playlist}] metadata missing, skip")
            continue
        p_index = faiss.read_index(idx_path)
        p_meta = np.load(meta_path, allow_pickle=True)

        top_k = min(5, p_index.ntotal)
        distances, indices = p_index.search(query_vector, top_k)
        results = [p_meta[i] for i in indices[0] if i != -1]
        leaked = [v for v, _ in results if video_playlist.get(v) != playlist]
        if leaked:
            print(f" [{playlist}] ERROR: results contain videos outside this playlist: {leaked}")
        else:
            top1 = f"{results[0][0]}/{results[0][1]} (L2={distances[0][0]:.4f})" if results else "no result"
            print(f" [{playlist}] top-{top_k} OK, all results inside playlist. Top-1: {top1}")

def main():
    config = load_config()
    
    # 1. Test local npy vs keyframe directories
    test_model_embeddings(config, "siglip2")
    test_model_embeddings(config, "beit3")
    test_model_embeddings(config, "dinov3")

    # 2. Test playlist indexes (structural, no model loading — fast)
    test_playlist_indexes(config, "siglip2")
    test_playlist_indexes(config, "beit3")
    test_playlist_indexes(config, "dinov3")

    # 3. Test FAISS index and retrieval queries (global + per-playlist, loads models)
    queries = [
        "a person standing",
        "outdoor scenery",
        "red car on the road"
    ]
    test_faiss_retrieval(config, "siglip2", queries)
    test_faiss_retrieval(config, "beit3", queries)
    test_faiss_retrieval(config, "dinov3", queries)

if __name__ == "__main__":
    main()
