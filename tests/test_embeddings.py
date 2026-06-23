import os
import sys
import numpy as np
import faiss
import torch
from glob import glob

# Append project root to sys.path
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if project_root not in sys.path:
    sys.path.append(project_root)

from utils.config_loader import load_config

# Set Hugging Face cache to ensure offline safety
config = load_config()
hf_cache = config["paths"].get("huggingface_cache_dir", "/tmp2/maitanha/vgu/huggingface_cache")
os.environ["HF_HOME"] = hf_cache
os.environ["HF_HUB_OFFLINE"] = "1"

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
        query_vector = extractor.encode_text(query)
        if query_vector is None:
            print(" - Error encoding query text.")
            continue
            
        if len(query_vector.shape) == 1:
            query_vector = query_vector.reshape(1, -1)
        elif len(query_vector.shape) == 3:
            query_vector = query_vector.squeeze(0)
            
        query_vector = query_vector.astype(np.float32)
        
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

def main():
    config = load_config()
    
    # 1. Test local npy vs keyframe directories
    test_model_embeddings(config, "siglip2")
    test_model_embeddings(config, "beit3")
    test_model_embeddings(config, "dinov3")
    
    # 2. Test FAISS index and retrieval queries
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
