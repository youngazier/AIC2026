import os
import sys
import numpy as np
import faiss
import torch
from fastapi import FastAPI, Query, HTTPException
from contextlib import asynccontextmanager

# Đảm bảo app chạy trên GPU 7
os.environ["CUDA_VISIBLE_DEVICES"] = "7"

project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), "../../"))
if project_root not in sys.path:
    sys.path.insert(0, project_root)


from utils.config_loader import load_config
from data_processing.feature_extraction.siglip2_embedding import SigLIP2Embedding
from data_processing.feature_extraction.beit3_embedding import Beit3Embedding

@asynccontextmanager
async def lifespan(app: FastAPI):
    # Load cấu hình hệ thống
    config = load_config()
    app.state.config = config
    
    # Khởi tạo mô hình duy nhất 1 lần và đưa lên GPU 7
    print("Initializing SigLIP2 model on GPU 7...")
    app.state.siglip2_model = SigLIP2Embedding()
    
    print("Initializing BEiT-3 model on GPU 7...")
    app.state.beit3_model = Beit3Embedding()
    
    index_dir = config["paths"]["index_dir"]
    
    # Load index và metadata cho SigLIP2
    siglip2_idx_path = os.path.join(index_dir, "siglip2_keyframe_embedding.index")
    siglip2_meta_path = os.path.join(index_dir, "siglip2_keyframe_metadata.npy")
    
    if os.path.exists(siglip2_idx_path) and os.path.exists(siglip2_meta_path):
        print("Loading SigLIP2 FAISS index and metadata...")
        app.state.siglip2_index = faiss.read_index(siglip2_idx_path)
        app.state.siglip2_metadata = np.load(siglip2_meta_path, allow_pickle=True)
    else:
        print(f"Warning: SigLIP2 index or metadata not found at {index_dir}")
        app.state.siglip2_index = None
        app.state.siglip2_metadata = None
        
    # Load index và metadata cho BEiT-3
    beit3_idx_path = os.path.join(index_dir, "beit3_keyframe_embedding.index")
    beit3_meta_path = os.path.join(index_dir, "beit3_keyframe_metadata.npy")
    
    if os.path.exists(beit3_idx_path) and os.path.exists(beit3_meta_path):
        print("Loading BEiT-3 FAISS index and metadata...")
        app.state.beit3_index = faiss.read_index(beit3_idx_path)
        app.state.beit3_metadata = np.load(beit3_meta_path, allow_pickle=True)
    else:
        print(f"Warning: BEiT-3 index or metadata not found at {index_dir}")
        app.state.beit3_index = None
        app.state.beit3_metadata = None
        
    yield
    
    print("Clean up resources, shutting down app...")

app = FastAPI(
    title="AIC 2026 Keyframe Retrieval API",
    description="FastAPI service for keyframe retrieval using SigLIP2 and BEiT-3 with FAISS.",
    version="1.0.0",
    lifespan=lifespan
)

@app.get("/search")
async def search(
    query: str = Query(..., description="Text query to search for keyframes"),
    model: str = Query("siglip2", enum=["siglip2", "beit3"], description="Model to use for query embedding and retrieval"),
    top_k: int = Query(10, ge=1, le=100, description="Number of nearest neighbors to retrieve")
):
    # Lấy model, index và metadata tương ứng
    if model == "siglip2":
        extractor = getattr(app.state, "siglip2_model", None)
        index = getattr(app.state, "siglip2_index", None)
        metadata = getattr(app.state, "siglip2_metadata", None)
    elif model == "beit3":
        extractor = getattr(app.state, "beit3_model", None)
        index = getattr(app.state, "beit3_index", None)
        metadata = getattr(app.state, "beit3_metadata", None)
    else:
        raise HTTPException(status_code=400, detail="Unsupported model.")

    if extractor is None:
        raise HTTPException(status_code=500, detail=f"Model {model} was not initialized properly.")
        
    if index is None or metadata is None:
        raise HTTPException(
            status_code=503, 
            detail=f"FAISS index or metadata for {model} is not loaded/available. Please run feature extraction first."
        )

    # Trích xuất vector từ text query
    query_vector = extractor.encode_text(query)
    if query_vector is None:
        raise HTTPException(status_code=500, detail="Failed to encode query text.")

    # Đảm bảo định dạng vector truy vấn đúng (1, dimension)
    if len(query_vector.shape) == 1:
        query_vector = query_vector.reshape(1, -1)
    elif len(query_vector.shape) == 3:
        query_vector = query_vector.squeeze(0)

    query_vector = query_vector.astype(np.float32)

    # Thực hiện tìm kiếm láng giềng gần nhất (FAISS L2 search)
    distances, indices = index.search(query_vector, top_k)

    results = []
    for dist, idx in zip(distances[0], indices[0]):
        # faiss trả về -1 nếu không tìm thấy đủ láng giềng
        if idx == -1:
            continue
        if idx < len(metadata):
            video_id, keyframe_id = metadata[idx]
            results.append({
                "video_id": str(video_id),
                "keyframe_id": str(keyframe_id),
                "score": float(dist)  # Khoảng cách L2 (càng nhỏ càng giống)
            })

    return {
        "query": query,
        "model": model,
        "results": results
    }

if __name__ == "__main__":
    import uvicorn
    # Chạy trên port 8000
    uvicorn.run("test_retrieval:app", host="0.0.0.0", port=8000, reload=True)
