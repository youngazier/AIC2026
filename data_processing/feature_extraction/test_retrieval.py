"""
Web app retrieval đơn giản: nhập text query -> encode -> search FAISS -> trả về
top-k keyframe kèm ẢNH thật để xem ngay trên trình duyệt.

Chạy (từ thư mục này, với venv_embedding):
    python test_retrieval.py
Rồi mở http://localhost:8000/  (giao diện tìm kiếm)
API JSON thuần vẫn dùng được: GET /search?query=...&model=siglip2&top_k=10
"""
import os
import sys

# Đảm bảo app chạy trên GPU 7
os.environ["CUDA_VISIBLE_DEVICES"] = "7"

project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), "../../"))
if project_root not in sys.path:
    sys.path.insert(0, project_root)

from utils.config_loader import load_config

# Set HF cache TRƯỚC khi import các module model.
# KHÔNG bật HF_HUB_OFFLINE: repo timm/ViT-gopt-16-SigLIP2-384 không có config.json,
# chế độ offline làm transformers ném OSError thay vì bỏ qua file thiếu đó.
_config = load_config()
os.environ.setdefault("HF_HOME", _config["paths"].get("huggingface_cache_dir", "/tmp2/maitanha/vgu/huggingface_cache"))

import re
from contextlib import asynccontextmanager
from glob import glob

import faiss
import numpy as np
from fastapi import FastAPI, HTTPException, Query
from fastapi.responses import FileResponse, HTMLResponse

from data_processing.feature_extraction.siglip2_embedding import SigLIP2Embedding
from data_processing.feature_extraction.beit3_embedding import Beit3Embedding
from data_processing.feature_extraction.dinov3_embedding import DinoV3Embedding

MODEL_CLASSES = {
    "siglip2": SigLIP2Embedding,
    "beit3": Beit3Embedding,
    "dinov3": DinoV3Embedding,
}

SAFE_ID = re.compile(r"^[A-Za-z0-9_-]+$")  # chặn path traversal khi serve ảnh


@asynccontextmanager
async def lifespan(app: FastAPI):
    config = load_config()
    app.state.config = config
    index_dir = config["paths"]["index_dir"]

    # engines[model] = {"extractor": ..., "index": ..., "metadata": ...}
    app.state.engines = {}
    for name, cls in MODEL_CLASSES.items():
        idx_path = os.path.join(index_dir, f"{name}_keyframe_embedding.index")
        meta_path = os.path.join(index_dir, f"{name}_keyframe_metadata.npy")
        if not (os.path.exists(idx_path) and os.path.exists(meta_path)):
            print(f"Warning: index/metadata for '{name}' not found at {index_dir} — skipping this model.")
            continue
        print(f"Initializing {name} model on GPU...")
        extractor = cls()
        print(f"Loading {name} FAISS index and metadata...")
        app.state.engines[name] = {
            "extractor": extractor,
            "index": faiss.read_index(idx_path),
            "metadata": np.load(meta_path, allow_pickle=True),
        }

    if not app.state.engines:
        print("ERROR: no model has index+metadata. Run run_embedding.py first.")
    yield
    print("Clean up resources, shutting down app...")


app = FastAPI(
    title="AIC 2026 Keyframe Retrieval",
    description="Nhập text query, xem top-k keyframe khớp nhất (SigLIP2 / BEiT-3 / DINOv3 + FAISS).",
    version="2.0.0",
    lifespan=lifespan,
)


HTML_PAGE = """<!DOCTYPE html>
<html lang="vi">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>AIC 2026 — Keyframe Retrieval</title>
<style>
  * { box-sizing: border-box; }
  body { font-family: system-ui, sans-serif; margin: 0; background: #111; color: #eee; }
  header { padding: 16px 24px; background: #1b1b1b; border-bottom: 1px solid #333; }
  h1 { font-size: 18px; margin: 0 0 12px; }
  form { display: flex; gap: 8px; flex-wrap: wrap; }
  input[type=text] { flex: 1; min-width: 260px; padding: 10px 12px; font-size: 15px;
    background: #222; color: #eee; border: 1px solid #444; border-radius: 8px; }
  select, input[type=number] { padding: 10px; background: #222; color: #eee;
    border: 1px solid #444; border-radius: 8px; }
  input[type=number] { width: 80px; }
  button { padding: 10px 20px; font-size: 15px; background: #2563eb; color: #fff;
    border: 0; border-radius: 8px; cursor: pointer; }
  button:disabled { background: #555; cursor: wait; }
  #status { padding: 10px 24px; color: #aaa; min-height: 20px; }
  #grid { display: grid; grid-template-columns: repeat(auto-fill, minmax(240px, 1fr));
    gap: 14px; padding: 0 24px 40px; }
  .card { background: #1b1b1b; border: 1px solid #333; border-radius: 10px; overflow: hidden; }
  .card img { width: 100%; aspect-ratio: 16/9; object-fit: cover; display: block; background: #000; }
  .card .info { padding: 8px 10px; font-size: 13px; line-height: 1.5; }
  .card .rank { color: #60a5fa; font-weight: 600; }
  .card .score { color: #aaa; }
</style>
</head>
<body>
<header>
  <h1>AIC 2026 — Keyframe Retrieval</h1>
  <form id="f">
    <input type="text" id="query" placeholder="Mô tả phân cảnh, VD: a red car on the road" required>
    <select id="model"></select>
    <input type="number" id="topk" value="10" min="1" max="100" title="top-k">
    <button id="btn" type="submit">Tìm kiếm</button>
  </form>
</header>
<div id="status"></div>
<div id="grid"></div>
<script>
const $ = id => document.getElementById(id);

// Đổ danh sách model đang thực sự được nạp
fetch("/models").then(r => r.json()).then(ms => {
  $("model").innerHTML = ms.map(m => `<option value="${m}">${m}</option>`).join("");
});

$("f").addEventListener("submit", async e => {
  e.preventDefault();
  const q = $("query").value.trim();
  if (!q) return;
  $("btn").disabled = true;
  $("grid").innerHTML = "";
  $("status").textContent = "Đang tìm kiếm...";
  try {
    const url = `/search?query=${encodeURIComponent(q)}&model=${$("model").value}&top_k=${$("topk").value}`;
    const res = await fetch(url);
    if (!res.ok) throw new Error((await res.json()).detail || res.statusText);
    const data = await res.json();
    $("status").textContent = `${data.results.length} kết quả cho "${data.query}" (model: ${data.model})`;
    $("grid").innerHTML = data.results.map((r, i) => `
      <div class="card">
        <a href="${r.image_url}" target="_blank"><img src="${r.image_url}" loading="lazy" alt=""></a>
        <div class="info">
          <span class="rank">#${i + 1}</span> ${r.video_id} / ${r.keyframe_id}<br>
          <span class="score">L2 = ${r.score.toFixed(4)}</span>
        </div>
      </div>`).join("");
  } catch (err) {
    $("status").textContent = "Lỗi: " + err.message;
  } finally {
    $("btn").disabled = false;
  }
});
</script>
</body>
</html>"""


@app.get("/", response_class=HTMLResponse)
async def home():
    """Giao diện tìm kiếm."""
    return HTML_PAGE


@app.get("/models")
async def list_models():
    """Danh sách model đã nạp thành công (có cả model lẫn index)."""
    return list(app.state.engines.keys())


@app.get("/search")
async def search(
    query: str = Query(..., description="Text query to search for keyframes"),
    model: str = Query("siglip2", description="Model dùng để encode query (xem GET /models)"),
    top_k: int = Query(10, ge=1, le=100, description="Số keyframe gần nhất cần lấy"),
):
    engine = app.state.engines.get(model)
    if engine is None:
        raise HTTPException(
            status_code=503,
            detail=f"Model '{model}' chưa được nạp. Các model khả dụng: {list(app.state.engines.keys())}",
        )

    query_vector = engine["extractor"].encode_text(query)
    if query_vector is None:
        raise HTTPException(status_code=500, detail="Failed to encode query text.")

    # Đưa vector về đúng shape (1, dim) float32
    if len(query_vector.shape) == 1:
        query_vector = query_vector.reshape(1, -1)
    elif len(query_vector.shape) == 3:
        query_vector = query_vector.squeeze(0)
    query_vector = query_vector.astype(np.float32)

    distances, indices = engine["index"].search(query_vector, top_k)

    metadata = engine["metadata"]
    results = []
    for dist, idx in zip(distances[0], indices[0]):
        if idx == -1 or idx >= len(metadata):
            continue
        video_id, keyframe_id = metadata[idx]
        results.append({
            "video_id": str(video_id),
            "keyframe_id": str(keyframe_id),
            "score": float(dist),  # L2 distance — càng nhỏ càng giống
            "image_url": f"/keyframe/{video_id}/{keyframe_id}",
        })

    return {"query": query, "model": model, "results": results}


@app.get("/keyframe/{video_id}/{keyframe_id}")
async def get_keyframe_image(video_id: str, keyframe_id: str):
    """Serve ảnh keyframe: data/keyframes/Keyframes_Lxx/{video_id}/{keyframe_id}.jpg"""
    if not (SAFE_ID.match(video_id) and SAFE_ID.match(keyframe_id)):
        raise HTTPException(status_code=400, detail="Invalid video/keyframe id.")

    keyframes_dir = app.state.config["paths"]["keyframes_dir"]
    candidates = glob(os.path.join(keyframes_dir, "*", video_id))
    video_dir = candidates[0] if candidates else os.path.join(keyframes_dir, video_id)

    image_path = os.path.join(video_dir, f"{keyframe_id}.jpg")
    if not os.path.exists(image_path):
        raise HTTPException(status_code=404, detail=f"Keyframe image not found: {video_id}/{keyframe_id}")
    return FileResponse(image_path, media_type="image/jpeg")


if __name__ == "__main__":
    import uvicorn
    # Truyền thẳng app object (không dùng chuỗi "module:app") để chạy được từ bất kỳ thư mục nào
    uvicorn.run(app, host="0.0.0.0", port=8000)
