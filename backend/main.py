"""Điểm khởi chạy backend AIC 2026.

    ./scripts/run_backend.sh                 # cách chạy chuẩn (tự activate venv)
    uvicorn backend.main:app --workers 1     # chạy tay — LUÔN LUÔN 1 worker

Mở http://localhost:8000/docs để thử API ngay trên trình duyệt (Swagger UI được sinh
tự động từ các Pydantic schema, dùng làm UI tạm khi frontend chưa xong).
"""
import logging

# PHẢI import trước mọi thứ khác: get_settings() dựng CUDA_VISIBLE_DEVICES và HF_HOME,
# mà hai biến đó chỉ có tác dụng nếu được set TRƯỚC khi torch/transformers được import.
from backend.core.config import get_settings

get_settings()

from fastapi import FastAPI  # noqa: E402
from fastapi.middleware.cors import CORSMiddleware  # noqa: E402

from backend.api.routes import search  # noqa: E402
from backend.core.lifespan import lifespan  # noqa: E402

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    datefmt="%H:%M:%S",
)

app = FastAPI(
    title="AIC 2026 — Search Backend",
    description=(
        "Known-Item Search trên tập video AIC. Kênh hiện có: tìm kiếm ngữ nghĩa bằng "
        "vector (FAISS). Kênh chữ (OCR/caption/ASR qua FTS5) sẽ được bật khi có dữ liệu."
    ),
    version="0.1.0",
    lifespan=lifespan,
)

# Frontend chạy ở origin khác (Vite/React dev server) nên cần CORS. Mở rộng vì hệ
# thống chỉ chạy trong mạng nội bộ lúc thi, không phơi ra Internet.
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=False,
    allow_methods=["*"],
    allow_headers=["*"],
)

app.include_router(search.router, prefix="/api/v1", tags=["search"])


@app.get("/", include_in_schema=False)
def root():
    return {
        "service": "AIC 2026 Search Backend",
        "docs": "/docs",
        "endpoints": [
            "POST /api/v1/search/kis",
            "POST /api/v1/search/similar",
            "GET  /api/v1/playlists",
            "GET  /api/v1/models",
            "GET  /api/v1/health",
            "GET  /api/v1/keyframes/{video_id}/{keyframe_name}.jpg",
            "GET  /api/v1/videos/{video_id}",
            "GET  /api/v1/videos/{video_id}/stream",
            "POST /api/v1/submissions",
            "GET  /api/v1/submissions",
        ],
    }
