"""Vòng đời ứng dụng: nạp FAISS index + text encoder ĐÚNG MỘT LẦN lúc startup.

Đây là lý do backend phải chạy `--workers 1`: mỗi worker là một tiến trình riêng và
sẽ nạp lại toàn bộ index (vài GB RAM) cùng một bản model trên GPU. Song song hóa đã
có threadpool của FastAPI lo (FAISS/PyTorch đều nhả GIL khi tính toán).
"""
import importlib
import logging
import time
from contextlib import asynccontextmanager

from fastapi import FastAPI

from backend.core.config import get_settings
from backend.core.database import SessionLocal, database_exists
from backend.repositories.faiss_repo import FAISSRepository, discover_models

logger = logging.getLogger(__name__)

# Model nào có TEXT ENCODER (encode câu truy vấn thành vector).
#
# DINOv3 CỐ Ý VẮNG MẶT: nó là model thị giác tự giám sát, `encode_text()` trong
# dinov3_embedding.py chỉ có `pass` (trả None). Index DINOv3 vẫn được nạp vì dùng
# được cho tìm-ảnh-giống-ảnh, nhưng tìm bằng text thì phải từ chối thẳng thay vì
# trả về kết quả rác từ một vector None.
TEXT_ENCODER_SPECS = {
    "siglip2": ("data_processing.feature_extraction.siglip2_embedding", "SigLIP2Embedding"),
    "beit3": ("data_processing.feature_extraction.beit3_embedding", "Beit3Embedding"),
}


def _build_encoder(model: str, settings):
    """Import và khởi tạo text encoder. Import LƯỜI để máy không GPU vẫn chạy được backend."""
    module_path, class_name = TEXT_ENCODER_SPECS[model]
    cls = getattr(importlib.import_module(module_path), class_name)

    params = settings.model_params.get(model, {}) or {}
    if model == "siglip2":
        return cls(
            model_name=params.get("model_name", "ViT-gopt-16-SigLIP2-384"),
            pretrained=params.get("pretrained", "webli"),
        )
    return cls()


def load_resources(app: FastAPI) -> None:
    settings = get_settings()
    app.state.settings = settings
    app.state.faiss_repos = {}
    app.state.encoders = {}

    if not database_exists():
        logger.error(
            "Không tìm thấy %s. Chạy `python scripts/init_db.py && python scripts/ingest_db.py` "
            "trước khi khởi động backend.", settings.db_path,
        )

    available = discover_models(settings.index_dir)
    requested = [m for m in settings.models if m in available]
    for missing in [m for m in settings.models if m not in available]:
        logger.warning("Model '%s' chưa có index trong %s — bỏ qua.", missing, settings.index_dir)
    if not requested:
        logger.error("Không nạp được model nào. Index hiện có: %s", available or "(trống)")

    # --- 1. FAISS index (global + playlist) ---
    for model in requested:
        started = time.perf_counter()
        try:
            app.state.faiss_repos[model] = FAISSRepository(
                model, settings.index_dir, use_mmap=settings.use_mmap
            )
            logger.info("Nạp xong index '%s' trong %.1fs", model, time.perf_counter() - started)
        except Exception:
            logger.exception("Nạp index '%s' thất bại — model này sẽ không dùng được.", model)

    # --- 2. Text encoder (chỉ cho model thực sự có) ---
    if settings.skip_encoders:
        logger.warning("AIC_SKIP_ENCODERS=1 — không nạp text encoder, chỉ dùng được tầng DB.")
    else:
        for model in app.state.faiss_repos:
            if model not in TEXT_ENCODER_SPECS:
                logger.info("Model '%s' không có text encoder (thị giác thuần) — bỏ qua.", model)
                continue
            started = time.perf_counter()
            try:
                app.state.encoders[model] = _build_encoder(model, settings)
                logger.info("Nạp xong encoder '%s' trong %.1fs", model, time.perf_counter() - started)
            except Exception:
                logger.exception("Nạp encoder '%s' thất bại — không tìm bằng text được.", model)

    # --- 3. Registry playlist: hợp của SQLite và các file index trên đĩa ---
    playlists = {}
    if database_exists():
        from backend.repositories.sqlite_repo import SQLiteRepository

        db = SessionLocal()
        try:
            for row in SQLiteRepository(db).list_playlists():
                playlists[row["playlist"]] = row["n_videos"]
        finally:
            db.close()
    app.state.playlist_videos = playlists
    app.state.indexed_playlists = {
        name for repo in app.state.faiss_repos.values() for name in repo.playlists
    }

    logger.info(
        "Sẵn sàng. Model: %s | encoder text: %s | playlist: %d",
        list(app.state.faiss_repos) or "(không có)",
        list(app.state.encoders) or "(không có)",
        len(playlists),
    )


@asynccontextmanager
async def lifespan(app: FastAPI):
    started = time.perf_counter()
    load_resources(app)
    logger.info("Startup mất %.1fs", time.perf_counter() - started)
    yield
    app.state.faiss_repos.clear()
    app.state.encoders.clear()
    logger.info("Đã giải phóng tài nguyên, tắt app.")
