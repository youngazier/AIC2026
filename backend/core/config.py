"""Cấu hình lõi của backend.

Đây là module ĐƯỢC IMPORT ĐẦU TIÊN trong mọi tiến trình backend, vì `get_settings()`
có nhiệm vụ dựng biến môi trường (CUDA_VISIBLE_DEVICES, HF_HOME) TRƯỚC khi bất kỳ
module nào import torch/transformers — sau khi torch đã import thì set mấy biến đó
không còn tác dụng.

Mọi đường dẫn vẫn lấy từ config.yaml (nguồn khai báo duy nhất), module này chỉ bọc
lại cho gọn và cho phép ghi đè bằng biến môi trường khi dev:

    AIC_BACKEND_MODELS="siglip2"   # chỉ nạp 1 model cho nhanh
    AIC_SKIP_ENCODERS=1            # không nạp text encoder (máy không GPU, chỉ dev tầng DB)
    AIC_CONFIG=/duong/dan/khac/config.yaml
"""
import os
import sys
from functools import lru_cache

# Cho phép chạy `uvicorn backend.main:app` từ bất kỳ thư mục nào mà vẫn import được
# utils.* và data_processing.* (các module dùng chung với pipeline tiền xử lý).
PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from utils.config_loader import load_config  # noqa: E402

_TRUTHY = {"1", "true", "yes", "on"}


class Settings:
    """Ảnh chụp cấu hình backend tại thời điểm khởi động."""

    def __init__(self):
        cfg = load_config()
        paths = cfg["paths"]
        backend = cfg.get("backend", {}) or {}

        self.raw = cfg
        self.project_root = PROJECT_ROOT

        # --- Đường dẫn ---
        # Suy ra từ processed_dir nếu config.yaml chưa khai db_path, để backend vẫn
        # chạy được trên các config rút gọn (VD dataset giả của bộ test).
        self.db_path: str = paths.get("db_path") or os.path.join(
            paths["processed_dir"], "metadata.db"
        )
        self.index_dir: str = paths["index_dir"]
        self.keyframes_dir: str = paths["keyframes_dir"]
        self.map_keyframes_dir: str = paths["map_keyframes_dir"]
        self.data_dir: str = paths["data_dir"]
        self.hf_cache_dir = paths.get("huggingface_cache_dir")
        # Video gốc: videos/Videos_LXX/LXX_VXXX.mp4 — dùng cho endpoint xem lại video.
        self.videos_dir: str = paths.get("videos_dir") or os.path.join(
            paths["data_dir"], "videos"
        )
        # Nơi ghi file CSV nộp bài. Suy ra từ project_root nếu config chưa khai, để
        # backend không chết chỉ vì thiếu một khai báo tùy chọn.
        self.submissions_dir: str = paths.get("submissions_dir") or os.path.join(
            PROJECT_ROOT, "submissions"
        )

        # --- Model ---
        env_models = os.environ.get("AIC_BACKEND_MODELS")
        self.models: list = (
            [m.strip() for m in env_models.split(",") if m.strip()]
            if env_models
            else list(backend.get("models") or ["siglip2"])
        )
        self.model_params: dict = cfg.get("models", {}) or {}
        self.skip_encoders: bool = os.environ.get("AIC_SKIP_ENCODERS", "").lower() in _TRUTHY

        # --- Tham số tìm kiếm ---
        self.gpu_device: str = str(backend.get("gpu_device", "7"))
        self.use_mmap: bool = bool(backend.get("use_mmap", False))
        self.default_topk: int = int(backend.get("default_topk", 100))
        self.max_topk: int = int(backend.get("max_topk", 500))
        self.overfetch: int = int(backend.get("overfetch", 3))
        self.k_rrf: int = int(backend.get("k_rrf", 60))

    def configure_runtime_env(self) -> None:
        """Set các biến môi trường phải có TRƯỚC khi import torch/transformers.

        CUDA_VISIBLE_DEVICES chỉ được set khi người dùng CHƯA set sẵn — ai chạy
        `CUDA_VISIBLE_DEVICES=3 uvicorn ...` thì lựa chọn của họ được tôn trọng.
        """
        os.environ.setdefault("CUDA_VISIBLE_DEVICES", self.gpu_device)
        if self.hf_cache_dir:
            os.environ.setdefault("HF_HOME", self.hf_cache_dir)


@lru_cache(maxsize=1)
def get_settings() -> Settings:
    settings = Settings()
    settings.configure_runtime_env()
    return settings
