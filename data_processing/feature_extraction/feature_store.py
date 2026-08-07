"""Cách đọc/ghi file .npy đặc trưng — NƠI KHAI BÁO DUY NHẤT.

Tách khỏi run_embedding.py vì hai lý do:

1. run_embedding.py import torch/open_clip ngay ở đầu file nên chỉ chạy được
   trong venv_embedding. Quy tắc "thế nào là một .npy đã hoàn chỉnh" thì chỉ cần
   numpy — nhốt nó trong đó đồng nghĩa với không kiểm thử được, mà đây đúng là
   quy tắc đã gây ra lỗi 9 video mất khỏi index siglip2.
2. Quy tắc đó phải giống hệt nhau ở bên ghi (run_embedding) và bên kiểm
   (check_integrity), y như luật lấy mẫu keyframe ở shot_sampling.py.
"""
import os

import numpy as np


def feature_path(features_dir: str, model: str, video_id: str) -> str:
    return os.path.join(features_dir, model, f"{video_id}.npy")


def is_complete(npy_path: str, n_keyframes: int) -> bool:
    """.npy hiện có ĐÃ là kết quả đầy đủ cho đúng `n_keyframes` ảnh chưa?

    Điều kiện cũ trong run_embedding.py chỉ là `os.path.exists(npy_path)`. Vì
    thế một file rỗng shape (0, dim) — sinh ra khi encode chạy lúc keyframe chưa
    trích xong — được coi là "đã xong" VĨNH VIỄN: mọi lần chạy sau đều đếm nó
    vào `skipped`, và video đó lặng lẽ biến mất khỏi FAISS index.

    n_keyframes == 0 luôn trả về False: không có ảnh thì không có kết quả nào
    đáng gọi là hoàn chỉnh, kể cả một mảng rỗng.
    """
    if n_keyframes <= 0 or not os.path.exists(npy_path):
        return False
    try:
        arr = np.load(npy_path, mmap_mode="r")
    except Exception:
        return False           # file hỏng thì coi như chưa có, sẽ encode lại
    return arr.ndim == 2 and arr.shape[0] == n_keyframes


def save_atomic(npy_path: str, array: np.ndarray) -> None:
    """Ghi .npy nguyên tử: qua file tạm rồi rename.

    rename trên cùng filesystem là thao tác nguyên tử, nên tên chính thức hoặc
    chưa tồn tại, hoặc trỏ tới một file đầy đủ — không bao giờ là một .npy cụt
    do tiến trình bị kill giữa lúc ghi.
    """
    os.makedirs(os.path.dirname(npy_path), exist_ok=True)
    tmp_path = f"{npy_path}.tmp.npy"     # np.save giữ nguyên tên khi đã có .npy
    np.save(tmp_path, array)
    os.replace(tmp_path, npy_path)
