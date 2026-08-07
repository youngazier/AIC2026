"""Lớp truy cập FAISS index (xem docs/database_architecture.md mục 1 & 3.1).

Một FAISSRepository quản lý TOÀN BỘ index của MỘT model: 1 index global + N index
playlist. Layout file do build_index.py quy định:

    {index_dir}/{model}_keyframe_embedding.index          (+ _keyframe_metadata.npy)
    {index_dir}/playlist/{model}_{playlist}.index         (+ _keyframe_metadata.npy)

Mỗi index có file metadata .npy SONG HÀNH RIÊNG, vì thứ tự vector trong index global
khác với trong index playlist. FAISS id chỉ có nghĩa trong phạm vi index sinh ra nó.
"""
import logging
import os
import time
from glob import glob

import faiss
import numpy as np

logger = logging.getLogger(__name__)

GLOBAL_SCOPE = "global"


class FAISSRepository:
    """Quản lý index của một model. Chỉ đọc, an toàn khi search từ nhiều thread."""

    def __init__(self, model_name: str, index_dir: str, use_mmap: bool = False):
        self.model_name = model_name
        self.index_dir = index_dir
        self.indexes: dict = {}    # scope -> faiss.Index  (scope = "global" | tên playlist)
        self.metadata: dict = {}   # scope -> mảng numpy các cặp (video_id, keyframe_name)
        self._pos_cache: dict = {}  # scope -> {(video_id, kf_name): faiss_id}, dựng khi cần

        io_flag = faiss.IO_FLAG_MMAP if use_mmap else 0

        global_index = os.path.join(index_dir, f"{model_name}_keyframe_embedding.index")
        global_meta = os.path.join(index_dir, f"{model_name}_keyframe_metadata.npy")
        if not (os.path.exists(global_index) and os.path.exists(global_meta)):
            raise FileNotFoundError(
                f"Thiếu index global của '{model_name}' trong {index_dir}. "
                f"Chạy data_processing/feature_extraction/build_index.py --model {model_name}."
            )
        self._load(GLOBAL_SCOPE, global_index, global_meta, io_flag)

        prefix = f"{model_name}_"
        for index_path in sorted(glob(os.path.join(index_dir, "playlist", f"{prefix}*.index"))):
            playlist = os.path.basename(index_path)[len(prefix): -len(".index")]
            meta_path = os.path.join(
                index_dir, "playlist", f"{prefix}{playlist}_keyframe_metadata.npy"
            )
            if not os.path.exists(meta_path):
                logger.warning("[%s] %s thiếu file metadata song hành — bỏ qua.", model_name, playlist)
                continue
            self._load(playlist, index_path, meta_path, io_flag)

    def _load(self, scope: str, index_path: str, meta_path: str, io_flag: int) -> None:
        started = time.perf_counter()
        index = faiss.read_index(index_path, io_flag)
        meta = np.load(meta_path, allow_pickle=True)
        if index.ntotal != len(meta):
            raise ValueError(
                f"[{self.model_name}/{scope}] index có {index.ntotal} vector nhưng metadata có "
                f"{len(meta)} dòng. Hai file lệch pha — build lại index trước khi chạy backend."
            )
        self.indexes[scope] = index
        self.metadata[scope] = meta
        logger.info(
            "[%s/%s] %d vector, dim=%d (%.1fs)",
            self.model_name, scope, index.ntotal, index.d, time.perf_counter() - started,
        )

    # --- Thông tin ---------------------------------------------------------- #
    @property
    def playlists(self) -> list:
        """Registry động: dùng để validate request và đổ dropdown cho UI."""
        return sorted(s for s in self.indexes if s != GLOBAL_SCOPE)

    @property
    def dim(self) -> int:
        return self.indexes[GLOBAL_SCOPE].d

    @property
    def ntotal(self) -> int:
        return self.indexes[GLOBAL_SCOPE].ntotal

    def has_scope(self, playlist=None) -> bool:
        return (playlist or GLOBAL_SCOPE) in self.indexes

    # --- Tìm kiếm ----------------------------------------------------------- #
    def position_map(self, scope: str) -> dict:
        """{(video_id, kf_name): faiss_id} của một scope, dựng LƯỜI.

        Chỉ cần cho chế độ lọc cứng bằng IDSelectorBatch (Chặng B). Dựng sẵn cho mọi
        scope lúc startup sẽ tốn vài trăm MB RAM cho thứ chưa dùng đến.
        """
        if scope not in self._pos_cache:
            self._pos_cache[scope] = {
                (str(v), str(k)): i for i, (v, k) in enumerate(self.metadata[scope])
            }
        return self._pos_cache[scope]

    def get_vector(self, video_id: str, keyframe_name: str, playlist=None):
        """Lấy lại vector ĐÃ LƯU của một keyframe. Trả None nếu frame không có trong scope.

        Dùng cho tìm-ảnh-giống-ảnh: index là IndexFlatL2 nên lưu nguyên vector gốc,
        `reconstruct()` trả về đúng vector lúc build index. Nhờ vậy không phải nạp lại
        ảnh JPEG rồi đẩy qua GPU — vừa nhanh hơn vài trăm lần, vừa đảm bảo vector truy
        vấn KHỚP TUYỆT ĐỐI với vector trong index (encode lại có thể lệch chút ít do
        resize/interpolation khác lần trước).
        """
        scope = playlist or GLOBAL_SCOPE
        if scope not in self.indexes:
            raise KeyError(f"Model '{self.model_name}' không có index cho playlist '{scope}'")

        faiss_id = self.position_map(scope).get((video_id, keyframe_name))
        if faiss_id is None:
            return None
        return self.indexes[scope].reconstruct(int(faiss_id)).reshape(1, -1)

    def search(self, query_vector: np.ndarray, top_k: int = 100,
               playlist=None, allowed_pairs=None) -> list:
        """Trả về [(video_id, keyframe_name, distance)] sắp tăng dần theo distance.

        LƯU Ý VỀ THANG ĐIỂM: index là IndexFlatL2 nên FAISS trả về L2 BÌNH PHƯƠNG,
        CÀNG NHỎ CÀNG TỐT. Vector đã chuẩn hóa (norm=1) nên quy đổi được sang cosine:
        cosine = 1 - distance / 2.

        - playlist=None -> search index global; ngược lại search index của playlist đó.
        - allowed_pairs -> tập (video_id, kf_name) từ lọc cứng OCR/Object, được đổi
          sang FAISS id CỦA ĐÚNG index đang search rồi đưa vào IDSelector.
        """
        scope = playlist or GLOBAL_SCOPE
        if scope not in self.indexes:
            raise KeyError(f"Model '{self.model_name}' không có index cho playlist '{scope}'")

        index, meta = self.indexes[scope], self.metadata[scope]

        query = np.ascontiguousarray(query_vector, dtype=np.float32)
        if query.ndim == 1:
            query = query.reshape(1, -1)
        if query.shape[1] != index.d:
            raise ValueError(
                f"Query vector {query.shape[1]} chiều nhưng index '{self.model_name}' "
                f"cần {index.d} chiều."
            )

        top_k = min(top_k, index.ntotal)
        if top_k <= 0:
            return []

        if allowed_pairs is not None:
            pos = self.position_map(scope)
            ids = np.array([pos[p] for p in allowed_pairs if p in pos], dtype="int64")
            if ids.size == 0:
                return []
            params = faiss.SearchParameters(sel=faiss.IDSelectorBatch(ids))
            distances, ids_out = index.search(query, min(top_k, ids.size), params=params)
        else:
            distances, ids_out = index.search(query, top_k)

        results = []
        for distance, faiss_id in zip(distances[0], ids_out[0]):
            if faiss_id == -1:  # FAISS trả -1 khi không đủ ứng viên
                continue
            video_id, keyframe_name = meta[faiss_id]
            results.append((str(video_id), str(keyframe_name), float(distance)))
        return results


def discover_models(index_dir: str) -> list:
    """Các model đang có index global trong thư mục index (không hard-code danh sách)."""
    suffix = "_keyframe_embedding.index"
    return sorted(
        os.path.basename(p)[: -len(suffix)]
        for p in glob(os.path.join(index_dir, f"*{suffix}"))
    )
