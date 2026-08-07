"""Kiểm thử quy tắc lưu/nhận .npy đặc trưng (feature_store.py).

Trọng tâm là `is_complete`: chính điều kiện lỏng lẻo ở chỗ này ("file tồn tại là
đã xong") đã khiến 9 video mất hẳn khỏi FAISS index siglip2 mà không lần chạy
nào báo lỗi.

Chạy: pytest tests/test_feature_store.py -v
"""
import sys
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from data_processing.feature_extraction.feature_store import (  # noqa: E402
    feature_path,
    is_complete,
    save_atomic,
)


@pytest.fixture
def npy(tmp_path):
    return str(tmp_path / "L01_V001.npy")


class TestIsComplete:
    def test_dung_so_vector_thi_hoan_chinh(self, npy):
        np.save(npy, np.zeros((4, 8), dtype="float32"))
        assert is_complete(npy, 4)

    def test_file_rong_khong_bao_gio_hoan_chinh(self, npy):
        """Đúng dạng file đã gây ra sự cố: có mặt, đọc được, nhưng 0 vector."""
        np.save(npy, np.empty((0, 8), dtype="float32"))
        assert not is_complete(npy, 4)
        assert not is_complete(npy, 0)

    def test_thieu_vector(self, npy):
        np.save(npy, np.zeros((3, 8), dtype="float32"))
        assert not is_complete(npy, 4)

    def test_thua_vector(self, npy):
        """Dư cũng là sai: .npy còn sót của lần chạy khi có nhiều keyframe hơn."""
        np.save(npy, np.zeros((5, 8), dtype="float32"))
        assert not is_complete(npy, 4)

    def test_file_chua_ton_tai(self, npy):
        assert not is_complete(npy, 4)

    def test_file_hong_coi_nhu_chua_co(self, npy):
        Path(npy).write_bytes(b"khong phai npy")
        assert not is_complete(npy, 4)

    def test_mang_sai_so_chieu(self, npy):
        np.save(npy, np.zeros(4, dtype="float32"))     # 1 chiều
        assert not is_complete(npy, 4)

    def test_khong_co_keyframe_thi_khong_hoan_chinh(self, npy):
        """Không có ảnh thì không kết quả nào đáng gọi là xong — kể cả mảng rỗng."""
        np.save(npy, np.empty((0, 8), dtype="float32"))
        assert not is_complete(npy, 0)


class TestSaveAtomic:
    def test_ghi_va_doc_lai_dung(self, npy):
        arr = np.arange(32, dtype="float32").reshape(4, 8)
        save_atomic(npy, arr)
        assert np.array_equal(np.load(npy), arr)

    def test_khong_de_lai_file_tam(self, tmp_path, npy):
        save_atomic(npy, np.zeros((2, 8), dtype="float32"))
        assert sorted(p.name for p in tmp_path.iterdir()) == ["L01_V001.npy"]

    def test_tu_tao_thu_muc(self, tmp_path):
        target = str(tmp_path / "features" / "siglip2" / "L01_V001.npy")
        save_atomic(target, np.zeros((2, 8), dtype="float32"))
        assert Path(target).exists()

    def test_ghi_de_ban_cu(self, npy):
        save_atomic(npy, np.zeros((2, 8), dtype="float32"))
        save_atomic(npy, np.ones((5, 8), dtype="float32"))
        assert np.load(npy).shape == (5, 8)


def test_feature_path():
    assert feature_path("/data/features", "siglip2", "L01_V001") == \
        "/data/features/siglip2/L01_V001.npy"
