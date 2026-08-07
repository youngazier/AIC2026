"""Kiểm thử check_integrity.py bằng phương pháp TIÊM LỖI (fault injection).

Một trình kiểm tra không bao giờ báo lỗi thì vô dụng, mà điều đó lại không nhìn
ra được từ việc chạy nó trên dữ liệu thật đang PASS. Cách duy nhất biết nó có
tác dụng là cố tình làm hỏng dữ liệu rồi xem nó có kêu không.

Mỗi test ở đây làm đúng ba việc:
    1. lấy một bản sao dataset giả đang sạch  (fixture `dataset`)
    2. tiêm MỘT lỗi cụ thể
    3. khẳng định check_integrity trả về đúng mã lỗi đó và exit code 1

Chạy: pytest tests/test_check_integrity.py -v
"""
import json
import os
import shutil

import numpy as np
import pytest

from conftest import (
    MODEL,
    VIDEOS,
    build_index,
    codes,
    jpg_path,
    keyframe_dir,
    map_path,
    npy_path,
    run_verify,
    scenes_path,
    seal,
    video_path,
    write_jpg,
)

VID = "L01_V001"        # 4 keyframe: n = 1..4


# ---------------------------------------------------------------------------
# Mốc chuẩn: dataset sạch phải PASS. Nếu test này hỏng thì mọi test dưới vô nghĩa
# (chúng sẽ "bắt được lỗi" chỉ vì nền vốn đã bẩn).
# ---------------------------------------------------------------------------
def test_dataset_sach_thi_pass(dataset):
    rc, report, out = run_verify(dataset)
    assert rc == 0, f"dataset giả lẽ ra phải sạch:\n{out}"
    assert report["n_errors"] == 0
    assert codes(report, "ERROR") == set()
    assert report["totals"]["expected_keyframes"] == report["totals"]["actual_keyframes"] == 8


def test_bao_cao_liet_ke_dung_so_video(dataset):
    _, report, _ = run_verify(dataset)
    assert report["n_videos_checked"] == len(VIDEOS)
    assert set(report["per_video"]) == set(VIDEOS)


# ---------------------------------------------------------------------------
# STAGE manifest — toàn vẹn dữ liệu nguồn
# ---------------------------------------------------------------------------
def test_video_bien_mat_sau_khi_da_vao_manifest(dataset):
    run_verify(dataset)                       # lần 1: dựng manifest
    video_path(dataset, VID).unlink()         # ai đó xoá nhầm video gốc
    rc, report, _ = run_verify(dataset)
    assert "VIDEO_DISAPPEARED" in codes(report, "ERROR")
    assert rc == 1


def test_video_bi_doi_noi_dung(dataset):
    run_verify(dataset)
    with open(video_path(dataset, VID), "ab") as f:
        f.write(b"\x00" * 4096)               # kích thước đổi = nội dung đổi
    rc, report, _ = run_verify(dataset)
    assert "VIDEO_SIZE_CHANGED" in codes(report, "ERROR")
    assert rc == 1


def test_video_hong_khong_decode_duoc(dataset):
    """Checksum khớp không có nghĩa là file dùng được — cần --deep-decode."""
    video_path(dataset, VID).write_bytes(b"khong phai mp4" * 500)
    rc, report, _ = run_verify(dataset, "--deep-decode")
    assert {"PROBE_FAILED", "DECODE_ERROR"} & codes(report, "ERROR")
    assert rc == 1


def test_manifest_ghi_lai_sha256_khi_bat_hash(dataset):
    run_verify(dataset, "--hash")
    manifest = json.loads((dataset / "data_processed" / "_manifest" / "videos.json").read_text())
    assert all(len(e["sha256"]) == 64 for e in manifest["videos"].values())


# ---------------------------------------------------------------------------
# STAGE transnet
# ---------------------------------------------------------------------------
def test_thieu_scenes_file(dataset):
    scenes_path(dataset, VID).unlink()
    rc, report, _ = run_verify(dataset)
    assert "SCENES_MISSING" in codes(report, "ERROR")
    assert rc == 1


def test_thieu_predictions_file(dataset):
    (dataset / "data_processed" / "transnetv2" / "predictions" / f"{VID}.predictions.txt").unlink()
    rc, report, _ = run_verify(dataset)
    assert "PRED_MISSING" in codes(report, "ERROR")
    assert rc == 1


def test_scenes_bi_cat_cut(dataset):
    """TransNetV2 dừng giữa chừng nhưng vẫn ghi file — số shot vẫn 'hợp lệ'."""
    scenes_path(dataset, VID).write_text("0 99\n", encoding="utf-8")
    rc, report, _ = run_verify(dataset)
    assert "SCENES_TRUNCATED" in codes(report, "ERROR")
    assert rc == 1


def test_shot_chong_lan(dataset):
    scenes_path(dataset, VID).write_text("0 249\n200 299\n", encoding="utf-8")
    rc, report, _ = run_verify(dataset)
    assert "SHOT_OVERLAP" in codes(report, "ERROR")
    assert rc == 1


def test_ho_vai_frame_giua_shot_khong_bi_bao_dong(dataset):
    """TransNetV2 loại 1-2 frame chuyển cảnh — đó là bình thường, không phải lỗi."""
    scenes_path(dataset, VID).write_text("0 248\n250 299\n", encoding="utf-8")
    seal(dataset, VID)                                  # scenes đổi -> niêm phong lại
    _, report, _ = run_verify(dataset, "--stage", "transnet")
    assert "SHOT_GAP" not in codes(report)


def test_ho_ca_doan_dai_thi_bi_canh_bao(dataset):
    scenes_path(dataset, VID).write_text("0 99\n200 299\n", encoding="utf-8")
    seal(dataset, VID)
    _, report, _ = run_verify(dataset, "--stage", "transnet")
    assert "SHOT_GAP" in codes(report, "WARN")


# ---------------------------------------------------------------------------
# STAGE keyframe — map csv
# ---------------------------------------------------------------------------
def test_thieu_map_csv(dataset):
    map_path(dataset, VID).unlink()
    rc, report, _ = run_verify(dataset)
    assert "MAP_MISSING" in codes(report, "ERROR")
    assert rc == 1


def test_thieu_mot_dong_csv(dataset):
    rows = map_path(dataset, VID).read_text().splitlines()
    map_path(dataset, VID).write_text("\n".join(rows[:2] + rows[3:]) + "\n", encoding="utf-8")
    rc, report, _ = run_verify(dataset)
    assert "MAP_ROW_MISSING" in codes(report, "ERROR")
    assert rc == 1


def test_thua_mot_dong_csv(dataset):
    with open(map_path(dataset, VID), "a") as f:
        f.write("99,99,1.0,25.0,50\n")
    rc, report, _ = run_verify(dataset)
    assert "MAP_ROW_EXTRA" in codes(report, "ERROR")
    assert rc == 1


def test_frame_idx_sai(dataset):
    """Lỗi mà mọi phép ĐẾM đều bỏ lọt: số dòng đúng, nội dung sai."""
    rows = map_path(dataset, VID).read_text().splitlines()
    cols = rows[1].split(",")
    cols[4] = str(int(cols[4]) + 7)
    rows[1] = ",".join(cols)
    map_path(dataset, VID).write_text("\n".join(rows) + "\n", encoding="utf-8")
    rc, report, _ = run_verify(dataset)
    assert "MAP_FRAME_IDX_MISMATCH" in codes(report, "ERROR")
    assert rc == 1


def test_frame_idx_vuot_ngoai_video(dataset):
    rows = map_path(dataset, VID).read_text().splitlines()
    cols = rows[1].split(",")
    cols[4] = "999999"
    rows[1] = ",".join(cols)
    map_path(dataset, VID).write_text("\n".join(rows) + "\n", encoding="utf-8")
    _, report, _ = run_verify(dataset)
    assert "MAP_FRAME_OUT_OF_RANGE" in codes(report, "ERROR")


def test_fps_trong_csv_lech_ffprobe(dataset):
    text = map_path(dataset, VID).read_text().replace("25.0", "30.0")
    map_path(dataset, VID).write_text(text, encoding="utf-8")
    _, report, _ = run_verify(dataset)
    assert "MAP_FPS_MISMATCH" in codes(report, "WARN")


def test_header_csv_sai(dataset):
    rows = map_path(dataset, VID).read_text().splitlines()
    rows[0] = "a,b,c,d,e"
    map_path(dataset, VID).write_text("\n".join(rows) + "\n", encoding="utf-8")
    _, report, _ = run_verify(dataset)
    assert "MAP_HEADER" in codes(report, "ERROR")


# ---------------------------------------------------------------------------
# STAGE keyframe — ảnh .jpg
# ---------------------------------------------------------------------------
def test_thieu_anh(dataset):
    jpg_path(dataset, VID, 2).unlink()
    rc, report, _ = run_verify(dataset)
    assert "KF_MISSING" in codes(report, "ERROR")
    assert rc == 1


def test_anh_thua(dataset):
    write_jpg(jpg_path(dataset, VID, 99), seed=1)
    rc, report, _ = run_verify(dataset)
    assert "KF_EXTRA" in codes(report, "ERROR")
    assert rc == 1


def test_thieu_va_thua_bu_tru_nhau_van_bi_bat(dataset):
    """Chính là lỗ hổng của bản cũ: so số ĐẾM thì hai lỗi này triệt tiêu nhau."""
    jpg_path(dataset, VID, 2).unlink()
    write_jpg(jpg_path(dataset, VID, 99), seed=1)
    rc, report, _ = run_verify(dataset)
    assert {"KF_MISSING", "KF_EXTRA"} <= codes(report, "ERROR")
    assert rc == 1


def test_anh_ghi_do_dang(dataset):
    p = jpg_path(dataset, VID, 1)
    p.write_bytes(p.read_bytes()[:200])
    rc, report, _ = run_verify(dataset)
    assert "KF_TRUNCATED" in codes(report, "ERROR")
    assert rc == 1


def test_anh_du_lon_nhung_hong(dataset):
    """Kích thước hợp lệ, mở ra mới biết hỏng — stat không đủ."""
    jpg_path(dataset, VID, 1).write_bytes(os.urandom(5000))
    rc, report, _ = run_verify(dataset)
    assert "KF_CORRUPT" in codes(report, "ERROR")
    assert rc == 1


def test_fast_bo_qua_kiem_tra_noi_dung_anh(dataset):
    """--fast là đánh đổi có ý thức: nhanh hơn nhưng không mở ảnh ra xem."""
    jpg_path(dataset, VID, 1).write_bytes(os.urandom(5000))
    _, report, _ = run_verify(dataset, "--fast")
    assert "KF_CORRUPT" not in codes(report)


def test_anh_den_tuyen(dataset):
    write_jpg(jpg_path(dataset, VID, 1), seed=0, blank=True)
    _, report, _ = run_verify(dataset, "--deep-jpg")
    assert "KF_BLANK" in codes(report, "WARN")


def test_thieu_ca_thu_muc_keyframe(dataset):
    shutil.rmtree(keyframe_dir(dataset, VID))
    rc, report, _ = run_verify(dataset)
    assert "KF_DIR_MISSING" in codes(report, "ERROR")
    assert rc == 1


# ---------------------------------------------------------------------------
# STAGE keyframe — sentinel _SUCCESS
# ---------------------------------------------------------------------------
def test_thieu_sentinel_chi_la_canh_bao(dataset):
    """Dữ liệu trích trước khi có cơ chế sentinel: đáng lưu ý, chưa phải lỗi."""
    (keyframe_dir(dataset, VID) / "_SUCCESS").unlink()
    rc, report, _ = run_verify(dataset)
    assert "SENTINEL_MISSING" in codes(report, "WARN")
    assert rc == 0


def test_thieu_sentinel_thanh_loi_khi_strict(dataset):
    (keyframe_dir(dataset, VID) / "_SUCCESS").unlink()
    rc, _, _ = run_verify(dataset, "--strict")
    assert rc == 1


def test_sentinel_lech_so_luong(dataset):
    seal(dataset, VID, n_keyframes=999)
    rc, report, _ = run_verify(dataset)
    assert "SENTINEL_STALE" in codes(report, "ERROR")
    assert rc == 1


def test_scenes_doi_sau_khi_trich_keyframe(dataset):
    """TransNetV2 chạy lại, ranh giới shot đổi -> keyframe cũ đã hết hạn.

    Không có sentinel thì trường hợp này vô hình: thư mục vẫn đầy ảnh, csv vẫn
    khớp ảnh, chỉ có điều chúng thuộc về một phiên bản scenes.txt khác.
    """
    scenes_path(dataset, VID).write_text("0 249\n250 298\n", encoding="utf-8")
    rc, report, _ = run_verify(dataset)
    assert "SENTINEL_STALE" in codes(report, "ERROR")
    assert rc == 1


# ---------------------------------------------------------------------------
# STAGE features
# ---------------------------------------------------------------------------
def test_thieu_npy(dataset):
    npy_path(dataset, VID).unlink()
    rc, report, _ = run_verify(dataset)
    assert "NPY_MISSING" in codes(report, "ERROR")
    assert rc == 1


def test_npy_rong(dataset):
    """Đúng lỗi có thật trong dataset của dự án: 9 video mất hẳn khỏi index."""
    np.save(npy_path(dataset, VID), np.empty((0, 8), dtype="float32"))
    rc, report, _ = run_verify(dataset)
    assert {"NPY_EMPTY", "NPY_COUNT_MISMATCH"} <= codes(report, "ERROR")
    assert rc == 1


def test_npy_thieu_vector(dataset):
    arr = np.load(npy_path(dataset, VID))
    np.save(npy_path(dataset, VID), arr[:-1])
    rc, report, _ = run_verify(dataset)
    assert "NPY_COUNT_MISMATCH" in codes(report, "ERROR")
    assert rc == 1


def test_npy_chua_nan(dataset):
    arr = np.load(npy_path(dataset, VID))
    arr[0, 0] = np.nan
    np.save(npy_path(dataset, VID), arr)
    rc, report, _ = run_verify(dataset)
    assert "NPY_NOT_FINITE" in codes(report, "ERROR")
    assert rc == 1


def test_npy_co_vector_toan_khong(dataset):
    """Ảnh nguồn hỏng mà model vẫn trả về output — không exception nào cả."""
    arr = np.load(npy_path(dataset, VID))
    arr[1] = 0.0
    np.save(npy_path(dataset, VID), arr)
    rc, report, _ = run_verify(dataset)
    assert "NPY_ZERO_VECTOR" in codes(report, "ERROR")
    assert rc == 1


def test_so_chieu_khong_dong_nhat_giua_cac_video(dataset):
    """Trộn hai phiên bản model khác nhau trong cùng một thư mục features."""
    arr = np.load(npy_path(dataset, "L01_V002"))
    np.save(npy_path(dataset, "L01_V002"), np.hstack([arr, arr]))
    _, report, _ = run_verify(dataset)
    assert "DIM_INCONSISTENT" in codes(report, "ERROR")


def test_loi_keyframe_khong_de_ra_loi_features(dataset):
    """Lỗi hệ quả bị hạ xuống WARN để ERROR dồn vào nguyên nhân gốc."""
    shutil.rmtree(keyframe_dir(dataset, VID))
    _, report, _ = run_verify(dataset)
    cascaded = [f for f in report["findings"] if f["code"] == "NPY_COUNT_MISMATCH"]
    assert cascaded and all(f["severity"] == "WARN" for f in cascaded)


# ---------------------------------------------------------------------------
# STAGE index — cần faiss nên chạy bằng python của venv_embedding
# ---------------------------------------------------------------------------
@pytest.fixture
def indexed(dataset):
    if not build_index(dataset):
        pytest.skip("chưa có venv_embedding — bỏ qua các test cần faiss")
    return dataset


def _verify_index(root, *extra):
    from conftest import VENV_PY
    return run_verify(root, "--stage", "index", "--fast", *extra, python=VENV_PY)


def test_index_dung_thi_pass(indexed):
    rc, report, out = _verify_index(indexed)
    assert rc == 0, out
    assert codes(report, "ERROR") == set()


def test_hoan_vi_hai_keyframe_lien_nhau_bi_bat(indexed):
    """Phép kiểm quan trọng nhất của stage index.

    Hoán vị hai keyframe cạnh nhau trong metadata KHÔNG làm đổi số vector, số
    video, số chiều hay bất kỳ con số nào — mọi phép đếm vẫn PASS. Chỉ có việc
    đối chiếu từng vector với .npy mới thấy. Và phải quét TOÀN BỘ: lấy mẫu vài
    trăm vị trí sẽ bỏ lọt rồi in ra chữ PASS.
    """
    meta_file = indexed / "data_processed" / "index" / f"{MODEL}_keyframe_metadata.npy"
    meta = np.load(meta_file, allow_pickle=True)
    meta[[0, 1]] = meta[[1, 0]]
    np.save(meta_file, meta)

    rc, report, _ = _verify_index(indexed)
    assert "INDEX_MISALIGNED" in codes(report, "ERROR")
    assert rc == 1


def test_video_co_npy_nhung_khong_co_trong_index(indexed):
    meta_file = indexed / "data_processed" / "index" / f"{MODEL}_keyframe_metadata.npy"
    meta = np.load(meta_file, allow_pickle=True)
    keep = np.array([row for row in meta if row[0] != "L01_V002"])
    np.save(meta_file, keep)
    rc, report, _ = _verify_index(indexed, "--stage", "features")
    assert {"INDEX_META_DESYNC", "INDEX_VIDEO_NOT_COVERED"} & codes(report, "ERROR")
    assert rc == 1


def test_metadata_lech_do_dai_so_voi_index(indexed):
    meta_file = indexed / "data_processed" / "index" / f"{MODEL}_keyframe_metadata.npy"
    meta = np.load(meta_file, allow_pickle=True)
    np.save(meta_file, np.vstack([meta, meta[:1]]))
    rc, report, _ = _verify_index(indexed)
    assert "INDEX_META_DESYNC" in codes(report, "ERROR")
    assert rc == 1


def test_index_trung_lap_keyframe(indexed):
    meta_file = indexed / "data_processed" / "index" / f"{MODEL}_keyframe_metadata.npy"
    meta = np.load(meta_file, allow_pickle=True)
    meta[1] = meta[0]
    np.save(meta_file, meta)
    rc, report, _ = _verify_index(indexed)
    assert "INDEX_DUPLICATE" in codes(report, "ERROR")
    assert rc == 1


def test_thieu_file_index(indexed):
    (indexed / "data_processed" / "index" / f"{MODEL}_keyframe_embedding.index").unlink()
    rc, report, _ = _verify_index(indexed)
    assert "INDEX_MISSING" in codes(report, "ERROR")
    assert rc == 1
