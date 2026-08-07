"""Kiểm thử phía SINH dữ liệu: keyframe_extraction.py.

check_integrity.py phát hiện lỗi sau khi đã xảy ra; những cơ chế ở đây ngăn lỗi
phát sinh ngay từ đầu. Ba tính chất được kiểm:

    1. Ghi nguyên tử   — kết quả đi qua vùng staging `.partial`, chỉ rename sang
                         tên chính thức khi đã đủ.
    2. Sentinel        — `_SUCCESS` ghi SAU CÙNG; không có nó nghĩa là chưa xong,
                         bất kể thư mục trông đầy đặn thế nào.
    3. Cổng nghiệm thu — trích thiếu frame thì KHÔNG niêm phong và exit != 0.

Chạy: pytest tests/test_keyframe_extraction.py -v
"""
import hashlib
import json
import os
import shutil
import subprocess
import sys

import pytest

from conftest import (
    KEYFRAME_SCRIPT,
    PROJECT_ROOT,
    codes,
    keyframe_dir,
    map_path,
    run_verify,
    scenes_path,
)

VID = "L01_V001"


def run_extract(root, *extra):
    """Chạy keyframe_extraction.py trên dataset giả -> (exit_code, output)."""
    env = dict(os.environ, AIC_CONFIG=str(root / "config.yaml"))
    proc = subprocess.run(
        [sys.executable, str(KEYFRAME_SCRIPT), "-v", VID, *extra],
        capture_output=True, text=True, env=env, cwd=str(PROJECT_ROOT),
    )
    return proc.returncode, proc.stdout + proc.stderr


def wipe_keyframes(root, vid=VID):
    """Xoá sạch đầu ra keyframe để lần chạy sau phải trích lại thật."""
    shutil.rmtree(keyframe_dir(root, vid), ignore_errors=True)
    map_path(root, vid).unlink(missing_ok=True)


def sentinel(root, vid=VID):
    path = keyframe_dir(root, vid) / "_SUCCESS"
    return json.loads(path.read_text()) if path.exists() else None


def jpgs(root, vid=VID):
    d = keyframe_dir(root, vid)
    return sorted(p.name for p in d.glob("*.jpg")) if d.exists() else []


# ---------------------------------------------------------------------------
def test_trich_xuat_sinh_du_va_niem_phong(dataset):
    wipe_keyframes(dataset)
    rc, out = run_extract(dataset)
    assert rc == 0, out

    assert jpgs(dataset) == ["0001.jpg", "0002.jpg", "0003.jpg", "0004.jpg"]
    assert map_path(dataset, VID).exists()

    s = sentinel(dataset)
    assert s is not None, "_SUCCESS phải được ghi khi trích đủ"
    assert s["n_keyframes"] == 4
    assert s["scenes_sha256"] == hashlib.sha256(
        scenes_path(dataset, VID).read_bytes()).hexdigest()


def test_ket_qua_trich_xuat_qua_duoc_check_integrity(dataset):
    """Vòng khép kín: bên ghi và bên kiểm phải nhất trí với nhau."""
    wipe_keyframes(dataset)
    assert run_extract(dataset)[0] == 0
    rc, report, out = run_verify(dataset, "--stage", "keyframe")
    assert rc == 0, out
    assert codes(report, "ERROR") == set()


def test_chay_lai_thi_bo_qua_khong_dung_vao_file(dataset):
    """Idempotent: đã có _SUCCESS khớp đầu vào thì không trích lại."""
    wipe_keyframes(dataset)
    assert run_extract(dataset)[0] == 0
    before = {p.name: p.stat().st_mtime_ns for p in keyframe_dir(dataset, VID).iterdir()}

    rc, out = run_extract(dataset)
    assert rc == 0
    assert "Skipping" in out
    after = {p.name: p.stat().st_mtime_ns for p in keyframe_dir(dataset, VID).iterdir()}
    assert before == after, "lần chạy thứ hai không được ghi đè gì cả"


def test_scenes_doi_thi_trich_lai(dataset):
    """Ranh giới shot đổi -> keyframe cũ hết hạn, phải trích lại chứ không 'đã có'."""
    wipe_keyframes(dataset)
    assert run_extract(dataset)[0] == 0
    old = sentinel(dataset)

    # 2 shot ngắn -> 2 keyframe, thay cho 4 keyframe trước đó
    scenes_path(dataset, VID).write_text("0 49\n50 99\n", encoding="utf-8")
    rc, out = run_extract(dataset)
    assert rc == 0, out

    assert jpgs(dataset) == ["0001.jpg", "0002.jpg"]
    new = sentinel(dataset)
    assert new["n_keyframes"] == 2
    assert new["scenes_sha256"] != old["scenes_sha256"]


def test_khong_niem_phong_khi_trich_thieu(dataset):
    """Cổng nghiệm thu: scenes trỏ ra ngoài video -> không đọc được frame.

    Đây là kịch bản sinh ra chính lỗi im lặng trong pipeline: nhánh OpenCV gặp
    `cap.read()` trả về False thì bỏ qua cả .jpg lẫn dòng csv, nên csv và jpg vẫn
    khớp nhau hoàn hảo trong khi dữ liệu đã mất. Chỉ có sentinel + đối chiếu với
    scenes.txt mới lộ ra.
    """
    wipe_keyframes(dataset)
    scenes_path(dataset, VID).write_text("0 999\n", encoding="utf-8")   # video chỉ có 300 frame

    rc, out = run_extract(dataset)
    assert rc == 1, "trích thiếu frame phải exit != 0"
    assert sentinel(dataset) is None, "không được niêm phong kết quả thiếu"
    assert len(jpgs(dataset)) < 3

    # Phần trích được vẫn phải nằm ở vị trí chính thức để còn chẩn đoán tiếp.
    rc, report, _ = run_verify(dataset, "--stage", "keyframe")
    assert "KF_MISSING" in codes(report, "ERROR")
    assert rc == 1


def test_don_sach_vung_staging_con_sot(dataset):
    """Job trước bị kill giữa chừng để lại .partial — lần sau phải làm lại từ đầu."""
    wipe_keyframes(dataset)
    staging = keyframe_dir(dataset, VID).parent / f"{VID}.partial"
    staging.mkdir(parents=True)
    (staging / "0001.jpg").write_bytes(b"rac cua lan chay truoc")
    (staging / "9999.jpg").write_bytes(b"rac")

    rc, out = run_extract(dataset)
    assert rc == 0, out
    assert not staging.exists(), "vùng staging phải được dọn sau khi commit"
    assert jpgs(dataset) == ["0001.jpg", "0002.jpg", "0003.jpg", "0004.jpg"]
    assert (keyframe_dir(dataset, VID) / "0001.jpg").stat().st_size > 1024


def test_khong_de_lai_partial_khi_thanh_cong(dataset):
    wipe_keyframes(dataset)
    assert run_extract(dataset)[0] == 0
    leftovers = list(keyframe_dir(dataset, VID).parent.glob("*.partial"))
    leftovers += list(map_path(dataset, VID).parent.glob("*.partial"))
    assert leftovers == []


def test_thieu_scenes_file_thi_bao_loi(dataset):
    wipe_keyframes(dataset)
    scenes_path(dataset, VID).unlink()
    rc, out = run_extract(dataset)
    assert rc == 1
    assert sentinel(dataset) is None
