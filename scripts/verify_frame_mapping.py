#!/usr/bin/env python3
"""[5c] Kiểm chứng ánh xạ keyframe -> frame_idx bằng cách so ảnh với video gốc.

    python scripts/verify_frame_mapping.py            # 20 keyframe ngẫu nhiên
    python scripts/verify_frame_mapping.py -n 100     # kỹ hơn
    python scripts/verify_frame_mapping.py --keep-frames /tmp/kiemtra   # giữ ảnh để nhìn tận mắt

VÌ SAO CẦN SCRIPT NÀY: mọi lỗi khác trong hệ thống đều lộ ra ngay (crash, không có
kết quả). Riêng lỗi lệch offset giữa keyframe_name và frame_idx thì hệ thống vẫn tìm
kiếm "hoàn hảo" — chỉ có con số nộp cho BTC là sai. Không có cách nào phát hiện bằng
mắt thường khi nhìn kết quả tìm kiếm.

CÁCH KIỂM: với mỗi keyframe lấy ngẫu nhiên từ DB, dùng ffmpeg trích đúng frame số
`frame_idx` từ file .mp4 gốc, rồi so pixel với file .jpg mà pipeline đã lưu. Trùng
frame -> sai lệch gần như bằng 0. Lệch frame -> khác biệt lớn (trừ cảnh tĩnh, nên
script báo cáo phân bố chứ không chỉ một con số).
"""
import argparse
import os
import random
import subprocess
import sys
import tempfile
from glob import glob

import numpy as np
from PIL import Image
from sqlalchemy import text

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from backend.core.config import get_settings
from backend.core.database import engine

# Ngưỡng trên thang 0-255. Cảnh giống hệt nhau sau khi nén JPEG 2 lần vẫn lệch vài đơn vị.
MAE_OK = 8.0
MAE_SUSPICIOUS = 20.0


def find_video_file(videos_dir: str, video_id: str):
    hits = glob(os.path.join(videos_dir, "*", f"{video_id}.mp4"))
    return hits[0] if hits else None


def find_keyframe_file(keyframes_dir: str, video_id: str, keyframe_name: str):
    hits = glob(os.path.join(keyframes_dir, "*", video_id, f"{keyframe_name}.jpg"))
    return hits[0] if hits else None


def extract_frame(video_path: str, frame_idx: int, out_path: str) -> bool:
    """Trích đúng frame thứ `frame_idx` (đếm từ 0) bằng bộ lọc select của ffmpeg."""
    cmd = [
        "ffmpeg", "-v", "error", "-y",
        "-i", video_path,
        "-vf", f"select=eq(n\\,{frame_idx})",
        "-vsync", "0", "-frames:v", "1",
        out_path,
    ]
    subprocess.run(cmd, check=False, capture_output=True)
    return os.path.exists(out_path) and os.path.getsize(out_path) > 0


def mean_abs_diff(path_a: str, path_b: str) -> float:
    """MAE trên ảnh xám đã resize về cùng kích thước nhỏ (bỏ qua khác biệt độ phân giải)."""
    a = np.asarray(Image.open(path_a).convert("L").resize((160, 90)), dtype=np.float32)
    b = np.asarray(Image.open(path_b).convert("L").resize((160, 90)), dtype=np.float32)
    return float(np.abs(a - b).mean())


def main():
    parser = argparse.ArgumentParser(description="[5c] Kiểm chứng frame_idx bằng ffmpeg.")
    parser.add_argument("-n", "--samples", type=int, default=20, help="Số keyframe kiểm tra")
    parser.add_argument("--seed", type=int, default=42, help="Seed để lặp lại được kết quả")
    parser.add_argument("--keep-frames", type=str, default="", help="Thư mục giữ ảnh để nhìn tận mắt")
    args = parser.parse_args()

    settings = get_settings()
    videos_dir = settings.raw["paths"]["videos_dir"]
    random.seed(args.seed)

    with engine.connect() as conn:
        total = conn.execute(text("SELECT COUNT(*) FROM keyframes")).scalar()
        if not total:
            sys.exit("Bảng keyframes rỗng. Chạy scripts/ingest_db.py trước.")
        rows = conn.execute(
            text(
                "SELECT video_id, keyframe_name, frame_idx, timestamp "
                "FROM keyframes ORDER BY RANDOM() LIMIT :n"
            ),
            {"n": args.samples},
        ).fetchall()

    outdir = args.keep_frames or tempfile.mkdtemp(prefix="aic_verify_")
    os.makedirs(outdir, exist_ok=True)

    results, skipped = [], []
    for video_id, kf_name, frame_idx, ts in rows:
        video_path = find_video_file(videos_dir, video_id)
        jpg_path = find_keyframe_file(settings.keyframes_dir, video_id, kf_name)
        if not video_path or not jpg_path:
            skipped.append((video_id, kf_name, "thiếu file video hoặc jpg"))
            continue

        extracted = os.path.join(outdir, f"{video_id}_{kf_name}_f{frame_idx}.jpg")
        if not extract_frame(video_path, frame_idx, extracted):
            skipped.append((video_id, kf_name, "ffmpeg không trích được frame"))
            continue

        mae = mean_abs_diff(jpg_path, extracted)
        results.append((video_id, kf_name, frame_idx, ts, mae))
        flag = "OK  " if mae <= MAE_OK else ("NGỜ " if mae <= MAE_SUSPICIOUS else "SAI ")
        print(f"{flag} {video_id}/{kf_name}  frame={frame_idx:<7} t={ts:<9.3f} MAE={mae:6.2f}")

    if not results:
        sys.exit("Không kiểm được mẫu nào. " + str(skipped[:3]))

    maes = np.array([r[4] for r in results])
    n_ok = int((maes <= MAE_OK).sum())
    n_sus = int(((maes > MAE_OK) & (maes <= MAE_SUSPICIOUS)).sum())
    n_bad = int((maes > MAE_SUSPICIOUS).sum())

    print(f"\n--- {len(results)} mẫu (bỏ qua {len(skipped)}) ---")
    print(f"MAE trung vị: {np.median(maes):.2f} | tệ nhất: {maes.max():.2f}")
    print(f"OK (<={MAE_OK}): {n_ok}   nghi ngờ: {n_sus}   sai rõ (>{MAE_SUSPICIOUS}): {n_bad}")
    print(f"Ảnh đã trích nằm ở: {outdir}")

    if n_bad == 0 and n_sus <= max(1, len(results) // 10):
        print("\nKẾT LUẬN: ánh xạ keyframe_name -> frame_idx ĐÚNG.")
        return 0
    print(
        "\nKẾT LUẬN: CÓ VẤN ĐỀ. Mở vài ảnh trong thư mục trên so với file .jpg gốc trước khi\n"
        "tin bất kỳ kết quả nào — frame_idx là con số nộp cho BTC."
    )
    return 1


if __name__ == "__main__":
    sys.exit(main())
