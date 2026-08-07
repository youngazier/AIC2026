#!/usr/bin/env python3
"""[5b] Nạp dữ liệu mồi vào metadata.db.

    source data_processing/feature_extraction/venv_embedding/bin/activate
    python scripts/init_db.py --drop     # tạo cấu trúc trước
    python scripts/ingest_db.py          # rồi nạp dữ liệu

Nguồn dữ liệu (đều đã có sẵn, script chỉ ghép lại):

    data/media-info*/media-info/*.json          -> bảng videos (metadata BTC)
    data/video_to_playlist.json                 -> videos.playlist (+ khử chuẩn hóa xuống keyframes)
    data_processed/map_keyframes/LXX_VXXX.csv   -> frame_idx, timestamp, shot_idx, fps
    data_processed/index/*_keyframe_metadata.npy -> DANH SÁCH KEYFRAME CHÍNH THỨC

QUY ƯỚC QUAN TRỌNG — vì sao danh sách keyframe lấy từ file .npy chứ không từ CSV:
file .npy là metadata song hành của FAISS index, tức đúng những keyframe THỰC SỰ đã
được encode thành vector (build_index.py đã lọc theo file .jpg tồn tại trên đĩa). Nếu
lấy từ CSV, DB có thể chứa dòng không có vector tương ứng -> FAISS không bao giờ trả
về, hoặc tệ hơn là lệch ánh xạ. DB phải là tấm gương của index, không phải ngược lại.

Ánh xạ keyframe_name <-> dòng CSV: keyframe_name = f"{n:04d}" với `n` là cột đầu của
CSV (1-based, liên tục). Script VERIFY lại quy ước này cho từng video trước khi ghi,
vì sai lệch ở đây tạo ra hệ thống tìm kiếm "trông rất đúng" nhưng nộp sai frame.
"""
import argparse
import csv
import json
import os
import sys
from glob import glob

import numpy as np
from sqlalchemy import func, select, text

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from backend.core.config import get_settings
from backend.core.database import SessionLocal, database_exists, engine
from backend.database.models import Keyframe, Video
from backend.database.schema import detect_schema_drift, format_drift_report

BATCH_SIZE = 5000


# --------------------------------------------------------------------------- #
# Đọc các nguồn dữ liệu
# --------------------------------------------------------------------------- #
def find_media_info_files(data_dir: str) -> dict:
    """{video_id: đường dẫn json}. Tự dò mọi batch media-info* để dùng lại cho b2/b3."""
    files = {}
    for pattern in ("media-info*/media-info/*.json", "media-info*/*.json"):
        for path in sorted(glob(os.path.join(data_dir, pattern))):
            files[os.path.basename(path)[: -len(".json")]] = path
    return files


def normalize_publish_date(raw) -> str:
    """'01/08/2024' (DD/MM/YYYY) -> '2024-08-01'. So sánh/lọc theo thời gian được ngay."""
    if not raw:
        return ""
    raw = str(raw).strip()
    if len(raw) == 10 and raw[4] == "-" and raw[7] == "-":  # đã là ISO
        return raw
    parts = raw.replace("-", "/").split("/")
    if len(parts) == 3 and len(parts[2]) == 4:
        day, month, year = parts
        return f"{year}-{int(month):02d}-{int(day):02d}"
    return raw  # định dạng lạ: giữ nguyên còn hơn đoán bừa


def load_keyframe_metadata(index_dir: str, model_hint: str = "siglip2") -> np.ndarray:
    """Đọc file metadata song hành của FAISS index global."""
    preferred = os.path.join(index_dir, f"{model_hint}_keyframe_metadata.npy")
    candidates = [preferred] if os.path.exists(preferred) else sorted(
        glob(os.path.join(index_dir, "*_keyframe_metadata.npy"))
    )
    if not candidates:
        sys.exit(
            f"Không tìm thấy *_keyframe_metadata.npy trong {index_dir}\n"
            f"Chạy data_processing/feature_extraction/build_index.py trước."
        )
    print(f"Danh sách keyframe lấy từ: {os.path.basename(candidates[0])}")
    return np.load(candidates[0], allow_pickle=True)


def read_map_keyframes(csv_path: str) -> list:
    """CSV -> [{'n', 'shot_idx', 'pts_time', 'fps', 'frame_idx'}] theo đúng thứ tự dòng."""
    with open(csv_path, newline="", encoding="utf-8") as f:
        return list(csv.DictReader(f))


# --------------------------------------------------------------------------- #
# Ingest
# --------------------------------------------------------------------------- #
def ingest(force: bool = False, limit: int = 0, ignore_drift: bool = False) -> None:
    settings = get_settings()
    data_dir = settings.data_dir

    if not database_exists():
        sys.exit(f"Chưa có {settings.db_path}. Chạy `python scripts/init_db.py` trước.")

    # Kiểm tra TRƯỚC khi ghi bất cứ thứ gì: nếu cấu trúc trong DB lệch với models.py
    # thì mọi thông báo lỗi phía sau đều khó hiểu (hoặc tệ hơn: ghi trót lọt vào một
    # bảng thiếu cột và ta tưởng dữ liệu đã đầy đủ).
    problems = detect_schema_drift(engine)
    if problems and not ignore_drift:
        print(format_drift_report(problems), file=sys.stderr)
        sys.exit(1)
    if problems:
        print(f"--ignore-drift: bỏ qua {len(problems)} điểm lệch schema.", file=sys.stderr)

    session = SessionLocal()

    existing = session.scalar(select(func.count()).select_from(Keyframe))
    if existing and not force:
        session.close()
        sys.exit(
            f"metadata.db đã có {existing} keyframe. Chạy lại với --force để xóa và nạp lại,\n"
            f"hoặc `python scripts/init_db.py --drop` để làm sạch hoàn toàn."
        )
    if existing:
        print(f"--force: xóa {existing} keyframe cũ ...")
        session.execute(text("DELETE FROM keyframes"))
        session.execute(text("DELETE FROM videos"))
        session.commit()

    # --- 1. Danh sách keyframe chính thức, gom theo video ---
    meta = load_keyframe_metadata(settings.index_dir)
    kf_by_video: dict = {}
    for video_id, kf_name in meta:
        kf_by_video.setdefault(str(video_id), []).append(str(kf_name))
    video_ids = sorted(kf_by_video)
    if limit:
        video_ids = video_ids[:limit]
        print(f"--limit {limit}: chỉ nạp {len(video_ids)} video đầu tiên")
    print(f"{len(video_ids)} video, {sum(len(kf_by_video[v]) for v in video_ids)} keyframe")

    # --- 2. Metadata BTC + playlist ---
    media_files = find_media_info_files(data_dir)
    print(f"{len(media_files)} file media-info")

    playlist_path = os.path.join(data_dir, "video_to_playlist.json")
    if os.path.exists(playlist_path):
        with open(playlist_path, encoding="utf-8") as f:
            video_playlist = json.load(f)
        print(f"{len(video_playlist)} video có nhãn playlist")
    else:
        video_playlist = {}
        print(
            f"CẢNH BÁO: không có {playlist_path} — cột playlist sẽ rỗng, không lọc theo\n"
            f"          playlist được. Chạy data_processing/build_playlist_mapping.py trước."
        )

    # --- 3. Dựng rows, verify ánh xạ CSV trong lúc duyệt ---
    video_rows, keyframe_rows = [], []
    problems = {"no_json": [], "no_csv": [], "count_mismatch": [], "name_mismatch": [], "no_playlist": []}

    for video_id in video_ids:
        kf_names = sorted(kf_by_video[video_id])

        info = {}
        if video_id in media_files:
            with open(media_files[video_id], encoding="utf-8") as f:
                info = json.load(f)
        else:
            problems["no_json"].append(video_id)

        csv_path = os.path.join(settings.map_keyframes_dir, f"{video_id}.csv")
        if not os.path.exists(csv_path):
            problems["no_csv"].append(video_id)
            continue
        rows = read_map_keyframes(csv_path)

        # VERIFY: keyframe_name phải là f"{n:04d}" của đúng dòng CSV tương ứng.
        by_n = {f"{int(r['n']):04d}": r for r in rows}
        if len(rows) != len(kf_names):
            problems["count_mismatch"].append((video_id, len(rows), len(kf_names)))
        missing = [k for k in kf_names if k not in by_n]
        if missing:
            problems["name_mismatch"].append((video_id, missing[:3]))
            continue

        fps = float(rows[0]["fps"]) if rows else 0.0
        playlist = video_playlist.get(video_id)
        if playlist is None:
            problems["no_playlist"].append(video_id)

        video_rows.append(
            {
                "video_id": video_id,
                "playlist": playlist,
                "title": info.get("title", ""),
                "author": info.get("author", ""),
                "channel_id": info.get("channel_id", ""),
                "watch_url": info.get("watch_url", ""),
                "thumbnail_url": info.get("thumbnail_url", ""),
                "publish_date": normalize_publish_date(info.get("publish_date")),
                "fps": fps,
                "duration": float(info["length"]) if info.get("length") else None,
                "total_frames": None,  # cần ffprobe, chưa làm ở Chặng A
            }
        )

        for kf_name in kf_names:
            row = by_n[kf_name]
            keyframe_rows.append(
                {
                    "video_id": video_id,
                    "keyframe_name": kf_name,
                    "frame_idx": int(row["frame_idx"]),
                    "timestamp": float(row["pts_time"]),
                    "shot_idx": int(row["shot_idx"]) if row.get("shot_idx") else None,
                    "playlist": playlist,
                    "object_summary": "",
                    "ocr_summary": "",
                    "caption": "",
                    "asr_text": "",
                }
            )

    # --- 4. Ghi, gói trong MỘT transaction cho mỗi bảng ---
    print(f"Ghi {len(video_rows)} video ...")
    with engine.begin() as conn:
        conn.execute(Video.__table__.insert(), video_rows)

    print(f"Ghi {len(keyframe_rows)} keyframe (trigger FTS5 chạy kèm) ...")
    with engine.begin() as conn:
        for start in range(0, len(keyframe_rows), BATCH_SIZE):
            conn.execute(Keyframe.__table__.insert(), keyframe_rows[start : start + BATCH_SIZE])
            done = min(start + BATCH_SIZE, len(keyframe_rows))
            print(f"  {done}/{len(keyframe_rows)}", end="\r", flush=True)
    print()

    session.close()
    report_problems(problems)
    verify()


def report_problems(problems: dict) -> None:
    labels = {
        "no_json": "video không có media-info json (metadata để rỗng)",
        "no_csv": "video không có map_keyframes csv (BỎ QUA HOÀN TOÀN)",
        "count_mismatch": "lệch số lượng dòng CSV vs số keyframe đã encode",
        "name_mismatch": "keyframe_name không khớp cột n của CSV (BỎ QUA VIDEO)",
        "no_playlist": "video chưa có nhãn playlist (chỉ nằm trong index global)",
    }
    any_problem = False
    for key, items in problems.items():
        if items:
            any_problem = True
            print(f"CẢNH BÁO: {len(items)} {labels[key]}. VD: {items[:5]}")
    if not any_problem:
        print("Không có bất thường nào trong lúc ghép dữ liệu.")


def verify() -> None:
    """Đối chiếu số liệu ngay sau khi ghi — rẻ, và bắt được lỗi trigger/ánh xạ sớm."""
    print("\n--- Kiểm tra ---")
    with engine.connect() as conn:
        n_videos = conn.execute(text("SELECT COUNT(*) FROM videos")).scalar()
        n_kf = conn.execute(text("SELECT COUNT(*) FROM keyframes")).scalar()
        n_fts = conn.execute(text("SELECT COUNT(*) FROM keyframes_fts")).scalar()
        n_kf_videos = conn.execute(text("SELECT COUNT(DISTINCT video_id) FROM keyframes")).scalar()
        playlists = conn.execute(
            text("SELECT playlist, COUNT(*) FROM videos GROUP BY playlist ORDER BY 2 DESC")
        ).fetchall()
        sample = conn.execute(
            text(
                "SELECT video_id, keyframe_name, frame_idx, timestamp, shot_idx, playlist "
                "FROM keyframes ORDER BY id LIMIT 3"
            )
        ).fetchall()

    print(f"videos          : {n_videos}")
    print(f"keyframes       : {n_kf}  (trên {n_kf_videos} video)")
    print(f"keyframes_fts   : {n_fts}  {'OK — trigger đã đồng bộ' if n_fts == n_kf else 'LỆCH! trigger INSERT có vấn đề'}")
    print("playlist        :")
    for name, count in playlists:
        print(f"  {name or '(chưa gán)'}: {count} video")
    print("3 dòng đầu      :")
    for row in sample:
        print(f"  {tuple(row)}")


def main():
    parser = argparse.ArgumentParser(
        description="[5b] Nạp videos + keyframes vào metadata.db.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument("--force", action="store_true", help="Xóa dữ liệu cũ rồi nạp lại")
    parser.add_argument("--limit", type=int, default=0, help="Chỉ nạp N video đầu (smoke test)")
    parser.add_argument(
        "--ignore-drift",
        action="store_true",
        help="Vẫn nạp dù cấu trúc trong DB lệch với code (chỉ dùng khi bạn CỐ Ý)",
    )
    args = parser.parse_args()
    ingest(force=args.force, limit=args.limit, ignore_drift=args.ignore_drift)


if __name__ == "__main__":
    main()
