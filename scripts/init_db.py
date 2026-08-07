#!/usr/bin/env python3
"""[5a] Tạo cấu trúc metadata.db: bảng thường + bảng ảo FTS5 + trigger đồng bộ.

Script này CHỈ tạo cấu trúc, không nạp dữ liệu (việc đó là của ingest_db.py).

    source data_processing/feature_extraction/venv_embedding/bin/activate
    python scripts/init_db.py            # tạo, rồi ĐỐI CHIẾU cấu trúc thật với code
    python scripts/init_db.py --drop     # XÓA HẲN file .db rồi tạo lại từ đầu

QUY TẮC QUAN TRỌNG: hễ sửa backend/database/models.py hoặc backend/database/schema.py
thì LUÔN chạy kèm --drop. SQLite không tự ALTER bảng đã tồn tại, nên chạy không --drop
sẽ giữ nguyên cấu trúc cũ. Script sẽ phát hiện và báo lỗi to nếu bạn quên — nhưng vẫn
nên nhớ, vì dựng lại toàn bộ chỉ mất vài giây (metadata.db là dữ liệu DẪN XUẤT).

Định nghĩa schema nằm ở backend/database/{models,schema}.py, không nằm ở đây.
"""
import argparse
import os
import sys

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from sqlalchemy import text

from backend.core.config import get_settings
from backend.core.database import engine
from backend.database.models import Base
from backend.database.schema import create_raw_objects, detect_schema_drift, format_drift_report


def init_db(drop: bool = False, ignore_drift: bool = False) -> int:
    settings = get_settings()
    db_path = settings.db_path

    if drop:
        removed = False
        for suffix in ("", "-wal", "-shm"):
            if os.path.exists(db_path + suffix):
                os.remove(db_path + suffix)
                removed = True
        print(f"Đã xóa {db_path}" if removed else f"Chưa có {db_path} để xóa")

    os.makedirs(os.path.dirname(db_path), exist_ok=True)

    print(f"Tạo bảng trong {db_path} ...")
    Base.metadata.create_all(bind=engine)
    with engine.begin() as conn:
        create_raw_objects(conn)

    # Đối chiếu NGAY sau khi tạo: đây là điểm duy nhất phát hiện được việc "code đã đổi
    # nhưng file .db thì chưa" — vì mọi lệnh tạo ở trên đều im lặng bỏ qua thứ đã tồn tại.
    problems = detect_schema_drift(engine)
    if problems and not ignore_drift:
        print(format_drift_report(problems), file=sys.stderr)
        return 1
    if problems:
        print(f"--ignore-drift: bỏ qua {len(problems)} điểm lệch schema.", file=sys.stderr)

    with engine.connect() as conn:
        tables = [
            r[0] for r in conn.execute(
                text("SELECT name FROM sqlite_master WHERE type='table' ORDER BY name")
            )
        ]
        triggers = [
            r[0] for r in conn.execute(
                text("SELECT name FROM sqlite_master WHERE type='trigger' ORDER BY name")
            )
        ]

    print(f"Bảng   : {', '.join(tables)}")
    print(f"Trigger: {', '.join(triggers)}")
    print(
        f"Schema: LỆCH {len(problems)} điểm (đang bị bỏ qua theo yêu cầu)"
        if problems
        else "Schema: khớp với code."
    )
    print("Xong. Bước tiếp theo: python scripts/ingest_db.py")
    return 0


def main():
    parser = argparse.ArgumentParser(
        description="[5a] Tạo cấu trúc metadata.db (bảng + FTS5 + trigger).",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument(
        "--drop",
        action="store_true",
        help="XÓA file .db hiện có rồi tạo lại từ đầu (mất toàn bộ dữ liệu đã ingest)",
    )
    parser.add_argument(
        "--ignore-drift",
        action="store_true",
        help="Vẫn tiếp tục dù cấu trúc trong DB lệch với code (chỉ dùng khi bạn CỐ Ý)",
    )
    args = parser.parse_args()
    sys.exit(init_db(drop=args.drop, ignore_drift=args.ignore_drift))


if __name__ == "__main__":
    main()
