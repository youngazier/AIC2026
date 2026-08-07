"""Định nghĩa phần schema mà SQLAlchemy không mô tả được, + lớp phát hiện lệch schema.

Bảng thường nằm ở models.py; ở đây là bảng ảo FTS5 và các trigger đồng bộ (raw SQL).

VÌ SAO CẦN PHÁT HIỆN LỆCH SCHEMA: cả ba cơ chế tạo cấu trúc đều mang tính "IF NOT
EXISTS", tức là chúng KHÔNG BAO GIỜ sửa thứ đã tồn tại:

    Base.metadata.create_all()          -> chỉ tạo bảng còn thiếu, không ALTER
    CREATE VIRTUAL TABLE IF NOT EXISTS  -> giữ nguyên cột/tokenizer cũ
    CREATE TRIGGER IF NOT EXISTS        -> giữ nguyên thân trigger cũ

Nên nếu sửa models.py hoặc sửa file này rồi chạy init_db.py mà QUÊN --drop, script sẽ
in ra "Xong" một cách vui vẻ trong khi DB vẫn nguyên cấu trúc cũ. Trường hợp thêm cột
thì ingest sẽ lỗi khó hiểu; trường hợp bỏ cột thì chạy trót lọt và ta tưởng schema mới
đã có hiệu lực — đó mới là kiểu hỏng nguy hiểm. Module này so cấu trúc mong muốn với
cấu trúc thật trong file .db và báo to khi lệch.
"""
import re

from sqlalchemy import inspect, text

from backend.database.models import Base

# Các cột được FTS5 lập chỉ mục (tách từ). Thứ tự phải khớp giữa CREATE và các trigger.
FTS_INDEXED_COLUMNS = ["object_summary", "caption", "ocr_summary", "asr_text"]
# Các cột mang theo chỉ để đọc ra / lọc WHERE, KHÔNG tách từ.
FTS_UNINDEXED_COLUMNS = ["video_id", "keyframe_name", "playlist"]
FTS_ALL_COLUMNS = FTS_UNINDEXED_COLUMNS + FTS_INDEXED_COLUMNS

_ALL = ", ".join(FTS_ALL_COLUMNS)
_ALL_NEW = ", ".join(f"new.{c}" for c in FTS_ALL_COLUMNS)
_IDX = ", ".join(FTS_INDEXED_COLUMNS)
_IDX_OLD = ", ".join(f"old.{c}" for c in FTS_INDEXED_COLUMNS)

CREATE_FTS = f"""
CREATE VIRTUAL TABLE IF NOT EXISTS keyframes_fts USING fts5(
    {", ".join(f"{c} UNINDEXED" for c in FTS_UNINDEXED_COLUMNS)},
    {_IDX},
    content='keyframes',
    content_rowid='id',
    tokenize = "unicode61 remove_diacritics 2"
);
"""

# --- Trigger đồng bộ bảng gốc -> bảng ảo ---------------------------------------
# Bắt buộc phải có ĐỦ CẢ BA. Thiếu AFTER UPDATE là bug âm thầm nguy hiểm nhất ở đây:
# ocr_summary/caption/object_summary được UPDATE vào SAU (Chặng C), nếu không có
# trigger update thì FTS5 vẫn giữ chuỗi rỗng -> search OCR không ra gì mà không báo lỗi.
#
# Với external-content FTS5, muốn gỡ một dòng khỏi chỉ mục phải chèn dòng lệnh đặc
# biệt 'delete' KÈM ĐÚNG GIÁ TRỊ CŨ của các cột được index — FTS5 không tự đọc lại
# bảng gốc để biết cần gỡ token nào.
TRIGGERS = {
    "keyframes_ai": f"""
    CREATE TRIGGER IF NOT EXISTS keyframes_ai AFTER INSERT ON keyframes BEGIN
        INSERT INTO keyframes_fts(rowid, {_ALL}) VALUES (new.id, {_ALL_NEW});
    END;
    """,
    "keyframes_ad": f"""
    CREATE TRIGGER IF NOT EXISTS keyframes_ad AFTER DELETE ON keyframes BEGIN
        INSERT INTO keyframes_fts(keyframes_fts, rowid, {_IDX})
        VALUES ('delete', old.id, {_IDX_OLD});
    END;
    """,
    "keyframes_au": f"""
    CREATE TRIGGER IF NOT EXISTS keyframes_au AFTER UPDATE ON keyframes BEGIN
        INSERT INTO keyframes_fts(keyframes_fts, rowid, {_IDX})
        VALUES ('delete', old.id, {_IDX_OLD});
        INSERT INTO keyframes_fts(rowid, {_ALL}) VALUES (new.id, {_ALL_NEW});
    END;
    """,
}

# Tên -> câu CREATE mong muốn, dùng cho cả lúc tạo lẫn lúc đối chiếu.
RAW_SQL_OBJECTS = {"keyframes_fts": CREATE_FTS, **TRIGGERS}

_IF_NOT_EXISTS_RE = re.compile(r"\bIF\s+NOT\s+EXISTS\b\s*", re.IGNORECASE)


def create_raw_objects(conn) -> None:
    """Tạo bảng ảo FTS5 và các trigger (idempotent)."""
    for sql in RAW_SQL_OBJECTS.values():
        conn.execute(text(sql))


def _normalize_sql(sql) -> str:
    """Đưa câu SQL về dạng so sánh được.

    SQLite lưu nguyên văn câu CREATE trong sqlite_master, chỉ bỏ `IF NOT EXISTS` và
    dấu `;` cuối. Nên chỉ cần bỏ đúng hai thứ đó rồi gom khoảng trắng là so được —
    không cần parser SQL.
    """
    if not sql:
        return ""
    sql = _IF_NOT_EXISTS_RE.sub("", sql)
    sql = re.sub(r"\s+", " ", sql).strip()
    return sql.rstrip(";").strip()


def detect_schema_drift(engine) -> list:
    """So cấu trúc mong muốn (models.py + file này) với cấu trúc thật trong .db.

    Trả về danh sách mô tả các điểm lệch; danh sách rỗng nghĩa là khớp hoàn toàn.
    Chỉ ĐỌC, không sửa gì.
    """
    problems = []
    inspector = inspect(engine)
    existing_tables = set(inspector.get_table_names())

    # --- 1. Bảng thường: cột và index ---
    for table in Base.metadata.sorted_tables:
        if table.name not in existing_tables:
            problems.append(f"bảng '{table.name}' chưa tồn tại trong DB")
            continue

        actual_columns = {c["name"] for c in inspector.get_columns(table.name)}
        expected_columns = {c.name for c in table.columns}
        missing = sorted(expected_columns - actual_columns)
        extra = sorted(actual_columns - expected_columns)
        if missing:
            problems.append(f"bảng '{table.name}' THIẾU cột: {missing}")
        if extra:
            problems.append(f"bảng '{table.name}' có cột THỪA (models.py đã bỏ): {extra}")

        # Index chỉ báo khi THIẾU. SQLite tự sinh thêm sqlite_autoindex_* cho ràng buộc
        # UNIQUE/PK, báo "thừa" cho những cái đó chỉ tạo báo động giả.
        actual_indexes = {i["name"] for i in inspector.get_indexes(table.name)}
        missing_indexes = sorted({i.name for i in table.indexes} - actual_indexes)
        if missing_indexes:
            problems.append(f"bảng '{table.name}' THIẾU index: {missing_indexes}")

    # --- 2. Bảng ảo FTS5 và trigger: so nguyên văn câu CREATE ---
    with engine.connect() as conn:
        rows = conn.execute(text("SELECT name, sql FROM sqlite_master")).fetchall()
    stored = {name: sql for name, sql in rows if name in RAW_SQL_OBJECTS}

    for name, wanted_sql in RAW_SQL_OBJECTS.items():
        actual_sql = stored.get(name)
        if actual_sql is None:
            problems.append(f"'{name}' chưa tồn tại trong DB")
        elif _normalize_sql(actual_sql) != _normalize_sql(wanted_sql):
            problems.append(
                f"'{name}' trong DB KHÁC với định nghĩa hiện tại trong backend/database/schema.py"
            )

    return problems


def format_drift_report(problems: list) -> str:
    lines = [
        "",
        "=" * 78,
        " LỆCH SCHEMA — cấu trúc trong metadata.db KHÔNG khớp với code",
        "=" * 78,
    ]
    lines += [f"  - {p}" for p in problems]
    lines += [
        "",
        " Nguyên nhân gần như chắc chắn: bạn đã sửa backend/database/models.py hoặc",
        " backend/database/schema.py rồi chạy init_db.py mà KHÔNG có --drop. SQLite",
        " không tự ALTER bảng đã tồn tại, nên thay đổi của bạn chưa có hiệu lực.",
        "",
        " CÁCH SỬA (metadata.db là dữ liệu DẪN XUẤT, dựng lại hết chỉ mất vài giây):",
        "     python scripts/init_db.py --drop",
        "     python scripts/ingest_db.py",
        "",
        " Nếu bạn CỐ Ý muốn giữ cấu trúc cũ, chạy lại kèm --ignore-drift.",
        "=" * 78,
    ]
    return "\n".join(lines)
