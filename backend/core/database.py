"""Kết nối SQLite (SQLAlchemy engine + session).

Workload của hệ thống là CHỈ ĐỌC lúc chạy thi (ghi duy nhất một lần ở bước ingest),
nên toàn bộ cấu hình dưới đây tối ưu cho đọc.
"""
import os

from sqlalchemy import create_engine, event
from sqlalchemy.orm import sessionmaker

from backend.core.config import get_settings

_settings = get_settings()

# check_same_thread=False: FastAPI chạy các endpoint khai báo `def` (không async)
# trong threadpool, nên một connection có thể được tạo ở thread này và dùng ở thread
# khác. An toàn vì mỗi request lấy một Session riêng qua get_db().
engine = create_engine(
    f"sqlite:///{_settings.db_path}",
    connect_args={"check_same_thread": False},
    future=True,
)


@event.listens_for(engine, "connect")
def _set_sqlite_pragma(dbapi_connection, _connection_record):
    """PRAGMA phải set trên TỪNG connection (không phải một lần cho cả engine)."""
    cursor = dbapi_connection.cursor()
    # WAL: người đọc không bị chặn bởi người ghi, và ngược lại.
    cursor.execute("PRAGMA journal_mode=WAL")
    # NORMAL: bớt fsync -> ingest 150k dòng nhanh hơn hẳn. Rủi ro chỉ xảy ra khi mất
    # điện đúng lúc ghi, mà dữ liệu thì build lại được từ pipeline nên chấp nhận được.
    cursor.execute("PRAGMA synchronous=NORMAL")
    # 64MB page cache cho mỗi connection (giá trị âm = tính bằng KB).
    cursor.execute("PRAGMA cache_size=-64000")
    cursor.execute("PRAGMA foreign_keys=ON")
    # Bảng tạm nằm trong RAM: có lợi cho các câu FTS5/GROUP BY ở tầng search.
    cursor.execute("PRAGMA temp_store=MEMORY")
    cursor.close()


SessionLocal = sessionmaker(bind=engine, autocommit=False, autoflush=False, future=True)


def get_db():
    """FastAPI dependency: mỗi request một Session, đóng lại khi request kết thúc."""
    db = SessionLocal()
    try:
        yield db
    finally:
        db.close()


def database_exists() -> bool:
    """metadata.db đã tồn tại chưa (để báo lỗi tử tế lúc startup thay vì crash)."""
    return os.path.exists(_settings.db_path)
