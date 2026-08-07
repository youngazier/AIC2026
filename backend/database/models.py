"""Định nghĩa ORM cho metadata.db (xem docs/database_architecture.md mục 2).

Bảng ảo FTS5 `keyframes_fts` và các trigger đồng bộ KHÔNG khai báo ở đây: SQLAlchemy
không mô tả được virtual table, chúng nằm ở backend/database/schema.py dưới dạng raw SQL.

SỬA FILE NÀY XONG PHẢI CHẠY `python scripts/init_db.py --drop` (rồi ingest lại).
SQLite không tự ALTER bảng đã tồn tại, nên chạy không --drop thì thay đổi của bạn
KHÔNG có hiệu lực — schema.py có lớp kiểm tra sẽ chặn lại, nhưng đừng dựa vào nó.
"""
from sqlalchemy import Column, Float, ForeignKey, Index, Integer, String
from sqlalchemy.orm import declarative_base, relationship

Base = declarative_base()


class Video(Base):
    """Thông tin một video (mức cao nhất), gộp metadata BTC + số đo từ pipeline."""

    __tablename__ = "videos"

    video_id = Column(String, primary_key=True, index=True)  # Ví dụ: "L21_V001"

    # --- Nhãn dẫn xuất (suy ra từ channel_id + title, xem mục 2.5) ---
    playlist = Column(String, index=True)                    # "60-Giay", "Mon-Ngon-Moi-Ngay"...

    # --- Metadata gốc từ media-info/*.json của BTC ---
    title = Column(String, default="")                       # "60 Giây Sáng - Ngày 01082024"
    author = Column(String, default="", index=True)          # Tên kênh: "60 Giây Official"
    channel_id = Column(String, default="", index=True)      # Tín hiệu gom playlist ổn định nhất
    watch_url = Column(String, default="")                   # Mở YouTube gốc để kiểm chứng khi thi
    thumbnail_url = Column(String, default="")               # Ảnh bìa cho danh sách kết quả
    publish_date = Column(String, default="", index=True)    # ĐÃ chuẩn hóa ISO: "2024-08-01"

    # --- Số đo kỹ thuật ---
    fps = Column(Float, nullable=False)                      # Lấy từ map_keyframes/*.csv
    duration = Column(Float)                                 # Giây, theo field `length` của BTC
    # Chỉ điền được sau khi chạy ffprobe trên file video thật (chưa làm ở Chặng A).
    # Để NULL còn hơn điền một con số suy diễn rồi sau này tin nhầm là số đo thật.
    total_frames = Column(Integer)

    keyframes = relationship("Keyframe", back_populates="video", cascade="all, delete-orphan")


class Keyframe(Base):
    """Một khung hình đại diện.

    Khóa liên kết với FAISS là CẶP (video_id, keyframe_name) — xem mục 2.2.1. Cột
    `id` tự tăng chỉ dùng nội bộ SQLite (làm rowid cho FTS5 và làm khóa ngoại).
    """

    __tablename__ = "keyframes"

    id = Column(Integer, primary_key=True, autoincrement=True)
    video_id = Column(String, ForeignKey("videos.video_id"), nullable=False, index=True)

    keyframe_name = Column(String, nullable=False)   # "0052" — khớp tên file 0052.jpg
    frame_idx = Column(Integer, nullable=False)      # Frame thật trong video — ĐÂY LÀ SỐ NỘP CHO BTC
    timestamp = Column(Float, nullable=False)        # Thời điểm xuất hiện (giây)
    shot_idx = Column(Integer)                       # Shot chứa keyframe (TransNetV2), phục vụ gom cụm thời gian

    # --- Khử chuẩn hóa để FTS5 search trên MỘT bảng, không phải JOIN ---
    playlist = Column(String, index=True)            # Chép xuống từ videos.playlist để lọc nhanh
    object_summary = Column(String, default="")      # "person, car, dog"
    ocr_summary = Column(String, default="")         # Toàn bộ chữ trên frame đã làm sạch
    caption = Column(String, default="")             # Mô tả sinh bởi VLM
    asr_text = Column(String, default="")            # Lời thoại quanh thời điểm frame (Whisper)

    video = relationship("Video", back_populates="keyframes")
    detected_objects = relationship("DetectedObject", back_populates="keyframe", cascade="all, delete-orphan")
    ocr_texts = relationship("OCRText", back_populates="keyframe", cascade="all, delete-orphan")

    __table_args__ = (
        # Index quan trọng nhất của cả DB: mỗi query KIS tra ~100 lần theo cặp khóa
        # này để hydrate kết quả FAISS. UNIQUE vừa tăng tốc vừa chặn ingest trùng.
        Index("ix_keyframes_video_kf", "video_id", "keyframe_name", unique=True),
    )


class DetectedObject(Base):
    """Kết quả Object Detection (RAM++ / GroundingDINO) — dữ liệu đến ở Chặng C."""

    __tablename__ = "detected_objects"

    id = Column(Integer, primary_key=True, autoincrement=True)
    keyframe_id = Column(Integer, ForeignKey("keyframes.id"), nullable=False, index=True)

    label = Column(String, nullable=False, index=True)
    confidence = Column(Float, nullable=False)

    # Bounding box chuẩn hóa theo tỷ lệ (0.0 -> 1.0) so với kích thước ảnh gốc
    box_x1 = Column(Float)
    box_y1 = Column(Float)
    box_x2 = Column(Float)
    box_y2 = Column(Float)

    keyframe = relationship("Keyframe", back_populates="detected_objects")


class OCRText(Base):
    """Từng dòng chữ nhận diện được kèm bbox (để UI vẽ khung) — dữ liệu đến ở Chặng C.

    Bản gộp-và-làm-sạch của các dòng này nằm ở `keyframes.ocr_summary`; FTS5 index
    cột đó chứ không index bảng này.
    """

    __tablename__ = "ocr_texts"

    id = Column(Integer, primary_key=True, autoincrement=True)
    keyframe_id = Column(Integer, ForeignKey("keyframes.id"), nullable=False, index=True)

    text = Column(String, nullable=False)
    confidence = Column(Float, nullable=False)

    box_x1 = Column(Float)
    box_y1 = Column(Float)
    box_x2 = Column(Float)
    box_y2 = Column(Float)

    keyframe = relationship("Keyframe", back_populates="ocr_texts")
