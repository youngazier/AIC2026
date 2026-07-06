from sqlalchemy import Column, Integer, String, Float, ForeignKey
from sqlalchemy.orm import declarative_base, relationship

Base = declarative_base()

class Video(Base):
    """
    Bảng lưu trữ thông tin về Video (Dữ liệu mức cao nhất)
    """
    __tablename__ = 'videos'
    
    video_id = Column(String, primary_key=True, index=True)  # Ví dụ: "L21_V001"
    playlist = Column(String, index=True)                    # Ví dụ: "News", "Sports"
    fps = Column(Float, nullable=False)
    total_frames = Column(Integer, nullable=False)
    duration = Column(Float, nullable=False)                 # Tính bằng giây
    
    # Khai báo Relationship (1 Video có nhiều Keyframes)
    keyframes = relationship("Keyframe", back_populates="video", cascade="all, delete-orphan")


class Keyframe(Base):
    """
    Bảng lưu trữ từng khung hình (Keyframe). 
    QUAN TRỌNG: Trường 'id' sẽ được dùng để ánh xạ 1:1 với ID trong FAISS Index.
    """
    __tablename__ = 'keyframes'
    
    id = Column(Integer, primary_key=True, autoincrement=True)
    video_id = Column(String, ForeignKey('videos.video_id'), nullable=False, index=True)
    
    keyframe_name = Column(String, nullable=False)           # Ví dụ: "0052"
    frame_idx = Column(Integer, nullable=False)              # Index thực tế của frame trong video
    timestamp = Column(Float, nullable=False)                # Thời điểm xuất hiện (giây)
    
    # Denormalization (Khử chuẩn hóa) để tăng tốc độ tìm kiếm:
    # Nối tất cả nhãn vật thể thành 1 chuỗi (VD: "person, car, dog") để lập chỉ mục FTS5 dễ dàng.
    object_summary = Column(String, default="")              
    # Lưu câu miêu tả tổng hợp từ VLM (Image Captioning)
    caption = Column(String, default="")                     
    
    # Khai báo Relationship
    video = relationship("Video", back_populates="keyframes")
    detected_objects = relationship("DetectedObject", back_populates="keyframe", cascade="all, delete-orphan")
    ocr_texts = relationship("OCRText", back_populates="keyframe", cascade="all, delete-orphan")


class DetectedObject(Base):
    """
    Bảng lưu trữ kết quả Object Detection (Từ RAM++ & GroundingDINO)
    """
    __tablename__ = 'detected_objects'
    
    id = Column(Integer, primary_key=True, autoincrement=True)
    keyframe_id = Column(Integer, ForeignKey('keyframes.id'), nullable=False, index=True)
    
    label = Column(String, nullable=False, index=True)       # Tên vật thể (VD: "person", "dog")
    confidence = Column(Float, nullable=False)               # Độ tin cậy (0.0 -> 1.0)
    
    # Bounding Box đã được chuẩn hóa theo tỷ lệ (0.0 -> 1.0) so với kích thước gốc của ảnh
    box_x1 = Column(Float)
    box_y1 = Column(Float)
    box_x2 = Column(Float)
    box_y2 = Column(Float)
    
    # Khai báo Relationship
    keyframe = relationship("Keyframe", back_populates="detected_objects")


class OCRText(Base):
    """
    Bảng lưu trữ kết quả nhận diện chữ viết (Từ PaddleOCR/VietOCR)
    """
    __tablename__ = 'ocr_texts'
    
    id = Column(Integer, primary_key=True, autoincrement=True)
    keyframe_id = Column(Integer, ForeignKey('keyframes.id'), nullable=False, index=True)
    
    text = Column(String, nullable=False)                    # Chữ viết phát hiện được
    confidence = Column(Float, nullable=False)               # Độ tin cậy (0.0 -> 1.0)
    
    # Bounding Box (Chuẩn hóa) để UI frontend có thể vẽ khung nổi bật chữ nếu cần
    box_x1 = Column(Float)
    box_y1 = Column(Float)
    box_x2 = Column(Float)
    box_y2 = Column(Float)
    
    # Khai báo Relationship
    keyframe = relationship("Keyframe", back_populates="ocr_texts")
