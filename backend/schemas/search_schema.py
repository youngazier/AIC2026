"""Hợp đồng Request/Response của API tìm kiếm (xem docs/database_architecture.md mục 4).

Đây cũng chính là tài liệu cho frontend: FastAPI sinh ra trang /docs từ đúng các
class dưới đây, không có bản mô tả nào khác phải giữ cho đồng bộ.

PHẠM VI HIỆN TẠI (Chặng A): chỉ có kênh tìm kiếm ngữ nghĩa (vector). Các trường của
chế độ hybrid — ocr_query, object_filters, fusion_weights — CỐ Ý CHƯA khai báo: khai
sẵn một trường mà backend lặng lẽ bỏ qua còn tệ hơn là chưa có, vì frontend sẽ tưởng
nó có tác dụng. Chúng được thêm vào ở Chặng B/C cùng lúc với phần cài đặt thật.
"""
from enum import Enum

from pydantic import BaseModel, Field


class ModelName(str, Enum):
    """Tập model cố định trong code (mỗi model cần một encoder được cài đặt sẵn).

    Kế thừa (str, Enum) chứ không phải StrEnum vì đang chạy Python 3.10.
    Ngược lại, `playlist` là DỮ LIỆU nên validate động qua registry — không Enum.
    """

    SIGLIP2 = "siglip2"
    BEIT3 = "beit3"
    DINOV3 = "dinov3"


class KISRequest(BaseModel):
    text_query: str = Field(
        ...,
        min_length=1,
        description="Mô tả phân cảnh cần tìm. Text encoder mạnh tiếng Anh hơn tiếng Việt.",
        examples=["a woman in a red shirt riding a bicycle"],
    )
    models: list[ModelName] = Field(
        default=[ModelName.SIGLIP2],
        description=(
            "Model dùng để encode câu truy vấn. Gửi nhiều model (chế độ 'hybrid' trên UI) "
            "thì kết quả các model được trộn bằng Reciprocal Rank Fusion."
        ),
    )
    playlist: str | None = Field(
        default=None,
        description="Giới hạn trong một chương trình. Xem danh sách hợp lệ ở GET /api/v1/playlists.",
    )
    topk: int = Field(default=100, ge=1, le=500, description="Số kết quả trả về")
    exclude_videos: list[str] = Field(
        default=[],
        description="Loại các video đã kiểm tra và xác nhận sai (hữu ích khi thi KIS nhiều lượt).",
    )


class SimilarRequest(BaseModel):
    """Tìm keyframe giống một keyframe đã có (ảnh -> ảnh).

    Không nhận file ảnh upload: frame nguồn LUÔN là một frame đã nằm trong index, nên
    chỉ cần định danh nó. Vector được lấy thẳng từ index thay vì encode lại ảnh.
    """

    video_id: str = Field(..., examples=["L21_V001"])
    keyframe_name: str = Field(..., examples=["0052"])
    models: list[ModelName] = Field(
        default=[ModelName.DINOV3],
        description=(
            "Mặc định DINOv3 — model thị giác tự giám sát, mạnh nhất cho so ảnh với ảnh."
        ),
    )
    playlist: str | None = None
    topk: int = Field(default=50, ge=1, le=500)
    exclude_videos: list[str] = Field(default=[])
    exclude_self: bool = Field(
        default=True,
        description="Bỏ chính frame nguồn khỏi kết quả (nó luôn đứng đầu với distance = 0).",
    )


class KeyframeResult(BaseModel):
    rank: int = Field(..., description="Thứ hạng, bắt đầu từ 1")

    video_id: str
    keyframe_name: str
    frame_idx: int = Field(..., description="Frame thật trong video — ĐÂY LÀ SỐ NỘP CHO BTC")
    timestamp: float = Field(..., description="Thời điểm xuất hiện, tính bằng giây")
    shot_idx: int | None = None
    playlist: str | None = None

    score: float = Field(..., description="Điểm dùng để xếp hạng, CÀNG LỚN CÀNG TỐT")
    distance: float = Field(..., description="Khoảng cách L2 bình phương thô từ FAISS, càng nhỏ càng tốt")
    matched_channels: list[str] = Field(
        default=[], description="Kênh nào đã bầu chọn frame này (Chặng A luôn chỉ có 1 kênh visual)"
    )

    image_url: str = Field(..., description="Đường dẫn ảnh keyframe để hiển thị trên UI")
    title: str | None = None
    author: str | None = None
    watch_url: str | None = None
    thumbnail_url: str | None = None
    publish_date: str | None = None


class SearchResponse(BaseModel):
    query: str
    models: list[ModelName]
    playlist: str | None = None
    topk: int
    total: int = Field(..., description="Số kết quả thực sự trả về")
    elapsed_ms: float
    timings_ms: dict = Field(
        default={}, description="Thời gian từng chặng: encode / faiss / fuse / hydrate"
    )
    results: list[KeyframeResult]


class PlaylistInfo(BaseModel):
    name: str
    n_videos: int
    # Có index FAISS riêng hay không. Playlist chỉ có trong SQLite mà thiếu index thì
    # lọc theo nó sẽ không search vector được -> UI nên vô hiệu hóa lựa chọn đó.
    has_index: bool


class ModelInfo(BaseModel):
    name: str
    loaded: bool = Field(..., description="Index FAISS đã nạp xong chưa")
    supports_text: bool = Field(
        ..., description="Có text encoder không. DINOv3 là model thị giác thuần -> False"
    )
    dim: int | None = None
    n_vectors: int | None = None
    playlists: list[str] = []


class HealthResponse(BaseModel):
    status: str
    database: bool
    n_videos: int
    n_keyframes: int
    models: list[ModelInfo]


class VideoInfo(BaseModel):
    """Thông tin cần để phát video và quy đổi giây <-> frame trên UI.

    `fps` là mấu chốt: người dùng tua video tới đúng khoảnh khắc rồi bấm lấy frame, UI
    tính frame = round(currentTime * fps). Sai fps là sai frame_idx, mà frame_idx sai
    thì R-Score = 0 dù tìm đúng video.
    """

    video_id: str
    playlist: str | None = None
    title: str | None = None
    author: str | None = None
    fps: float | None = None
    duration: float | None = None
    total_frames: int | None = None
    watch_url: str | None = None
    has_file: bool = Field(..., description="File .mp4 có thật trên đĩa hay không")
    stream_url: str = Field(..., description="Đường dẫn phát video (hỗ trợ tua bằng HTTP Range)")


class QueryType(str, Enum):
    KIS = "kis"
    QNA = "qna"
    TRAKE = "trake"


class SubmissionRow(BaseModel):
    video_id: str = Field(..., examples=["L21_V001"])
    frames: list[int] = Field(
        ...,
        min_length=1,
        description="KIS/Q&A đúng 1 frame; TRAKE là chuỗi frame theo thứ tự sự kiện.",
        examples=[[1000]],
    )
    answer: str | None = Field(default=None, description="Bắt buộc với Q&A, bỏ trống với loại khác")


class SubmissionRequest(BaseModel):
    name: str = Field(
        ..., description="Tên file (không cần đuôi .csv). Chỉ chữ, số, '_' và '-'.",
        examples=["query-1-kis"],
    )
    query_type: QueryType = QueryType.KIS
    rows: list[SubmissionRow] = Field(
        ...,
        min_length=1,
        max_length=100,
        description="Tối đa 100 dòng theo quy định BTC. THỨ TỰ DÒNG chính là thứ hạng nộp.",
    )
    overwrite: bool = True


class SubmissionResponse(BaseModel):
    name: str
    path: str = Field(..., description="Đường dẫn tuyệt đối của file đã ghi trên máy chủ")
    n_rows: int
    warnings: list[str] = Field(
        default=[],
        description="Nghi vấn phát hiện khi đối chiếu metadata.db (không chặn việc ghi file)",
    )


class SubmissionFile(BaseModel):
    name: str
    size_bytes: int
    modified_at: float
