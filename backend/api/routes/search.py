"""Endpoint tìm kiếm KIS.

Route ở đây CỐ Ý MỎNG: nhận request, ủy quyền cho service, dịch lỗi nghiệp vụ sang mã
HTTP. Không có logic xếp hạng, không có SQL, không đụng vào mảng numpy.

Vì sao khai báo `def` chứ không phải `async def`: bên trong là công việc CPU/GPU đồng
bộ (encode text, FAISS search). `async def` sẽ chạy thẳng trên event loop và đóng băng
toàn bộ server trong lúc tính; `def` được FastAPI tự đẩy sang threadpool, mà FAISS và
PyTorch đều nhả GIL nên chúng chạy song song thật.
"""
import logging
import os
import re
import time

from fastapi import APIRouter, Depends, HTTPException, Request
from fastapi.responses import FileResponse

from backend.api.dependencies import get_search_service, get_sqlite_repo
from backend.core.database import database_exists
from backend.repositories.sqlite_repo import SQLiteRepository
from backend.schemas.search_schema import (
    HealthResponse,
    KISRequest,
    ModelInfo,
    PlaylistInfo,
    SearchResponse,
    SimilarRequest,
    SubmissionFile,
    SubmissionRequest,
    SubmissionResponse,
    VideoInfo,
)
from backend.services import submission_service
from backend.services.search_service import SearchRequestError, SearchService

logger = logging.getLogger(__name__)
router = APIRouter()

# Chặn path traversal: id chỉ được chứa chữ, số, gạch dưới, gạch ngang.
SAFE_ID = re.compile(r"^[A-Za-z0-9_-]+$")


def _video_path(settings, video_id: str) -> str:
    """videos/Videos_LXX/LXX_VXXX.mp4 — suy thư mục batch từ tiền tố id, khỏi glob đĩa."""
    batch = f"Videos_{video_id.split('_')[0]}"
    return os.path.join(settings.videos_dir, batch, f"{video_id}.mp4")


@router.post("/search/kis", response_model=SearchResponse, summary="Known-Item Search bằng text")
def search_kis(request: KISRequest, service: SearchService = Depends(get_search_service)):
    started = time.perf_counter()
    try:
        outcome = service.execute_kis(
            text_query=request.text_query,
            models=request.models,
            playlist=request.playlist,
            topk=request.topk,
            exclude_videos=request.exclude_videos,
        )
    except SearchRequestError as exc:
        raise HTTPException(status_code=400, detail=str(exc))

    return SearchResponse(
        query=request.text_query,
        models=request.models,
        playlist=request.playlist,
        topk=request.topk,
        total=len(outcome["results"]),
        elapsed_ms=round((time.perf_counter() - started) * 1000, 2),
        timings_ms=outcome["timings_ms"],
        results=outcome["results"],
    )


@router.post(
    "/search/similar",
    response_model=SearchResponse,
    summary="Tìm keyframe giống một keyframe (ảnh -> ảnh, mặc định DINOv3)",
)
def search_similar(request: SimilarRequest, service: SearchService = Depends(get_search_service)):
    started = time.perf_counter()
    try:
        outcome = service.execute_similar(
            video_id=request.video_id,
            keyframe_name=request.keyframe_name,
            models=request.models,
            playlist=request.playlist,
            topk=request.topk,
            exclude_videos=request.exclude_videos,
            exclude_self=request.exclude_self,
        )
    except SearchRequestError as exc:
        raise HTTPException(status_code=400, detail=str(exc))

    return SearchResponse(
        query=f"{request.video_id}/{request.keyframe_name}",
        models=request.models,
        playlist=request.playlist,
        topk=request.topk,
        total=len(outcome["results"]),
        elapsed_ms=round((time.perf_counter() - started) * 1000, 2),
        timings_ms=outcome["timings_ms"],
        results=outcome["results"],
    )


@router.get("/playlists", response_model=list[PlaylistInfo], summary="Danh sách playlist hợp lệ")
def list_playlists(request: Request, repo: SQLiteRepository = Depends(get_sqlite_repo)):
    """Registry ĐỘNG (không phải Enum): playlist là dữ liệu, đổi theo từng batch của BTC.

    `has_index=False` nghĩa là playlist có trong DB nhưng chưa build index FAISS — UI
    nên vô hiệu hóa lựa chọn đó vì lọc theo nó sẽ bị từ chối.
    """
    indexed = request.app.state.indexed_playlists
    return [
        PlaylistInfo(
            name=row["playlist"], n_videos=row["n_videos"], has_index=row["playlist"] in indexed
        )
        for row in repo.list_playlists()
    ]


@router.get("/models", response_model=list[ModelInfo], summary="Model đã nạp và khả năng của nó")
def list_models(request: Request):
    state = request.app.state
    infos = []
    for name, repo in sorted(state.faiss_repos.items()):
        infos.append(
            ModelInfo(
                name=name,
                loaded=True,
                supports_text=name in state.encoders,
                dim=repo.dim,
                n_vectors=repo.ntotal,
                playlists=repo.playlists,
            )
        )
    return infos


@router.get("/health", response_model=HealthResponse, summary="Trạng thái hệ thống")
def health(request: Request, repo: SQLiteRepository = Depends(get_sqlite_repo)):
    stats = repo.stats() if database_exists() else {"n_videos": 0, "n_keyframes": 0}
    return HealthResponse(
        status="ok" if request.app.state.encoders and stats["n_keyframes"] else "degraded",
        database=database_exists(),
        n_videos=stats["n_videos"],
        n_keyframes=stats["n_keyframes"],
        models=list_models(request),
    )


@router.get("/keyframes/{video_id}/{keyframe_name}.jpg", summary="Ảnh keyframe")
def get_keyframe_image(video_id: str, keyframe_name: str, request: Request):
    if not (SAFE_ID.match(video_id) and SAFE_ID.match(keyframe_name)):
        raise HTTPException(status_code=400, detail="video_id hoặc keyframe_name không hợp lệ")

    keyframes_dir = request.app.state.settings.keyframes_dir
    # Layout: keyframes_transnetv2/Keyframes_LXX/LXX_VXXX/0052.jpg. Tên thư mục batch
    # (Keyframes_LXX) suy được từ tiền tố video_id nên không cần glob toàn bộ đĩa.
    batch = f"Keyframes_{video_id.split('_')[0]}"
    path = os.path.join(keyframes_dir, batch, video_id, f"{keyframe_name}.jpg")
    if not os.path.exists(path):
        raise HTTPException(status_code=404, detail=f"Không có ảnh {video_id}/{keyframe_name}")
    # Keyframe là file tĩnh, sinh một lần lúc index và không bao giờ đổi. `immutable` cho
    # trình duyệt dùng thẳng cache mà KHÔNG bắn request kiểm tra (kể cả 304 cũng tốn một
    # socket) — quan trọng vì trình duyệt chỉ mở 6 kết nối cho mỗi origin, và mỗi lần tìm
    # kiếm lại là vài chục ảnh tranh nhau đúng 6 khe đó với request video.
    return FileResponse(
        path,
        media_type="image/jpeg",
        headers={"Cache-Control": "public, max-age=31536000, immutable"},
    )


@router.get("/videos/{video_id}", response_model=VideoInfo, summary="Thông tin video (fps, độ dài)")
def get_video_info(video_id: str, request: Request,
                   repo: SQLiteRepository = Depends(get_sqlite_repo)):
    if not SAFE_ID.match(video_id):
        raise HTTPException(status_code=400, detail="video_id không hợp lệ")

    video = repo.get_video(video_id) if database_exists() else None
    if video is None:
        raise HTTPException(status_code=404, detail=f"Không có video '{video_id}' trong metadata.db")

    return VideoInfo(
        video_id=video_id,
        playlist=video.get("playlist"),
        title=video.get("title"),
        author=video.get("author"),
        fps=video.get("fps"),
        duration=video.get("duration"),
        total_frames=video.get("total_frames"),
        watch_url=video.get("watch_url"),
        has_file=os.path.exists(_video_path(request.app.state.settings, video_id)),
        stream_url=f"/api/v1/videos/{video_id}/stream",
    )


@router.get("/videos/{video_id}/stream", summary="Phát video gốc (hỗ trợ tua)")
def stream_video(video_id: str, request: Request):
    """Trả file .mp4 để thẻ <video> phát trực tiếp.

    Không tự cắt đoạn quanh frame: FileResponse của Starlette đã xử lý header `Range`,
    nên trình duyệt tự tải đúng khúc nó cần và tua tức thì. Frontend chỉ việc set
    `video.currentTime = frame_idx / fps` — rẻ hơn và chính xác hơn nhiều so với việc
    backend transcode ra một clip con.
    """
    if not SAFE_ID.match(video_id):
        raise HTTPException(status_code=400, detail="video_id không hợp lệ")

    path = _video_path(request.app.state.settings, video_id)
    if not os.path.exists(path):
        raise HTTPException(status_code=404, detail=f"Không có file video cho '{video_id}'")
    # KHÔNG truyền `filename=`: Starlette sẽ sinh `Content-Disposition: attachment`, tức
    # là báo "đây là file để tải về", không phải luồng để phát. Thẻ <video> gặp header
    # này thì tuỳ trình duyệt mà im lặng bỏ qua hoặc bắn `error` — chính là lỗi
    # "Trình duyệt không phát được file video này".
    return FileResponse(
        path,
        media_type="video/mp4",
        headers={"Cache-Control": "public, max-age=86400"},
    )


@router.post("/submissions", response_model=SubmissionResponse, summary="Ghi file CSV nộp bài")
def create_submission(payload: SubmissionRequest, request: Request,
                      repo: SQLiteRepository = Depends(get_sqlite_repo)):
    rows = [row.model_dump() for row in payload.rows]
    query_type = payload.query_type.value

    warnings = (
        submission_service.check_rows(query_type, rows, repo.get_video)
        if database_exists() else []
    )
    try:
        written = submission_service.write_submission(
            submissions_dir=request.app.state.settings.submissions_dir,
            name=payload.name,
            query_type=query_type,
            rows=rows,
            overwrite=payload.overwrite,
        )
    except submission_service.SubmissionError as exc:
        raise HTTPException(status_code=400, detail=str(exc))
    except OSError as exc:
        raise HTTPException(status_code=500, detail=f"Không ghi được file: {exc}")

    return SubmissionResponse(**written, warnings=warnings)


@router.get("/submissions", response_model=list[SubmissionFile], summary="Các file đã nộp")
def list_submissions(request: Request):
    directory = request.app.state.settings.submissions_dir
    if not os.path.isdir(directory):
        return []
    files = []
    for name in sorted(os.listdir(directory)):
        if not name.endswith(".csv"):
            continue
        stat = os.stat(os.path.join(directory, name))
        files.append(SubmissionFile(name=name, size_bytes=stat.st_size, modified_at=stat.st_mtime))
    return sorted(files, key=lambda f: f.modified_at, reverse=True)
