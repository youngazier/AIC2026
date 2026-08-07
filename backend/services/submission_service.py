"""Sinh file CSV nộp bài cho BTC (xem docs/Thong-tin-vong-So-tuyen-AIC2026.pdf mục 2.1).

Định dạng mỗi dòng, KHÔNG có dòng tiêu đề:

    KIS     <video_id>,<frame_idx>
    QnA     <video_id>,<frame_idx>,<answer>
    TRAKE   <video_id>,<frame_1>,<frame_2>,...,<frame_n>

THỨ TỰ DÒNG LÀ ĐIỂM SỐ, không phải chuyện thẩm mỹ: Final Score = trung bình của R@k
với k ∈ {1,5,20,50,100}, nên câu trả lời đúng nằm ở dòng 1 ăn điểm gấp nhiều lần cũng
câu đó nằm ở dòng 60. Service này ghi ĐÚNG thứ tự frontend gửi lên, không tự sắp xếp
lại theo score hay theo bất cứ thứ gì khác.

Trần 100 dòng là quy định của BTC ("đội thi được gửi tối đa 100 câu trả lời"), nên
đây là lỗi 400 chứ không phải tự cắt bớt: cắt im lặng sẽ vứt mất đáp án mà người dùng
tưởng là đã nộp.
"""
import csv
import logging
import os
import re

logger = logging.getLogger(__name__)

MAX_ROWS = 100

# Tên file do người dùng tự gõ -> phải chặn path traversal (../../etc/passwd) và mọi
# ký tự có nghĩa với shell/filesystem. Cho phép đúng chữ, số, gạch dưới, gạch ngang.
SAFE_NAME = re.compile(r"^[A-Za-z0-9_-]{1,64}$")
SAFE_VIDEO_ID = re.compile(r"^[A-Za-z0-9_-]{1,64}$")


class SubmissionError(ValueError):
    """Dữ liệu nộp sai định dạng -> HTTP 400."""


def _clean_name(name: str) -> str:
    stem = (name or "").strip()
    if stem.lower().endswith(".csv"):
        stem = stem[: -len(".csv")]
    if not SAFE_NAME.match(stem):
        raise SubmissionError(
            f"Tên file '{name}' không hợp lệ. Chỉ dùng chữ, số, '_' và '-' (tối đa 64 ký tự)."
        )
    return stem


def _clean_frames(frames, row_no: int) -> list:
    if not frames:
        raise SubmissionError(f"Dòng {row_no}: phải có ít nhất một frame.")
    cleaned = []
    for frame in frames:
        try:
            value = int(frame)
        except (TypeError, ValueError):
            raise SubmissionError(f"Dòng {row_no}: frame '{frame}' không phải số nguyên.")
        if value < 0:
            raise SubmissionError(f"Dòng {row_no}: frame {value} âm.")
        cleaned.append(value)
    return cleaned


def build_rows(query_type: str, rows: list) -> list:
    """Đổi list dict từ frontend thành list list chuỗi sẵn sàng ghi CSV."""
    kind = (query_type or "kis").lower()
    if kind not in {"kis", "qna", "trake"}:
        raise SubmissionError(f"Loại truy vấn '{query_type}' không hợp lệ (kis | qna | trake).")
    if not rows:
        raise SubmissionError("Danh sách nộp đang trống.")
    if len(rows) > MAX_ROWS:
        raise SubmissionError(
            f"Nộp {len(rows)} dòng nhưng BTC chỉ nhận tối đa {MAX_ROWS}. "
            "Bỏ bớt các kết quả kém tự tin nhất ở cuối danh sách."
        )

    built = []
    for row_no, row in enumerate(rows, start=1):
        video_id = str(row.get("video_id") or "").strip()
        if not SAFE_VIDEO_ID.match(video_id):
            raise SubmissionError(f"Dòng {row_no}: video_id '{video_id}' không hợp lệ.")

        frames = _clean_frames(row.get("frames"), row_no)

        if kind == "trake":
            built.append([video_id, *[str(f) for f in frames]])
            continue

        if len(frames) != 1:
            raise SubmissionError(
                f"Dòng {row_no}: {kind.upper()} chỉ nhận đúng 1 frame, đang có {len(frames)}."
            )
        if kind == "kis":
            built.append([video_id, str(frames[0])])
        else:  # qna
            answer = str(row.get("answer") or "").strip()
            if not answer:
                raise SubmissionError(
                    f"Dòng {row_no}: truy vấn Q&A bắt buộc có câu trả lời "
                    "(nộp thiếu answer thì R-Score = 0)."
                )
            built.append([video_id, str(frames[0]), answer])

    return built


def write_submission(submissions_dir: str, name: str, query_type: str, rows: list,
                     overwrite: bool = True) -> dict:
    """Ghi file CSV, trả về thông tin file đã ghi."""
    stem = _clean_name(name)
    built = build_rows(query_type, rows)

    os.makedirs(submissions_dir, exist_ok=True)
    path = os.path.join(submissions_dir, f"{stem}.csv")
    if os.path.exists(path) and not overwrite:
        raise SubmissionError(f"File '{stem}.csv' đã tồn tại. Đổi tên khác hoặc bật ghi đè.")

    # newline="" là bắt buộc của module csv trên mọi nền tảng; thiếu nó Windows sẽ
    # sinh ra dòng trống xen kẽ. lineterminator="\n" để file giống nhau ở mọi máy.
    with open(path, "w", newline="", encoding="utf-8") as handle:
        csv.writer(handle, lineterminator="\n").writerows(built)

    logger.info("Đã ghi %s (%d dòng, loại %s)", path, len(built), query_type)
    return {"name": f"{stem}.csv", "path": path, "n_rows": len(built)}


def check_rows(query_type: str, rows: list, get_video) -> list:
    """Đối chiếu với metadata.db, trả về danh sách CẢNH BÁO (không chặn việc ghi file).

    Cố ý là cảnh báo chứ không phải lỗi: DB có thể chưa ingest hết batch mới của BTC,
    và lúc thi thì thà nộp được file có nghi vấn còn hơn bị chặn vì một video mà hệ
    thống chưa biết. Nhưng gõ nhầm 'L21_V01' thành 'L21_V1' là lỗi rất dễ mắc và rất
    đắt, nên vẫn phải nói cho người dùng biết.
    """
    warnings = []
    cache = {}
    for row_no, row in enumerate(rows, start=1):
        video_id = str(row.get("video_id") or "").strip()
        if video_id not in cache:
            try:
                cache[video_id] = get_video(video_id)
            except Exception:  # DB lỗi thì bỏ qua bước kiểm tra, đừng làm hỏng việc nộp
                logger.exception("Không tra cứu được video '%s' để kiểm tra", video_id)
                cache[video_id] = None
        video = cache[video_id]

        if video is None:
            warnings.append(f"Dòng {row_no}: không tìm thấy video '{video_id}' trong metadata.db.")
            continue

        total = video.get("total_frames")
        if not total:
            continue
        for frame in row.get("frames") or []:
            try:
                value = int(frame)
            except (TypeError, ValueError):
                continue
            if value >= total:
                warnings.append(
                    f"Dòng {row_no}: frame {value} vượt quá độ dài video '{video_id}' "
                    f"({total} frame)."
                )
    return warnings
