"""Luật chọn keyframe từ ranh giới shot của TransNetV2 — NƠI KHAI BÁO DUY NHẤT.

`keyframe_extraction.py` (bên GHI keyframe) và `check_integrity.py` (bên KIỂM)
nằm cùng thư mục này và đều phải gọi hàm ở đây. Lý do giống hệt ghi chú về
đường dẫn trong config.yaml:
nếu mỗi bên tự chép lại luật, sớm muộn hai bản sẽ lệch nhau — và lúc đó trình
kiểm tra sẽ xác nhận một sự thật sai, tệ hơn nhiều so với không kiểm tra gì.

Tập kỳ vọng (expected set) của cả pipeline keyframe được suy ra từ đây, tức là
từ ĐẦU VÀO (scenes.txt), không phải từ đầu ra (map csv / thư mục .jpg). Đó là
điều kiện tiên quyết để phép đối chiếu có ý nghĩa.
"""
from typing import Iterable, List, Tuple

# Số frame lấy ra tuỳ độ dài shot (đơn vị: frame).
#   shot ngắn   (< 75)        -> 1 frame giữa shot
#   shot vừa    (75..200)     -> 2 frame ở 25% và 75%
#   shot dài    (> 200)       -> 3 frame ở 20%, 50%, 80%
SHORT_SHOT_FRAMES = 75
MEDIUM_SHOT_FRAMES = 200


def sample_shot_frames(left: int, right: int) -> List[int]:
    """Trả về danh sách frame index cần trích cho một shot [left, right]."""
    duration = right - left
    if duration < SHORT_SHOT_FRAMES:
        return [left + duration // 2]
    if duration <= MEDIUM_SHOT_FRAMES:
        return [left + int(duration * 0.25), left + int(duration * 0.75)]
    return [
        left + int(duration * 0.20),
        left + int(duration * 0.50),
        left + int(duration * 0.80),
    ]


def parse_scene_lines(lines: Iterable[str]) -> Tuple[List[Tuple[int, int, int]], List[int]]:
    """Đọc nội dung file `*.scenes.txt` -> (shots, bad_lines).

    shots     : [(line_no, left, right)] với line_no đếm từ 1 theo DÒNG THÔ.
                Giữ số dòng thô là cố ý: cột `shot_idx` trong map csv chính là
                số này, kể cả khi ở giữa có dòng hỏng bị bỏ qua.
    bad_lines : số thứ tự các dòng có nội dung nhưng không parse được.
    """
    shots: List[Tuple[int, int, int]] = []
    bad_lines: List[int] = []
    for i, line in enumerate(lines):
        parts = line.strip().split()
        if len(parts) < 2:
            if line.strip():  # dòng trắng là bình thường, dòng rác thì không
                bad_lines.append(i + 1)
            continue
        try:
            left, right = int(parts[0]), int(parts[1])
        except ValueError:
            bad_lines.append(i + 1)
            continue
        shots.append((i + 1, left, right))
    return shots, bad_lines


def expected_keyframes(lines: Iterable[str]):
    """Dựng tập keyframe KỲ VỌNG từ nội dung scenes.txt.

    Trả về (expected, shots, bad_lines) với
        expected = [(n, shot_idx, frame_idx)] đã đánh số n từ 1
    Mỗi phần tử ứng với đúng một dòng trong map csv và đúng một file
    `{n:04d}.jpg` trong thư mục keyframe.
    """
    lines = list(lines)
    shots, bad_lines = parse_scene_lines(lines)
    expected = []
    n = 1
    for line_no, left, right in shots:
        for frame_idx in sample_shot_frames(left, right):
            expected.append((n, line_no, frame_idx))
            n += 1
    return expected, shots, bad_lines
