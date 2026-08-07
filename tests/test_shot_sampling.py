"""Kiểm tra luật chọn keyframe (shot_sampling.py).

Đây là bài test QUAN TRỌNG NHẤT của bộ này dù trông đơn giản nhất: mọi kiểm tra
đầy đủ ở check_integrity.py đều dựng tập kỳ vọng từ luật này. Luật sai thì trình
kiểm tra sẽ xác nhận một sự thật sai, còn tệ hơn không kiểm gì.

Vì vậy các con số dưới đây được viết TAY, không suy ra từ chính hàm đang test.
"""
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from data_processing.keyframe_extraction.shot_sampling import (  # noqa: E402
    expected_keyframes,
    parse_scene_lines,
    sample_shot_frames,
)


class TestSampleShotFrames:
    @pytest.mark.parametrize("left,right,want", [
        # shot ngắn (< 75 frame) -> 1 frame ở giữa
        (0, 10, [5]),
        (0, 74, [37]),
        (100, 174, [137]),
        # shot vừa (75..200) -> 2 frame ở 25% và 75%
        (0, 75, [18, 56]),
        (0, 200, [50, 150]),
        (1000, 1100, [1025, 1075]),
        # shot dài (> 200) -> 3 frame ở 20%, 50%, 80%
        (0, 201, [40, 100, 160]),
        (0, 1000, [200, 500, 800]),
    ])
    def test_ranh_gioi_do_dai(self, left, right, want):
        assert sample_shot_frames(left, right) == want

    def test_frame_luon_nam_trong_shot(self):
        for left, right in [(0, 3), (0, 74), (0, 75), (0, 200), (0, 201), (5, 5000)]:
            for frame in sample_shot_frames(left, right):
                assert left <= frame <= right

    def test_shot_dai_1_frame(self):
        """Shot chỉ có một frame vẫn phải sinh ra đúng frame đó, không lỗi."""
        assert sample_shot_frames(42, 42) == [42]


class TestParseSceneLines:
    def test_giu_so_dong_tho(self):
        """shot_idx trong map csv chính là số dòng thô, kể cả khi có dòng trắng."""
        shots, bad = parse_scene_lines(["0 9\n", "\n", "10 19\n"])
        assert shots == [(1, 0, 9), (3, 10, 19)]
        assert bad == []

    def test_dong_rac_bi_ghi_nhan(self):
        shots, bad = parse_scene_lines(["0 9\n", "khong-phai-so\n", "abc def\n", "10 19\n"])
        assert shots == [(1, 0, 9), (4, 10, 19)]
        assert bad == [2, 3]

    def test_file_rong(self):
        assert parse_scene_lines([]) == ([], [])


class TestExpectedKeyframes:
    def test_danh_so_n_lien_tuc_tu_1(self):
        expected, shots, _ = expected_keyframes(["0 249\n", "250 299\n"])
        assert [n for n, _, _ in expected] == [1, 2, 3, 4]
        assert len(shots) == 2

    def test_ket_qua_day_du(self):
        """(n, shot_idx, frame_idx) — viết tay, đối chiếu với luật ở trên."""
        expected, _, _ = expected_keyframes(["0 249\n", "250 299\n"])
        assert expected == [
            (1, 1, 49),    # shot 1 dài 249 -> 20%
            (2, 1, 124),   # -> 50%
            (3, 1, 199),   # -> 80%
            (4, 2, 274),   # shot 2 dài 49 -> giữa shot
        ]

    def test_frame_idx_tang_dan(self):
        expected, _, _ = expected_keyframes(["0 99\n", "100 199\n", "200 999\n"])
        frames = [f for _, _, f in expected]
        assert frames == sorted(frames)

    def test_dong_hong_khong_lam_lech_n(self):
        """Dòng rác bị bỏ qua nhưng n vẫn liên tục — không được thủng số."""
        expected, _, bad = expected_keyframes(["0 9\n", "rac\n", "10 19\n"])
        assert bad == [2]
        assert [n for n, _, _ in expected] == [1, 2]
        assert [s for _, s, _ in expected] == [1, 3]   # shot_idx theo dòng thô
