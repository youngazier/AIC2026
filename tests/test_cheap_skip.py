"""Item 5 tests: cheap skip checks, --verify deep checks, and summary output line."""

from pathlib import Path
import shutil

from PIL import Image
import pytest

from data_processing.cli import main
from data_processing.phase2_feature_extraction.models import MODELS


def test_phase2_skip_opens_zero_keyframe_images(
    processed_tree: Path, stub_embedder, monkeypatch
):
    monkeypatch.setitem(MODELS, "clip", (stub_embedder, stub_embedder.DEFAULT_MODEL))
    monkeypatch.setattr(
        Image,
        "open",
        lambda *args, **kwargs: pytest.fail("PIL.Image.open should not be called on Phase 2 skip"),
    )
    assert main(["phase2", "--all"]) == 0
    assert stub_embedder.calls == 0


def test_deleted_keyframe_fails_quick_check_and_reprocesses(processed_tree: Path, stub_detector):
    missing = processed_tree / "data_processed/keyframes/Videos_L21/L21_V001/00001.jpg"
    missing.unlink()

    assert main(["phase1", "--all"]) == 0
    assert stub_detector.calls == 1
    assert missing.is_file()


def test_truncated_header_fails_quick_and_damaged_pixels_caught_by_verify(
    processed_tree: Path, stub_detector
):
    keyframe = processed_tree / "data_processed/keyframes/Videos_L21/L21_V001/00000.jpg"
    original = keyframe.read_bytes()

    # 1. Truncated header: quick check rejects and rebuilds.
    keyframe.write_bytes(original[:4])
    assert main(["phase1", "--video", "L21_V001"]) == 0
    assert stub_detector.calls == 1

    # 2. Intact header, truncated scan pixels: quick check accepts, --verify catches.
    raw = keyframe.read_bytes()
    sos = raw.rindex(b"\xff\xda")
    damaged = raw[: sos + 20]
    keyframe.write_bytes(damaged)
    with Image.open(keyframe) as opened:
        assert opened.format == "JPEG"
    with pytest.raises(OSError):
        with Image.open(keyframe) as opened:
            opened.load()

    stub_detector.calls = 0
    assert main(["phase1", "--video", "L21_V001"]) == 0
    assert stub_detector.calls == 0

    assert main(["phase1", "--video", "L21_V001", "--verify"]) == 0
    assert stub_detector.calls == 1
    with Image.open(keyframe) as repaired:
        repaired.load()


def test_verify_on_good_output_skips_phase1_and_phase2(
    processed_tree: Path, stub_detector, stub_embedder, monkeypatch
):
    monkeypatch.setitem(MODELS, "clip", (stub_embedder, stub_embedder.DEFAULT_MODEL))
    assert main(["phase1", "--all", "--verify"]) == 0
    assert stub_detector.calls == 0

    assert main(["phase2", "--all", "--verify"]) == 0
    assert stub_embedder.calls == 0


def test_keyframe_count_mismatch_reprocesses(processed_tree: Path, stub_detector):
    kf_dir = processed_tree / "data_processed/keyframes/Videos_L21/L21_V001"
    extra = kf_dir / "00003.jpg"
    shutil.copy2(kf_dir / "00000.jpg", extra)

    assert main(["phase1", "--all"]) == 0
    assert stub_detector.calls == 1
    assert not extra.exists()


def test_end_of_run_summary_line_matches_reality(
    data_tree: Path, stub_detector, stub_embedder, monkeypatch, capsys
):
    monkeypatch.setitem(MODELS, "clip", (stub_embedder, stub_embedder.DEFAULT_MODEL))

    assert main(["phase1", "--video", "L21_V001", "L21_V002"]) == 0
    assert "processed 2, skipped 0, failed 0" in capsys.readouterr().out

    assert main(["phase1", "--all"]) == 0
    assert "processed 1, skipped 2, failed 0" in capsys.readouterr().out

    assert main(["phase2", "--video", "L21_V001"]) == 0
    assert "processed 1, skipped 0, failed 0" in capsys.readouterr().out

    assert main(["phase2", "--all"]) == 0
    assert "processed 2, skipped 1, failed 0" in capsys.readouterr().out
