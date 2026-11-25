"""Tests for Step 8 (Items 8 & 9): team merge workflow and tiny-video E2E."""

from contextlib import contextmanager
import json
import logging
from pathlib import Path
import shutil
import subprocess

import faiss
import numpy as np
import pytest
from sqlalchemy import func, select

from backend.app.database import create_catalog_engine
from backend.app.models import Keyframe, Shot, Video
from data_processing import cli
from data_processing.phase1_keyframe_extraction import pipeline as phase1_pipeline
from data_processing.phase2_feature_extraction import models as model_registry
from data_processing.phase2_feature_extraction import pipeline as phase2_pipeline
from data_processing.phase2_feature_extraction.build_index import LOGGER as INDEX_LOGGER
from data_processing.phase2_feature_extraction.clip_embedding import CLIPEmbedding


REAL_WEIGHTS = phase1_pipeline.DEFAULT_WEIGHTS


@contextmanager
def _use_tree(monkeypatch, root: Path, *, weights: Path | None = None):
    with monkeypatch.context() as scoped:
        scoped.setattr(phase1_pipeline, "PROJECT_ROOT", root)
        scoped.setattr(phase1_pipeline, "DEFAULT_OUTPUT_ROOT", root / "data_processed/transnetv2")
        scoped.setattr(phase1_pipeline, "DEFAULT_WEIGHTS", weights or (root / "weights.pth"))
        scoped.setattr(phase2_pipeline, "PROJECT_ROOT", root)
        scoped.setattr(phase2_pipeline, "DEFAULT_PHASE1_ROOT", root / "data_processed/transnetv2")
        scoped.setattr(phase2_pipeline, "DEFAULT_KEYFRAME_ROOT", root / "data_processed/keyframes")
        scoped.setattr(phase2_pipeline, "DEFAULT_OUTPUT_ROOT", root / "data_processed/features")
        yield


def _populate_member(root: Path, videos: tuple[tuple[str, str], ...], tiny_video: Path) -> None:
    for batch, filename in videos:
        target = root / "data/videos" / batch / filename
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(tiny_video, target)
    (root / "data_processed").mkdir(parents=True, exist_ok=True)
    (root / "weights.pth").write_bytes(b"stub-transnetv2-weights")


def _build_two_member_trees(tmp_path: Path, tiny_video: Path, monkeypatch, stub_embedder) -> tuple[Path, Path]:
    monkeypatch.setitem(
        model_registry.MODELS,
        "clip",
        (stub_embedder, stub_embedder.DEFAULT_MODEL),
    )
    member_a = tmp_path / "member_a"
    member_b = tmp_path / "member_b"
    _populate_member(member_a, (("Videos_L21", "L21_V001.mp4"), ("Videos_L21", "L21_V002.mp4")), tiny_video)
    _populate_member(member_b, (("Videos_L22", "L22_V001.mp4"),), tiny_video)

    with _use_tree(monkeypatch, member_a):
        assert cli.main(["all", "--batch", "Videos_L21", "--model", "clip", "--no-phase3"]) == 0
    with _use_tree(monkeypatch, member_b):
        assert cli.main(["all", "--batch", "Videos_L22", "--model", "clip", "--no-phase3"]) == 0
    return member_a, member_b


def _assert_merged_outputs(merged: Path, stub_detector, stub_embedder, monkeypatch) -> None:
    index_path = merged / "data_processed/indexes/clip/global.index"
    rows_path = merged / "data_processed/indexes/clip/global_rows.json"
    assert faiss.read_index(str(index_path)).ntotal == 9
    rows = json.loads(rows_path.read_text())
    assert {row["video_name"] for row in rows} == {"L21_V001", "L21_V002", "L22_V001"}
    assert {row["batch"] for row in rows} == {"Videos_L21", "Videos_L22"}

    engine = create_catalog_engine(merged / "data_processed/database/catalog.sqlite3")
    try:
        with engine.connect() as conn:
            assert conn.scalar(select(func.count()).select_from(Video)) == 3
            assert conn.scalar(select(func.count()).select_from(Shot)) == 9
            assert conn.scalar(select(func.count()).select_from(Keyframe)) == 9
    finally:
        engine.dispose()

    stub_detector.calls = 0
    stub_embedder.calls = 0
    with _use_tree(monkeypatch, merged):
        assert cli.main(["phase1", "--all"]) == 0
        assert cli.main(["phase2", "--all", "--model", "clip"]) == 0
    assert stub_detector.calls == 0
    assert stub_embedder.calls == 0


def test_tar_merge_workflow_indexes_catalogs_and_skips(
    tmp_path: Path, tiny_video: Path, stub_detector, stub_embedder, monkeypatch
):
    member_a, member_b = _build_two_member_trees(tmp_path, tiny_video, monkeypatch, stub_embedder)
    tar_a = tmp_path / "Videos_L21.tgz"
    tar_b = tmp_path / "Videos_L22.tgz"

    subprocess.run(
        [
            "tar",
            "-C",
            str(member_a / "data_processed"),
            "-czf",
            str(tar_a),
            "transnetv2/Videos_L21",
            "keyframes/Videos_L21",
            "features/clip",
        ],
        check=True,
    )
    subprocess.run(
        [
            "tar",
            "-C",
            str(member_b / "data_processed"),
            "-czf",
            str(tar_b),
            "transnetv2/Videos_L22",
            "keyframes/Videos_L22",
            "features/clip",
        ],
        check=True,
    )

    merged = tmp_path / "merged"
    _populate_member(
        merged,
        (("Videos_L21", "L21_V001.mp4"), ("Videos_L21", "L21_V002.mp4"), ("Videos_L22", "L22_V001.mp4")),
        tiny_video,
    )
    subprocess.run(["tar", "-C", str(merged / "data_processed"), "-xzf", str(tar_a)], check=True)
    subprocess.run(["tar", "-C", str(merged / "data_processed"), "-xzf", str(tar_b)], check=True)

    with _use_tree(monkeypatch, merged):
        assert cli.main(["phase3", "--model", "clip"]) == 0

    _assert_merged_outputs(merged, stub_detector, stub_embedder, monkeypatch)


def test_cp_merge_workflow_skips_and_warns_on_missing_features(
    tmp_path: Path, tiny_video: Path, stub_detector, stub_embedder, monkeypatch, caplog
):
    member_a, member_b = _build_two_member_trees(tmp_path, tiny_video, monkeypatch, stub_embedder)

    merged = tmp_path / "merged_cp"
    _populate_member(
        merged,
        (("Videos_L21", "L21_V001.mp4"), ("Videos_L21", "L21_V002.mp4"), ("Videos_L22", "L22_V001.mp4")),
        tiny_video,
    )
    for src_root in (member_a / "data_processed", member_b / "data_processed"):
        for sub in ("transnetv2", "keyframes", "features"):
            subprocess.run(
                ["cp", "-r", str(src_root / sub), str(merged / "data_processed") + "/"],
                check=True,
            )

    with _use_tree(monkeypatch, merged):
        assert cli.main(["phase3", "--model", "clip"]) == 0

    _assert_merged_outputs(merged, stub_detector, stub_embedder, monkeypatch)

    (merged / "data_processed/features/clip/L22_V001.npy").unlink()
    (merged / "data_processed/features/clip/L22_V001.json").unlink()
    with caplog.at_level(logging.WARNING, logger=INDEX_LOGGER.name):
        with _use_tree(monkeypatch, merged):
            assert cli.main(["phase3", "--model", "clip"]) == 0

    assert "L22_V001" in "\n".join(caplog.messages)
    assert faiss.read_index(str(merged / "data_processed/indexes/clip/global.index")).ntotal == 6


def test_fast_tiny_video_end_to_end_with_stubs(
    tmp_path: Path, tiny_video: Path, stub_detector, stub_embedder, monkeypatch
):
    monkeypatch.setitem(
        model_registry.MODELS,
        "clip",
        (stub_embedder, stub_embedder.DEFAULT_MODEL),
    )
    root = tmp_path / "e2e_fast"
    _populate_member(root, (("Videos_L21", "L21_V001.mp4"),), tiny_video)

    with _use_tree(monkeypatch, root):
        assert cli.main(["all", "--batch", "Videos_L21", "--model", "clip", "--device", "cpu"]) == 0

    index = faiss.read_index(str(root / "data_processed/indexes/clip/global.index"))
    rows = json.loads((root / "data_processed/indexes/clip/global_rows.json").read_text())
    assert index.ntotal == 3

    query = stub_embedder().encode_text("red").numpy()[None, :].astype(np.float32)
    _, indices = index.search(query, 3)
    assert rows[int(indices[0][0])]["shot_index"] == 1


@pytest.mark.slow
def test_tiny_video_end_to_end_real_models(tmp_path: Path, tiny_video: Path, monkeypatch):
    if not REAL_WEIGHTS.is_file():
        pytest.skip(f"TransNetV2 weights not found at {REAL_WEIGHTS}")

    root = tmp_path / "e2e_real"
    target = root / "data/videos/Videos_L21/L21_V001.mp4"
    target.parent.mkdir(parents=True)
    shutil.copy2(tiny_video, target)
    (root / "data_processed").mkdir()

    with _use_tree(monkeypatch, root, weights=REAL_WEIGHTS):
        exit_code = cli.main(["all", "--batch", "Videos_L21", "--model", "clip", "--device", "cpu"])
    assert exit_code == 0

    shots = json.loads((root / "data_processed/transnetv2/Videos_L21/L21_V001/shots.json").read_text())
    assert len(shots["shots"]) == 3
    for i in range(3):
        assert (root / f"data_processed/keyframes/Videos_L21/L21_V001/{i:05d}.jpg").is_file()

    vectors = np.load(root / "data_processed/features/clip/L21_V001.npy", allow_pickle=False)
    assert vectors.shape == (3, 512)

    index = faiss.read_index(str(root / "data_processed/indexes/clip/global.index"))
    assert index.ntotal == 3
    rows = json.loads((root / "data_processed/indexes/clip/global_rows.json").read_text())

    engine = create_catalog_engine(root / "data_processed/database/catalog.sqlite3")
    try:
        with engine.connect() as conn:
            assert conn.scalar(select(func.count()).select_from(Video)) == 1
            assert conn.scalar(select(func.count()).select_from(Shot)) == 3
            assert conn.scalar(select(func.count()).select_from(Keyframe)) == 3
    finally:
        engine.dispose()

    query = CLIPEmbedding(device="cpu").encode_text("red").numpy()[None, :].astype(np.float32)
    _, indices = index.search(query, 3)
    assert rows[int(indices[0][0])]["shot_index"] == 1
