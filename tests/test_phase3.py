"""Item 3 tests: phase3 command, SQLite catalog import, safe index build, and indexes/<model>/ layout."""

import json
import logging
import os
from pathlib import Path
import shutil

import faiss
import numpy as np
import pytest

from backend.app.catalog.queries import CatalogQueries
from backend.app.database import create_catalog_engine, create_session_factory
from data_processing.cli import main
from data_processing.phase2_feature_extraction.build_index import build_for_model


def test_phase3_builds_global_index_rows_and_manifest(processed_tree: Path, capsys):
    assert main(["phase3", "--model", "clip"]) == 0
    out = capsys.readouterr().out
    assert "Videos_L21" in out
    assert "Videos_L22" in out

    model_index_dir = processed_tree / "data_processed/indexes/clip"
    index_path = model_index_dir / "global.index"
    rows_path = model_index_dir / "global_rows.json"
    manifest_path = model_index_dir / "manifest.json"
    assert index_path.is_file()
    assert rows_path.is_file()
    assert manifest_path.is_file()

    index = faiss.read_index(str(index_path))
    rows = json.loads(rows_path.read_text())
    manifest = json.loads(manifest_path.read_text())
    assert index.ntotal == 9
    assert index.d == 8
    assert len(rows) == 9
    assert manifest["vector_count"] == 9
    assert manifest["dimension"] == 8

    for i, row in enumerate(rows):
        assert row["index"] == i
        shots = json.loads(
            (processed_tree / "data_processed/transnetv2" / row["batch"] / row["video_name"] / "shots.json").read_text()
        )
        shot = shots["shots"][row["shot_index"]]
        assert row["frame_index"] == shot["keyframe"]["frame_index"]
        assert row["timestamp"] == shot["keyframe"]["timestamp"]
        assert row["image_path"] == shot["keyframe"]["image_path"]
        assert (processed_tree / "data_processed" / row["image_path"]).is_file()


def test_phase3_builds_playlist_indexes(processed_tree: Path):
    assert main(["phase3", "--model", "clip"]) == 0
    model_index_dir = processed_tree / "data_processed/indexes/clip"
    global_rows = json.loads((model_index_dir / "global_rows.json").read_text())
    global_keys = {(r["video_name"], r["shot_index"]): r for r in global_rows}

    news_idx = faiss.read_index(str(model_index_dir / "playlist/news.index"))
    news_rows = json.loads((model_index_dir / "playlist/news_rows.json").read_text())
    sports_idx = faiss.read_index(str(model_index_dir / "playlist/sports.index"))
    sports_rows = json.loads((model_index_dir / "playlist/sports_rows.json").read_text())

    assert news_idx.ntotal == len(news_rows) == 6
    assert sports_idx.ntotal == len(sports_rows) == 3
    for row in news_rows + sports_rows:
        key = (row["video_name"], row["shot_index"])
        assert key in global_keys
        assert row["timestamp"] == global_keys[key]["timestamp"]
        assert row["image_path"] == global_keys[key]["image_path"]


def test_phase3_skips_unchanged_index(processed_tree: Path):
    assert main(["phase3", "--model", "clip", "--no-catalog"]) == 0
    index_path = processed_tree / "data_processed/indexes/clip/global.index"
    manifest_path = processed_tree / "data_processed/indexes/clip/manifest.json"
    mtime_before = index_path.stat().st_mtime_ns
    manifest_mtime_before = manifest_path.stat().st_mtime_ns

    assert main(["phase3", "--model", "clip", "--no-catalog"]) == 0
    assert index_path.stat().st_mtime_ns == mtime_before
    assert manifest_path.stat().st_mtime_ns == manifest_mtime_before


def test_truncated_global_index_is_rebuilt_with_reason_logged(processed_tree: Path, caplog):
    assert main(["phase3", "--model", "clip", "--no-catalog"]) == 0
    index_path = processed_tree / "data_processed/indexes/clip/global.index"
    index_path.write_bytes(b"bad!")

    with caplog.at_level(logging.INFO):
        assert main(["phase3", "--model", "clip", "--no-catalog"]) == 0
    assert "global.index" in caplog.text
    assert faiss.read_index(str(index_path)).ntotal == 9


def test_wrong_length_global_rows_rebuilds(processed_tree: Path, caplog):
    assert main(["phase3", "--model", "clip", "--no-catalog"]) == 0
    rows_path = processed_tree / "data_processed/indexes/clip/global_rows.json"
    rows_path.write_text("[]")

    with caplog.at_level(logging.INFO):
        assert main(["phase3", "--model", "clip", "--no-catalog"]) == 0
    assert len(json.loads(rows_path.read_text())) == 9


def test_missing_manifest_rebuilds(processed_tree: Path):
    assert main(["phase3", "--model", "clip", "--no-catalog"]) == 0
    manifest_path = processed_tree / "data_processed/indexes/clip/manifest.json"
    manifest_path.unlink()

    assert main(["phase3", "--model", "clip", "--no-catalog"]) == 0
    assert manifest_path.is_file()
    assert json.loads(manifest_path.read_text())["vector_count"] == 9


def test_added_feature_file_rebuilds_and_grows_ntotal(processed_tree: Path, tmp_path: Path):
    npy = processed_tree / "data_processed/features/clip/L22_V001.npy"
    meta = npy.with_suffix(".json")
    backup_npy = tmp_path / "L22_V001.npy"
    backup_meta = tmp_path / "L22_V001.json"
    shutil.move(npy, backup_npy)
    shutil.move(meta, backup_meta)

    assert main(["phase3", "--model", "clip", "--no-catalog"]) == 0
    index_path = processed_tree / "data_processed/indexes/clip/global.index"
    assert faiss.read_index(str(index_path)).ntotal == 6

    shutil.move(backup_npy, npy)
    shutil.move(backup_meta, meta)
    assert main(["phase3", "--model", "clip", "--no-catalog"]) == 0
    assert faiss.read_index(str(index_path)).ntotal == 9


def test_leftover_temp_file_deleted_at_startup(processed_tree: Path):
    assert main(["phase3", "--model", "clip", "--no-catalog"]) == 0
    model_index_dir = processed_tree / "data_processed/indexes/clip"
    stale_global = model_index_dir / ".global.index.deadbeef.tmp"
    stale_playlist = model_index_dir / "playlist/.news.index.deadbeef.tmp"
    stale_global.write_bytes(b"stale")
    stale_playlist.write_bytes(b"stale")

    assert main(["phase3", "--model", "clip", "--no-catalog"]) == 0
    assert not stale_global.exists()
    assert not stale_playlist.exists()


def test_crash_before_manifest_replace_recovers_on_next_run(processed_tree: Path, monkeypatch):
    real_replace = os.replace

    def fail_on_manifest(src, dst, *args, **kwargs):
        if Path(dst).name == "manifest.json":
            raise OSError("simulated crash before manifest")
        return real_replace(src, dst, *args, **kwargs)

    with monkeypatch.context() as scoped:
        scoped.setattr(os, "replace", fail_on_manifest)
        assert main(["phase3", "--model", "clip", "--no-catalog"]) == 1

    model_index_dir = processed_tree / "data_processed/indexes/clip"
    assert not (model_index_dir / "manifest.json").exists()

    assert main(["phase3", "--model", "clip", "--no-catalog"]) == 0
    assert (model_index_dir / "manifest.json").is_file()
    assert faiss.read_index(str(model_index_dir / "global.index")).ntotal == 9
    assert len(json.loads((model_index_dir / "global_rows.json").read_text())) == 9


def test_warns_about_phase1_video_missing_features(processed_tree: Path, caplog):
    (processed_tree / "data_processed/features/clip/L22_V001.npy").unlink()
    (processed_tree / "data_processed/features/clip/L22_V001.json").unlink()

    with caplog.at_level(logging.WARNING):
        assert main(["phase3", "--model", "clip", "--no-catalog"]) == 0
    assert "L22_V001" in caplog.text
    assert any(r.levelno == logging.WARNING and "L22_V001" in r.getMessage() for r in caplog.records)
    assert faiss.read_index(str(processed_tree / "data_processed/indexes/clip/global.index")).ntotal == 6


@pytest.mark.parametrize("case", ["dimension", "precision"])
def test_refuses_mixed_precisions_or_dimensions(tmp_path: Path, case: str):
    model_dir = tmp_path / "features/clip"
    model_dir.mkdir(parents=True)
    specs = (
        ("L21_V001", np.array([[1.0, 0.0, 0.0]], dtype=np.float32), "fp32"),
        (
            "L21_V002",
            np.array([[1.0, 0.0, 0.0, 0.0]] if case == "dimension" else [[1.0, 0.0, 0.0]], dtype=np.float32),
            "fp32" if case == "dimension" else "fp16",
        ),
    )
    for name, vec, precision in specs:
        npy = model_dir / f"{name}.npy"
        np.save(npy, vec, allow_pickle=False)
        npy.with_suffix(".json").write_text(json.dumps({
            "status": "COMPLETED",
            "model_name": "openai/clip-vit-base-patch32",
            "representation_version": "clip-l2-v1",
            "precision": precision,
            "batch": "Videos_L21",
            "video_name": name,
            "embedding_count": 1,
            "embedding_dimension": int(vec.shape[1]),
            "records": [{
                "embedding_index": 0,
                "keyframe_name": "00000",
                "shot_index": 0,
                "frame_index": 0,
                "timestamp": 0.0,
                "image_path": f"keyframes/Videos_L21/{name}/00000.jpg",
            }],
        }))

    with pytest.raises(ValueError, match="incompatible|precision|dimension"):
        build_for_model(
            "clip",
            features_root=tmp_path / "features",
            index_root=tmp_path / "indexes",
            playlist_mapping=None,
        )


def test_phase3_populates_catalog_matching_shots_json_and_idempotent(processed_tree: Path):
    assert main(["phase3", "--model", "clip"]) == 0
    db_path = processed_tree / "data_processed/database/catalog.sqlite3"
    engine = create_catalog_engine(db_path)
    try:
        queries = CatalogQueries(create_session_factory(engine))
        videos = queries.list_videos(limit=100)
        assert videos.total == 3
        for video in videos.items:
            shots_json = json.loads((processed_tree / "data_processed" / video.manifest_path).read_text())
            assert video.shot_count == len(shots_json["shots"])
            kfs = queries.list_keyframes(video.batch, video.video_name, limit=100)
            assert kfs.total == len(shots_json["shots"])
            for kf in kfs.items:
                assert (processed_tree / "data_processed" / kf.image_path).is_file()
    finally:
        engine.dispose()

    assert main(["phase3", "--model", "clip"]) == 0
    engine = create_catalog_engine(db_path)
    try:
        queries = CatalogQueries(create_session_factory(engine))
        assert queries.list_videos(limit=100).total == 3
        assert sum(v.shot_count for v in queries.list_videos(limit=100).items) == 9
    finally:
        engine.dispose()


def test_corrupt_shots_json_fails_catalog_for_that_video_imports_others_and_still_builds_index(
    processed_tree: Path, capsys
):
    bad_shots = processed_tree / "data_processed/transnetv2/Videos_L22/L22_V001/shots.json"
    bad_shots.write_text("{corrupt")

    rc = main(["phase3", "--model", "clip"])
    captured = capsys.readouterr()
    assert rc == 1
    assert "L22_V001" in captured.err + captured.out

    engine = create_catalog_engine(processed_tree / "data_processed/database/catalog.sqlite3")
    try:
        queries = CatalogQueries(create_session_factory(engine))
        videos = queries.list_videos(limit=100)
        assert videos.total == 2
        assert {v.video_name for v in videos.items} == {"L21_V001", "L21_V002"}
    finally:
        engine.dispose()

    assert faiss.read_index(str(processed_tree / "data_processed/indexes/clip/global.index")).ntotal == 9


def test_no_index_and_no_catalog_flags(processed_tree: Path):
    assert main(["phase3", "--model", "clip", "--no-index"]) == 0
    db_path = processed_tree / "data_processed/database/catalog.sqlite3"
    assert db_path.is_file()
    assert not (processed_tree / "data_processed/indexes/clip").exists()

    db_path.unlink()
    assert main(["phase3", "--model", "clip", "--no-catalog"]) == 0
    assert (processed_tree / "data_processed/indexes/clip/global.index").is_file()
    assert not db_path.exists()


def test_empty_database_file_migrated_by_alembic(processed_tree: Path):
    db_path = processed_tree / "data_processed/database/catalog.sqlite3"
    db_path.parent.mkdir(parents=True, exist_ok=True)
    db_path.write_bytes(b"")

    assert main(["phase3", "--model", "clip", "--no-index"]) == 0
    engine = create_catalog_engine(db_path)
    try:
        queries = CatalogQueries(create_session_factory(engine))
        assert queries.list_videos(limit=100).total == 3
    finally:
        engine.dispose()
