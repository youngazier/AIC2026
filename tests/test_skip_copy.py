"""Item 4 tests: skip checks rely on sizes and content hashes, never mtimes."""

import hashlib
import json
import os
from pathlib import Path
import shutil
import subprocess
import time

from backend.app.catalog.validation import prepare_catalog_video
from data_processing.phase1_keyframe_extraction import pipeline as phase1_pipeline
from data_processing.phase2_feature_extraction import pipeline as phase2_pipeline
from data_processing.phase2_feature_extraction.build_index import build_for_model


def test_sidecars_have_no_mtimes_and_bumped_versions(processed_tree: Path):
    assert phase1_pipeline.PIPELINE_VERSION == "phase1-middle-v4"
    assert phase2_pipeline.PIPELINE_VERSION == "phase2-features-v3"

    shots_path = processed_tree / "data_processed/transnetv2/Videos_L21/L21_V001/shots.json"
    shots = json.loads(shots_path.read_text())
    assert "source_mtime_ns" not in shots
    assert "checkpoint_mtime_ns" not in shots
    assert shots["source_key"] == "data/videos/Videos_L21/L21_V001.mp4"
    assert shots["source_size"] == (processed_tree / shots["source_key"]).stat().st_size
    assert shots["checkpoint_size"] == (processed_tree / "weights.pth").stat().st_size
    assert shots["upstream_revision"] == phase1_pipeline.UPSTREAM_REVISION
    assert shots["pipeline_version"] == "phase1-middle-v4"

    feature_meta_path = processed_tree / "data_processed/features/clip/L21_V001.json"
    feature_meta = json.loads(feature_meta_path.read_text())
    assert "input_shots_mtime_ns" not in feature_meta
    assert "input_manifest_mtime_ns" not in feature_meta
    assert feature_meta["input_shots_sha256"] == hashlib.sha256(shots_path.read_bytes()).hexdigest()
    assert feature_meta["pipeline_version"] == "phase2-features-v3"


def test_cp_r_output_tree_skips_every_phase(
    processed_tree: Path, tmp_path: Path, monkeypatch, stub_detector, stub_embedder
):
    index_summary = build_for_model(
        "clip",
        features_root=processed_tree / "data_processed/features",
        index_root=processed_tree / "data_processed/indexes",
        playlist_mapping=processed_tree / "data/video_to_playlist.json",
    )
    assert index_summary == {"built": 3, "skipped": 0}

    # Backdate files in the source tree so plain `cp -r` produces newer mtimes.
    old_ts = time.time() - 120
    for item in (processed_tree / "data_processed").rglob("*"):
        if item.is_file():
            os.utime(item, (old_ts, old_ts))

    other = tmp_path / "other"
    other.mkdir()
    shutil.copytree(processed_tree / "data", other / "data")
    shutil.copy(processed_tree / "weights.pth", other / "weights.pth")
    subprocess.run(
        ["cp", "-r", str(processed_tree / "data_processed"), str(other / "data_processed")],
        check=True,
    )

    orig_shots = processed_tree / "data_processed/transnetv2/Videos_L21/L21_V001/shots.json"
    copied_shots = other / "data_processed/transnetv2/Videos_L21/L21_V001/shots.json"
    orig_npy = processed_tree / "data_processed/features/clip/L21_V001.npy"
    copied_npy = other / "data_processed/features/clip/L21_V001.npy"
    assert copied_shots.stat().st_mtime_ns != orig_shots.stat().st_mtime_ns
    assert copied_npy.stat().st_mtime_ns != orig_npy.stat().st_mtime_ns

    monkeypatch.setattr(phase1_pipeline, "PROJECT_ROOT", other)
    monkeypatch.setattr(phase2_pipeline, "PROJECT_ROOT", other)

    assert phase1_pipeline.run_pipeline(
        other / "data/videos",
        output_root=other / "data_processed/transnetv2",
        weights=other / "weights.pth",
    ) == {"processed": 0, "skipped": 3, "failed": 0}
    assert stub_detector.calls == 0

    assert phase2_pipeline.run_pipeline(
        other / "data_processed/transnetv2",
        keyframe_root=other / "data_processed/keyframes",
        output_root=other / "data_processed/features",
        embedder_factory=stub_embedder,
    ) == {"processed": 0, "skipped": 3, "failed": 0}
    assert stub_embedder.calls == 0

    assert build_for_model(
        "clip",
        features_root=other / "data_processed/features",
        index_root=other / "data_processed/indexes",
        playlist_mapping=other / "data/video_to_playlist.json",
    ) == {"built": 0, "skipped": 3}

    prepared = prepare_catalog_video(copied_shots, project_root=other)
    assert prepared.video.video_name == "L21_V001"


def test_touch_source_mp4_still_skips_phase1_and_catalog(processed_tree: Path, stub_detector):
    source = processed_tree / "data/videos/Videos_L21/L21_V001.mp4"
    bumped = source.stat().st_mtime + 60
    os.utime(source, (bumped, bumped))

    assert phase1_pipeline.run_pipeline(
        processed_tree / "data/videos",
        output_root=processed_tree / "data_processed/transnetv2",
        weights=processed_tree / "weights.pth",
    ) == {"processed": 0, "skipped": 3, "failed": 0}
    assert stub_detector.calls == 0

    shots = processed_tree / "data_processed/transnetv2/Videos_L21/L21_V001/shots.json"
    assert prepare_catalog_video(shots, project_root=processed_tree).video.video_name == "L21_V001"


def test_replace_source_mp4_different_size_reprocesses(processed_tree: Path):
    ffmpeg = shutil.which("ffmpeg")
    assert ffmpeg is not None
    source = processed_tree / "data/videos/Videos_L21/L21_V001.mp4"
    old_size = source.stat().st_size
    subprocess.run(
        [
            ffmpeg,
            "-y",
            "-v",
            "error",
            "-f",
            "lavfi",
            "-i",
            "color=c=blue:size=320x180:rate=30:duration=2",
            "-c:v",
            "mpeg4",
            "-q:v",
            "3",
            str(source),
        ],
        check=True,
    )
    assert source.stat().st_size != old_size

    assert phase1_pipeline.run_pipeline(
        processed_tree / "data/videos",
        output_root=processed_tree / "data_processed/transnetv2",
        weights=processed_tree / "weights.pth",
    ) == {"processed": 1, "skipped": 2, "failed": 0}


def test_touch_weights_skips_but_weights_size_or_revision_reprocesses(
    processed_tree: Path, monkeypatch
):
    weights = processed_tree / "weights.pth"
    bumped = weights.stat().st_mtime + 60
    os.utime(weights, (bumped, bumped))

    kwargs = dict(
        output_root=processed_tree / "data_processed/transnetv2",
        weights=weights,
    )
    assert phase1_pipeline.run_pipeline(processed_tree / "data/videos", **kwargs) == {
        "processed": 0, "skipped": 3, "failed": 0,
    }

    weights.write_bytes(weights.read_bytes() + b"-modified")
    assert phase1_pipeline.run_pipeline(processed_tree / "data/videos", **kwargs) == {
        "processed": 3, "skipped": 0, "failed": 0,
    }
    assert phase1_pipeline.run_pipeline(processed_tree / "data/videos", **kwargs) == {
        "processed": 0, "skipped": 3, "failed": 0,
    }

    monkeypatch.setattr(phase1_pipeline, "UPSTREAM_REVISION", "different-revision")
    assert phase1_pipeline.run_pipeline(processed_tree / "data/videos", **kwargs) == {
        "processed": 3, "skipped": 0, "failed": 0,
    }


def test_touch_shots_json_still_skips_phase2(processed_tree: Path, stub_embedder):
    shots = processed_tree / "data_processed/transnetv2/Videos_L21/L21_V001/shots.json"
    bumped = shots.stat().st_mtime + 60
    os.utime(shots, (bumped, bumped))

    assert phase2_pipeline.run_pipeline(
        processed_tree / "data_processed/transnetv2",
        keyframe_root=processed_tree / "data_processed/keyframes",
        output_root=processed_tree / "data_processed/features",
        embedder_factory=stub_embedder,
    ) == {"processed": 0, "skipped": 3, "failed": 0}
    assert stub_embedder.calls == 0


def test_change_one_byte_in_shots_json_reprocesses_that_video(
    processed_tree: Path, stub_embedder
):
    shots = processed_tree / "data_processed/transnetv2/Videos_L21/L21_V001/shots.json"
    raw = shots.read_text()
    assert '"threshold": 0.5' in raw
    shots.write_text(raw.replace('"threshold": 0.5', '"threshold": 0.6', 1))

    assert phase2_pipeline.run_pipeline(
        processed_tree / "data_processed/transnetv2",
        keyframe_root=processed_tree / "data_processed/keyframes",
        output_root=processed_tree / "data_processed/features",
        embedder_factory=stub_embedder,
    ) == {"processed": 1, "skipped": 2, "failed": 0}


def test_copy_features_with_cp_and_rebuild_index(processed_tree: Path, tmp_path: Path):
    copied_features = tmp_path / "features"
    subprocess.run(
        ["cp", "-r", str(processed_tree / "data_processed/features"), str(copied_features)],
        check=True,
    )
    indexes = tmp_path / "indexes"
    assert build_for_model(
        "clip", features_root=copied_features, index_root=indexes, playlist_mapping=None
    ) == {"built": 1, "skipped": 0}

    manifest = json.loads((indexes / "clip/manifest.json").read_text())
    for source in manifest["sources"]:
        feature_file = copied_features / "clip" / source["feature_file"]
        assert "feature_mtime_ns" not in source
        assert "metadata_mtime_ns" not in source
        assert source["feature_size"] == feature_file.stat().st_size
        assert source["metadata_sha256"] == hashlib.sha256(
            feature_file.with_suffix(".json").read_bytes()
        ).hexdigest()

    assert build_for_model(
        "clip", features_root=copied_features, index_root=indexes, playlist_mapping=None
    ) == {"built": 0, "skipped": 1}


def test_old_pipeline_version_triggers_reprocess(
    processed_tree: Path, stub_detector, stub_embedder
):
    shots = processed_tree / "data_processed/transnetv2/Videos_L21/L21_V001/shots.json"
    data = json.loads(shots.read_text())
    data["pipeline_version"] = "phase1-middle-v3"
    shots.write_text(json.dumps(data))

    assert phase1_pipeline.run_pipeline(
        processed_tree / "data/videos",
        output_root=processed_tree / "data_processed/transnetv2",
        weights=processed_tree / "weights.pth",
    ) == {"processed": 1, "skipped": 2, "failed": 0}

    # Rebuilding with the same inputs produces identical shots.json bytes, so Phase 2 skips.
    assert phase2_pipeline.run_pipeline(
        processed_tree / "data_processed/transnetv2",
        keyframe_root=processed_tree / "data_processed/keyframes",
        output_root=processed_tree / "data_processed/features",
        embedder_factory=stub_embedder,
    ) == {"processed": 0, "skipped": 3, "failed": 0}

    feature_meta_path = processed_tree / "data_processed/features/clip/L21_V001.json"
    feature_meta = json.loads(feature_meta_path.read_text())
    feature_meta["pipeline_version"] = "phase2-features-v2"
    feature_meta_path.write_text(json.dumps(feature_meta))

    assert phase2_pipeline.run_pipeline(
        processed_tree / "data_processed/transnetv2",
        keyframe_root=processed_tree / "data_processed/keyframes",
        output_root=processed_tree / "data_processed/features",
        embedder_factory=stub_embedder,
    ) == {"processed": 1, "skipped": 2, "failed": 0}
