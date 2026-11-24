"""Items 1 & 2 tests: shared video/batch/all selectors and model registry."""

import json
from pathlib import Path
import shutil

import numpy as np
import pytest

from data_processing.cli import main
from data_processing.phase2_feature_extraction.build_index import build_for_model
from data_processing.phase2_feature_extraction.clip_embedding import CLIPEmbedding


def test_video_selector_by_names(data_tree: Path, stub_detector, stub_embedder, monkeypatch):
    from data_processing.phase2_feature_extraction.models import MODELS

    monkeypatch.setitem(MODELS, "clip", (stub_embedder, stub_embedder.DEFAULT_MODEL))
    assert main(["phase1", "--video", "L21_V001", "L21_V002"]) == 0
    assert (data_tree / "data_processed/transnetv2/Videos_L21/L21_V001/shots.json").is_file()
    assert (data_tree / "data_processed/transnetv2/Videos_L21/L21_V002/shots.json").is_file()
    assert not (data_tree / "data_processed/transnetv2/Videos_L22/L22_V001").exists()

    assert main(["phase2", "--video", "L21_V001", "L21_V002"]) == 0
    assert (data_tree / "data_processed/features/clip/L21_V001.npy").is_file()
    assert (data_tree / "data_processed/features/clip/L21_V002.npy").is_file()
    assert not (data_tree / "data_processed/features/clip/L22_V001.npy").exists()


def test_video_selector_accepts_mp4_path(data_tree: Path, stub_detector):
    mp4 = data_tree / "data/videos/Videos_L21/L21_V001.mp4"
    assert main(["phase1", "--video", str(mp4)]) == 0
    assert (data_tree / "data_processed/transnetv2/Videos_L21/L21_V001/shots.json").is_file()
    assert not (data_tree / "data_processed/transnetv2/Videos_L21/L21_V002").exists()


def test_video_selector_unknown_name_errors_and_processes_nothing(
    data_tree: Path, stub_detector, caplog
):
    assert main(["phase1", "--video", "L21_V001", "L99_V999"]) != 0
    assert "L99_V999" in caplog.text
    assert stub_detector.calls == 0
    assert not (data_tree / "data_processed/transnetv2/Videos_L21/L21_V001").exists()


def test_batch_selector_multiple_batches(data_tree: Path, stub_detector):
    assert main(["phase1", "--batch", "Videos_L21", "Videos_L22"]) == 0
    assert stub_detector.calls == 3
    for batch, name in (
        ("Videos_L21", "L21_V001"),
        ("Videos_L21", "L21_V002"),
        ("Videos_L22", "L22_V001"),
    ):
        assert (data_tree / "data_processed/transnetv2" / batch / name / "shots.json").is_file()


def test_batch_selector_nonexistent_batch_errors_and_processes_nothing(
    data_tree: Path, stub_detector, caplog
):
    assert main(["phase1", "--batch", "Videos_L21", "Videos_L99"]) != 0
    assert "Videos_L99" in caplog.text
    assert stub_detector.calls == 0


@pytest.mark.parametrize("bad", ["videos_l21", "L21"])
def test_batch_selector_rejects_malformed_names(data_tree: Path, stub_detector, bad):
    assert main(["phase1", "--batch", bad]) != 0
    assert stub_detector.calls == 0


def test_all_selector_ignores_zips_and_other_folders(data_tree: Path, stub_detector):
    (data_tree / "data/videos/Videos_L23.zip").write_bytes(b"zip-bytes")
    (data_tree / "data/videos/Videos_L21/archive.zip").write_bytes(b"zip-bytes")
    other = data_tree / "data/videos/other_folder/extra.mp4"
    other.parent.mkdir(parents=True)
    shutil.copy2(data_tree / "data/videos/Videos_L21/L21_V001.mp4", other)

    assert main(["phase1", "--all"]) == 0
    assert stub_detector.calls == 3
    assert not (data_tree / "data_processed/transnetv2/unbatched").exists()


def test_no_selector_rejected(data_tree: Path, stub_detector):
    assert main(["phase1"]) != 0
    assert main(["phase2"]) != 0
    assert stub_detector.calls == 0


@pytest.mark.parametrize(
    "argv_fn",
    [
        lambda root: ["phase1", "--video", "L21_V001", "--batch", "Videos_L21"],
        lambda root: ["phase1", "--batch", "Videos_L21", "--all"],
        lambda root: ["phase1", "--video", "L21_V001", "--all"],
        lambda root: ["phase1", str(root / "data/videos/Videos_L21/L21_V001.mp4"), "--all"],
        lambda root: ["phase2", "--video", "L21_V001", "--batch", "Videos_L21"],
        lambda root: ["phase2", "--batch", "Videos_L21", "--all"],
    ],
)
def test_multiple_selector_kinds_rejected(data_tree: Path, stub_detector, argv_fn):
    assert main(argv_fn(data_tree)) != 0
    assert stub_detector.calls == 0


def test_phase2_reports_video_missing_phase1_output_and_continues(
    data_tree: Path, stub_detector, stub_embedder, caplog, monkeypatch
):
    from data_processing.phase2_feature_extraction.models import MODELS

    monkeypatch.setitem(MODELS, "clip", (stub_embedder, stub_embedder.DEFAULT_MODEL))
    assert main(["phase1", "--video", "L21_V001", "L22_V001"]) == 0
    shutil.rmtree(data_tree / "data/videos")

    rc = main(["phase2", "--video", "L21_V001", "L21_V002", "L22_V001"])
    assert rc == 1
    assert "L21_V002" in caplog.text
    assert "no Phase 1 output" in caplog.text
    assert (data_tree / "data_processed/features/clip/L21_V001.npy").is_file()
    assert (data_tree / "data_processed/features/clip/L22_V001.npy").is_file()
    assert not (data_tree / "data_processed/features/clip/L21_V002.npy").exists()


def test_phase2_selectors_work_when_data_videos_missing_or_empty(
    data_tree: Path, stub_detector, stub_embedder, monkeypatch
):
    from data_processing.phase2_feature_extraction.models import MODELS

    monkeypatch.setitem(MODELS, "clip", (stub_embedder, stub_embedder.DEFAULT_MODEL))
    assert main(["phase1", "--all"]) == 0

    # 1. data/videos/ completely missing
    shutil.rmtree(data_tree / "data/videos")
    assert main(["phase2", "--batch", "Videos_L21"]) == 0
    assert (data_tree / "data_processed/features/clip/L21_V001.npy").is_file()
    assert (data_tree / "data_processed/features/clip/L21_V002.npy").is_file()
    assert not (data_tree / "data_processed/features/clip/L22_V001.npy").exists()

    # 2. data/videos/ present but empty
    (data_tree / "data/videos").mkdir(parents=True)
    assert main(["phase2", "--batch", "Videos_L21", "Videos_L22"]) == 0
    assert (data_tree / "data_processed/features/clip/L22_V001.npy").is_file()
    assert main(["phase2", "--video", "L21_V001"]) == 0
    assert main(["phase2", "--all"]) == 0


def test_phase2_nonexistent_batch_errors_without_data_videos(processed_tree: Path, caplog):
    shutil.rmtree(processed_tree / "data/videos")
    assert main(["phase2", "--batch", "Videos_L99"]) == 1
    assert "Videos_L99" in caplog.text


def test_legacy_single_path_still_works(
    data_tree: Path, stub_detector, stub_embedder, monkeypatch
):
    from data_processing.phase2_feature_extraction.models import MODELS

    monkeypatch.setitem(MODELS, "clip", (stub_embedder, stub_embedder.DEFAULT_MODEL))
    assert main(["phase1", str(data_tree / "data/videos/Videos_L21/L21_V001.mp4")]) == 0
    assert main(["phase2", str(data_tree / "data_processed/transnetv2/Videos_L21/L21_V001")]) == 0
    assert (data_tree / "data_processed/features/clip/L21_V001.npy").is_file()


def test_model_registry_resolves_clip_and_rejects_unknown(processed_tree: Path, caplog):
    from data_processing.phase2_feature_extraction.models import MODELS, resolve_model

    assert MODELS["clip"] == (CLIPEmbedding, "openai/clip-vit-base-patch32")
    assert resolve_model("clip") == (CLIPEmbedding, "openai/clip-vit-base-patch32")
    assert main(["phase2", "--video", "L21_V001", "--model", "nonexistent"]) == 1
    assert "clip" in caplog.text


def test_second_registered_model_writes_separate_features(
    processed_tree: Path, stub_embedder, monkeypatch
):
    from data_processing.phase2_feature_extraction.models import MODELS

    monkeypatch.setitem(MODELS, "altmodel", (stub_embedder, "org/alt-v1"))
    assert main(["phase2", "--batch", "Videos_L21", "--model", "altmodel"]) == 0
    alt_meta = processed_tree / "data_processed/features/altmodel/L21_V001.json"
    assert alt_meta.with_suffix(".npy").is_file()
    assert json.loads(alt_meta.read_text())["model_name"] == "org/alt-v1"
    assert (processed_tree / "data_processed/features/clip/L21_V001.npy").is_file()


def test_checkpoint_overrides_only_weights_name(
    processed_tree: Path, stub_embedder, monkeypatch
):
    from data_processing.phase2_feature_extraction.models import MODELS

    monkeypatch.setitem(MODELS, "clip", (stub_embedder, "openai/clip-vit-base-patch32"))
    assert main([
        "phase2", "--video", "L21_V001", "--model", "clip",
        "--checkpoint", "local/custom-clip", "--force",
    ]) == 0
    meta = json.loads(
        (processed_tree / "data_processed/features/clip/L21_V001.json").read_text()
    )
    assert meta["model_name"] == "local/custom-clip"
    assert stub_embedder.loads == 1


def test_build_index_refuses_mixed_models(tmp_path: Path):
    model_dir = tmp_path / "features/clip"
    model_dir.mkdir(parents=True)
    for name, model_name in (("L21_V001", "model-a"), ("L21_V002", "model-b")):
        npy = model_dir / f"{name}.npy"
        np.save(npy, np.array([[1.0, 0.0, 0.0]], dtype=np.float32), allow_pickle=False)
        npy.with_suffix(".json").write_text(json.dumps({
            "status": "COMPLETED",
            "model_name": model_name,
            "representation_version": "v1",
            "batch": "Videos_L21",
            "video_name": name,
            "embedding_count": 1,
            "embedding_dimension": 3,
            "records": [{
                "embedding_index": 0,
                "keyframe_name": "00000",
                "shot_index": 0,
                "frame_index": 0,
                "timestamp": 0.0,
                "image_path": f"keyframes/Videos_L21/{name}/00000.jpg",
            }],
        }))
    with pytest.raises(ValueError, match="incompatible"):
        build_for_model(
            "clip",
            features_root=tmp_path / "features",
            index_root=tmp_path / "indexes",
            playlist_mapping=None,
        )
