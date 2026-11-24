"""Filesystem lifecycle tests for Phase 2 feature extraction."""

import json
from pathlib import Path

import numpy as np
from PIL import Image
import pytest
import torch

from data_processing.cli import main
from data_processing.phase2_feature_extraction.pipeline import run_pipeline, validate_result


class FakeEmbedding:
    loads = 0

    def __init__(self, model_name, device):
        type(self).loads += 1
        self.device = torch.device("cpu")
        self.model = None

    def encode_images_batch(self, images):
        rows = torch.arange(1, len(images) * 3 + 1, dtype=torch.float32).reshape(-1, 3)
        return torch.nn.functional.normalize(rows, dim=1)


@pytest.fixture(params=["single_file", "legacy"])
def phase1(tmp_path: Path, request):
    phase1_root = tmp_path / "transnetv2"
    keyframe_root = tmp_path / "keyframes"
    result = phase1_root / "Videos_L21/L21_V001"
    images = keyframe_root / "Videos_L21/L21_V001"
    result.mkdir(parents=True)
    images.mkdir(parents=True)
    records = []
    artifacts = []
    for index in range(3):
        name = f"{index:05d}.jpg"
        Image.new("RGB", (16, 12), (index * 40, 20, 100)).save(images / name)
        artifacts.append(name)
        records.append({
            "shot_index": index,
            "keyframe": {
                "frame_index": index * 10,
                "timestamp": index / 2,
                "image_path": f"keyframes/Videos_L21/L21_V001/{name}",
            },
        })
    manifest = {
        "mode": "phase1", "status": "COMPLETED", "batch": "Videos_L21",
        "video_name": "L21_V001", "pipeline_version": "phase1-test",
        "shot_count": 3, "keyframe_artifacts": artifacts,
        "artifacts": ["shots.json"],
    }
    if request.param == "legacy":
        (result / "shots.json").write_text(json.dumps(records))
        (result / "manifest.json").write_text(json.dumps(manifest))
    else:
        (result / "shots.json").write_text(json.dumps({
            "mode": "phase1", "batch": "Videos_L21", "video_name": "L21_V001",
            "pipeline_version": "phase1-middle-v4", "shots": records,
        }))
    return phase1_root, keyframe_root, result


def test_pipeline_writes_ordered_validated_embeddings(phase1, tmp_path):
    phase1_root, keyframe_root, _ = phase1
    output = tmp_path / "embeddings"
    summary = run_pipeline(
        phase1_root, keyframe_root=keyframe_root, output_root=output,
        batch_size=2, embedder_factory=FakeEmbedding,
    )
    assert summary == {"processed": 1, "skipped": 0, "failed": 0}
    result = output / "clip/L21_V001.npy"
    manifest = validate_result(result)
    assert manifest["embedding_count"] == 3
    assert manifest["embedding_dimension"] == 3
    vectors = np.load(result, allow_pickle=False)
    np.testing.assert_allclose(np.linalg.norm(vectors, axis=1), 1, atol=1e-6)
    records = json.loads(result.with_suffix(".json").read_text())["records"]
    assert [record["shot_index"] for record in records] == [0, 1, 2]


def test_valid_result_is_skipped_and_force_rebuilds(phase1, tmp_path):
    phase1_root, keyframe_root, _ = phase1
    output = tmp_path / "embeddings"
    kwargs = dict(keyframe_root=keyframe_root, output_root=output,
                  embedder_factory=FakeEmbedding)
    assert run_pipeline(phase1_root, **kwargs)["processed"] == 1
    assert run_pipeline(phase1_root, **kwargs)["skipped"] == 1
    assert run_pipeline(phase1_root, force=True, **kwargs)["processed"] == 1


def test_changed_keyframe_fails_before_loading_model(phase1, tmp_path):
    phase1_root, keyframe_root, _ = phase1
    FakeEmbedding.loads = 0
    (keyframe_root / "Videos_L21/L21_V001/00001.jpg").write_bytes(b"damaged")
    summary = run_pipeline(
        phase1_root, keyframe_root=keyframe_root, output_root=tmp_path / "embeddings",
        embedder_factory=FakeEmbedding,
    )
    assert summary == {"processed": 0, "skipped": 0, "failed": 1}
    assert FakeEmbedding.loads == 0


def test_changed_shots_rebuild_embeddings(phase1, tmp_path):
    phase1_root, keyframe_root, source = phase1
    output = tmp_path / "embeddings"
    kwargs = dict(keyframe_root=keyframe_root, output_root=output, embedder_factory=FakeEmbedding)
    assert run_pipeline(phase1_root, **kwargs)["processed"] == 1
    path = source / "shots.json"
    data = json.loads(path.read_text())
    records = data["shots"] if isinstance(data, dict) else data
    records[0]["keyframe"]["timestamp"] = .1
    path.write_text(json.dumps(data))
    assert run_pipeline(phase1_root, **kwargs)["processed"] == 1
    metadata = validate_result(output / "clip/L21_V001.npy")
    assert metadata["records"][0]["timestamp"] == .1


def test_rejects_invalid_batch_size(phase1, tmp_path):
    phase1_root, keyframe_root, _ = phase1
    with pytest.raises(ValueError, match="Batch size"):
        run_pipeline(
            phase1_root, keyframe_root=keyframe_root,
            output_root=tmp_path / "embeddings", batch_size=0,
            embedder_factory=FakeEmbedding,
        )


def test_phase2_cli_routes_batch_and_options(monkeypatch, tmp_path):
    captured = {}

    def fake_run(input_path, **kwargs):
        captured["input_path"] = input_path
        captured.update(kwargs)
        return {"processed": 1, "skipped": 0, "failed": 0}

    monkeypatch.setattr(
        "data_processing.phase2_feature_extraction.pipeline.DEFAULT_PHASE1_ROOT",
        tmp_path / "transnetv2",
    )
    monkeypatch.setattr("data_processing.phase2_feature_extraction.pipeline.run_pipeline", fake_run)
    assert main(["phase2", "--batch", "Videos_L21", "--batch-size", "8",
                 "--device", "gpu"]) == 0
    assert captured["input_path"] == tmp_path / "transnetv2/Videos_L21"
    assert captured["batch_size"] == 8
    assert captured["device"] == "cuda"
    assert captured["model_key"] == "clip"
