"""Tests for Step 7 (Item 6): Phase 2 DataLoader, batch size, precision, workers, and timing."""

import json
import logging
from pathlib import Path

import numpy as np
from PIL import Image
import pytest
import torch
import torch.nn.functional as F

from data_processing import cli
from data_processing.phase2_feature_extraction import pipeline as phase2_pipeline
from data_processing.phase2_feature_extraction.clip_embedding import CLIPEmbedding


def _make_phase1_video(
    root: Path,
    batch: str,
    video_name: str,
    count: int,
    *,
    bad_jpeg_index: int | None = None,
) -> Path:
    result_dir = root / "transnetv2" / batch / video_name
    keyframe_dir = root / "keyframes" / batch / video_name
    result_dir.mkdir(parents=True, exist_ok=True)
    keyframe_dir.mkdir(parents=True, exist_ok=True)
    shots = []
    for index in range(count):
        name = f"{index:05d}.jpg"
        target = keyframe_dir / name
        if index == bad_jpeg_index:
            target.write_bytes(b"corrupt-jpeg-bytes")
        else:
            color = ((index * 37) % 256, (index * 71) % 256, (index * 113) % 256)
            Image.new("RGB", (24, 18), color).save(target)
        shots.append({
            "shot_index": index,
            "keyframe": {
                "frame_index": index * 10,
                "timestamp": float(index),
                "image_path": f"keyframes/{batch}/{video_name}/{name}",
            },
        })
    (result_dir / "shots.json").write_text(json.dumps({
        "mode": "phase1",
        "batch": batch,
        "video_name": video_name,
        "pipeline_version": "phase1-middle-v4",
        "shot_count": count,
        "keyframe_artifacts": [f"{index:05d}.jpg" for index in range(count)],
        "shots": shots,
    }))
    return result_dir


def test_workers_0_and_2_produce_identical_bytes_and_sidecars(tmp_path: Path, stub_embedder):
    root = tmp_path / "data_processed"
    _make_phase1_video(root, "Videos_L21", "L21_V001", 20)
    out0 = root / "features0"
    out2 = root / "features2"

    summary0 = phase2_pipeline.run_pipeline(
        root / "transnetv2",
        keyframe_root=root / "keyframes",
        output_root=out0,
        batch_size=8,
        workers=0,
        embedder_factory=stub_embedder,
    )
    summary2 = phase2_pipeline.run_pipeline(
        root / "transnetv2",
        keyframe_root=root / "keyframes",
        output_root=out2,
        batch_size=8,
        workers=2,
        embedder_factory=stub_embedder,
    )
    assert summary0 == {"processed": 1, "skipped": 0, "failed": 0}
    assert summary2 == {"processed": 1, "skipped": 0, "failed": 0}

    assert (out0 / "clip/L21_V001.npy").read_bytes() == (out2 / "clip/L21_V001.npy").read_bytes()

    side0 = json.loads((out0 / "clip/L21_V001.json").read_text())
    side2 = json.loads((out2 / "clip/L21_V001.json").read_text())
    for side in (side0, side2):
        side.pop("started_at")
        side.pop("finished_at")
    assert side0 == side2


@pytest.mark.parametrize("count", [1, 63, 65, 130])
def test_various_keyframe_counts_at_batch_size_64(tmp_path: Path, stub_embedder, count: int):
    root = tmp_path / "data_processed"
    _make_phase1_video(root, "Videos_L21", "L21_V001", count)
    output_root = root / "features"

    summary = phase2_pipeline.run_pipeline(
        root / "transnetv2",
        keyframe_root=root / "keyframes",
        output_root=output_root,
        batch_size=64,
        workers=0,
        embedder_factory=stub_embedder,
    )
    assert summary == {"processed": 1, "skipped": 0, "failed": 0}
    vectors = np.load(output_root / "clip/L21_V001.npy", allow_pickle=False)
    assert vectors.shape == (count, 8)
    expected = torch.stack([
        stub_embedder._to_vector(root / f"keyframes/Videos_L21/L21_V001/{i:05d}.jpg")
        for i in range(count)
    ]).numpy()
    np.testing.assert_allclose(vectors, expected, rtol=1e-6, atol=1e-6)


def test_batch_size_1_vs_64_match(tmp_path: Path, stub_embedder):
    root = tmp_path / "data_processed"
    _make_phase1_video(root, "Videos_L21", "L21_V001", 15)

    phase2_pipeline.run_pipeline(
        root / "transnetv2",
        keyframe_root=root / "keyframes",
        output_root=root / "out1",
        batch_size=1,
        embedder_factory=stub_embedder,
    )
    phase2_pipeline.run_pipeline(
        root / "transnetv2",
        keyframe_root=root / "keyframes",
        output_root=root / "out64",
        batch_size=64,
        embedder_factory=stub_embedder,
    )
    v1 = np.load(root / "out1/clip/L21_V001.npy", allow_pickle=False)
    v64 = np.load(root / "out64/clip/L21_V001.npy", allow_pickle=False)
    np.testing.assert_allclose(v1, v64, rtol=1e-6, atol=1e-6)


def test_one_bad_jpeg_fails_only_its_video(tmp_path: Path, stub_embedder):
    root = tmp_path / "data_processed"
    _make_phase1_video(root, "Videos_L21", "L21_V001", 4)
    _make_phase1_video(root, "Videos_L21", "L21_V002", 4, bad_jpeg_index=2)
    _make_phase1_video(root, "Videos_L22", "L22_V001", 4)
    out = root / "features"

    summary = phase2_pipeline.run_pipeline(
        root / "transnetv2",
        keyframe_root=root / "keyframes",
        output_root=out,
        embedder_factory=stub_embedder,
    )
    assert summary == {"processed": 2, "skipped": 0, "failed": 1}
    assert (out / "clip/L21_V001.npy").is_file()
    assert not (out / "clip/L21_V002.npy").exists()
    assert (out / "clip/L22_V001.npy").is_file()


def test_dataloader_fp32_matches_legacy_encode_images_batch(tmp_path: Path, monkeypatch):
    class DeterministicImageProcessor:
        def __call__(self, *, images, return_tensors="pt"):
            items = images if isinstance(images, list) else [images]
            tensors = [
                torch.from_numpy(np.asarray(img.convert("RGB").resize((8, 8)), dtype=np.float32) / 255.0)
                .permute(2, 0, 1)
                for img in items
            ]
            return {"pixel_values": torch.stack(tensors, dim=0)}

    class DeterministicProcessor:
        def __init__(self):
            self.image_processor = DeterministicImageProcessor()

        @classmethod
        def from_pretrained(cls, model_name):
            return cls()

        def __call__(self, *, images=None, text=None, return_tensors="pt", **kwargs):
            return self.image_processor(images=images, return_tensors=return_tensors)

    class DeterministicModel:
        @classmethod
        def from_pretrained(cls, model_name):
            return cls()

        def to(self, device):
            return self

        def eval(self):
            return self

        def get_image_features(self, pixel_values):
            return pixel_values.mean(dim=(2, 3)) + 0.1

    monkeypatch.setattr(
        "data_processing.phase2_feature_extraction.clip_embedding.CLIPProcessor",
        DeterministicProcessor,
    )
    monkeypatch.setattr(
        "data_processing.phase2_feature_extraction.clip_embedding.CLIPModel",
        DeterministicModel,
    )

    root = tmp_path / "data_processed"
    _make_phase1_video(root, "Videos_L21", "L21_V001", 10)
    kf_paths = sorted((root / "keyframes/Videos_L21/L21_V001").glob("*.jpg"))

    legacy = CLIPEmbedding(device="cpu").encode_images_batch(kf_paths).numpy()

    out = root / "features"
    summary = phase2_pipeline.run_pipeline(
        root / "transnetv2",
        keyframe_root=root / "keyframes",
        output_root=out,
        batch_size=4,
        workers=0,
        precision="fp32",
        embedder_factory=CLIPEmbedding,
    )
    assert summary == {"processed": 1, "skipped": 0, "failed": 0}
    new_vecs = np.load(out / "clip/L21_V001.npy", allow_pickle=False)
    np.testing.assert_allclose(new_vecs, legacy, rtol=1e-6, atol=1e-6)


def test_device_gpu_without_cuda_raises_clear_error(data_tree: Path, monkeypatch):
    monkeypatch.setattr(torch.cuda, "is_available", lambda: False)
    with pytest.raises(ValueError, match="CUDA"):
        CLIPEmbedding(device="cuda")

    _make_phase1_video(data_tree / "data_processed", "Videos_L21", "L21_V001", 2)
    assert cli.main(["phase2", "--batch", "Videos_L21", "--device", "gpu"]) == 1


def test_sidecar_precision_rebuilds_and_logs_timing_and_cuda_name(tmp_path: Path, stub_embedder, monkeypatch, caplog):
    root = tmp_path / "data_processed"
    _make_phase1_video(root, "Videos_L21", "L21_V001", 5)
    out = root / "features"

    class CudaNamedStub(stub_embedder):
        def __init__(self, model_name=stub_embedder.DEFAULT_MODEL, device="cpu"):
            super().__init__(model_name=model_name, device=device)
            self.device = torch.device("cuda")

    monkeypatch.setattr(torch.cuda, "get_device_name", lambda *args: "Fake GTX 1650 Ti")

    with caplog.at_level(logging.INFO, logger=phase2_pipeline.LOGGER.name):
        summary = phase2_pipeline.run_pipeline(
            root / "transnetv2",
            keyframe_root=root / "keyframes",
            output_root=out,
            precision="fp32",
            workers=0,
            embedder_factory=CudaNamedStub,
        )
    assert summary == {"processed": 1, "skipped": 0, "failed": 0}
    sidecar = json.loads((out / "clip/L21_V001.json").read_text())
    assert sidecar["precision"] == "fp32"

    messages = "\n".join(caplog.messages)
    assert "Fake GTX 1650 Ti" in messages
    assert "images/s" in messages
    assert "L21_V001" in messages and "s" in messages

    # Changing precision rebuilds instead of skipping.
    summary_fp16 = phase2_pipeline.run_pipeline(
        root / "transnetv2",
        keyframe_root=root / "keyframes",
        output_root=out,
        precision="fp16",
        workers=0,
        embedder_factory=CudaNamedStub,
    )
    assert summary_fp16 == {"processed": 1, "skipped": 0, "failed": 0}
    assert json.loads((out / "clip/L21_V001.json").read_text())["precision"] == "fp16"


@pytest.mark.slow
def test_dataloader_fp32_matches_saved_videos_l21_reference(tmp_path: Path):
    repo_root = Path(__file__).resolve().parents[1]
    ref_npy = repo_root / "data_processed/features/clip/L21_V001.npy"
    kf_root = repo_root / "data_processed/keyframes"
    result_dir = repo_root / "data_processed/transnetv2/Videos_L21/L21_V001"
    if not ref_npy.is_file() or not result_dir.is_dir():
        pytest.skip("Saved Videos_L21 reference L21_V001 files not present")

    ref_vecs = np.load(ref_npy, allow_pickle=False)
    out = tmp_path / "features"
    summary = phase2_pipeline.run_pipeline(
        result_dir,
        keyframe_root=kf_root,
        output_root=out,
        precision="fp32",
        device="cpu",
    )
    assert summary == {"processed": 1, "skipped": 0, "failed": 0}
    new_vecs = np.load(out / "clip/L21_V001.npy", allow_pickle=False)
    np.testing.assert_allclose(new_vecs, ref_vecs, rtol=1e-5, atol=1e-5)


@pytest.mark.gpu
def test_fp16_cosine_similarity_and_sidecar_on_gpu(tmp_path: Path):
    if not torch.cuda.is_available():
        pytest.skip("CUDA device required")

    repo_root = Path(__file__).resolve().parents[1]
    real_kf = sorted((repo_root / "data_processed/keyframes/Videos_L21").rglob("*.jpg"))[:200]
    root = tmp_path / "data_processed"
    if len(real_kf) == 200:
        result_dir = root / "transnetv2/Videos_L21/SAMPLE200"
        kf_dir = root / "keyframes/Videos_L21/SAMPLE200"
        result_dir.mkdir(parents=True)
        kf_dir.mkdir(parents=True)
        shots = []
        for idx, src in enumerate(real_kf):
            dst = kf_dir / f"{idx:05d}.jpg"
            dst.symlink_to(src)
            shots.append({
                "shot_index": idx,
                "keyframe": {
                    "frame_index": idx * 10,
                    "timestamp": float(idx),
                    "image_path": f"keyframes/Videos_L21/SAMPLE200/{idx:05d}.jpg",
                },
            })
        (result_dir / "shots.json").write_text(json.dumps({
            "mode": "phase1",
            "batch": "Videos_L21",
            "video_name": "SAMPLE200",
            "pipeline_version": "phase1-middle-v4",
            "shot_count": 200,
            "keyframe_artifacts": [f"{i:05d}.jpg" for i in range(200)],
            "shots": shots,
        }))
    else:
        _make_phase1_video(root, "Videos_L21", "SAMPLE200", 200)

    out32 = root / "out32"
    out16 = root / "out16"
    assert phase2_pipeline.run_pipeline(
        root / "transnetv2", keyframe_root=root / "keyframes", output_root=out32,
        device="cuda", precision="fp32",
    ) == {"processed": 1, "skipped": 0, "failed": 0}
    assert phase2_pipeline.run_pipeline(
        root / "transnetv2", keyframe_root=root / "keyframes", output_root=out16,
        device="cuda", precision="fp16",
    ) == {"processed": 1, "skipped": 0, "failed": 0}

    v32 = np.load(out32 / "clip/SAMPLE200.npy", allow_pickle=False)
    v16 = np.load(out16 / "clip/SAMPLE200.npy", allow_pickle=False)
    assert v16.dtype == np.float32
    cos = (v32 * v16).sum(axis=1)
    assert float(cos.min()) >= 0.999
    assert json.loads((out16 / "clip/SAMPLE200.json").read_text())["precision"] == "fp16"


def test_default_workers_is_zero_on_cpu_and_cuda(tmp_path: Path, monkeypatch):
    assert phase2_pipeline.DEFAULT_WORKERS == 0

    seen_num_workers: list[int] = []
    real_dataloader = phase2_pipeline.DataLoader

    def recording_dataloader(*args, **kwargs):
        seen_num_workers.append(kwargs.get("num_workers", 0))
        return real_dataloader(*args, **kwargs)

    monkeypatch.setattr(phase2_pipeline, "DataLoader", recording_dataloader)

    class FakeCudaEmbedder:
        device = torch.device("cuda", 0)

        def encode_images_batch(self, images):
            return torch.ones((len(images), 8), dtype=torch.float32)

    img = tmp_path / "00000.jpg"
    Image.new("RGB", (8, 8), color=(10, 20, 30)).save(img)

    phase2_pipeline._encode_images(
        FakeCudaEmbedder(), [img], batch_size=4, workers=None, precision="fp32"
    )
    assert seen_num_workers == [0]
