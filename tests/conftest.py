"""Shared pytest fixtures for fast unit and integration tests."""

from pathlib import Path
import shutil
import subprocess

import numpy as np
from PIL import Image
import pytest
import torch
import torch.nn.functional as F

from data_processing.phase1_keyframe_extraction import pipeline as phase1_pipeline
from data_processing.phase1_keyframe_extraction.transnetv2.inference import TransNetV2Torch
from data_processing.phase2_feature_extraction import pipeline as phase2_pipeline
from data_processing.phase2_feature_extraction.base_embedding import BaseEmbedding, ImageInput


@pytest.fixture(scope="session")
def tiny_video(tmp_path_factory) -> Path:
    """Build a 6-second 320x180 MP4 with three 2-second hard-cut clips."""
    ffmpeg = shutil.which("ffmpeg")
    if ffmpeg is None:
        pytest.skip("ffmpeg is required to build the tiny_video fixture")
    output = tmp_path_factory.mktemp("media") / "tiny.mp4"
    subprocess.run(
        [
            ffmpeg,
            "-y",
            "-v",
            "error",
            "-f",
            "lavfi",
            "-i",
            "testsrc=size=320x180:rate=30:duration=2",
            "-f",
            "lavfi",
            "-i",
            "color=c=red:size=320x180:rate=30:duration=2",
            "-f",
            "lavfi",
            "-i",
            "smptebars=size=320x180:rate=30:duration=2",
            "-filter_complex",
            "[0:v][1:v][2:v]concat=n=3:v=1:a=0,format=yuv420p[v]",
            "-map",
            "[v]",
            "-c:v",
            "mpeg4",
            "-q:v",
            "3",
            str(output),
        ],
        check=True,
    )
    return output


@pytest.fixture
def data_tree(tmp_path: Path, tiny_video: Path, monkeypatch) -> Path:
    """Create a temporary project tree with 3 videos across 2 batches."""
    root = tmp_path / "project"
    for batch, name in (
        ("Videos_L21", "L21_V001.mp4"),
        ("Videos_L21", "L21_V002.mp4"),
        ("Videos_L22", "L22_V001.mp4"),
    ):
        target = root / "data/videos" / batch / name
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(tiny_video, target)
    (root / "data_processed").mkdir(parents=True, exist_ok=True)
    (root / "weights.pth").write_bytes(b"stub-transnetv2-weights")
    (root / "data/video_to_playlist.json").write_text(
        '{"L21_V001": "news", "L21_V002": "news", "L22_V001": "sports"}\n'
    )
    monkeypatch.setattr(phase1_pipeline, "PROJECT_ROOT", root)
    monkeypatch.setattr(phase2_pipeline, "PROJECT_ROOT", root)
    return root


@pytest.fixture
def stub_detector(monkeypatch):
    """Patch TransNetV2Torch with a deterministic stub returning 3 shots on tiny_video."""

    class StubDetector:
        calls = 0
        device = "cpu"
        visualize_predictions = staticmethod(TransNetV2Torch.visualize_predictions)

        def __init__(self, *args, **kwargs):
            pass

        def predict_frames(self, frames: np.ndarray):
            type(self).calls += 1
            single = np.zeros(len(frames), dtype=np.float32)
            if len(frames) > 120:
                single[59] = 0.9
                single[119] = 0.9
            elif len(frames) > 2:
                single[len(frames) // 2] = 0.9
            return single, single.copy()

    StubDetector.calls = 0
    monkeypatch.setattr(phase1_pipeline, "TransNetV2Torch", StubDetector)
    return StubDetector


@pytest.fixture
def stub_embedder():
    """Return a deterministic 8-d unit-vector embedder class with no model weights."""

    class StubEmbedder(BaseEmbedding):
        DEFAULT_MODEL = "openai/clip-vit-base-patch32"
        calls = 0
        loads = 0

        def __init__(self, model_name: str = DEFAULT_MODEL, device: str = "cpu"):
            type(self).loads += 1
            self.model_name = model_name
            self.device = torch.device("cpu")
            self.model = None

        @staticmethod
        def _to_vector(image: ImageInput) -> torch.Tensor:
            if isinstance(image, (str, Path)):
                with Image.open(image) as opened:
                    rgb = np.asarray(opened.convert("RGB"), dtype=np.float32) / 255.0
            elif isinstance(image, Image.Image):
                rgb = np.asarray(image.convert("RGB"), dtype=np.float32) / 255.0
            else:
                rgb = np.asarray(image, dtype=np.float32) / 255.0
            means = rgb.mean(axis=(0, 1))
            stds = rgb.std(axis=(0, 1))
            vec = torch.tensor(
                [
                    float(means[0]),
                    float(means[1]),
                    float(means[2]),
                    float(stds[0]) + 0.1,
                    float(stds[1]) + 0.1,
                    float(stds[2]) + 0.1,
                    float(means[0] - means[2]),
                    1.0,
                ],
                dtype=torch.float32,
            )
            return F.normalize(vec, dim=0)

        def encode_text(self, text: str | list[str]) -> torch.Tensor:
            items = [text] if isinstance(text, str) else list(text)
            rows = []
            for item in items:
                base = torch.tensor(
                    [1.0 if "red" in item.lower() else 0.2, 0.2, 0.2, 0.1, 0.1, 0.1, 0.5, 1.0],
                    dtype=torch.float32,
                )
                rows.append(F.normalize(base, dim=0))
            stacked = torch.stack(rows, dim=0)
            return stacked[0] if isinstance(text, str) else stacked

        def encode_image(self, image: ImageInput) -> torch.Tensor:
            return self._to_vector(image)

        def encode_images_batch(self, images: list[ImageInput]) -> torch.Tensor:
            type(self).calls += 1
            return torch.stack([self._to_vector(img) for img in images], dim=0)

    StubEmbedder.calls = 0
    StubEmbedder.loads = 0
    return StubEmbedder


@pytest.fixture
def processed_tree(data_tree: Path, stub_detector, stub_embedder) -> Path:
    """Run stub Phase 1 and Phase 2 over data_tree and reset call counters."""
    summary1 = phase1_pipeline.run_pipeline(
        data_tree / "data/videos",
        output_root=data_tree / "data_processed/transnetv2",
        weights=data_tree / "weights.pth",
    )
    assert summary1 == {"processed": 3, "skipped": 0, "failed": 0}
    summary2 = phase2_pipeline.run_pipeline(
        data_tree / "data_processed/transnetv2",
        keyframe_root=data_tree / "data_processed/keyframes",
        output_root=data_tree / "data_processed/features",
        embedder_factory=stub_embedder,
    )
    assert summary2 == {"processed": 3, "skipped": 0, "failed": 0}
    stub_detector.calls = 0
    stub_embedder.calls = 0
    stub_embedder.loads = 0
    return data_tree
