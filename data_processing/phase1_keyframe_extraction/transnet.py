"""Validated detector artifacts, independent of the CLI and database."""

from pathlib import Path

import numpy as np

from .metadata import decode_for_detection
from .transnetv2.inference import TransNetV2Torch, predictions_to_scenes

UPSTREAM_REVISION = "b4927ebfa576791521c6fb02217b4a622c731c37"
def validate_arrays(predictions: np.ndarray, scenes: np.ndarray, timestamps: np.ndarray | None,
                    threshold: float) -> None:
    count = len(predictions)
    if (predictions.shape != (count, 2) or not count
            or not np.isfinite(predictions).all()
            or np.any((predictions < 0) | (predictions > 1))):
        raise ValueError("Predictions must contain two finite probabilities per decoded frame")
    if timestamps is not None and (timestamps.shape != (count,) or not np.isfinite(timestamps).all()
            or np.any(timestamps < 0) or np.any(np.diff(timestamps) < 0)):
        raise ValueError("Prediction count and valid presentation timestamps must agree")
    if (scenes.ndim != 2 or scenes.shape[1] != 2 or not len(scenes)
            or not np.issubdtype(scenes.dtype, np.integer)
            or np.any(scenes[:, 0] < 0) or np.any(scenes[:, 0] > scenes[:, 1])
            or np.any(scenes[:, 1] >= count)
            or np.any(scenes[1:, 0] <= scenes[:-1, 1])):
        raise ValueError("Shots must be ordered, nonoverlapping inclusive integer intervals")
    if not np.array_equal(scenes, predictions_to_scenes(predictions[:, 0], threshold)):
        raise ValueError("Shot intervals do not match the saved probabilities and threshold")


def read_detection(output_dir: Path, stem: str, threshold: float):
    predictions = np.loadtxt(output_dir / f"{stem}.predictions.txt", ndmin=2, dtype=np.float32)
    raw_scenes = np.loadtxt(output_dir / f"{stem}.scenes.txt", ndmin=2)
    if not np.isfinite(raw_scenes).all() or not np.equal(raw_scenes, np.floor(raw_scenes)).all():
        raise ValueError("Shot bounds must be finite integers")
    scenes = raw_scenes.astype(np.int64)
    validate_arrays(predictions, scenes, None, threshold)
    return predictions, scenes


def detect_shots(detector: TransNetV2Torch, video_path: Path, metadata: dict,
                 output_dir: Path, *, threshold: float = 0.5, visualize: bool = False):
    frames, timestamps = decode_for_detection(video_path, metadata)
    single, many = detector.predict_frames(frames)
    predictions = np.column_stack((single, many))
    scenes = predictions_to_scenes(single, threshold)
    validate_arrays(predictions, scenes, timestamps, threshold)
    output_dir.mkdir(parents=True, exist_ok=True)
    stem = video_path.stem
    # Nine significant digits round-trip the model's float32 probabilities.
    np.savetxt(output_dir / f"{stem}.predictions.txt", predictions, fmt="%.9g")
    np.savetxt(output_dir / f"{stem}.scenes.txt", scenes, fmt="%d")
    if visualize:
        detector.visualize_predictions(frames, (single, many)).save(output_dir / f"{stem}.vis.png")
    # Validate the serialized files as well as the in-memory outputs.
    predictions, scenes = read_detection(output_dir, stem, threshold)
    validate_arrays(predictions, scenes, timestamps, threshold)
    return predictions, scenes, timestamps
