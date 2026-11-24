"""Inference adapted from the pinned upstream implementation in UPSTREAM.md."""

import logging
from pathlib import Path

import numpy as np
import torch

from .model import TransNetV2

LOGGER = logging.getLogger(__name__)


def resolve_device(device: str) -> torch.device:
    """Resolve auto/CPU/GPU selection and fail clearly for unavailable CUDA."""
    device = "cuda" if device == "gpu" else device
    if device not in {"auto", "cpu", "cuda"}:
        raise ValueError("Device must be one of: auto, cpu, gpu, cuda")
    selected = torch.device(
        ("cuda" if torch.cuda.is_available() else "cpu") if device == "auto" else device
    )
    if selected.type == "cuda" and not torch.cuda.is_available():
        raise ValueError("GPU/CUDA was requested but is unavailable; use --device cpu")
    return selected


def predictions_to_scenes(predictions: np.ndarray, threshold: float = 0.5) -> np.ndarray:
    """Preserve upstream inclusive boundaries and transition gaps."""
    predictions = np.asarray(predictions)
    if (predictions.ndim != 1 or not len(predictions)
            or not np.isfinite(predictions).all()
            or np.any((predictions < 0) | (predictions > 1))):
        raise ValueError("Expected nonempty, finite one-dimensional probabilities in [0, 1]")
    if not np.isfinite(threshold) or not 0 <= threshold <= 1:
        raise ValueError("Threshold must be finite and within [0, 1]")
    binary = (predictions > threshold).astype(np.uint8)
    scenes, previous, start = [], 0, 0
    for index, transition in enumerate(binary):
        if previous == 1 and transition == 0:
            start = index
        if previous == 0 and transition == 1 and index != 0:
            scenes.append([start, index])
        previous = transition
    if transition == 0:
        scenes.append([start, index])
    if not scenes:
        scenes = [[0, len(predictions) - 1]]
    return np.asarray(scenes, dtype=np.int64)


class TransNetV2Torch:
    def __init__(self, weights_path: Path, device: str = "auto"):
        self.device = resolve_device(device)
        self.model = TransNetV2()
        self.model.load_state_dict(torch.load(weights_path, map_location="cpu", weights_only=True))
        self.model.eval().to(self.device)
        device_name = torch.cuda.get_device_name(self.device) if self.device.type == "cuda" else "CPU"
        LOGGER.info("Loaded TransNetV2 on %s (%s)", self.device, device_name)

    @torch.inference_mode()
    def predict_raw(self, frames_tensor):
        logits, outputs = self.model(frames_tensor)
        return logits.cpu().numpy(), outputs["many_hot"].cpu().numpy()

    def predict_frames(self, frames: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        if (frames.ndim != 4 or frames.shape[1:] != (27, 48, 3)
                or frames.dtype != np.uint8 or not len(frames)):
            raise ValueError("Expected nonempty uint8 RGB frames with shape [N, 27, 48, 3]")
        total = len(frames)
        single, many = [], []
        # Clip indices to repeat the first/last frame, matching upstream padding.
        for start in range(0, total, 50):
            indices = np.clip(np.arange(start - 25, start + 75), 0, total - 1)
            window = np.ascontiguousarray(frames[indices][None])
            logits, many_logits = self.predict_raw(torch.from_numpy(window).to(self.device))
            single.append(logits[0, 25:75, 0])
            many.append(many_logits[0, 25:75, 0])
            if start % 2500 == 0 or start + 50 >= total:
                LOGGER.info("TransNetV2 frames %d/%d", min(start + 50, total), total)
        # Numerically stable sigmoid, including very large-magnitude logits.
        single = np.concatenate(single)[:total]
        many = np.concatenate(many)[:total]
        return np.exp(-np.logaddexp(0, -single)), np.exp(-np.logaddexp(0, -many))

    predictions_to_scenes = staticmethod(predictions_to_scenes)

    @staticmethod
    def visualize_predictions(frames: np.ndarray, predictions):
        from PIL import Image, ImageDraw
        if isinstance(predictions, np.ndarray):
            predictions = [predictions]
        ih, iw, _ = frames.shape[1:]
        width = 25
        pad_with = width - len(frames) % width if len(frames) % width != 0 else 0
        frames = np.pad(frames, [(0, pad_with), (0, 1), (0, len(predictions)), (0, 0)])
        predictions = [np.pad(x, (0, pad_with)) for x in predictions]
        height = len(frames) // width
        img = frames.reshape([height, width, ih + 1, iw + len(predictions), 3])
        img = np.concatenate(np.split(np.concatenate(np.split(img, height), axis=2)[0], width), axis=2)[0, :-1]
        img = Image.fromarray(img)
        draw = ImageDraw.Draw(img)
        for i, pred in enumerate(zip(*predictions)):
            x, y = (i % width, i // width)
            x, y = (x * (iw + len(predictions)) + iw, y * (ih + 1) + ih - 1)
            for j, p in enumerate(pred):
                color = [0, 0, 0]
                color[(j + 1) % 3] = 255
                value = round(p * (ih - 1))
                if value != 0:
                    draw.line((x + j, y, x + j, y - value), fill=tuple(color), width=1)
        return img
