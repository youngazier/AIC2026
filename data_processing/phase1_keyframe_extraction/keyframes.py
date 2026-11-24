"""Extract one full-resolution middle JPEG per inclusive shot."""

from pathlib import Path

import numpy as np
from PIL import Image

from .metadata import iter_video_frames


def shot_records(scenes: np.ndarray, timestamps: np.ndarray) -> list[dict]:
    return [
        {"shot_index": index, "start_frame": int(start), "end_frame": int(end),
         "start_timestamp": float(timestamps[start]), "end_timestamp": float(timestamps[end])}
        for index, (start, end) in enumerate(scenes)
    ]


def middle_frame(start: int, end: int) -> int:
    if start < 0 or end < start:
        raise ValueError("Expected 0 <= start <= end")
    return start + (end - start) // 2


def keyframe_filename(shot_index: int) -> str:
    return f"{shot_index:05d}.jpg"


def validate_shots(records: list[dict], scenes: np.ndarray) -> None:
    """Check saved boundaries and playback times without a per-frame sidecar."""
    if not isinstance(records, list) or len(records) != len(scenes):
        raise ValueError("Shot count disagrees with detector intervals")
    previous = 0.0
    for index, (record, (start, end)) in enumerate(zip(records, scenes, strict=True)):
        if (not isinstance(record, dict) or record.get("shot_index") != index
                or record.get("start_frame") != int(start) or record.get("end_frame") != int(end)):
            raise ValueError("Shot metadata disagrees with detector intervals")
        times = [record["start_timestamp"], record["end_timestamp"]]
        if not all(isinstance(t, (int, float)) and np.isfinite(t) for t in times):
            raise ValueError("Shot timestamps must be finite seconds")
        if not previous <= times[0] <= times[1]:
            raise ValueError("Shot timestamps must be ordered and nonnegative")
        previous = times[1]


def extract_middle_keyframes(video_path: Path, metadata: dict, scenes: np.ndarray,
                             timestamps: np.ndarray, image_dir: Path,
                             image_path_prefix: str) -> list[dict]:
    records = shot_records(scenes, timestamps)
    targets = {middle_frame(record["start_frame"], record["end_frame"]): record
               for record in records}
    image_dir.mkdir(parents=True, exist_ok=True)
    count = 0
    for index, frame, timestamp in iter_video_frames(video_path, metadata):
        if index >= len(timestamps) or not np.isclose(timestamp, timestamps[index], rtol=0, atol=1e-6):
            raise ValueError(f"Detection/extraction frame timing mismatch at frame {index}")
        count += 1
        if index not in targets:
            continue
        record = targets[index]
        name = keyframe_filename(record["shot_index"])
        frame.to_image().save(image_dir / name, format="JPEG", quality=95, subsampling=0)
        record["keyframe"] = {
            "role": "middle", "frame_index": index, "timestamp": timestamp,
            "image_path": f"{image_path_prefix}/{name}",
        }
    if count != len(timestamps):
        raise ValueError(f"Detection/extraction frame counts differ: {len(timestamps)} vs {count}")
    validate_keyframes(image_dir, image_path_prefix, records, scenes, timestamps, metadata)
    return records


def validate_keyframes(image_dir: Path, image_path_prefix: str, records: list[dict],
                       scenes: np.ndarray, timestamps: np.ndarray | None, metadata: dict) -> None:
    validate_shots(records, scenes)
    if timestamps is not None:
        for record, expected in zip(records, shot_records(scenes, timestamps), strict=True):
            if any(record.get(key) != value for key, value in expected.items()):
                raise ValueError("Shot metadata disagrees with decoded timestamps")
    expected_names = set()
    for record in records:
        index = middle_frame(record["start_frame"], record["end_frame"])
        name = keyframe_filename(record["shot_index"])
        expected_names.add(name)
        keyframe = record.get("keyframe")
        if not isinstance(keyframe, dict):
            raise ValueError("Shot is missing its middle keyframe")
        time = keyframe.get("timestamp")
        if (not isinstance(time, (int, float)) or not np.isfinite(time)
                or not record["start_timestamp"] <= time <= record["end_timestamp"]):
            raise ValueError("Keyframe timestamp must lie within the shot")
        expected_time = float(timestamps[index]) if timestamps is not None else time
        if keyframe != {"role": "middle", "frame_index": index,
                        "timestamp": expected_time, "image_path": f"{image_path_prefix}/{name}"}:
            raise ValueError("Expected exactly one middle keyframe with the correct source position")
        with Image.open(image_dir / name) as image:
            image.load()
            if image.format != "JPEG" or image.size != (metadata["width"], metadata["height"]):
                raise ValueError(f"Invalid full-resolution JPEG: {name}")
    if {p.name for p in image_dir.iterdir()} != expected_names:
        raise ValueError("Keyframe directory must contain exactly one middle JPEG per shot")
