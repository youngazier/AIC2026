"""Source metadata and shared display-order decoding for both processing passes."""

import json
import math
from pathlib import Path
import subprocess

import av
import numpy as np

VIDEO_EXTENSIONS = {".mp4", ".mkv", ".avi", ".mov", ".webm", ".flv"}


def probe_video(path: Path) -> dict:
    """Choose the first non-cover-art video stream by absolute stream index."""
    result = subprocess.run(
        ["ffprobe", "-v", "error", "-show_streams", "-show_format", "-of", "json", str(path)],
        capture_output=True, text=True, check=False,
    )
    if result.returncode:
        raise ValueError(f"ffprobe failed for {path}: {result.stderr.strip()}")
    info = json.loads(result.stdout)
    stream = next((s for s in info.get("streams", [])
                   if s.get("codec_type") == "video"
                   and not s.get("disposition", {}).get("attached_pic")), None)
    if stream is None or min(stream.get("width", 0), stream.get("height", 0)) <= 0:
        raise ValueError(f"No usable video stream in {path}")
    rotations = [float(s.get("rotation", 0)) for s in stream.get("side_data_list", [])]
    rotations.append(float(stream.get("tags", {}).get("rotate", 0)))
    if any(r % 360 for r in rotations):
        raise ValueError("Rotated videos are not supported yet; refusing ambiguous image orientation")
    container = info.get("format", {})
    duration = float(container.get("duration", stream.get("duration", "nan")))
    origin = float(container.get("start_time", stream.get("start_time", 0)))
    if not math.isfinite(duration) or duration <= 0 or not math.isfinite(origin):
        raise ValueError(f"Invalid duration or timeline origin in {path}")
    numerator, denominator = map(int, stream.get("avg_frame_rate", "0/0").split("/"))
    return {
        "stream_index": stream["index"], "codec": stream.get("codec_name"),
        "width": stream["width"], "height": stream["height"],
        "duration_seconds": duration,
        "average_fps": numerator / denominator if denominator else None,
        "timeline_origin_seconds": origin,
        "orientation": "coded pixels; nonzero rotation rejected",
    }


def iter_video_frames(path: Path, metadata: dict):
    """Yield (zero-based index, PyAV frame, playback seconds), without resampling."""
    previous = -math.inf
    with av.open(str(path)) as container:
        stream = container.streams[metadata["stream_index"]]
        stream.thread_type = "AUTO"
        stream.codec_context.thread_count = 4
        for index, frame in enumerate(container.decode(stream)):
            if frame.is_corrupt or frame.pts is None or frame.time_base is None:
                raise ValueError(f"Corrupt frame or missing presentation timestamp at frame {index}")
            if (frame.width, frame.height) != (metadata["width"], metadata["height"]):
                raise ValueError(f"Video dimensions changed at frame {index}")
            timestamp = float(frame.pts * frame.time_base) - metadata["timeline_origin_seconds"]
            if not math.isfinite(timestamp) or timestamp < -1e-6 or timestamp < previous:
                raise ValueError(f"Invalid/nonmonotonic playback timestamp at frame {index}")
            timestamp = max(0.0, timestamp)
            previous = timestamp
            yield index, frame, timestamp


def decode_for_detection(path: Path, metadata: dict) -> tuple[np.ndarray, np.ndarray]:
    frames, timestamps = [], []
    for _, frame, timestamp in iter_video_frames(path, metadata):
        frames.append(frame.to_ndarray(width=48, height=27, format="rgb24", interpolation="BICUBIC"))
        timestamps.append(timestamp)
    if not frames:
        raise ValueError(f"Video decoded to zero frames: {path}")
    return np.stack(frames), np.asarray(timestamps, dtype=np.float64)
