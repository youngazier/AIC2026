"""Sequential video processing with one small result directory per video."""

from collections.abc import Sequence
from contextlib import contextmanager
import fcntl
import json
import logging
import os
from pathlib import Path
import re
import shutil
from uuid import uuid4

import numpy as np
from PIL import Image

from .keyframes import extract_middle_keyframes, shot_records, validate_keyframes, validate_shots
from .metadata import VIDEO_EXTENSIONS, probe_video
from .transnet import UPSTREAM_REVISION, detect_shots, read_detection
from .transnetv2.inference import TransNetV2Torch

LOGGER = logging.getLogger(__name__)
PROJECT_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_OUTPUT_ROOT = PROJECT_ROOT / "data_processed" / "transnetv2"
DEFAULT_WEIGHTS = Path(__file__).resolve().parent / "transnetv2" / "transnetv2-pytorch-weights.pth"
PIPELINE_VERSION = "phase1-middle-v4"


def video_identity(path: Path) -> tuple[str, str]:
    batch = path.parent.name
    if re.fullmatch(r"Videos_L\d+", batch):
        if not re.fullmatch(rf"{re.escape(batch.removeprefix('Videos_'))}_V\d+", path.stem):
            raise ValueError(f"Video name {path.name} does not match batch {batch}")
    else:
        batch = "unbatched"
    return batch, path.stem


def discover_videos(input_path: Path) -> list[Path]:
    if input_path.is_file():
        if input_path.suffix.lower() not in VIDEO_EXTENSIONS:
            raise ValueError(f"Unsupported video extension: {input_path}")
        return [input_path.resolve()]
    if not input_path.is_dir():
        raise ValueError(f"Input does not exist: {input_path}")
    videos = sorted(p.resolve() for p in input_path.rglob("*")
                    if p.is_file() and p.suffix.lower() in VIDEO_EXTENSIONS)
    if not videos:
        raise ValueError(f"No supported, unpacked videos found in {input_path}")
    return videos


@contextmanager
def processing_lock(output_root: Path):
    output_root.mkdir(parents=True, exist_ok=True)
    with (output_root / ".processing.lock").open("a") as lock:
        try:
            fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError as exc:
            raise ValueError(f"Another processor is using {output_root}") from exc
        try:
            yield
        finally:
            fcntl.flock(lock, fcntl.LOCK_UN)


def _remove_entry(path: Path) -> None:
    if path.is_dir() and not path.is_symlink():
        shutil.rmtree(path, ignore_errors=True)
    else:
        path.unlink(missing_ok=True)


def _cleanup_leftovers(*roots: Path) -> None:
    """Remove leftover .work/ directories and .*.backup-* crash artifacts."""
    for root in roots:
        if not root.is_dir():
            continue
        _remove_entry(root / ".work")
        for entry in root.iterdir():
            if entry.name.startswith(".") and ".backup-" in entry.name:
                _remove_entry(entry)
            elif entry.is_dir() and not entry.name.startswith("."):
                for child in entry.iterdir():
                    if child.name.startswith(".") and ".backup-" in child.name:
                        _remove_entry(child)


def validate_result(result_dir: Path, keyframe_dir: Path, result: dict,
                    *, decode_images: bool = True) -> None:
    stem = result["video_name"]
    required = {"shots.json", f"{stem}.predictions.txt", f"{stem}.scenes.txt"}
    predictions, scenes = read_detection(result_dir, stem, result["threshold"])
    if len(predictions) != result["frame_count"]:
        raise ValueError("Frame count disagrees with detector outputs")
    records, metadata = result["shots"], result["metadata"]
    if not isinstance(metadata, dict):
        raise ValueError("Malformed video metadata")
    validate_shots(records, scenes)
    if result["mode"] == "phase1":
        validate_keyframes(keyframe_dir, f"keyframes/{result['batch']}/{stem}",
                           records, scenes, None, metadata, decode_images=decode_images)
    elif result["mode"] != "transnetv2" or any("keyframe" in record for record in records):
        raise ValueError("Invalid processing mode or keyframe records")
    if result["visualize"]:
        required.add(f"{stem}.vis.png")
        with Image.open(result_dir / f"{stem}.vis.png") as image:
            if decode_images:
                image.load()
            elif image.format != "PNG":
                raise ValueError("Invalid visualization PNG")
    if {p.name for p in result_dir.iterdir()} != required:
        raise ValueError("Result must contain exactly the expected artifacts")


def completed_result(video_dir: Path, keyframe_dir: Path, expected: dict,
                     mode: str, visualize: bool, *, verify: bool = False) -> bool:
    try:
        result = json.loads((video_dir / "shots.json").read_text())
        if not isinstance(result, dict) or any(result.get(k) != v for k, v in expected.items()):
            return False
        if mode == "phase1" and result.get("mode") != "phase1":
            return False
        if visualize and not result.get("visualize"):
            return False
        validate_result(video_dir, keyframe_dir, result, decode_images=verify)
        return True
    except (OSError, ValueError, KeyError, TypeError, IndexError) as exc:
        LOGGER.debug("No reusable result for %s: %s", video_dir, exc)
        return False


def publish_result(staging_dir: Path, video_dir: Path, staging_keyframes: Path | None,
                   keyframe_dir: Path) -> None:
    """Replace results and keyframes together, restoring previous outputs on failure."""
    token = uuid4().hex
    pairs = [(staging_dir, video_dir)]
    if staging_keyframes is not None or keyframe_dir.exists():
        pairs.append((staging_keyframes, keyframe_dir))
    backups, published = [], []
    try:
        for staged, target in pairs:
            target.parent.mkdir(parents=True, exist_ok=True)
            if target.exists():
                backup = target.with_name(f".{target.name}.backup-{token}")
                target.rename(backup)
                backups.append((backup, target))
            if staged is not None:
                staged.rename(target)
                published.append(target)
    except BaseException:
        for target in published:
            shutil.rmtree(target)
        for backup, target in reversed(backups):
            backup.rename(target)
        raise
    for backup, _ in backups:
        shutil.rmtree(backup)


def process_video(video_path: Path, *, output_root: Path, weights: Path,
                  get_detector, mode: str,
                  threshold: float, force: bool, visualize: bool,
                  verify: bool = False) -> tuple[str, Path]:
    batch, video_name = video_identity(video_path)
    video_dir = output_root / batch / video_name
    keyframe_dir = output_root.parent / "keyframes" / batch / video_name
    video_dir.parent.mkdir(parents=True, exist_ok=True)
    expected = {
        "source_key": Path(os.path.relpath(video_path, PROJECT_ROOT)).as_posix(),
        "source_size": video_path.stat().st_size,
        "checkpoint_size": weights.stat().st_size,
        "pipeline_version": PIPELINE_VERSION, "threshold": threshold,
        "upstream_revision": UPSTREAM_REVISION,
    }
    if not force and completed_result(video_dir, keyframe_dir, expected, mode, visualize, verify=verify):
        LOGGER.info("SKIPPED %s: %s", video_name, video_dir)
        return "skipped", video_dir

    work_root = output_root / ".work"
    staging_root = work_root / f"{video_name}-{uuid4().hex}"
    result_dir, staged_keyframes = staging_root / "transnetv2", staging_root / "keyframes"
    result_dir.mkdir(parents=True)
    LOGGER.info("PROCESSING %s (%s)", video_name, mode)
    try:
        metadata = probe_video(video_path)
        detector = get_detector()
        predictions, scenes, timestamps = detect_shots(
            detector, video_path, metadata, result_dir, threshold=threshold, visualize=visualize,
        )
        records = (extract_middle_keyframes(
                       video_path, metadata, scenes, timestamps, staged_keyframes,
                       f"keyframes/{batch}/{video_name}",
                   ) if mode == "phase1" else shot_records(scenes, timestamps))
        result = dict(expected, video_name=video_name, batch=batch, mode=mode,
                      visualize=visualize, frame_count=len(predictions), metadata=metadata, shots=records)
        (result_dir / "shots.json").write_text(
            json.dumps(result, indent=2, ensure_ascii=False, allow_nan=False) + "\n")
        validate_result(result_dir, staged_keyframes, result)
        publish_result(result_dir, video_dir,
                       staged_keyframes if mode == "phase1" else None, keyframe_dir)
        LOGGER.info("COMPLETED %s: %d shots, %s", video_name, len(scenes), video_dir)
        return "processed", video_dir
    finally:
        shutil.rmtree(staging_root, ignore_errors=True)
        try:
            work_root.rmdir()
        except OSError:
            pass


def run_pipeline(input_path: Path | Sequence[Path], *, output_root: Path = DEFAULT_OUTPUT_ROOT,
                 weights: Path = DEFAULT_WEIGHTS, mode: str = "phase1", threshold: float = 0.5,
                 device: str = "auto", force: bool = False, visualize: bool = False,
                 verify: bool = False) -> dict[str, int]:
    if mode not in {"phase1", "transnetv2"}:
        raise ValueError("Mode must be phase1 or transnetv2")
    if not np.isfinite(threshold) or not 0 <= threshold <= 1:
        raise ValueError("Threshold must be finite and within [0, 1]")
    output_root, weights = output_root.resolve(), weights.resolve()
    if output_root.is_relative_to(PROJECT_ROOT / "data"):
        raise ValueError("data/ is reserved for competition inputs; choose an output directory outside it")
    if not weights.is_file():
        raise ValueError(f"Checkpoint missing: {weights}. See docs/03_VIDEO_PROCESSING.md for setup")
    if isinstance(input_path, (str, Path)):
        videos = discover_videos(Path(input_path))
    else:
        videos = []
        for path in input_path:
            resolved = Path(path).resolve()
            if not resolved.is_file() or resolved.suffix.lower() not in VIDEO_EXTENSIONS:
                raise ValueError(f"Unsupported or missing video file: {path}")
            videos.append(resolved)
        if not videos:
            raise ValueError("No input videos provided")
    destinations = set()
    for path in videos:
        identity = video_identity(path)
        if identity in destinations:
            raise ValueError(f"Multiple input videos would share an output directory: {identity}")
        destinations.add(identity)
    detector = None

    def get_detector():
        nonlocal detector
        if detector is None:
            detector = TransNetV2Torch(weights, device=device)
        return detector

    summary = {"processed": 0, "skipped": 0, "failed": 0}
    with processing_lock(output_root):
        _cleanup_leftovers(output_root, output_root.parent / "keyframes")
        for index, path in enumerate(videos, 1):
            LOGGER.info("Video %d/%d: %s", index, len(videos), path)
            try:
                status, _ = process_video(
                    path, output_root=output_root, weights=weights, get_detector=get_detector,
                    mode=mode, threshold=threshold, force=force, visualize=visualize, verify=verify,
                )
                summary[status] += 1
            except Exception:
                summary["failed"] += 1
                LOGGER.exception("FAILED %s", path)
    return summary
