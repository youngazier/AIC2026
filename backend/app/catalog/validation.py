"""Prepare validated catalog rows without opening a database transaction."""

from dataclasses import dataclass
import json
import math
from pathlib import Path, PurePosixPath

from backend.app.database import PROJECT_ROOT
from data_processing.phase1_keyframe_extraction.metadata import VIDEO_EXTENSIONS
from data_processing.phase1_keyframe_extraction.pipeline import validate_result, video_identity


class CatalogValidationError(ValueError):
    """A published Phase 1 result cannot safely be imported."""


@dataclass(frozen=True)
class PreparedVideo:
    batch: str
    video_name: str
    video_path: str
    manifest_path: str
    manifest_mtime_ns: int
    stream_index: int
    codec: str | None
    width: int
    height: int
    duration_seconds: float
    average_fps: float | None
    timeline_origin_seconds: float
    frame_count: int
    shot_count: int


@dataclass(frozen=True)
class PreparedShot:
    batch: str
    video_name: str
    shot_index: int
    start_frame: int
    end_frame: int
    start_timestamp: float
    end_timestamp: float


@dataclass(frozen=True)
class PreparedKeyframe:
    batch: str
    video_name: str
    shot_index: int
    role: str
    frame_index: int
    timestamp: float
    image_path: str


@dataclass(frozen=True)
class PreparedCatalogVideo:
    video: PreparedVideo
    shots: tuple[PreparedShot, ...]
    keyframes: tuple[PreparedKeyframe, ...]


def _integer(value: object, name: str, minimum: int = 0) -> int:
    if type(value) is not int or value < minimum:
        raise CatalogValidationError(f"{name} must be an integer >= {minimum}")
    return value


def _seconds(value: object, name: str, *, minimum: float | None = None) -> float:
    if type(value) not in (int, float) or not math.isfinite(value):
        raise CatalogValidationError(f"{name} must be finite numeric seconds")
    if minimum is not None and value < minimum:
        raise CatalogValidationError(f"{name} must be >= {minimum}")
    return float(value)


def _key(value: object, name: str) -> str:
    if (not isinstance(value, str) or not value or "\\" in value
            or PurePosixPath(value).is_absolute()
            or any(part in ("", ".", "..") for part in value.split("/"))):
        raise CatalogValidationError(f"{name} must be a relative POSIX key without traversal")
    return value


def _file(base: Path, key: str, approved_root: Path) -> Path:
    path = base / key
    resolved = path.resolve(strict=True)
    if not resolved.is_relative_to(approved_root) or not resolved.is_file():
        raise CatalogValidationError(f"File escapes its approved root or is not a file: {key}")
    return path


def _signature(path: Path) -> tuple[int, int, int, int]:
    stat = path.stat()
    return stat.st_dev, stat.st_ino, stat.st_size, stat.st_mtime_ns


def prepare_catalog_video(
    manifest_path: str | Path, *, project_root: Path = PROJECT_ROOT,
) -> PreparedCatalogVideo:
    """Validate a published shots.json and return immutable, database-ready rows.

    Source keys are relative to project_root; generated keys are relative to
    data_processed. No database is accessed. The importer supplies imported_at
    only when committing. Concurrent Phase 1 publication must be serialized by
    the importer; signature checks here also detect changes during validation.
    """
    try:
        return _prepare_catalog_video(Path(manifest_path), project_root.resolve())
    except CatalogValidationError:
        raise
    except (OSError, ValueError, KeyError, TypeError, IndexError, OverflowError) as exc:
        raise CatalogValidationError(f"Invalid Phase 1 catalog input {manifest_path}: {exc}") from exc


def _prepare_catalog_video(manifest: Path, root: Path) -> PreparedCatalogVideo:
    processed = root / "data_processed"
    video_root = root / "data/videos"
    # Approved roots themselves must not redirect outside the project.
    for approved in (processed, video_root):
        if not approved.resolve().is_relative_to(root):
            raise CatalogValidationError("Configured media root escapes the project")
    for approved in (processed / "transnetv2", processed / "keyframes"):
        if not approved.resolve().is_relative_to(processed.resolve()):
            raise CatalogValidationError("Generated artifact root escapes data_processed")
    if not manifest.is_absolute():
        manifest = root / manifest
    manifest_key = _key(manifest.relative_to(processed).as_posix(), "manifest_path")
    _file(processed, manifest_key, (processed / "transnetv2").resolve())
    before = _signature(manifest)
    result = json.loads(manifest.read_text())
    if not isinstance(result, dict) or result.get("mode") != "phase1":
        raise CatalogValidationError("Expected a published Phase 1 shots.json object")
    batch = _key(result["batch"], "batch")
    name = _key(result["video_name"], "video_name")
    if "/" in batch or "/" in name or batch.startswith(".") or name.startswith("."):
        raise CatalogValidationError("Batch and video name must be visible single path components")
    if manifest_key != f"transnetv2/{batch}/{name}/shots.json":
        raise CatalogValidationError("Manifest directory disagrees with video identity")
    for field in ("pipeline_version", "upstream_revision"):
        if not isinstance(result.get(field), str) or not result[field].strip():
            raise CatalogValidationError(f"Missing processing provenance: {field}")
    threshold = _seconds(result["threshold"], "threshold", minimum=0)
    if threshold > 1 or type(result["visualize"]) is not bool:
        raise CatalogValidationError("Invalid threshold or visualization setting")
    for field in ("source_size", "checkpoint_size"):
        _integer(result[field], field)
    source_key = _key(result["source_key"], "source_key")
    source = _file(root, source_key, video_root.resolve())
    if source.suffix.lower() not in VIDEO_EXTENSIONS or video_identity(source) != (batch, name):
        raise CatalogValidationError("Source video identity or extension disagrees with manifest")
    if source.stat().st_size != result["source_size"]:
        raise CatalogValidationError("Source video changed since Phase 1 processing")

    metadata = result["metadata"]
    stream_index = _integer(metadata["stream_index"], "stream_index")
    width = _integer(metadata["width"], "width", 1)
    height = _integer(metadata["height"], "height", 1)
    duration = _seconds(metadata["duration_seconds"], "duration_seconds", minimum=0)
    if duration == 0:
        raise CatalogValidationError("duration_seconds must be positive")
    origin = _seconds(metadata["timeline_origin_seconds"], "timeline_origin_seconds")
    fps = metadata.get("average_fps")
    if fps is not None:
        fps = _seconds(fps, "average_fps", minimum=0)
    codec = metadata.get("codec")
    if codec is not None and (not isinstance(codec, str) or not codec.strip()):
        raise CatalogValidationError("codec must be a nonempty string or null")
    frame_count = _integer(result["frame_count"], "frame_count", 1)
    records = result["shots"]
    if not isinstance(records, list) or not records:
        raise CatalogValidationError("Expected a nonempty shot list")
    if "shot_count" in result and _integer(result["shot_count"], "shot_count", 1) != len(records):
        raise CatalogValidationError("shot_count disagrees with shot records")

    shots, keyframes = [], []
    watched = {manifest: before, source: _signature(source)}
    # Validate containment before the Phase 1 validator opens any artifact.
    for artifact in manifest.parent.iterdir():
        key = _key(artifact.relative_to(processed).as_posix(), "artifact_path")
        _file(processed, key, (processed / "transnetv2").resolve())
        watched.setdefault(artifact, _signature(artifact))
    for index, record in enumerate(records):
        shot_index = _integer(record["shot_index"], "shot_index")
        if shot_index != index:
            raise CatalogValidationError("Shot indices must be contiguous and ordered")
        start = _integer(record["start_frame"], "start_frame")
        end = _integer(record["end_frame"], "end_frame")
        if not start <= end < frame_count:
            raise CatalogValidationError("Shot frames lie outside the video")
        start_time = _seconds(record["start_timestamp"], "start_timestamp", minimum=0)
        end_time = _seconds(record["end_timestamp"], "end_timestamp", minimum=0)
        keyframe = record["keyframe"]
        frame = _integer(keyframe["frame_index"], "frame_index")
        timestamp = _seconds(keyframe["timestamp"], "timestamp", minimum=0)
        image_key = _key(keyframe["image_path"], "image_path")
        if image_key != f"keyframes/{batch}/{name}/{index:05d}.jpg":
            raise CatalogValidationError("Keyframe path disagrees with shot identity")
        image = _file(processed, image_key, (processed / "keyframes").resolve())
        watched[image] = _signature(image)
        shots.append(PreparedShot(batch, name, index, start, end, start_time, end_time))
        keyframes.append(PreparedKeyframe(batch, name, index, keyframe["role"], frame, timestamp, image_key))

    validate_result(manifest.parent, processed / "keyframes" / batch / name, result)
    if any(_signature(path) != signature for path, signature in watched.items()):
        raise CatalogValidationError("Phase 1 inputs changed during validation; retry import")
    video = PreparedVideo(batch, name, source_key, manifest_key, before[3], stream_index,
                          codec, width, height, duration, fps, origin, frame_count, len(shots))
    return PreparedCatalogVideo(video, tuple(shots), tuple(keyframes))
