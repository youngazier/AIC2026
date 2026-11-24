"""Encode Phase 1 keyframes into one NumPy feature file per video."""

from contextlib import contextmanager
from datetime import datetime, timezone
import fcntl
import json
import logging
from pathlib import Path
import shutil
from uuid import uuid4

import numpy as np
from PIL import Image
import torch

from .clip_embedding import CLIPEmbedding


LOGGER = logging.getLogger(__name__)
PROJECT_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_PHASE1_ROOT = PROJECT_ROOT / "data_processed" / "transnetv2"
DEFAULT_KEYFRAME_ROOT = PROJECT_ROOT / "data_processed" / "keyframes"
DEFAULT_OUTPUT_ROOT = PROJECT_ROOT / "data_processed" / "features"
DEFAULT_MODEL_KEY = "clip"
PIPELINE_VERSION = "phase2-features-v2"
REPRESENTATION_VERSION = "clip-l2-v1"


def _now() -> str:
    return datetime.now(timezone.utc).isoformat()


def _write_json(path: Path, value) -> None:
    path.write_text(json.dumps(value, indent=2, ensure_ascii=False, allow_nan=False) + "\n")


@contextmanager
def _processing_lock(output_root: Path):
    output_root.mkdir(parents=True, exist_ok=True)
    with (output_root / ".processing.lock").open("a") as lock:
        try:
            fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError as exc:
            raise ValueError(f"Another Phase 2 processor is using {output_root}") from exc
        try:
            yield
        finally:
            fcntl.flock(lock, fcntl.LOCK_UN)


def read_phase1_result(result_dir: Path) -> tuple[dict, list[dict]]:
    """Read the single-file format, with support for existing legacy outputs."""
    result = json.loads((result_dir / "shots.json").read_text())
    if isinstance(result, dict):
        return result, result["shots"]
    manifest = json.loads((result_dir / "manifest.json").read_text())
    if not isinstance(manifest, dict) or manifest.get("status") != "COMPLETED":
        raise ValueError(f"Legacy Phase 1 result is incomplete: {result_dir}")
    return manifest, result


def discover_phase1_results(input_path: Path) -> list[Path]:
    """Return completed Phase 1 result directories in deterministic order."""
    input_path = input_path.resolve()
    if input_path.is_file() and input_path.name in {"shots.json", "manifest.json"}:
        candidates = [input_path.parent]
    elif input_path.is_dir() and (input_path / "shots.json").is_file():
        candidates = [input_path]
    elif input_path.is_dir():
        candidates = sorted(path.parent for path in input_path.rglob("shots.json")
                            if not any(part.startswith(".") for part in path.relative_to(input_path).parts))
    else:
        raise ValueError(f"Phase 1 input does not exist: {input_path}")
    results, identities = [], set()
    for result_dir in candidates:
        try:
            result, _ = read_phase1_result(result_dir)
        except (OSError, ValueError, KeyError, TypeError):
            continue
        if result.get("mode") != "phase1":
            continue
        identity = result.get("video_name")
        if identity in identities:
            raise ValueError(f"Duplicate Phase 1 video identity: {identity}")
        identities.add(identity)
        results.append(result_dir)
    if not results:
        raise ValueError(f"No completed Phase 1 results found in {input_path}")
    return results


def validate_phase1_input(result_dir: Path, keyframe_root: Path) -> tuple[dict, list[dict], list[Path]]:
    """Validate Phase 1 provenance and return its exact ordered image list."""
    try:
        result, records = read_phase1_result(result_dir)
    except (OSError, ValueError, KeyError, TypeError) as exc:
        raise ValueError(f"Unreadable Phase 1 result: {result_dir}") from exc
    required = {"batch", "video_name", "pipeline_version"}
    if result.get("mode") != "phase1" or not required.issubset(result):
        raise ValueError(f"Phase 1 result is not completed or lacks provenance: {result_dir}")
    if (not isinstance(records, list) or not records
            or not all(isinstance(record, dict) for record in records)
            or len(records) != result.get("shot_count", len(records))):
        raise ValueError(f"Phase 1 shot records are incomplete: {result_dir}")

    image_dir = keyframe_root / result["batch"] / result["video_name"]
    expected_names = [f"{index:05d}.jpg" for index in range(len(records))]
    if result.get("keyframe_artifacts", expected_names) != expected_names:
        raise ValueError(f"Phase 1 result has an invalid keyframe set: {result_dir}")
    if {path.name for path in image_dir.iterdir()} != set(expected_names):
        raise ValueError(f"Phase 1 images do not match the shot list: {result_dir}")
    images = []
    for index, (record, name) in enumerate(zip(records, expected_names, strict=True)):
        try:
            relative = record["keyframe"]["image_path"]
        except (KeyError, TypeError) as exc:
            raise ValueError(f"Shot {index} has no keyframe reference") from exc
        if (record.get("shot_index") != index
                or relative != f"keyframes/{result['batch']}/{result['video_name']}/{name}"):
            raise ValueError(f"Shot {index} and its keyframe filename disagree")
        path = image_dir / name
        if not path.is_file():
            raise ValueError(f"Missing Phase 1 keyframe: {path}")
        try:
            with Image.open(path) as image:
                image.verify()
        except (OSError, ValueError) as exc:
            raise ValueError(f"Unreadable Phase 1 keyframe: {path}") from exc
        images.append(path)
    return result, records, images


def validate_vectors(vectors: np.ndarray, expected_count: int) -> None:
    """Reject incomplete, nonfinite, or unnormalized feature matrices."""
    if (vectors.ndim != 2 or vectors.shape[0] != expected_count or vectors.shape[1] < 1
            or vectors.dtype != np.float32 or not np.isfinite(vectors).all()):
        raise ValueError("Features must be a finite float32 matrix with one row per keyframe")
    if not np.allclose(np.linalg.vector_norm(vectors, axis=1), 1.0, rtol=1e-4, atol=1e-5):
        raise ValueError("Features must be nonzero and L2-normalized")


def metadata_path(feature_path: Path) -> Path:
    return feature_path.with_suffix(".json")


def validate_result(feature_path: Path, expected: dict | None = None) -> dict:
    """Validate a per-video feature file and its JSON row mapping."""
    sidecar = metadata_path(feature_path)
    try:
        metadata = json.loads(sidecar.read_text())
        vectors = np.load(feature_path, allow_pickle=False)
    except (OSError, json.JSONDecodeError, ValueError) as exc:
        raise ValueError(f"Unreadable Phase 2 feature: {feature_path}") from exc
    if not isinstance(metadata, dict) or metadata.get("status") != "COMPLETED":
        raise ValueError("Phase 2 feature is not completed")
    if expected and any(metadata.get(key) != value for key, value in expected.items()):
        raise ValueError("Phase 2 provenance does not match the requested run")
    records = metadata.get("records")
    if (not isinstance(records, list) or not all(isinstance(record, dict) for record in records)
            or len(records) != metadata.get("embedding_count")):
        raise ValueError("Feature records and count disagree")
    if [record.get("embedding_index") for record in records] != list(range(len(records))):
        raise ValueError("Feature records are not in canonical order")
    validate_vectors(vectors, len(records))
    if vectors.shape[1] != metadata.get("embedding_dimension"):
        raise ValueError("Feature dimension and metadata disagree")
    return metadata


def _publish_pair(staged_feature: Path, staged_metadata: Path,
                  feature_path: Path, sidecar: Path) -> None:
    """Publish the array and metadata together, restoring old files on failure."""
    token = uuid4().hex
    feature_backup = feature_path.with_name(f".{feature_path.name}.backup-{token}")
    metadata_backup = sidecar.with_name(f".{sidecar.name}.backup-{token}")
    if feature_path.exists():
        feature_path.rename(feature_backup)
    if sidecar.exists():
        sidecar.rename(metadata_backup)
    try:
        staged_feature.replace(feature_path)
        staged_metadata.replace(sidecar)
    except BaseException:
        feature_path.unlink(missing_ok=True)
        sidecar.unlink(missing_ok=True)
        if feature_backup.exists():
            feature_backup.rename(feature_path)
        if metadata_backup.exists():
            metadata_backup.rename(sidecar)
        raise
    feature_backup.unlink(missing_ok=True)
    metadata_backup.unlink(missing_ok=True)


def process_result(result_dir: Path, *, keyframe_root: Path, model_dir: Path,
                   get_embedder, model_name: str, batch_size: int,
                   force: bool = False) -> tuple[str, Path]:
    """Encode one video and atomically publish its feature/metadata pair."""
    phase1, shots, images = validate_phase1_input(result_dir, keyframe_root)
    video_name = phase1["video_name"]
    feature_path = model_dir / f"{video_name}.npy"
    sidecar = metadata_path(feature_path)
    expected = {
        "pipeline_version": PIPELINE_VERSION,
        "representation_version": REPRESENTATION_VERSION,
        "model_name": model_name,
        "input_shots_mtime_ns": (result_dir / "shots.json").stat().st_mtime_ns,
        "batch": phase1["batch"],
        "video_name": video_name,
    }
    # Old results keep their provenance in a separate file until Phase 1 reruns.
    if (result_dir / "manifest.json").is_file():
        expected["input_manifest_mtime_ns"] = (result_dir / "manifest.json").stat().st_mtime_ns
    if not force:
        try:
            validate_result(feature_path, expected)
            LOGGER.info("SKIPPED %s: %s", video_name, feature_path)
            return "skipped", feature_path
        except ValueError:
            pass

    embedder = get_embedder()
    work_dir = model_dir / ".work" / f"{video_name}-{uuid4().hex}"
    work_dir.mkdir(parents=True)
    staged_feature, staged_metadata = work_dir / feature_path.name, work_dir / sidecar.name
    LOGGER.info("PROCESSING %s: %d keyframes", video_name, len(images))
    started_at = _now()
    try:
        chunks = []
        for start in range(0, len(images), batch_size):
            tensor = embedder.encode_images_batch(images[start:start + batch_size])
            chunks.append(tensor.detach().to(device="cpu", dtype=torch.float32).numpy())
        vectors = np.concatenate(chunks, axis=0).astype(np.float32, copy=False)
        validate_vectors(vectors, len(images))
        np.save(staged_feature, vectors, allow_pickle=False)
        records = [{
            "embedding_index": index,
            "keyframe_name": path.stem,
            "shot_index": shot["shot_index"],
            "frame_index": shot["keyframe"]["frame_index"],
            "timestamp": shot["keyframe"]["timestamp"],
            "image_path": shot["keyframe"]["image_path"],
        } for index, (shot, path) in enumerate(zip(shots, images, strict=True))]
        metadata = dict(expected, status="COMPLETED", started_at=started_at, finished_at=_now(),
                        device=str(embedder.device),
                        embedding_count=len(records), embedding_dimension=int(vectors.shape[1]),
                        dtype="float32", normalized=True,
                        input_pipeline_version=phase1["pipeline_version"], records=records)
        _write_json(staged_metadata, metadata)
        validate_result(staged_feature, expected)
        _publish_pair(staged_feature, staged_metadata, feature_path, sidecar)
        LOGGER.info("COMPLETED %s: %s", video_name, feature_path)
        return "processed", feature_path
    finally:
        shutil.rmtree(work_dir, ignore_errors=True)


def run_pipeline(input_path: Path = DEFAULT_PHASE1_ROOT, *,
                 keyframe_root: Path = DEFAULT_KEYFRAME_ROOT,
                 output_root: Path = DEFAULT_OUTPUT_ROOT,
                 model_name: str = CLIPEmbedding.DEFAULT_MODEL,
                 model_key: str = DEFAULT_MODEL_KEY, device: str = "auto",
                 batch_size: int = 64, force: bool = False,
                 embedder_factory=CLIPEmbedding) -> dict[str, int]:
    """Encode every selected Phase 1 result into model-scoped `.npy` files."""
    if batch_size < 1:
        raise ValueError("Batch size must be at least 1")
    allowed = "abcdefghijklmnopqrstuvwxyz0123456789-_"
    if not model_key or any(character not in allowed for character in model_key):
        raise ValueError("Model key may contain only lowercase letters, digits, '-' and '_'")
    input_path, keyframe_root, output_root = (Path(value).resolve() for value in
                                              (input_path, keyframe_root, output_root))
    if output_root.is_relative_to(PROJECT_ROOT / "data"):
        raise ValueError("data/ is reserved for competition inputs")
    results = discover_phase1_results(input_path)
    model_dir = output_root / model_key
    model_dir.mkdir(parents=True, exist_ok=True)
    embedder = None

    def get_embedder():
        nonlocal embedder
        if embedder is None:
            embedder = embedder_factory(model_name=model_name, device=device)
        return embedder

    summary = {"processed": 0, "skipped": 0, "failed": 0}
    with _processing_lock(model_dir):
        shutil.rmtree(model_dir / ".work", ignore_errors=True)
        for index, result in enumerate(results, 1):
            LOGGER.info("Phase 1 result %d/%d: %s", index, len(results), result)
            try:
                status, _ = process_result(
                    result, keyframe_root=keyframe_root, model_dir=model_dir,
                    get_embedder=get_embedder, model_name=model_name,
                    batch_size=batch_size, force=force,
                )
                summary[status] += 1
            except Exception:
                summary["failed"] += 1
                LOGGER.exception("FAILED %s", result)
    return summary
