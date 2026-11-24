"""Build global and optional playlist FAISS indexes from Phase 2 feature files."""

import argparse
from collections import Counter, defaultdict
import hashlib
import json
import logging
import os
from pathlib import Path
import re
from uuid import uuid4

import faiss
import numpy as np

from .pipeline import (
    DEFAULT_MODEL_KEY,
    DEFAULT_OUTPUT_ROOT,
    DEFAULT_PHASE1_ROOT,
    PROJECT_ROOT,
    read_phase1_result,
    validate_result,
)

LOGGER = logging.getLogger(__name__)
DEFAULT_INDEX_ROOT = PROJECT_ROOT / "data_processed" / "indexes"
DEFAULT_PLAYLIST_MAPPING = PROJECT_ROOT / "data" / "video_to_playlist.json"
INDEX_VERSION = "faiss-flat-ip-v2"


def discover_models(features_root: Path, requested: str) -> list[str]:
    """Select one model directory or every directory containing feature arrays."""
    features_root = features_root.resolve()
    if requested != "all":
        model_dir = features_root / requested
        if not any(model_dir.glob("*.npy")):
            raise ValueError(f"No feature files found in {model_dir}")
        return [requested]
    models = sorted(
        path.name for path in features_root.iterdir()
        if path.is_dir() and any(path.glob("*.npy"))
    ) if features_root.is_dir() else []
    if not models:
        raise ValueError(f"No model feature directories found in {features_root}")
    return models


def load_model_features(model_dir: Path) -> tuple[np.ndarray, list[dict], list[dict]]:
    """Load every valid video feature file in stable video/keyframe order."""
    vectors, records, sources = [], [], []
    dimensions, model_names, representations, precisions = set(), set(), set(), set()
    for feature_path in sorted(model_dir.glob("*.npy")):
        metadata = validate_result(feature_path)
        array = np.load(feature_path, allow_pickle=False)
        precision = metadata.get("precision", "fp32")
        dimensions.add(array.shape[1])
        model_names.add(metadata["model_name"])
        representations.add(metadata.get("representation_version"))
        precisions.add(precision)
        vectors.append(array)
        base_index = len(records)
        records.extend({
            "index": base_index + offset,
            "video_name": metadata["video_name"],
            "batch": metadata["batch"],
            **record,
        } for offset, record in enumerate(metadata["records"]))
        sources.append({
            "video_name": metadata["video_name"],
            "batch": metadata["batch"],
            "feature_file": feature_path.name,
            "model_name": metadata["model_name"],
            "representation_version": metadata["representation_version"],
            "precision": precision,
            "feature_size": feature_path.stat().st_size,
            "metadata_sha256": hashlib.sha256(
                feature_path.with_suffix(".json").read_bytes()
            ).hexdigest(),
        })
    if not vectors:
        raise ValueError(f"No valid feature files found in {model_dir}")
    if (len(dimensions) != 1 or len(model_names) != 1
            or len(representations) != 1 or len(precisions) != 1):
        raise ValueError(
            f"Feature files in {model_dir} use incompatible models, precisions, or dimensions"
        )
    return np.concatenate(vectors).astype(np.float32, copy=False), records, sources


def _safe_name(value: str) -> str:
    name = re.sub(r"[^A-Za-z0-9_-]+", "_", value).strip("_")
    if not name:
        raise ValueError(f"Playlist name cannot be used as a filename: {value!r}")
    return name


def _cleanup_temp_files(model_index_dir: Path) -> None:
    for directory in (model_index_dir, model_index_dir / "playlist"):
        if not directory.is_dir():
            continue
        for entry in directory.iterdir():
            if entry.is_file() and entry.name.startswith(".") and entry.name.endswith(".tmp"):
                entry.unlink(missing_ok=True)


def _report_batch_counts_and_missing(
    model_key: str, sources: list[dict], phase1_root: Path | None
) -> None:
    counts = Counter(source["batch"] for source in sources)
    for batch, count in sorted(counts.items()):
        print(f"Batch {batch}: {count} video(s)")

    if phase1_root is None or not phase1_root.is_dir():
        return
    loaded_videos = {source["video_name"] for source in sources}
    phase1_videos: set[str] = set()
    for shots_path in sorted(phase1_root.rglob("shots.json")):
        if any(part.startswith(".") for part in shots_path.relative_to(phase1_root).parts):
            continue
        try:
            header, _ = read_phase1_result(shots_path.parent)
        except (OSError, ValueError, KeyError, TypeError):
            continue
        if header.get("mode") == "phase1" and isinstance(header.get("video_name"), str):
            phase1_videos.add(header["video_name"])
    missing = sorted(phase1_videos - loaded_videos)
    if missing:
        LOGGER.warning(
            "Phase 1 output exists with no %s features for: %s",
            model_key,
            ", ".join(missing),
        )


def _validate_index_pair(
    index_path: Path, rows_path: Path, expected_count: int, expected_dim: int
) -> None:
    if not index_path.is_file():
        raise ValueError(f"Missing index file {index_path.name}")
    if not rows_path.is_file():
        raise ValueError(f"Missing rows file {rows_path.name}")
    try:
        index = faiss.read_index(str(index_path))
    except Exception as exc:
        raise ValueError(f"Failed to read {index_path.name}: {exc}") from exc
    if index.ntotal != expected_count or index.d != expected_dim:
        raise ValueError(
            f"{index_path.name} shape ({index.ntotal}, {index.d}) != expected ({expected_count}, {expected_dim})"
        )
    rows = json.loads(rows_path.read_text())
    if not isinstance(rows, list) or len(rows) != expected_count:
        raise ValueError(f"{rows_path.name} row count != {expected_count}")


def _can_skip_model_index(
    model_key: str,
    model_index_dir: Path,
    *,
    sources: list[dict],
    vector_count: int,
    dimension: int,
    playlist_specs: dict[str, tuple[list[int], list[dict]]],
) -> bool:
    manifest_path = model_index_dir / "manifest.json"
    if not manifest_path.is_file():
        LOGGER.info("Rebuilding %s index: missing manifest.json", model_key)
        return False
    try:
        manifest = json.loads(manifest_path.read_text())
        if manifest.get("index_version") != INDEX_VERSION:
            raise ValueError("index_version changed")
        if manifest.get("sources") != sources:
            raise ValueError("source features changed")
        if manifest.get("vector_count") != vector_count or manifest.get("dimension") != dimension:
            raise ValueError("vector count or dimension changed")
        expected_playlists = {
            name: len(selected) for name, (selected, _) in playlist_specs.items()
        }
        if manifest.get("playlists") != expected_playlists:
            raise ValueError("playlist mapping changed")

        _validate_index_pair(
            model_index_dir / "global.index",
            model_index_dir / "global_rows.json",
            vector_count,
            dimension,
        )
        playlist_dir = model_index_dir / "playlist"
        for safe_name, (selected, _) in playlist_specs.items():
            _validate_index_pair(
                playlist_dir / f"{safe_name}.index",
                playlist_dir / f"{safe_name}_rows.json",
                len(selected),
                dimension,
            )
    except (OSError, ValueError, RuntimeError, KeyError, TypeError, json.JSONDecodeError) as exc:
        LOGGER.info("Rebuilding %s index: %s", model_key, exc)
        return False
    LOGGER.info("[%s] unchanged; keeping %d-vector index", model_key, vector_count)
    return True


def _write_validated_pair(
    vectors: np.ndarray,
    rows: list[dict],
    index_path: Path,
    rows_path: Path,
    token: str,
) -> None:
    index = faiss.IndexFlatIP(vectors.shape[1])
    index.add(np.ascontiguousarray(vectors, dtype=np.float32))
    index_path.parent.mkdir(parents=True, exist_ok=True)
    temp_index = index_path.with_name(f".{index_path.name}.{token}.tmp")
    temp_rows = rows_path.with_name(f".{rows_path.name}.{token}.tmp")
    try:
        faiss.write_index(index, str(temp_index))
        temp_rows.write_text(json.dumps(rows, indent=2, ensure_ascii=False) + "\n")
        _validate_index_pair(temp_index, temp_rows, len(rows), vectors.shape[1])
        os.replace(temp_index, index_path)
        os.replace(temp_rows, rows_path)
    finally:
        temp_index.unlink(missing_ok=True)
        temp_rows.unlink(missing_ok=True)


def build_for_model(
    model_key: str,
    *,
    features_root: Path,
    index_root: Path,
    playlist_mapping: Path | None = DEFAULT_PLAYLIST_MAPPING,
    phase1_root: Path | None = None,
    force: bool = False,
) -> dict[str, int]:
    """Build the global index and every mapped playlist index under indexes/<model>/."""
    model_index_dir = index_root / model_key
    _cleanup_temp_files(model_index_dir)

    vectors, records, sources = load_model_features(features_root / model_key)
    _report_batch_counts_and_missing(model_key, sources, phase1_root)

    playlist_specs: dict[str, tuple[list[int], list[dict]]] = {}
    if playlist_mapping is not None and playlist_mapping.is_file():
        mapping = json.loads(playlist_mapping.read_text())
        if not isinstance(mapping, dict) or not all(
            isinstance(key, str) and isinstance(value, str) for key, value in mapping.items()
        ):
            raise ValueError("Playlist mapping must be a JSON object of video IDs to playlist names")
        positions: dict[str, list[int]] = defaultdict(list)
        for position, record in enumerate(records):
            playlist = mapping.get(record["video_name"])
            if playlist:
                positions[playlist].append(position)
        for playlist in sorted(positions):
            selected = positions[playlist]
            safe_name = _safe_name(playlist)
            playlist_rows = [
                {**records[position], "index": index}
                for index, position in enumerate(selected)
            ]
            playlist_specs[safe_name] = (selected, playlist_rows)
    elif playlist_mapping is not None:
        LOGGER.warning(
            "Playlist mapping not found at %s; only the global index was built",
            playlist_mapping,
        )

    total_targets = 1 + len(playlist_specs)
    dimension = int(vectors.shape[1])
    if not force and _can_skip_model_index(
        model_key,
        model_index_dir,
        sources=sources,
        vector_count=len(records),
        dimension=dimension,
        playlist_specs=playlist_specs,
    ):
        return {"built": 0, "skipped": total_targets}

    model_index_dir.mkdir(parents=True, exist_ok=True)
    token = uuid4().hex
    global_index = model_index_dir / "global.index"
    global_rows = model_index_dir / "global_rows.json"
    manifest_path = model_index_dir / "manifest.json"
    temp_manifest = manifest_path.with_name(f".{manifest_path.name}.{token}.tmp")

    try:
        _write_validated_pair(vectors, records, global_index, global_rows, token)
        playlist_dir = model_index_dir / "playlist"
        for safe_name, (selected, playlist_rows) in playlist_specs.items():
            _write_validated_pair(
                vectors[selected],
                playlist_rows,
                playlist_dir / f"{safe_name}.index",
                playlist_dir / f"{safe_name}_rows.json",
                token,
            )
        if playlist_dir.is_dir():
            expected_files = {
                name
                for safe_name in playlist_specs
                for name in (f"{safe_name}.index", f"{safe_name}_rows.json")
            }
            for entry in playlist_dir.iterdir():
                if entry.is_file() and entry.name not in expected_files:
                    entry.unlink(missing_ok=True)

        manifest = {
            "index_version": INDEX_VERSION,
            "model_key": model_key,
            "metric": "inner_product",
            "normalized_features": True,
            "vector_count": len(records),
            "dimension": dimension,
            "playlists": {name: len(selected) for name, (selected, _) in playlist_specs.items()},
            "sources": sources,
        }
        temp_manifest.write_text(json.dumps(manifest, indent=2, ensure_ascii=False) + "\n")
        json.loads(temp_manifest.read_text())
        os.replace(temp_manifest, manifest_path)
    finally:
        temp_manifest.unlink(missing_ok=True)

    LOGGER.info("[%s] wrote %d vectors (dim=%d) to %s", model_key, len(records), dimension, model_index_dir)
    return {"built": total_targets, "skipped": 0}


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(
        description="Build complete FAISS indexes from all available per-video feature files."
    )
    parser.add_argument(
        "--model", default=DEFAULT_MODEL_KEY,
        help="Model key to index, or 'all' for every feature directory",
    )
    parser.add_argument("--features-root", type=Path, default=DEFAULT_OUTPUT_ROOT)
    parser.add_argument("--index-root", type=Path, default=DEFAULT_INDEX_ROOT)
    parser.add_argument("--phase1-root", type=Path, default=DEFAULT_PHASE1_ROOT)
    parser.add_argument("--playlist-mapping", type=Path, default=DEFAULT_PLAYLIST_MAPPING)
    parser.add_argument("--no-playlists", action="store_true")
    parser.add_argument("--force", action="store_true")
    args = parser.parse_args(argv)
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    try:
        models = discover_models(args.features_root, args.model)
        total = {"built": 0, "skipped": 0}
        for model in models:
            result = build_for_model(
                model,
                features_root=args.features_root.resolve(),
                index_root=args.index_root.resolve(),
                playlist_mapping=None if args.no_playlists else args.playlist_mapping.resolve(),
                phase1_root=args.phase1_root.resolve(),
                force=args.force,
            )
            for key in total:
                total[key] += result[key]
    except (OSError, ValueError, json.JSONDecodeError) as exc:
        LOGGER.error("%s", exc)
        return 1
    print(f"Built: {total['built']}; skipped: {total['skipped']}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
