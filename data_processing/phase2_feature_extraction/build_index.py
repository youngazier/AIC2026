"""Build global and optional playlist FAISS indexes from Phase 2 feature files."""

import argparse
from collections import defaultdict
import json
import logging
from pathlib import Path
import re
from uuid import uuid4

import faiss
import numpy as np

from .pipeline import DEFAULT_MODEL_KEY, DEFAULT_OUTPUT_ROOT, PROJECT_ROOT, validate_result


LOGGER = logging.getLogger(__name__)
DEFAULT_INDEX_ROOT = PROJECT_ROOT / "data_processed" / "indexes"
DEFAULT_PLAYLIST_MAPPING = PROJECT_ROOT / "data" / "video_to_playlist.json"
INDEX_VERSION = "faiss-flat-ip-v1"


def discover_models(features_root: Path, requested: str) -> list[str]:
    """Select one model directory or every directory containing feature arrays."""
    features_root = features_root.resolve()
    if requested != "all":
        model_dir = features_root / requested
        if not any(model_dir.glob("*.npy")):
            raise ValueError(f"No feature files found in {model_dir}")
        return [requested]
    models = sorted(path.name for path in features_root.iterdir()
                    if path.is_dir() and any(path.glob("*.npy"))) if features_root.is_dir() else []
    if not models:
        raise ValueError(f"No model feature directories found in {features_root}")
    return models


def load_model_features(model_dir: Path) -> tuple[np.ndarray, list[dict], list[dict]]:
    """Load every valid video feature file in stable video/keyframe order."""
    vectors, records, sources = [], [], []
    dimensions, model_names, representations = set(), set(), set()
    for feature_path in sorted(model_dir.glob("*.npy")):
        metadata = validate_result(feature_path)
        array = np.load(feature_path, allow_pickle=False)
        dimensions.add(array.shape[1])
        model_names.add(metadata["model_name"])
        representations.add(metadata.get("representation_version"))
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
            "feature_file": feature_path.name,
            "model_name": metadata["model_name"],
            "representation_version": metadata["representation_version"],
            "feature_size": feature_path.stat().st_size,
            "feature_mtime_ns": feature_path.stat().st_mtime_ns,
            "metadata_mtime_ns": feature_path.with_suffix(".json").stat().st_mtime_ns,
        })
    if not vectors:
        raise ValueError(f"No valid feature files found in {model_dir}")
    if len(dimensions) != 1 or len(model_names) != 1 or len(representations) != 1:
        raise ValueError(f"Feature files in {model_dir} use incompatible models or dimensions")
    return np.concatenate(vectors).astype(np.float32, copy=False), records, sources


def _safe_name(value: str) -> str:
    name = re.sub(r"[^A-Za-z0-9_-]+", "_", value).strip("_")
    if not name:
        raise ValueError(f"Playlist name cannot be used as a filename: {value!r}")
    return name


def build_flat_index(vectors: np.ndarray, records: list[dict], sources: list[dict],
                     *, index_path: Path, metadata_path: Path, manifest_path: Path,
                     label: str, force: bool = False) -> bool:
    """Build an exact inner-product index unless its source files are unchanged."""
    if not force and index_path.is_file() and metadata_path.is_file() and manifest_path.is_file():
        try:
            previous = json.loads(manifest_path.read_text())
            if previous.get("sources") == sources and previous.get("vector_count") == len(records):
                LOGGER.info("[%s] unchanged; keeping %d-vector index", label, len(records))
                return False
        except (OSError, json.JSONDecodeError):
            pass
    if vectors.ndim != 2 or len(vectors) != len(records) or not len(records):
        raise ValueError(f"Cannot build {label}: vectors and metadata do not match")

    index = faiss.IndexFlatIP(vectors.shape[1])
    index.add(np.ascontiguousarray(vectors, dtype=np.float32))
    index_path.parent.mkdir(parents=True, exist_ok=True)
    token = uuid4().hex
    temporary_index = index_path.with_name(f".{index_path.name}.{token}.tmp")
    temporary_metadata = metadata_path.with_name(f".{metadata_path.name}.{token}.tmp")
    temporary_manifest = manifest_path.with_name(f".{manifest_path.name}.{token}.tmp")
    try:
        faiss.write_index(index, str(temporary_index))
        temporary_metadata.write_text(json.dumps(records, indent=2, ensure_ascii=False) + "\n")
        manifest = {
            "index_version": INDEX_VERSION,
            "metric": "inner_product",
            "normalized_features": True,
            "vector_count": int(index.ntotal),
            "dimension": int(index.d),
            "sources": sources,
        }
        temporary_manifest.write_text(json.dumps(manifest, indent=2, ensure_ascii=False) + "\n")
        temporary_index.replace(index_path)
        temporary_metadata.replace(metadata_path)
        temporary_manifest.replace(manifest_path)
    finally:
        temporary_index.unlink(missing_ok=True)
        temporary_metadata.unlink(missing_ok=True)
        temporary_manifest.unlink(missing_ok=True)
    LOGGER.info("[%s] wrote %d vectors (dim=%d) to %s", label, index.ntotal, index.d, index_path)
    return True


def build_for_model(model_key: str, *, features_root: Path, index_root: Path,
                    playlist_mapping: Path | None = DEFAULT_PLAYLIST_MAPPING,
                    force: bool = False) -> dict[str, int]:
    """Build the global index and every mapped playlist index for one model."""
    vectors, records, sources = load_model_features(features_root / model_key)
    prefix = index_root / f"{model_key}_keyframe"
    summary = {"built": 0, "skipped": 0}
    changed = build_flat_index(
        vectors, records, sources,
        index_path=prefix.with_name(prefix.name + "_embedding.index"),
        metadata_path=prefix.with_name(prefix.name + "_metadata.json"),
        manifest_path=prefix.with_name(prefix.name + "_index.json"),
        label=f"{model_key}/global", force=force,
    )
    summary["built" if changed else "skipped"] += 1

    if playlist_mapping is None or not playlist_mapping.is_file():
        if playlist_mapping is not None:
            LOGGER.warning("Playlist mapping not found at %s; only the global index was built",
                           playlist_mapping)
        return summary
    mapping = json.loads(playlist_mapping.read_text())
    if not isinstance(mapping, dict) or not all(isinstance(key, str) and isinstance(value, str)
                                                for key, value in mapping.items()):
        raise ValueError("Playlist mapping must be a JSON object of video IDs to playlist names")

    positions = defaultdict(list)
    for position, record in enumerate(records):
        playlist = mapping.get(record["video_name"])
        if playlist:
            positions[playlist].append(position)
    playlist_dir = index_root / "playlist"
    for playlist in sorted(positions):
        selected = positions[playlist]
        playlist_records = [{**records[position], "index": index}
                            for index, position in enumerate(selected)]
        video_ids = {record["video_name"] for record in playlist_records}
        playlist_sources = [source for source in sources
                            if source["video_name"] in video_ids]
        name = f"{model_key}_{_safe_name(playlist)}"
        changed = build_flat_index(
            vectors[selected], playlist_records, playlist_sources,
            index_path=playlist_dir / f"{name}.index",
            metadata_path=playlist_dir / f"{name}_metadata.json",
            manifest_path=playlist_dir / f"{name}_index.json",
            label=f"{model_key}/{playlist}", force=force,
        )
        summary["built" if changed else "skipped"] += 1
    return summary


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(
        description="Build complete FAISS indexes from all available per-video feature files."
    )
    parser.add_argument("--model", default=DEFAULT_MODEL_KEY,
                        help="Model key to index, or 'all' for every feature directory")
    parser.add_argument("--features-root", type=Path, default=DEFAULT_OUTPUT_ROOT)
    parser.add_argument("--index-root", type=Path, default=DEFAULT_INDEX_ROOT)
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
                model, features_root=args.features_root.resolve(),
                index_root=args.index_root.resolve(),
                playlist_mapping=None if args.no_playlists else args.playlist_mapping.resolve(),
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
