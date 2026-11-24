"""Tests for building FAISS indexes from per-video feature files."""

import json
from pathlib import Path

import faiss
import numpy as np

from data_processing.phase2_feature_extraction.build_index import build_for_model


def write_feature(model_dir: Path, video_name: str, batch: str, vectors: np.ndarray) -> None:
    feature = model_dir / f"{video_name}.npy"
    np.save(feature, vectors.astype(np.float32), allow_pickle=False)
    records = [{
        "embedding_index": index,
        "keyframe_name": f"{index:05d}",
        "shot_index": index,
        "frame_index": index * 10,
        "timestamp": float(index),
        "image_path": f"keyframes/{batch}/{video_name}/{index:05d}.jpg",
    } for index in range(len(vectors))]
    metadata = {
        "status": "COMPLETED", "model_name": "test/clip",
        "representation_version": "test-v1", "batch": batch,
        "video_name": video_name, "embedding_count": len(vectors),
        "embedding_dimension": vectors.shape[1], "records": records,
    }
    feature.with_suffix(".json").write_text(json.dumps(metadata))


def test_builds_global_and_playlist_indexes(tmp_path):
    features = tmp_path / "features"
    model_dir = features / "clip"
    model_dir.mkdir(parents=True)
    write_feature(model_dir, "L21_V001", "Videos_L21",
                  np.array([[1, 0, 0], [0, 1, 0]], dtype=np.float32))
    write_feature(model_dir, "L22_V001", "Videos_L22",
                  np.array([[0, 0, 1]], dtype=np.float32))
    mapping = tmp_path / "video_to_playlist.json"
    mapping.write_text(json.dumps({"L21_V001": "news", "L22_V001": "sports"}))
    indexes = tmp_path / "indexes"

    summary = build_for_model(
        "clip", features_root=features, index_root=indexes,
        playlist_mapping=mapping,
    )
    assert summary == {"built": 3, "skipped": 0}
    index = faiss.read_index(str(indexes / "clip_keyframe_embedding.index"))
    assert index.ntotal == 3
    assert index.d == 3
    records = json.loads((indexes / "clip_keyframe_metadata.json").read_text())
    assert [(record["video_name"], record["keyframe_name"]) for record in records] == [
        ("L21_V001", "00000"), ("L21_V001", "00001"), ("L22_V001", "00000")
    ]
    assert faiss.read_index(str(indexes / "playlist/clip_news.index")).ntotal == 2

    assert build_for_model(
        "clip", features_root=features, index_root=indexes,
        playlist_mapping=mapping,
    ) == {"built": 0, "skipped": 3}
