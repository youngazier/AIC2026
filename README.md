# AIC2026

Offline video preprocessing for the competition video retrieval project.

- `data/`: competition-provided videos, media-info, map-keyframes, and other source assets.
- `data_processing/phase1_keyframe_extraction/transnetv2/`: vendored TransNetV2 code and checkpoint.
- `data_processing/phase2_feature_extraction/`: shared embedding interface and CLIP ViT-B/32 implementation.
- `data_processed/transnetv2/`: predictions, scenes, `shots.json`, and optional visualizations.
- `data_processed/keyframes/`: extracted full-resolution keyframe JPEGs (`00000.jpg`, `00001.jpg`, ...).

```bash
uv sync

# Detect shots only.
uv run python -m data_processing.cli transnetv2 data/videos/Videos_L21/L21_V001.mp4

# Detect shots and save one full-resolution middle keyframe per shot.
uv run python -m data_processing.cli phase1 --batch Videos_L21

# Explicit device selection (auto, cpu, gpu, and cuda are supported).
uv run python -m data_processing.cli phase1 --batch Videos_L21 --device gpu

# Encode all valid Phase 1 keyframes in one batch with CLIP ViT-B/32.
uv run python -m data_processing.cli phase2 --batch Videos_L21 --device gpu

# Build a complete FAISS index from every encoded video.
uv run python -m data_processing.phase2_feature_extraction.build_index --model clip
```

See [Video processing](docs/03_VIDEO_PROCESSING.md) for checkpoint setup,
prerequisites, Docker usage, outputs, and reruns. See
[Feature extraction](docs/04_FEATURE_EXTRACTION.md) for using CLIP to encode
completed Phase 1 results, building the FAISS index, and direct text/image
encoding.

M3 includes SQLAlchemy catalog models, SQLite configuration, Alembic migrations,
validated atomic imports, and typed catalog queries. Initialize and import:

```bash
uv run alembic upgrade head
uv run python -m backend.app.catalog.cli import --input data_processed/transnetv2
```

Imports preserve committed rows on failure and reject duplicate video names
across batches. Retrieval and HTTP API integration are the next milestones.
See [Database](docs/06_DATABASE.md) for the implemented Python interfaces and
[TODOs](docs/TODOS.md) for the remaining milestones.
