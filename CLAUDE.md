# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project

Local multimodal video retrieval system for the HCM AI Challenge 2026 (text known-item search, VQA, and temporal event retrieval over news/broadcast video). Milestones M1–M3 (offline processing, embeddings/FAISS, SQLite catalog) are implemented; M4+ (Python retrieval, FastAPI API, React frontend, Nginx/Docker deployment) are planned. `docs/TODOS.md` is the source of truth for what is done vs. planned — check it before assuming a component exists. `docs/02_ARCHITECTURE.md` describes the target system; `docs/03`–`06` give the artifact and command contracts for each stage.

## Commands

Python ≥3.14, managed with `uv`. FFmpeg/`ffprobe` must be installed for Phase 1.

```bash
uv sync

# Tests
uv run pytest
uv run pytest tests/test_phase1.py
uv run pytest tests/test_phase1.py::test_middle_selection
uv run pytest -k catalog

# Phase 1: TransNetV2 shot detection + one middle keyframe per shot
uv run python -m data_processing.cli transnetv2 data/videos/Videos_L21/L21_V001.mp4   # shots only
uv run python -m data_processing.cli phase1 --batch Videos_L21 [--device auto|cpu|gpu|cuda] [--force]

# Phase 2: CLIP ViT-B/32 image embeddings, then FAISS index
uv run python -m data_processing.cli phase2 --batch Videos_L21 --device gpu
uv run python -m data_processing.phase2_feature_extraction.build_index --model clip

# Catalog (SQLite via Alembic)
uv run alembic upgrade head
uv run python -m backend.app.catalog.cli import --input data_processed/transnetv2
```

Phase 1 Docker images live in `data_processing/phase1_keyframe_extraction/compose.yaml`; run `docker compose` from that directory (`phase1` service, `phase1-gpu` under the `gpu` profile).

## Architecture

Pipeline of independent stages that communicate only through validated on-disk artifacts:

1. **Phase 1** (`data_processing/phase1_keyframe_extraction/`): `data/videos/Videos_L*/L*_V*.mp4` → `data_processed/transnetv2/<batch>/<video>/` (predictions, scenes, `shots.json`) and `data_processed/keyframes/<batch>/<video>/` (full-res JPEGs `00000.jpg`…). Vendored TransNetV2 code and weights are in `transnetv2/` inside that package.
2. **Phase 2** (`data_processing/phase2_feature_extraction/`): validates Phase 1 output, writes one L2-normalized `.npy` matrix per video plus a JSON sidecar (row mapping + model/input provenance) to `data_processed/features/<model>/`. `build_index.py` combines them into `IndexFlatIP` FAISS indexes (global + optional per-playlist from `data/video_to_playlist.json`) with row metadata and a manifest in `data_processed/indexes/`. New encoders implement `BaseEmbedding`.
3. **Catalog** (`backend/app/`): SQLAlchemy models (`videos`, `shots`, `keyframes`), Alembic migrations in `backend/migrations/`, DB at `data_processed/database/catalog.sqlite3`. `catalog/validation.py` builds immutable records from Phase 1 artifacts before any DB write; `catalog/importer.py` writes one video per short transaction; `catalog/queries.py` has typed read queries. The import CLI only accepts inputs under `data_processed/transnetv2` and ignores dot-prefixed (staging) dirs.

Cross-cutting invariants that the code relies on:

- **Publish atomically.** Each stage writes to a staging location, validates, then publishes; a lock file (e.g. `data_processed/transnetv2/.processing.lock`) serializes publication and catalog import. Failed runs must leave the previous valid artifacts intact. Reruns skip outputs that still validate unless `--force`.
- **Validated JSON is the source of truth.** SQLite is derived from Phase 1 `shots.json`; FAISS is rebuilt from Phase 2 sidecars. Catalog import does not touch features or indexes.
- **FAISS row numbers are not IDs** — they are only meaningful with the row metadata published alongside that exact index. Persist video/keyframe identifiers instead.
- **Use stored presentation timestamps** from decoding; never derive playback time from frame index × FPS.
- Video names must be unique across batches (importer rejects duplicates) because Phase 2 feature filenames and playlist mapping are keyed by video name only.
- `database.py` enables SQLite foreign keys and a 5s busy timeout on every connection; schema changes go through Alembic, not `create_all`.

The implemented store is SQLite + FAISS. `test/transnetv2_pytorch/` is an upstream reference checkout (with its own `.git`), not the project test suite — tests live in `tests/`.
