# Implementation TODOs — Video Retrieval Portfolio Project

This checklist tracks completed offline processing and the SQLite catalog,
followed by validated FAISS search and a FastAPI backend. Validated
Phase 1 JSON feeds the catalog; Phase 2 JSON/NumPy feeds index rebuilds.

## Current implementation

### M1 — Video processing

- [x] Scan supported downloaded video files and validate video name/batch names.
- [x] Probe video streams and retain decoded presentation timestamps.
- [x] Run TransNetV2 and validate shot intervals/predictions.
- [x] Extract one full-resolution middle keyframe per shot.
- [x] Write predictions, scenes, optional visualization, and `shots.json`; save middle keyframe JPEGs separately.
- [x] Use staging, output validation, locking, skip checks, and safe per-video reruns.
- [x] Document setup, commands, output layout, and rerun behavior in `03_VIDEO_PROCESSING.md`.

### M2 — Embeddings and index construction

- [x] Validate completed Phase 1 artifacts before embedding.
- [x] Encode keyframes with CLIP and save one normalized `.npy` matrix per video.
- [x] Save JSON row mappings and model/input provenance beside each matrix.
- [x] Build global and optional playlist FAISS inner-product indexes with JSON row metadata.
- [x] Compare source file names, feature sizes, and sidecar SHA-256 hashes to skip unchanged index construction.
- [x] Document feature/index commands and artifacts in `04_FEATURE_EXTRACTION.md`.

### M3 — SQLite catalog and schema

**Outcome:** completed Phase 1 artifacts produce a queryable, repeatable
SQLite catalog without changing the processing CLIs.

- [x] Add SQLAlchemy models for `videos`, `shots`, and `keyframes`.
- [x] Add the `ix_videos_video_name` model index and an Alembic initial migration matching [06_DATABASE.md](06_DATABASE.md).
- [x] Configure the SQLite path under `data_processed/database/`, foreign keys on every connection, a 5-second busy timeout, and a session factory for short write transactions.
- [x] Prepare immutable catalog records before database writes; validate published Phase 1 artifacts, readable keyframes, metadata/settings, source provenance, timestamps, identity, and path containment.
- [x] Build an importer that validates Phase 1 `shots.json` and readable keyframes before writing a video, its shots, and keyframes in one transaction.
- [x] Serialize artifact imports with Phase 1 publication; validate before opening the short write transaction.
- [x] Make imports idempotent, preserve the last committed rows on invalid input, and test reruns, changed artifacts, and rollback.
- [x] Reject duplicate video names across batches until Phase 2 feature filenames and playlist mapping are keyed by batch as well as video name.
- [x] Add typed catalog queries for paginated videos, video details, and keyframes.
- [x] Add a standalone catalog import CLI with deterministic discovery, per-video error reporting, and a nonzero exit status on failures.

M3 is implemented in `backend/app/database.py`, `backend/app/catalog/`, and
`backend/migrations/`. Tests cover validation, migration constraints, publication
locking, reruns, changed shot sets, duplicate names, rollback, queries, and CLI
reporting. The combined catalog and Phase 1 regression suite passed 74 tests.
Initialize the schema, then import published artifacts:

```bash
uv run alembic upgrade head
uv run python -m backend.app.catalog.cli import --input data_processed/transnetv2
```

Next milestone: M4 snapshot validation and Python retrieval. Catalog import does
not rebuild features or FAISS indexes; run those existing CLIs separately.

## Next implementation work

### M4 — FAISS snapshot validation and Python retrieval

**Outcome:** a natural-language query returns ranked keyframes with correct
SQLite video/shot references, without a web server.

- [x] Validate existing FAISS and row metadata files before an unchanged-run skip.
- [ ] Load the same CLIP checkpoint and representation used by the image features; encode and normalize text queries.
- [ ] Validate the FAISS index, manifest, row count/order, dimension, and model compatibility at startup.
- [ ] Check Phase 2 sidecar provenance against the current Phase 1 `shots.json` before serving search.
- [ ] Resolve returned rows through the matching metadata and SQLite video/shot/keyframe keys; verify image key and stored timestamp agree.
- [ ] Implement reusable unfiltered top-K retrieval returning batch, video name, shot/keyframe, frame index, stored timestamp, score, and media references.
- [ ] Handle missing/empty index, stale catalog, incompatible model, empty query, and invalid `top_k` with clear errors.
- [ ] Test representative queries and index/catalog mismatch cases; record relevance examples and query latency.
- [ ] Define complete filtered top-K behavior before exposing filters; playlist indexes have separate row mappings.

### M5 — Local FastAPI backend and browser

**Outcome:** a user can search, inspect keyframes, and open a source video at the matching timestamp.

- [ ] Add FastAPI configuration, startup/lifespan loading, request/response schemas, and thin routes over catalog and retrieval services.
- [ ] Implement `GET /api/v1/videos`, video detail/keyframes, and `POST /api/v1/search` backed by SQLite and the loaded FAISS snapshot.
- [ ] Serve media only from configured roots after resolving and validating paths.
- [ ] Support byte-range video responses so browser playback can seek.
- [ ] Add liveness and separate catalog/search readiness checks; return controlled 404/422/503 errors without leaking local paths.
- [ ] Test API contracts, pagination order, media path containment, and byte-range seeking.
- [ ] Build a small React search/results/playback interface if it adds value to the portfolio.
- [ ] Bind local development to localhost; define access control before remote exposure.

### M6 — Optional visual analysis

**Outcome:** selected retrieved frames can be inspected with explicit evidence and limits.

- [ ] Choose one detector or vision-language model based on representative target questions.
- [ ] Implement reusable frame inspection and visible-object counting over selected keyframes.
- [ ] Save useful results as JSON linked to source identifiers, timestamps, model, and prompt/configuration versions.
- [ ] Define whether counts are per frame, per shot, or across a selected result set; do not imply unique-object tracking.
- [ ] Show supporting frames, method, scope, and uncertainty in the response/UI.

### M7 — Evaluation and polish

- [ ] Create a small set of natural-language queries with relevant video/keyframe judgments.
- [ ] Measure Recall@K or another suitable ranking metric alongside manual relevance review.
- [ ] Measure extraction time, embedding time, FAISS memory, query latency, and video playback behavior.
- [ ] Compare model/index changes using retained provenance and reproducible input artifacts.
- [ ] Improve README run instructions and add a concise architecture/data-flow diagram for the portfolio.

## Storage and scope rules

- Keep competition inputs under `data/` and generated files under `data_processed/`.
- Keep validated Phase 1 JSON as the catalog import source and Phase 2 JSON/NumPy as the index source; use SQLite for video, shot, and keyframe queries.
- Keep source videos/keyframes as files; keep embeddings in per-video NumPy arrays.
- Treat the FAISS index and its metadata as rebuildable derived outputs; a FAISS row is not a permanent ID.
- Keep SQLite local and initialize it with a migration plus validated artifact import.
- Keep processing in the existing CLIs.
- Avoid Redis, Celery, Kubernetes, microservices, model registries, and generic storage frameworks until a measured requirement calls for them.

## Existing commands

```bash
# Detect shots and extract one middle keyframe per shot.
uv run python -m data_processing.cli phase1 --batch Videos_L21

# Encode completed keyframes with a registered model (default: clip).
uv run python -m data_processing.cli phase2 --batch Videos_L21 --model clip

# Build/update the FAISS index and SQLite catalog after processing features.
uv run python -m data_processing.cli phase3 --model clip

# Or run Phase 1 -> Phase 2 -> Phase 3 together:
uv run python -m data_processing.cli all --batch Videos_L21 --model clip
```

See [03_VIDEO_PROCESSING.md](03_VIDEO_PROCESSING.md) and
[04_FEATURE_EXTRACTION.md](04_FEATURE_EXTRACTION.md) for existing commands, and
[06_DATABASE.md](06_DATABASE.md) for the target database/index contracts.
