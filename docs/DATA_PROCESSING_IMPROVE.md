# Data Processing Improvement Plan

Scope: `data_processing/` (Phase 1, Phase 2, Phase 3 index build and SQLite
catalog import) and the Docker setup that runs them. Keep it simple. Every item below must earn its place by
serving the workflow in the next section.

The full test plan is in `docs/DATA_PROCESSING_TEST_PLAN.md`. Each item's
"Tests" lines below are the short version.

When an item lands, tick it in the checklist at the end and update
`docs/TODOS.md`, `docs/03_VIDEO_PROCESSING.md` and `docs/04_FEATURE_EXTRACTION.md`.

## Workflow

1. A person downloads videos into `data/videos/Videos_L*/`.
2. **Phase 1**: TransNetV2 finds shots and writes one keyframe per shot.
3. **Phase 2**: an embedding model (`--model clip`, and later 2 or 3 models) turns each keyframe into a vector. It writes one `.npy` file per video.
4. **Phase 3**: build the FAISS index and row metadata from all the `.npy` files, and import the videos, shots and keyframes from Phase 1 into the SQLite catalog (`data_processed/database/catalog.sqlite3`).

Rules that follow from how the team works:

- Each phase runs on its own, from the CLI or from Docker.
- Each phase can take one video, a list of videos, one batch, a list of batches, or everything.
- A video that is already processed is skipped, so rerunning any command is always safe.
- Up to 5 members process **different batches on their own machines**, in parallel. Their outputs are then copied into one `data_processed/` folder.
- **Phase 3 runs once, after all outputs are merged.** It is the only step that needs every member's data, so it is its own phase.
- Processing is offline. Nobody reads `data_processed/` while a phase runs. If a run crashes, rerun it.
- Platform is Linux (host or Docker). No Windows support.

## What exists today, and what is missing

Checked against the code on 2026-11-24:

| Today | Problem | Fix (item) |
|---|---|---|
| `phase1` and `phase2` accept one path or one `--batch`. | Can't pass a list of videos, a list of batches, or "all". No input means "everything", which is easy to start by accident. | Item 1: shared selectors `--video`, `--batch`, `--all`. |
| Phase 2 has `--model clip` but the model class is hard-coded. | Adding a second model means editing the CLI. | Item 2: a small model registry. |
| Index building is a separate module with its own flags and no `phase3` command. The SQLite catalog import is another separate command (`backend.app.catalog.cli import`) that nothing in the pipeline calls. | Neither is part of the CLI or Docker, so the database step is easy to forget. | Item 3: one `phase3` command that does both. |
| The Docker image copies only `data_processing/`. `.dockerignore` excludes `backend/`, `alembic.ini` and the migrations. | The catalog import can't run in the container. | Item 7: include them in the image. |
| Skip checks pin file **mtimes**: the source video's mtime and the weights file's mtime (Phase 1), and `shots.json`'s mtime (Phase 2). | Copying outputs between members' machines changes mtimes, so a rerun on the merged folder reprocesses videos that are already done. | Item 4: use sizes and content hashes. |
| The index skip check only compares manifest sources. It never opens the `.index` file. | A truncated index is kept as "unchanged". | Item 3: validate before skipping. |
| The index, row metadata and manifest are replaced in three separate steps, with the manifest not last. | A crash can leave a mismatched set that the next run can't detect. | Item 3: write the manifest last. |
| Phase 1 skip fully decodes every keyframe. Phase 2 verifies every keyframe image before it decides to skip. | "Run everything again to check" reads all JPEGs for ~870 videos. | Item 5: cheap skip check, `--verify` for the deep one. |
| Phase 2 runs decode, resize and encode in one thread, in fp32, batch 64. | The GPU waits on Pillow. | Item 6: DataLoader workers and a measured batch size. |
| Compose has 5 services (`phase1`, `phase2`, `index`, `phase1-gpu`, `phase2-gpu`). The Dockerfile pins package versions by hand. The old Phase 1 compose file is still there. | Duplicate setup, and versions can drift from `uv.lock`. | Item 7: one compose file with `phase1`, `phase2`, `phase3`, `all`, and a GPU override file. |
| Index metadata rows already hold `video_name`, `batch`, `timestamp`, `frame_index` and `image_path`. | Nothing to fix. The index is usable without joining the catalog. | Keep. |
| `pgvector` and `psycopg[binary]` are unused. The `uv_build<0.13` pin doesn't match uv 0.13.7. | Install weight and a build warning. | Item 9. |

## Command design

All commands run as `python -m data_processing.cli <command>` (or the matching
Docker service).

### Selecting what to process (Phase 1 and Phase 2)

Exactly one selector is required. There is no implicit "everything".

| Flag | Meaning |
|---|---|
| `--video L21_V001 L21_V002` | One or more videos by name. A path to an `.mp4` is also accepted. |
| `--batch Videos_L21 Videos_L22` | One or more batches. |
| `--all` | Every `Videos_L*` batch under `data/videos/`. |

Phase 1 reads the selected videos from `data/videos/`. Phase 2 selects from the
videos that have **completed Phase 1 output** and reports any selected video
that doesn't have it.

### Common options

| Flag | Applies to | Meaning |
|---|---|---|
| `--device auto\|cpu\|gpu` | 1, 2 | `auto` picks the GPU when it's available. |
| `--force` | 1, 2, 3 | Reprocess even when the output is valid. |
| `--verify` | 1, 2 | Deep skip check (decode every keyframe). Default is the quick check. |
| `--model clip` | 2, 3 | Which embedding model. Features and indexes are kept per model. |
| `--batch-size N` | 2 | Images per inference batch. Default is the measured best for the 4 GB GPU. |
| `--workers N` | 2 | Processes that decode and resize JPEGs while the model encodes. Default `min(4, cpu_count - 1)` on GPU and `0` on CPU. `0` loads in the main process. |
| `--no-phase3` | all | Skip the index and catalog (for members who process only their own batches). |
| `--no-index`, `--no-catalog` | 3 | Run only one part of Phase 3. |

"Number of batch for multi processing" is read as `--batch-size` and
`--workers` above. Phase 1 stays one video at a time, because the 4 GB GPU can't
share TransNetV2 between processes and members already parallelise across
machines. Say if you meant something else.

### Phase 3

```bash
python -m data_processing.cli phase3 --model clip [--no-playlists] [--force]
```

Phase 3 has two parts, and both need the merged data:

1. **Index**: builds the global index and the per-playlist indexes from every `features/<model>/*.npy` that exists. Before building, it prints how many videos each batch contributes. It warns about videos that have Phase 1 output but no features, so a missing member's batch can't slip by unnoticed.
2. **Catalog**: runs `alembic upgrade head`, then imports every video under `data_processed/transnetv2/` into `data_processed/database/catalog.sqlite3`. This is the existing importer (`backend/app/catalog/importer.py`), which writes one row per video, shot and keyframe from the validated `shots.json` files. Re-importing a video replaces its rows, so rerunning is safe. It doesn't depend on the model, so it runs once, not once per `--model`.

`--no-catalog` and `--no-index` skip one part. The catalog is derived from
Phase 1 only, so it could run as soon as Phase 1 is merged. It lives in Phase 3
because that is where the merged data is first complete, and the backend is
then ready to use as soon as Phase 3 finishes. The `data_processing` package
imports the backend code only inside this step.

### Everything in one go

```bash
python -m data_processing.cli all --batch Videos_L21 Videos_L22 --model clip --device gpu [--no-phase3]
```

Runs Phase 1, then Phase 2, then Phase 3 (index and catalog) for the selected
input. A member who only wants their own batches processed adds `--no-phase3`,
because the index and catalog should be built once on the merged data. The command stops at
the first phase with a stage-level error. A video that fails in one phase is
reported and doesn't stop the others. The exit code is nonzero if anything
failed, so the answer is always "fix it and rerun the same command".

---

## Items

### 1. Selectors

Add `--video`, `--batch` (multiple values) and `--all` to `phase1` and `phase2`
in one shared helper that returns a list of videos. Reject zero or several
selector kinds. `--batch` keeps the `Videos_L<number>` check. Existing
single-path usage (`phase1 data/videos/Videos_L21/L21_V001.mp4`) keeps working.

Tests: each selector against a temporary `data/videos` tree, including an
unknown video name and a batch that doesn't exist.

### 2. Model registry

A dict in `phase2_feature_extraction/models.py`:

```python
MODELS = {"clip": (CLIPEmbedding, "openai/clip-vit-base-patch32")}
```

`--model` picks from it, and `--checkpoint` still overrides the weights name.
Adding a model means one `BaseEmbedding` subclass and one line here. Features
go to `features/<model>/`, indexes to `indexes/<model>/`, so models never
mix. `build_index` keeps its existing refusal to mix models or dimensions.

### 3. Phase 3: SQLite catalog and a safe index build

**Catalog part** (reuses the existing backend code, no schema change):

- `phase3` runs `alembic upgrade head` (creates or updates the tables, and is a no-op when current), then calls the catalog importer on `data_processed/transnetv2`. The importer already does the rest:
  - It validates each video's `shots.json` against the Phase 1 artifacts **before** writing.
  - It writes one video per short transaction, into the `videos`, `shots` and `keyframes` tables.
  - It ignores dot-prefixed staging folders.
  - Re-importing a video deletes and rewrites that video's shot and keyframe rows and updates its `videos` row in place, so rerunning is safe. It does not skip unchanged videos. That is acceptable at a few hundred rows per video, but the plan checks the time for the full corpus once.
- The importer takes the Phase 1 lock for each video, so don't run it while Phase 1 is running on the same folder.
- Failed or invalid videos are listed in the output and make the exit code nonzero. The other videos are still imported.
- Both parts run even if the first reports failures, unless a `--no-*` flag is given, and the exit code reports both.
- The database is derived data. If it's damaged or the schema changes, delete `catalog.sqlite3` and rerun `phase3 --no-index`. Nothing else is needed.

**Index part**:

- `phase3` calls the existing index-building code, and the standalone module keeps working.
- New file layout, one folder per model:

  ```
  indexes/clip/
    global.index
    global_rows.json         # row i -> video_name, batch, timestamp, frame_index, image_path
    manifest.json
    playlist/<playlist>.index
    playlist/<playlist>_rows.json
  ```

  The old flat `clip_keyframe_*` files are left in place and deleted by hand once M4 reads the new layout.
- Write each file to a temporary name, validate it, then `os.replace` it. Write `manifest.json` **last**, so a crash leaves a missing or old manifest and the next run rebuilds.
- Skip only when the manifest's sources match the current features **and** `faiss.read_index` succeeds **and** `ntotal`, `d` and the rows file length match the manifest. Otherwise log the reason and rebuild.
- At startup, delete temporary files left by a crashed build.

This closes the M4 TODO "Validate existing FAISS and row metadata files before
an unchanged-run skip".

Tests: a truncated `.index`, a rows file of the wrong length and a missing
manifest are each rebuilt. An unchanged run skips. A missing-features warning
appears. For the catalog, a second `phase3` leaves the same row counts, and a
video with a corrupt `shots.json` is reported while the others are still
imported.

### 4. Make skip checks survive copying

Do this **before** the corpus run. Only `Videos_L21` (19 videos) is processed
today, so redoing it costs minutes and nothing else needs migrating.

- Phase 1 sidecar: drop `source_mtime_ns` and `checkpoint_mtime_ns`. Keep `source_size` and `source_key`. Add the weights file's size, and keep the existing `upstream_revision`. A different clone then doesn't invalidate results.
- Phase 2 sidecar: replace `input_shots_mtime_ns` with `input_shots_sha256` (the hash of `shots.json`, which is a small file).
- Index sources: use feature file size plus sidecar sha256 instead of mtimes.
- Bump `PIPELINE_VERSION` for Phase 1 and Phase 2 once, in this change, and reprocess `Videos_L21`.

After this, merging with plain `cp -r` or `rsync` is safe and rerunning on the
merged folder skips everything that is done.

Tests: copy a finished output tree with `cp -r` (which resets mtimes) and
check that every phase skips.

### 5. Cheap skip checks

- **Phase 2**: read `shots.json` first. If the sidecar matches and not `--force`, skip. Only when processing, validate the keyframe images.
- **Phase 1 quick check** (default): `shots.json` and the manifest validate, the keyframe count equals the shot count, and every keyframe file exists and opens with `Image.open` (header only, no pixel decode).
- `--verify` does today's full decode.
- Print the counts at the end of every run, for example `processed 3, skipped 16, failed 0`, so "did it skip?" is visible.

Tests: a deleted or truncated keyframe fails the quick check. `--verify` catches
pixel corruption that the quick check accepts.

### 6. Phase 2 throughput

- Load and preprocess images with a `DataLoader` (`--workers`, `pin_memory` on CUDA). Keep the `CLIPProcessor` transform so fp32 output is unchanged.
- A bad JPEG fails only its own video.
- `--batch-size` default: run `Videos_L21` at 64, 128, 256 and 512 on the GTX 1650 Ti, and pick the largest that stays under about 75% of VRAM. Record the numbers in `docs/04`.
- `--precision fp32|fp16`, recorded in the sidecar. Make fp16 the default only if embeddings of 200 sample keyframes keep a cosine similarity of at least 0.999 against fp32. Normalise in fp32 and save float32.
- Log one line per video with its seconds, and a total with images per second at the end. That is the only timing log. Also log `torch.cuda.get_device_name()` once on GPU runs. Check the container with:

  ```bash
  docker compose -f compose.yaml -f compose.gpu.yaml run --rm --entrypoint python phase2 \
    -c "import torch; print(torch.cuda.is_available(), torch.cuda.get_device_name())"
  ```

Tests with a stub embedder: `--workers 0` and `--workers 2` give identical
vectors, and one bad JPEG fails only its own video.

### 7. Docker

Replace the five services with one compose file, `data_processing/compose.yaml`:

| Service | Runs |
|---|---|
| `phase1` | `cli phase1 ...` |
| `phase2` | `cli phase2 ...` |
| `phase3` | `cli phase3 ...` |
| `all` | `cli all ...` |

```bash
cd data_processing
docker compose run --rm phase1 --batch Videos_L21 Videos_L22
docker compose run --rm phase2 --batch Videos_L21 --model clip
docker compose run --rm phase3 --model clip
docker compose run --rm all --batch Videos_L21 --model clip

# GPU: add the override file
docker compose -f compose.yaml -f compose.gpu.yaml run --rm phase2 --batch Videos_L21 --device gpu
```

`compose.gpu.yaml` only sets `gpus: all` and the CUDA wheel build argument.

- Mounts stay: `data/` read-only, `data_processed/` read-write (the SQLite file lives there too), and the Hugging Face cache.
- Add `backend/`, `alembic.ini` and the migrations to the image (`COPY` them, and allow them in `.dockerignore`), so `phase3` can create and fill the catalog in the container. Install the backend's dependencies (`sqlalchemy`, `alembic`) from `uv.lock` as well.
- Install Python packages in the Dockerfile from `uv.lock` with `uv export --frozen --no-dev --no-hashes`, keeping the torch install as its own step so the CPU and CUDA wheels still switch by build argument.
- Delete `phase1_keyframe_extraction/{Dockerfile,compose.yaml}`. Update the Docker line in `CLAUDE.md` and in `docs/03`.

### 8. Merging members' outputs

Document this in `docs/03`, with no extra code:

```bash
# on each member's machine, after their batches finish
tar -C data_processed -czf Videos_L22.tgz \
  transnetv2/Videos_L22 keyframes/Videos_L22 features/clip   # features: only their videos
# on the merging machine
tar -C data_processed -xzf Videos_L22.tgz
```

- Each member gets distinct batches, so no two members write the same file.
- Features are one file per video, so a merge is just adding files.
- After merging: `cli phase3 --model clip`, which builds the index and imports the catalog.
- Each member should `phase2` only their own batches, and archive only their own feature files (the member's `features/clip/` holds only what they made, unless they copied others in).

### 9. Housekeeping

- Remove `pgvector` and `psycopg[binary]`, run `uv lock`, and run the tests.
- Set `uv_build>=0.13,<0.14`.
- Register `slow` and `gpu` markers, excluded by default:

  ```toml
  [tool.pytest.ini_options]
  markers = ["slow: real-model tests", "gpu: needs CUDA"]
  addopts = "-m 'not slow and not gpu'"
  ```

- Mark tests that load real CLIP or TransNetV2 weights as `slow`.
- Delete leftover `.*.backup-*` entries and `.work/` folders at the start of Phase 1 and Phase 2 (about 10 lines each, under the existing lock). They are crash leftovers that only waste disk.
- One tiny-video end-to-end test: a 6-second 320×180 MP4 built with `ffmpeg -f lavfi` (three 2-second clips with hard cuts: `testsrc`, `color=red`, `smptebars`). The fast version runs `all` with a stub detector and embedder. The `slow` version uses the real models and expects 3 shots, 3 vectors, an index with `ntotal == 3`, and "red" as the top result.

## Later (only when needed)

These were in the earlier plan. They don't serve the workflow above, so they wait.

- WebP thumbnails, built from keyframes when the frontend (M5) needs them.
- Faststart copy of videos for browser playback. `L21_V001.mp4` has its index at the end, so playback will need it, but this is an M5 concern (about +90 GB).
- Phase 1 seek-based second decode, only if the per-video timing shows extraction dominates.
- Batch-qualified feature paths, only if non-competition videos are added.
- Watch-folder worker, crash-during-publish test matrix, Windows locking, versioned index builds.

## Order of work

Each step is one change with the tests run after it.

| # | Items | Why here |
|---|---|---|
| 1 | 9 (dependencies, markers) | Quick, and speeds up later test runs. |
| 2 | 4 mtime-free skip checks | Cheap now (19 videos), expensive after the full run. |
| 3 | 1 selectors, 2 model registry | The CLI shape everything else builds on. |
| 4 | 3 phase3 and safe index build | The only real correctness gap. |
| 5 | 5 cheap skip checks, 9 leftover cleanup | Makes reruns on all data fast. |
| 6 | 7 Docker, plus `all` command | Ready for members to use. |
| 7 | 6 Phase 2 throughput | Measure, set defaults, then tell members to start. |
| 8 | 8 merge doc, 9 end-to-end test | Final check of the whole path. |

Do 6 before the members start the big run, because Phase 2 speed and defaults
apply to everyone.

## Done when

- `cli all --batch Videos_L21 --device gpu` on an empty `data_processed/` leaves keyframes, features and an index that agree. A second run skips everything.
- Copying `data_processed/` with `cp -r` and rerunning any phase skips everything.
- A truncated `.index` is rebuilt by `phase3`.
- After `phase3`, the catalog has one `videos` row per processed video, and the shot and keyframe counts equal those in the `shots.json` files. A second `phase3` leaves the same counts.
- `phase3` lists any video that has Phase 1 output but no features.
- Every phase runs as a Docker service on CPU and with the GPU override.
- Phase 2 images per second, before and after item 6, are in `docs/04`.
- `uv run pytest` finishes in under 2 minutes, and `uv run pytest -m slow` passes.

## Checklist

- [x] 1. Selectors `--video`, `--batch`, `--all` for Phase 1 and 2
- [x] 2. Model registry
- [x] 3. `phase3` command (index and catalog import), safe index build, new `indexes/<model>/` layout
- [ ] 4. Skip checks without mtimes, `Videos_L21` reprocessed
  - [x] Skip checks without mtimes (Phase 1, Phase 2, and index sources)
  - [ ] `Videos_L21` reprocessed
- [x] 5. Quick skip check with `--verify`, Phase 2 skip before image checks
- [ ] 6. Phase 2 DataLoader, batch size, fp16 option, timing line
- [x] 7. One compose file, GPU override, `all` command, old Docker files removed
- [ ] 8. Merge instructions in `docs/03`
- [ ] 9. Housekeeping
  - [x] Dependencies (`pgvector`, `psycopg[binary]`, `uv_build` pin) and `slow`/`gpu` pytest markers
  - [x] Leftover cleanup (`.*.backup-*` and `.work/`)
  - [ ] Tiny-video E2E test
