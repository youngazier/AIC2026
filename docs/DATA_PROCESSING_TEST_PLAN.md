# Data Processing Test Plan

Tests for `docs/DATA_PROCESSING_IMPROVE.md`. Item numbers refer to that plan.
The goal is to prove that each change works, and that the four things the team
relies on still hold:

1. **Skip is safe.** A finished video is never reprocessed, and a damaged one is always caught.
2. **Merging works.** Outputs copied from other machines are accepted as they are.
3. **Selection is exact.** A command processes the videos it was asked to, and nothing else.
4. **Outputs agree.** Keyframes, features, index and catalog describe the same videos.

## Levels

| Level | What it uses | Runs | Marker |
|---|---|---|---|
| **Unit** | Stub detector, stub embedder, temp folders, tiny fake JPEGs. No models, no GPU, no Docker. | Every change, in under 2 minutes in total. | none |
| **Integration** | The 6-second tiny video (fixture below), still with stub models. Covers the full `all` path. | Every change. | none |
| **Slow** | Real TransNetV2 and real CLIP on CPU, on the tiny video. | Before each merge. | `slow` |
| **Acceptance** | The real `Videos_L21` batch (19 videos), real models, the GPU. | Once per milestone. Manual script below. | `gpu` or by hand |
| **Docker** | The compose services. | After item 7, and after each Dockerfile change. | by hand |

The 51 existing tests in `tests/` must keep passing at every step. When item 4
bumps the pipeline versions, update the expected values in those tests in the
same commit. Don't delete a test to make it pass.

Run commands:

```bash
uv run pytest                      # unit + integration (default, fast)
uv run pytest -m slow              # real models on the tiny video
uv run pytest -m gpu               # needs CUDA
uv run pytest --durations=20       # find slow tests that need a marker
```

## Shared fixtures

Put these in `tests/conftest.py`.

- **`tiny_video`** (session scope): a 6-second 320×180 MP4 made with `ffmpeg -f lavfi`. It has three 2-second clips with hard cuts: `testsrc`, then `color=red`, then `smptebars`. It is written to `tmp_path_factory` and skipped with a clear message if `ffmpeg` is missing.
- **`data_tree`**: a temporary project root with `data/videos/Videos_L21/L21_V001.mp4`, `L21_V002.mp4` and `Videos_L22/L22_V001.mp4` (copies or symlinks of `tiny_video`), plus an empty `data_processed/`.
- **`stub_detector`**: returns the known cuts (frames 0–59, 60–119, 120–179 at 30 fps), with no weights loaded.
- **`stub_embedder`**: returns a fixed 8-dimension unit vector derived from the keyframe's pixel mean, so different colours give different vectors and the same image always gives the same one.
- **`processed_tree`**: runs Phase 1 and Phase 2 with the stubs over `data_tree`. Tests that start from finished output use this instead of repeating setup.

## Tests by plan item

### Item 1: Selectors

| Test | Expect |
|---|---|
| `--video L21_V001 L21_V002` | Exactly those two are processed. |
| `--video` with a path to an `.mp4` | The same as the name. |
| `--video` with an unknown name | Clear error naming it, exit code nonzero, nothing processed. |
| `--batch Videos_L21 Videos_L22` | All three videos. |
| `--batch Videos_L99` (doesn't exist) | Clear error, nothing processed. |
| `--batch videos_l21` or `L21` | Rejected by the `Videos_L<number>` check. |
| `--all` | Every video in every `Videos_L*` folder. Zip files and other folders are ignored. |
| No selector | Error, nothing processed. |
| Two selector kinds at once | Error. |
| Phase 2 on a video with no Phase 1 output | Reported by name as "no Phase 1 output", counted as failed. The others are still processed. |
| Old form `phase1 data/videos/Videos_L21/L21_V001.mp4` | Still works. |

### Item 2: Model registry

- `--model clip` resolves to `CLIPEmbedding`, and an unknown model gives an error that lists the valid names.
- With a second stub model registered, features go to `features/<model>/` and are separate from `clip`.
- `--checkpoint` overrides only the weights name.
- Building an index never mixes two models (the existing refusal still fires).

### Item 3: Phase 3 (index and catalog)

Index:

| Test | Expect |
|---|---|
| Build from `processed_tree` | `global.index`, `global_rows.json` and `manifest.json` exist, `ntotal` equals the total keyframe count, and row `i` points to the right video, timestamp and image. |
| Playlist indexes | One per playlist in the mapping, and their rows are a subset of the global rows. |
| Run again with no change | Skips (check the log or the file mtimes). |
| Truncate `global.index` | Rebuilt, with the reason logged. |
| Edit `global_rows.json` to the wrong length | Rebuilt. |
| Delete `manifest.json` | Rebuilt. |
| Add one feature file | Rebuilt, and `ntotal` grows by that video's keyframe count. |
| Leave a temporary file from a "crashed" build | Deleted at the start of the next run. |
| Simulate a crash after the index is written but before the manifest (monkeypatch `os.replace` to raise on the manifest) | The next run rebuilds and ends in a valid state. |
| Video with Phase 1 output but no features | Warning that names it. The index is still built from the rest. |
| Features from two precisions or two dimensions | Refused with a clear error. |

Catalog:

| Test | Expect |
|---|---|
| `phase3` on `processed_tree` | One `videos` row per video, and shot and keyframe counts equal those in each `shots.json`. |
| Run `phase3` twice | Same row counts, no duplicates. |
| Corrupt one `shots.json` | That video is reported and the exit code is nonzero. The other videos are imported. |
| `--no-index` and `--no-catalog` | Only the other part runs. |
| Run on an empty database file | The tables are created by `alembic upgrade head`. |
| A catalog row's image path | The file exists on disk. |

### Item 4: Skip checks that survive copying

This is the most important group. Do it carefully.

| Test | Expect |
|---|---|
| `cp -r data_processed other/`, then run each phase on `other/` | Every video is skipped in Phase 1, Phase 2 and Phase 3. |
| `touch` the source `.mp4` (mtime only) | Phase 1 still skips. |
| Replace the source `.mp4` with different content of a different size | Phase 1 reprocesses it. |
| `touch` the weights file | Phase 1 still skips. Changing the weights contents does **not** skip (checked through the weights size and `upstream_revision`). |
| `touch shots.json` | Phase 2 still skips. |
| Change one byte in `shots.json` | Phase 2 reprocesses that video. |
| Copy features with `cp` and rebuild the index | The index builds and the manifest sources are valid. |
| An output written with the old pipeline version | Reprocessed. |

### Item 5: Cheap skip checks

| Test | Expect |
|---|---|
| Phase 2 skip path | No keyframe image is opened. Check by monkeypatching `PIL.Image.open` to fail the test if it's called. |
| Delete one keyframe | The quick check fails and Phase 1 reprocesses the video. |
| Truncate one keyframe file | Rejected by the quick check (the header won't open) or, if only the pixels are damaged, accepted by quick and **rejected by `--verify`**. |
| `--verify` on good output | Skips. |
| Keyframe count different from the shot count | Reprocessed. |
| End-of-run line | `processed N, skipped N, failed N` matches reality. |

### Item 6: Phase 2 throughput

| Test | Expect |
|---|---|
| `--workers 0` against `--workers 2` | Identical `.npy` bytes and identical sidecars (apart from started and finished times). |
| Videos with 1, 63, 65 and 130 keyframes at `--batch-size 64` | Each has the right row count and the right vectors. |
| `--batch-size 1` against `64` | Same vectors within float tolerance. |
| One bad JPEG | That video fails and is reported. The others finish. |
| Output with the DataLoader against the old path (use a saved reference file from `Videos_L21`) | Identical for fp32. |
| fp16 (`gpu` marker) | Per-row cosine similarity with fp32 is at least 0.999 on 200 sample keyframes. Sidecar records `precision`. |
| `--device gpu` with no CUDA | Clear error, not a crash. |

### Item 7: Docker

Done by hand, with the checklist below. `docker compose config` is also run
as a test, so a broken YAML file is caught in the default run.

- [ ] `docker compose build` (CPU) works.
- [ ] `docker compose run --rm phase1 --batch Videos_L21` processes, and a second run skips all.
- [ ] `phase2`, `phase3` and `all` each run the same way as the CLI.
- [ ] Files written to `data_processed/` belong to your user, not root.
- [ ] `data/` is read-only inside the container (a write to it fails).
- [ ] The Hugging Face cache is reused, so a second run doesn't download weights.
- [ ] With `-f compose.yaml -f compose.gpu.yaml`: `torch.cuda.is_available()` is `True` and the device name is printed.
- [ ] `phase3` creates and fills the catalog inside the container.
- [ ] A `.dockerignore` test: `data/`, `data_processed/`, `test/` and `.git` are not in the build context, and the TransNetV2 weights, `backend/`, `alembic.ini` and the migrations are.

### Item 8: Merge

An automated test of the real workflow:

1. Build two separate trees, `member_a/` with `Videos_L21` and `member_b/` with `Videos_L22`, each processed with the stubs.
2. Archive each member's batch folders and features, and extract both into `merged/`.
3. Run `phase3` on `merged/`.
4. Check that the index holds both batches, the catalog holds all videos, and a rerun of Phase 1 and Phase 2 on `merged/` skips everything.
5. Run the same check using plain `cp -r` (mtimes reset) instead of `tar`.
6. Remove one member's features and run `phase3`: it warns about the missing videos.

### Item 9: Housekeeping

- After removing the dependencies: `uv lock --check` passes and `uv sync` builds with no warning.
- Leftover cleanup: plant a `.X.backup-token` file, a `.X.backup-token` folder and a `.work/` folder. The next run removes all three and still skips finished videos.
- Marker setup: `uv run pytest` runs none of the `slow` or `gpu` tests, and `uv run pytest -m slow` runs only those.

## End-to-end tests (`all`)

- **Fast** (default): `all --batch Videos_L21 Videos_L22` on `data_tree` with the stubs. Expect 3 videos, shots, keyframes, features, an index with `ntotal == 9`, and 3 catalog videos. The exit code is 0. A second run skips every video.
- **Failure isolation**: make one video unreadable (random bytes with an `.mp4` name). Expect that video to fail in Phase 1, the others to complete, the catalog and index to hold the others, and a nonzero exit code. Fix the file and rerun: only that video is processed.
- **`--no-phase3`**: no index and no catalog are created.
- **Slow**: the same on the tiny video with real TransNetV2 and CLIP. Expect 3 shots, 3 keyframes, 3 vectors, an index with `ntotal == 3`, and a text query "red" whose top result is the red shot.

## Acceptance run on `Videos_L21`

Do this by hand when the plan's items are done. Record the numbers in
`docs/04_FEATURE_EXTRACTION.md`.

```bash
# 0. Keep the old output to compare against
cp -a data_processed /tmp/data_processed.before     # use the scratch area, not the repo

# 1. Clean run
rm -rf data_processed
uv run python -m data_processing.cli all --batch Videos_L21 --device gpu

# 2. Second run must skip everything and be fast
time uv run python -m data_processing.cli all --batch Videos_L21 --device gpu

# 3. Deep check
uv run python -m data_processing.cli phase1 --batch Videos_L21 --verify
uv run python -m data_processing.cli phase2 --batch Videos_L21 --verify

# 4. Kill test: start `all`, press Ctrl+C or `kill -9` partway, rerun, and compare final outputs
```

Pass when:

- [ ] Step 2 reports `processed 0` for every phase and finishes in under 1 minute.
- [ ] Step 3 reports no failures.
- [ ] The number of videos is 19 in `data_processed/transnetv2/Videos_L21`, `keyframes/Videos_L21` and `features/clip`.
- [ ] For each video: the shot count in `shots.json`, the keyframe JPEG count, the `.npy` row count and the catalog's keyframe count are equal.
- [ ] The index `ntotal` equals the sum of all `.npy` rows, and `faiss` search with a stored vector returns that same vector's row as the top hit (score about 1.0).
- [ ] A random sample of 10 index rows: the `image_path` exists and its `timestamp` matches the shot's keyframe in `shots.json`.
- [ ] Catalog and index agree: every `(video_name, keyframe)` in the index rows exists in the catalog.
- [ ] After a forced kill in step 4 and a rerun, the outputs match an uninterrupted run (same row counts and the same index `ntotal`).
- [ ] Copy `data_processed/` elsewhere with `cp -r`, run `all` there, and everything is skipped.
- [ ] Compare against the "before" copy: shot boundaries and keyframe timestamps are unchanged. The `.npy` vectors match to within float tolerance (item 4 changes only sidecar fields, not the numbers).
- [ ] Phase 2 images per second before and after item 6, and the chosen `--batch-size`, are written in `docs/04`.

## What a reviewer checks in each PR

- The tests named for that item above exist and fail without the change (try reverting the change and seeing them fail).
- No test is skipped, deleted or loosened without a reason in the commit message.
- `uv run pytest` passes with the default markers, and the run time stays under 2 minutes.
- Any change that alters a file format also bumps the pipeline version and updates the docs (`docs/03`, `docs/04`, `docs/TODOS.md`).
