# Video processing

Phase 1 runs TransNetV2 and saves one full-resolution middle keyframe per shot.
Competition inputs stay in `data/`; generated outputs go in `data_processed/`.

## Run

From the project root, install dependencies with `uv sync`. FFmpeg/`ffprobe`
must be installed. The TransNetV2 checkpoint is bundled in
`data_processing/phase1_keyframe_extraction/transnetv2/`.

```bash
# Detect shots and extract keyframes for one video.
uv run python -m data_processing.cli phase1 data/videos/Videos_L21/L21_V001.mp4

# Select specific videos, multiple batches, or all downloaded Videos_L* batches.
uv run python -m data_processing.cli phase1 --video L21_V001 L21_V002
uv run python -m data_processing.cli phase1 --batch Videos_L21 Videos_L22 --device gpu
uv run python -m data_processing.cli phase1 --all --device gpu

# Shot detection only, with an optional visualization.
uv run python -m data_processing.cli transnetv2 --batch Videos_L21 --visualize
```

Options: `--device auto|cpu|gpu|cuda` (default `auto`), `--threshold 0.5`,
`--visualize`, `--force`, `--verify`, `--weights PATH`, and `--output-root PATH`.
Specify exactly one selector: a video/directory argument, `--video`, `--batch`, or `--all`.

## Outputs

```text
data_processed/
├── transnetv2/Videos_L21/L21_V001/
│   ├── L21_V001.predictions.txt
│   ├── L21_V001.scenes.txt
│   ├── L21_V001.vis.png         # only with --visualize
│   └── shots.json
└── keyframes/Videos_L21/L21_V001/
    ├── 00000.jpg               # phase1 only: one JPEG per shot
    └── ...
```

- `predictions.txt`: two sigmoid probabilities per decoded frame, following
  the upstream format (single-frame and many-hot).
- `scenes.txt`: inclusive `[start_frame, end_frame]` intervals, with zero-based
  frame indices. Shots use the single-frame probability `> threshold`.
- `shots.json`: one object containing video identity, processing settings,
  technical `metadata`, and a `shots` list. Each shot stores its bounds and
  playback timestamps; Phase 1 also adds its middle keyframe's index, timestamp,
  and image path relative to `data_processed/`.

The middle frame is `start + (end - start) // 2`. Playback times come from decoded
presentation timestamps, not frame index divided by FPS. Frame timestamps stay
in memory; no timestamp array, separate metadata/manifest, or keyframe CSV is saved.

Valid unchanged results are skipped based on `source_key`, `source_size`,
`checkpoint_size`, `upstream_revision`, and `pipeline_version` (no file
modification times are pinned, so copied outputs still skip). By default, skip
checks verify artifact headers (`Image.open` without decoding pixels); pass
`--verify` for a full pixel decode of every keyframe. Leftover `.work/` and
`.*.backup-*` crash artifacts are cleaned up at the start of each run, and the
summary prints `processed N, skipped N, failed N`. Use `--force` to
rebuild. Rebuilding an old result replaces its extra per-video files with this
layout; failed rebuilds keep the previous outputs. Phase 2 accepts both the new
and legacy formats.

## Docker

Run Docker commands from `data_processing/`. The Compose configuration
mounts `data/` read-only at `/app/data`, writes all generated outputs (including
the SQLite catalog) to `data_processed/` through `/app/data_processed`, and
reuses the host Hugging Face cache directory.

Create the output directory before the first run:

```bash
cd data_processing
mkdir -p ../data_processed
```

### Build the image

```bash
docker compose build
```

This builds the smaller CPU image from `uv.lock` without downloading CUDA
libraries. Build the GPU image only when CUDA processing is needed:

```bash
docker compose -f compose.yaml -f compose.gpu.yaml build
```

### Run services

Arguments after the service name (`phase1`, `phase2`, `phase3`, `all`) match
`data_processing.cli`:

```bash
# Phase 1: specific batches or all downloaded Videos_L* batches
docker compose run --rm phase1 --batch Videos_L21 Videos_L22
docker compose run --rm phase1 --all

# Phase 2 and Phase 3
docker compose run --rm phase2 --batch Videos_L21 --model clip
docker compose run --rm phase3 --model clip

# Everything in one go (Phase 1 -> Phase 2 -> Phase 3)
docker compose run --rm all --batch Videos_L21 --model clip
```

To use an NVIDIA GPU, pass the GPU override file and `--device gpu`:

```bash
docker compose -f compose.yaml -f compose.gpu.yaml run --rm phase1 --batch Videos_L21 --device gpu
docker compose -f compose.yaml -f compose.gpu.yaml run --rm phase2 --batch Videos_L21 --device gpu
docker compose -f compose.yaml -f compose.gpu.yaml run --rm all --batch Videos_L21 --model clip --device gpu
```

The GPU override requires an NVIDIA GPU, compatible host drivers, and the NVIDIA
Container Toolkit. The CPU compose file requires none of these.

### Run on a single video

```bash
docker compose run --rm phase1 --video L21_V001
docker compose run --rm phase1 /app/data/videos/Videos_L21/L21_V001.mp4 --force
```
