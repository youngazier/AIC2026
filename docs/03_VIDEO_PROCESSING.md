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

# Process a batch. Omit --batch to process all downloaded videos.
uv run python -m data_processing.cli phase1 --batch Videos_L21 --device gpu

# Shot detection only, with an optional visualization.
uv run python -m data_processing.cli transnetv2 --batch Videos_L21 --visualize
```

Options: `--device auto|cpu|gpu|cuda` (default `auto`), `--threshold 0.5`,
`--visualize`, `--force`, `--weights PATH`, and `--output-root PATH`.
Use either a video/directory argument or `--batch`.

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
modification times are pinned, so copied outputs still skip). Use `--force` to
rebuild. Rebuilding an old result replaces its extra per-video files with this
layout; failed rebuilds keep the previous outputs. Phase 2 accepts both the new
and legacy formats.

## Docker

Run Docker commands from the Phase 1 directory. The Compose configuration
mounts `data/` read-only at `/app/data` and writes results to
`data_processed/` through `/app/data_processed`.

Create the output directory before the first run:

```bash
cd data_processing/phase1_keyframe_extraction
mkdir -p ../../data_processed
```

### Build the image

```bash
docker compose build phase1
```

This builds the smaller CPU image and does not download CUDA libraries. Build
the GPU image only when CUDA processing is needed:

```bash
docker compose --profile gpu build phase1-gpu
```

### Run on a directory of videos

The input path is the path inside the container. For example, this processes
every supported video under the host directory `data/videos/Videos_L21`:

```bash
docker compose run --rm phase1 \
  phase1 /app/data/videos/Videos_L21
```

To process all downloaded videos, omit the input path:

```bash
docker compose run --rm phase1 phase1
```

To use an NVIDIA GPU, run the GPU service instead:

```bash
docker compose --profile gpu run --rm phase1-gpu \
  phase1 /app/data/videos/Videos_L21 --device gpu
```

The GPU command requires an NVIDIA GPU, compatible host drivers, and the
NVIDIA Container Toolkit. The CPU service requires none of these.

### Run on a single video

```bash
docker compose run --rm phase1 \
  phase1 /app/data/videos/Videos_L21/L21_V001.mp4
```

For GPU processing, replace `phase1` after `--rm` with `phase1-gpu`, enable
the `gpu` profile, and add `--device gpu`:

```bash
docker compose --profile gpu run --rm phase1-gpu \
  phase1 /app/data/videos/Videos_L21/L21_V001.mp4 --device gpu
```

Pass other CLI options after the input path, for example `--visualize` or
`--force`. Replace the first `phase1` after the service name with `transnetv2`
to run shot detection without keyframe extraction.
