# Phase 2 — CLIP Feature Extraction

Phase 2 has two independent steps. Feature extraction validates completed Phase
1 keyframes and writes one NumPy matrix per video. Index construction scans all
of those matrices and builds a FAISS index. Keeping these steps separate means
the expensive GPU extraction work does not need to run again when rebuilding an
index.

## Setup

Install the project dependencies from the repository root:

```bash
uv sync
```

The default checkpoint is Hugging Face
[`openai/clip-vit-base-patch32`](https://huggingface.co/openai/clip-vit-base-patch32).
The first `CLIPEmbedding` instance downloads the model into the Hugging Face
cache; later instances reuse the cached files. An internet connection is
therefore required on first use unless the checkpoint is already cached.

## Run Phase 2

Phase 1 must finish successfully before Phase 2. Specify exactly one selector
(`--video`, `--batch`, `--all`, or a Phase 1 result path):

```bash
uv run python -m data_processing.cli phase2 --batch Videos_L21 Videos_L22
uv run python -m data_processing.cli phase2 --video L21_V001 L21_V002
uv run python -m data_processing.cli phase2 --all --model clip
```

To process one video result directory:

```bash
uv run python -m data_processing.cli phase2 \
  data_processed/transnetv2/Videos_L21/L21_V001
```

The default inference device is `auto`, which selects CUDA when available. Set
the device and inference batch size explicitly when needed:

```bash
uv run python -m data_processing.cli phase2 \
  --batch Videos_L21 \
  --device gpu \
  --batch-size 128
```

Reduce `--batch-size` if GPU memory is insufficient. `gpu` is an alias for
`cuda`; `cpu` forces CPU inference.

The command prints a summary and exits nonzero if any video fails:

```text
Processed: 4; skipped: 10; failed: 1
```

One bad video does not stop the remaining videos in the selected input.

## Output layout

The default feature output is grouped by a short model key:

```text
data_processed/features/
└── clip/
    ├── L21_V001.npy
    ├── L21_V001.json
    ├── L21_V002.npy
    └── L21_V002.json
```

- Each `.npy` is a float32 matrix with shape `(keyframe_count, 512)` for the
  default ViT-B/32 model. Every row is L2-normalized.
- The matching `.json` maps each row to its keyframe, shot, source frame,
  and timestamp. It also records model and Phase 1 provenance.

Results are built in a staging directory and published only after validation.
An existing valid result is preserved if a rebuild fails.

## Validation and reruns

Before loading CLIP, Phase 2 verifies that:

- `shots.json` identifies a Phase 1 result and its ordered shot list.
- `shots.json` contains one matching keyframe reference per shot.
- Every keyframe exists and is readable.

An unchanged valid Phase 2 result is skipped. A result is rebuilt when the
Phase 1 `shots.json` SHA-256 hash (`input_shots_sha256`), model identity,
representation version, pipeline version, or saved feature structure changes.
Use `--force` after replacing a valid keyframe while preserving `shots.json`:

```bash
uv run python -m data_processing.cli phase2 --batch Videos_L21 --force
```

If validation reports a missing or changed Phase 1 keyframe, rerun Phase 1 with
`--force` before retrying Phase 2.

## Custom paths and models

For a non-default Phase 1 layout, specify all relevant roots. `--output-root`
is the parent of the model-specific directory:

```bash
uv run python -m data_processing.cli phase2 /path/to/transnetv2 \
  --keyframe-root /path/to/keyframes \
  --output-root /path/to/features
```

The `--model` option selects a registered model from
`data_processing/phase2_feature_extraction/models.py` (`MODELS`), which also sets
the feature/index directory key. Override the default Hugging Face checkpoint
or local checkpoint directory with `--checkpoint`:

```bash
uv run python -m data_processing.cli phase2 --batch Videos_L21 \
  --model clip \
  --checkpoint openai/clip-vit-base-patch32
```

Use a different model key when storing a separately versioned representation,
for example `--model siglip2`. Do not mix incompatible checkpoints in one
directory.

Changing the checkpoint can change the vector dimension and representation.
Never compare or index embeddings produced by different checkpoints as if they
were compatible. Keep each generated JSON sidecar with its `.npy` feature file.

## Build the FAISS index

Run index construction only after feature extraction. The builder deliberately
has no video or batch filter: it always scans every valid `.npy` for the selected
model so a partial run cannot overwrite a complete index.

```bash
uv run python -m data_processing.phase2_feature_extraction.build_index \
  --model clip
```

The global output is:

```text
data_processed/indexes/
├── clip_keyframe_embedding.index
├── clip_keyframe_metadata.json
└── clip_keyframe_index.json
```

The FAISS index uses exact inner-product search. Because all CLIP features are
L2-normalized, inner product is cosine similarity. The metadata file maps every
FAISS row back to its video, keyframe, shot, timestamp, and source image. The
index manifest records source file names, feature sizes, and sidecar SHA-256
hashes for unchanged-run skipping.

Build every model directory currently containing features with:

```bash
uv run python -m data_processing.phase2_feature_extraction.build_index --model all
```

Use `--force` to rebuild an unchanged index. Custom locations are supported with
`--features-root` and `--index-root`.

### Optional playlist indexes

If `data/video_to_playlist.json` exists, the builder also creates one index per
playlist under `data_processed/indexes/playlist/`. The mapping must be a JSON
object from video name to playlist name:

```json
{
  "L21_V001": "news",
  "L21_V002": "sports"
}
```

Use `--no-playlists` to build only the global index. A missing default mapping
produces a warning and does not fail global index construction.

## Encode text and images directly

Run Python from the repository root:

```bash
uv run python
```

Then create one model instance and reuse it:

```python
from data_processing.phase2_feature_extraction import CLIPEmbedding

embedder = CLIPEmbedding()

# A single input returns a one-dimensional tensor with shape (512,).
text_vector = embedder.encode_text("a person riding a bicycle")
image_vector = embedder.encode_image(
    "data_processed/keyframes/Videos_L21/L21_V001/00000.jpg"
)

# A batch returns one row per input, with shape (batch_size, 512).
text_vectors = embedder.encode_text(["a red car", "people outdoors"])
image_vectors = embedder.encode_images_batch([
    "data_processed/keyframes/Videos_L21/L21_V001/00000.jpg",
    "data_processed/keyframes/Videos_L21/L21_V001/00001.jpg",
])
```

All returned values are `torch.float32` tensors on CPU. Each row is L2-normalized,
so text and image similarity can be calculated with a dot product:

```python
score = text_vector @ image_vector
print(float(score))
```

The similarity is useful for ranking images. It is not a calibrated probability
that an object or event is present.

## Accepted image inputs

`encode_image` and `encode_images_batch` accept:

- A string or `pathlib.Path` image path.
- A `PIL.Image.Image`.
- A two- or three-dimensional NumPy array supported by Pillow.

Images are converted to RGB and processed using the preprocessing configuration
provided with the selected CLIP checkpoint.

## Device selection in Python

The default `device="auto"` uses CUDA when PyTorch reports that CUDA is available,
and otherwise uses the CPU:

```python
cpu_embedder = CLIPEmbedding(device="cpu")
gpu_embedder = CLIPEmbedding(device="cuda")
```

Requesting CUDA on a machine where it is unavailable raises `ValueError`.

## Select another compatible checkpoint in Python

Pass a Hugging Face CLIP checkpoint name or local checkpoint directory:

```python
embedder = CLIPEmbedding(model_name="openai/clip-vit-base-patch32")
```

Changing the checkpoint can change the vector dimension and representation.
Never compare or index embeddings produced by different checkpoints as if they
were compatible. Record the exact model name with persisted vectors.

## Run the embedding tests

The focused tests use lightweight fake model objects and do not download the
real checkpoint:

```bash
uv run pytest -q tests/test_phase2_embeddings.py tests/test_phase2_pipeline.py \
  tests/test_phase2_index.py
```

## Current boundary

The filesystem pipeline now produces per-video feature matrices and global or
playlist FAISS indexes. A query service that encodes text and searches these
indexes is not included yet; it can use `CLIPEmbedding.encode_text` and the
generated metadata without rerunning image extraction.
