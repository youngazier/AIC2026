# syntax=docker/dockerfile:1
#
# One image for all offline processing: Phase 1 (TransNetV2 + keyframes),
# Phase 2 (CLIP features) and the FAISS index builder.
# Build it through data_processing/compose.yaml (see docs/03_VIDEO_PROCESSING.md).

FROM python:3.14-slim

# CPU wheels by default; compose passes a CUDA index for the GPU image.
ARG TORCH_INDEX_URL=https://download.pytorch.org/whl/cpu

ENV PYTHONUNBUFFERED=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1 \
    PIP_PREFER_BINARY=1 \
    # Hugging Face downloads CLIP here; compose mounts a host folder on it
    # so the model is downloaded once, not on every `docker compose run`.
    HF_HOME=/hf-cache

WORKDIR /app

# ffprobe/ffmpeg for Phase 1 video probing.
RUN apt-get update \
    && apt-get install -y --no-install-recommends ffmpeg \
    && rm -rf /var/lib/apt/lists/*

# Layers are ordered from least to most frequently changed, so editing
# Python code only rebuilds the final COPY layer.
RUN --mount=type=cache,target=/root/.cache/pip \
    pip install --index-url "${TORCH_INDEX_URL}" torch==2.14.0

# Versions match uv.lock; update both together.
RUN --mount=type=cache,target=/root/.cache/pip \
    pip install \
    av==19.0.1 \
    numpy==2.5.3 \
    pillow==12.3.0 \
    transformers==5.19.0 \
    faiss-cpu==1.15.1

RUN mkdir -p /hf-cache && chmod 777 /hf-cache

COPY data_processing ./data_processing

# `docker compose run --rm phase1 ...` → python -m data_processing.cli phase1 ...
ENTRYPOINT ["python", "-m", "data_processing.cli"]
CMD ["--help"]
