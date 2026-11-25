# syntax=docker/dockerfile:1
#
# One image for all offline processing: Phase 1 (TransNetV2 + keyframes),
# Phase 2 (embedding features), and Phase 3 (FAISS index + SQLite catalog).
# Build and run through data_processing/compose.yaml.

FROM python:3.14-slim

# CPU wheels by default; compose.gpu.yaml passes a CUDA index for GPU builds.
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

# Export runtime dependencies from uv.lock and install torch separately so
# TORCH_INDEX_URL selects CPU vs CUDA wheels without drifting package versions.
COPY pyproject.toml uv.lock ./
RUN --mount=type=cache,target=/root/.cache/pip \
    pip install "uv>=0.13,<0.14" \
    && uv export --frozen --no-dev --no-hashes --no-emit-project --no-header --no-annotate -o /tmp/requirements.txt \
    && grep -E '^torch==' /tmp/requirements.txt > /tmp/torch.txt \
    && grep -vE '^(torch|nvidia-|cuda-|triton)==' /tmp/requirements.txt > /tmp/deps.txt \
    && pip install --index-url "${TORCH_INDEX_URL}" -r /tmp/torch.txt \
    && pip install --no-deps -r /tmp/deps.txt \
    && rm -f /tmp/requirements.txt /tmp/torch.txt /tmp/deps.txt

RUN mkdir -p /hf-cache && chmod 777 /hf-cache

COPY alembic.ini ./
COPY backend ./backend
COPY data_processing ./data_processing

# `docker compose run --rm phase1 ...` → python -m data_processing.cli phase1 ...
ENTRYPOINT ["python", "-m", "data_processing.cli"]
CMD ["--help"]
