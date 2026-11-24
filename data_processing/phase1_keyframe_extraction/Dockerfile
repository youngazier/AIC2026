# syntax=docker/dockerfile:1

FROM python:3.14-slim

ARG TORCH_INDEX_URL=https://download.pytorch.org/whl/cpu

ENV PYTHONUNBUFFERED=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1 \
    PIP_PREFER_BINARY=1

WORKDIR /app

# Install system dependencies
RUN apt-get update \
    && apt-get install -y --no-install-recommends ffmpeg \
    && rm -rf /var/lib/apt/lists/*

# Install PyTorch with pip cache
RUN --mount=type=cache,target=/root/.cache/pip \
    pip install \
    --index-url "${TORCH_INDEX_URL}" \
    torch==2.14.0

# Install other Python dependencies with pip cache
RUN --mount=type=cache,target=/root/.cache/pip \
    pip install \
    av==19.0.1 \
    numpy==2.5.3 \
    pillow==12.3.0

# Copy only the required application code
COPY data_processing/phase1_keyframe_extraction \
     ./data_processing/phase1_keyframe_extraction

# Required for Python module imports
RUN touch data_processing/__init__.py

ENTRYPOINT ["python", "-m", "data_processing.phase1_keyframe_extraction.cli"]

CMD ["phase1"]