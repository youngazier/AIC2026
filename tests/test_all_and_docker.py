"""Tests for Step 6 (Item 7): unified `all` CLI command and Docker configuration."""

from pathlib import Path, PurePosixPath
import re
import shutil
import subprocess

import faiss
import pytest
from sqlalchemy import func, select

from backend.app.database import create_catalog_engine
from backend.app.models import Keyframe, Shot, Video
from data_processing import cli
from data_processing.phase2_feature_extraction import models as model_registry


REPO_ROOT = Path(__file__).resolve().parents[1]


def _patch_clip_stub(monkeypatch, stub_embedder) -> None:
    monkeypatch.setitem(
        model_registry.MODELS,
        "clip",
        (stub_embedder, stub_embedder.DEFAULT_MODEL),
    )


def _matches_dockerignore(patterns: list[str], rel_path: str) -> bool:
    """Evaluate Docker .dockerignore rules (last matching pattern wins)."""
    ignored = False
    posix = PurePosixPath(rel_path)
    parts = posix.parts
    prefixes = ["/".join(parts[:i]) for i in range(1, len(parts) + 1)]
    for raw in patterns:
        line = raw.strip()
        if not line or line.startswith("#"):
            continue
        negated = line.startswith("!")
        if negated:
            line = line[1:].strip()
        pattern = line.strip("/")
        if not pattern:
            continue
        matched = False
        if "**" in pattern:
            if pattern.endswith("/**"):
                base = pattern[:-3].strip("/")
                matched = any(p == base or p.startswith(base + "/") for p in prefixes)
            elif pattern.startswith("**/"):
                tail = pattern[3:]
                matched = any(PurePosixPath(p).match(tail) or PurePosixPath(p).name == tail for p in prefixes)
            else:
                matched = posix.match(pattern)
        else:
            matched = any(PurePosixPath(p).match(pattern) for p in prefixes)
        if matched:
            ignored = not negated
    return ignored


def test_all_processes_validates_and_skips_on_rerun(data_tree: Path, stub_detector, stub_embedder, monkeypatch):
    _patch_clip_stub(monkeypatch, stub_embedder)

    exit_code = cli.main(["all", "--batch", "Videos_L21", "Videos_L22", "--model", "clip", "--device", "cpu"])
    assert exit_code == 0

    for batch, video_name in (
        ("Videos_L21", "L21_V001"),
        ("Videos_L21", "L21_V002"),
        ("Videos_L22", "L22_V001"),
    ):
        assert (data_tree / "data_processed/transnetv2" / batch / video_name / "shots.json").is_file()
        assert (data_tree / "data_processed/keyframes" / batch / video_name / "00000.jpg").is_file()
        assert (data_tree / "data_processed/features/clip" / f"{video_name}.npy").is_file()

    index_path = data_tree / "data_processed/indexes/clip/global.index"
    assert index_path.is_file()
    assert faiss.read_index(str(index_path)).ntotal == 9

    engine = create_catalog_engine(data_tree / "data_processed/database/catalog.sqlite3")
    try:
        with engine.connect() as conn:
            assert conn.scalar(select(func.count()).select_from(Video)) == 3
            assert conn.scalar(select(func.count()).select_from(Shot)) == 9
            assert conn.scalar(select(func.count()).select_from(Keyframe)) == 9
    finally:
        engine.dispose()

    stub_detector.calls = 0
    stub_embedder.calls = 0
    rerun_code = cli.main(["all", "--batch", "Videos_L21", "Videos_L22", "--model", "clip", "--device", "cpu"])
    assert rerun_code == 0
    assert stub_detector.calls == 0
    assert stub_embedder.calls == 0


def test_all_no_phase3_skips_index_and_catalog(data_tree: Path, stub_detector, stub_embedder, monkeypatch):
    _patch_clip_stub(monkeypatch, stub_embedder)

    exit_code = cli.main(["all", "--batch", "Videos_L21", "--model", "clip", "--no-phase3"])
    assert exit_code == 0
    assert (data_tree / "data_processed/keyframes/Videos_L21/L21_V001/00000.jpg").is_file()
    assert (data_tree / "data_processed/features/clip/L21_V001.npy").is_file()
    assert (data_tree / "data_processed/features/clip/L21_V002.npy").is_file()
    assert not (data_tree / "data_processed/indexes").exists()
    assert not (data_tree / "data_processed/database").exists()


def test_all_failure_isolation_and_recovery(
    data_tree: Path, tiny_video: Path, stub_detector, stub_embedder, monkeypatch
):
    _patch_clip_stub(monkeypatch, stub_embedder)
    broken = data_tree / "data/videos/Videos_L21/L21_V002.mp4"
    broken.write_bytes(b"corrupt-mp4-data")

    exit_code = cli.main(["all", "--all", "--model", "clip"])
    assert exit_code != 0

    assert (data_tree / "data_processed/features/clip/L21_V001.npy").is_file()
    assert (data_tree / "data_processed/features/clip/L22_V001.npy").is_file()
    assert not (data_tree / "data_processed/features/clip/L21_V002.npy").exists()
    assert faiss.read_index(str(data_tree / "data_processed/indexes/clip/global.index")).ntotal == 6

    engine = create_catalog_engine(data_tree / "data_processed/database/catalog.sqlite3")
    try:
        with engine.connect() as conn:
            assert conn.scalar(select(func.count()).select_from(Video)) == 2
    finally:
        engine.dispose()

    shutil.copy2(tiny_video, broken)
    stub_detector.calls = 0
    stub_embedder.calls = 0
    recovery_code = cli.main(["all", "--all", "--model", "clip"])
    assert recovery_code == 0
    assert stub_detector.calls == 1
    assert stub_embedder.calls == 1
    assert faiss.read_index(str(data_tree / "data_processed/indexes/clip/global.index")).ntotal == 9

    engine = create_catalog_engine(data_tree / "data_processed/database/catalog.sqlite3")
    try:
        with engine.connect() as conn:
            assert conn.scalar(select(func.count()).select_from(Video)) == 3
    finally:
        engine.dispose()


def test_all_stops_on_stage_level_error(data_tree: Path, stub_detector, stub_embedder, monkeypatch):
    _patch_clip_stub(monkeypatch, stub_embedder)

    assert cli.main(["all", "--batch", "Videos_L99", "--model", "clip"]) != 0
    assert cli.main(["all", "--batch", "Videos_L21", "--model", "unknown"]) != 0
    assert not (data_tree / "data_processed/indexes").exists()
    assert not (data_tree / "data_processed/database").exists()


def test_all_continues_when_entire_batch_fails_phase1(
    data_tree: Path, stub_detector, stub_embedder, monkeypatch
):
    _patch_clip_stub(monkeypatch, stub_embedder)
    broken = data_tree / "data/videos/Videos_L22/L22_V001.mp4"
    broken.write_bytes(b"corrupt-mp4-data")

    exit_code = cli.main(["all", "--batch", "Videos_L21", "Videos_L22", "--model", "clip"])
    assert exit_code != 0

    assert (data_tree / "data_processed/features/clip/L21_V001.npy").is_file()
    assert (data_tree / "data_processed/features/clip/L21_V002.npy").is_file()
    assert not (data_tree / "data_processed/features/clip/L22_V001.npy").exists()
    assert faiss.read_index(str(data_tree / "data_processed/indexes/clip/global.index")).ntotal == 6

    engine = create_catalog_engine(data_tree / "data_processed/database/catalog.sqlite3")
    try:
        with engine.connect() as conn:
            assert conn.scalar(select(func.count()).select_from(Video)) == 2
    finally:
        engine.dispose()

    assert cli.main(["phase2", "--batch", "Videos_L99", "--model", "clip"]) != 0
    assert cli.main(["phase2", "--batch", "Videos_L22", "--model", "clip"]) != 0


def test_dockerfile_filter_excludes_nvidia_and_cuda_packages():
    dockerfile = (REPO_ROOT / "data_processing/Dockerfile").read_text()
    match = re.search(r"grep -vE '([^']+)' /tmp/requirements\.txt", dockerfile)
    assert match is not None, "Expected grep -vE filter in data_processing/Dockerfile"
    filter_re = re.compile(match.group(1))

    exported = subprocess.run(
        [
            "uv", "export", "--frozen", "--no-dev", "--no-hashes",
            "--no-emit-project", "--no-header", "--no-annotate",
        ],
        cwd=REPO_ROOT,
        capture_output=True,
        text=True,
        check=True,
    )
    filtered = [line for line in exported.stdout.splitlines() if not filter_re.search(line)]
    assert filtered
    assert not any(line.startswith(("nvidia-", "cuda-", "torch==", "triton==")) for line in filtered)


def test_all_unbatched_path_completes_all_phases(
    data_tree: Path, tiny_video: Path, stub_detector, stub_embedder, monkeypatch
):
    _patch_clip_stub(monkeypatch, stub_embedder)
    outside = data_tree / "data/videos/custom_clip.mp4"
    shutil.copy2(tiny_video, outside)

    exit_code = cli.main(["all", str(outside), "--model", "clip"])
    assert exit_code == 0
    assert (data_tree / "data_processed/transnetv2/unbatched/custom_clip/shots.json").is_file()
    assert (data_tree / "data_processed/keyframes/unbatched/custom_clip/00000.jpg").is_file()
    assert (data_tree / "data_processed/features/clip/custom_clip.npy").is_file()
    assert faiss.read_index(str(data_tree / "data_processed/indexes/clip/global.index")).ntotal == 3

    engine = create_catalog_engine(data_tree / "data_processed/database/catalog.sqlite3")
    try:
        with engine.connect() as conn:
            assert conn.scalar(select(func.count()).select_from(Video)) == 1
    finally:
        engine.dispose()


def test_compose_configs_are_valid():
    assert not (REPO_ROOT / "data_processing/phase1_keyframe_extraction/Dockerfile").exists()
    assert not (REPO_ROOT / "data_processing/phase1_keyframe_extraction/compose.yaml").exists()

    compose_file = REPO_ROOT / "data_processing/compose.yaml"
    gpu_override = REPO_ROOT / "data_processing/compose.gpu.yaml"
    assert compose_file.is_file()
    assert gpu_override.is_file()

    if shutil.which("docker") is None:
        pytest.skip("docker CLI is not installed")
    compose_check = subprocess.run(["docker", "compose", "version"], capture_output=True, text=True)
    if compose_check.returncode != 0:
        pytest.skip("docker compose is not available")

    cpu = subprocess.run(
        ["docker", "compose", "-f", str(compose_file), "config"],
        capture_output=True, text=True,
    )
    assert cpu.returncode == 0, cpu.stderr
    assert 'shm_size: "4294967296"' in cpu.stdout

    cpu_services = subprocess.run(
        ["docker", "compose", "-f", str(compose_file), "config", "--services"],
        capture_output=True, text=True,
    )
    assert cpu_services.returncode == 0, cpu_services.stderr
    assert set(cpu_services.stdout.splitlines()) == {"phase1", "phase2", "phase3", "all"}

    gpu = subprocess.run(
        ["docker", "compose", "-f", str(compose_file), "-f", str(gpu_override), "config"],
        capture_output=True, text=True,
    )
    assert gpu.returncode == 0, gpu.stderr
    assert 'shm_size: "4294967296"' in gpu.stdout
    assert "TORCH_INDEX_URL: https://download.pytorch.org/whl/cu130" in gpu.stdout
    assert "driver: nvidia" in gpu.stdout or "gpus:" in gpu.stdout or "NVIDIA" in gpu.stdout.upper() or "capabilities:" in gpu.stdout


def test_dockerignore_build_context():
    patterns = (REPO_ROOT / ".dockerignore").read_text().splitlines()

    excluded = (
        "data/videos/Videos_L21/L21_V001.mp4",
        "data_processed/features/clip/L21_V001.npy",
        "test/transnetv2_pytorch/README.md",
        ".git/HEAD",
        "backend/__pycache__/database.cpython-314.pyc",
    )
    included = (
        "data_processing/cli.py",
        "data_processing/phase1_keyframe_extraction/transnetv2/transnetv2-pytorch-weights.pth",
        "backend/app/catalog/importer.py",
        "backend/migrations/versions/0001_create_video_shot_and_keyframe_catalog.py",
        "alembic.ini",
        "pyproject.toml",
        "uv.lock",
    )

    for rel in excluded:
        assert _matches_dockerignore(patterns, rel), f"Expected {rel} to be excluded by .dockerignore"
    for rel in included:
        assert not _matches_dockerignore(patterns, rel), f"Expected {rel} to be included in Docker context"
