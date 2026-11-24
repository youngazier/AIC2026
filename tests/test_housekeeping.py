"""Housekeeping tests for dependencies, build configuration, and pytest markers."""

from pathlib import Path
import shutil
import subprocess
import sys
import tomllib

import pytest

PROJECT_ROOT = Path(__file__).resolve().parents[1]


def test_unused_postgres_dependencies_removed():
    pyproject = tomllib.loads((PROJECT_ROOT / "pyproject.toml").read_text())
    dependencies = pyproject["project"]["dependencies"]
    assert not any(
        dep.startswith(("pgvector", "psycopg")) for dep in dependencies
    )

    lock = tomllib.loads((PROJECT_ROOT / "uv.lock").read_text())
    locked_names = {pkg["name"] for pkg in lock.get("package", [])}
    assert locked_names.isdisjoint({"pgvector", "psycopg", "psycopg-binary"})


def test_uv_build_pin_and_lockfile_in_sync():
    pyproject = tomllib.loads((PROJECT_ROOT / "pyproject.toml").read_text())
    assert pyproject["build-system"]["requires"] == ["uv_build>=0.13,<0.14"]

    uv_bin = shutil.which("uv") or "uv"
    result = subprocess.run(
        [uv_bin, "lock", "--check"],
        cwd=PROJECT_ROOT,
        capture_output=True,
        text=True,
    )
    assert result.returncode == 0, result.stderr


def test_slow_and_gpu_markers_registered_and_excluded_by_default(tmp_path):
    sample = tmp_path / "test_marker_behavior.py"
    sample.write_text(
        "import pytest\n\n"
        "def test_fast_default():\n"
        "    pass\n\n"
        "@pytest.mark.slow\n"
        "def test_slow_only():\n"
        "    pass\n\n"
        "@pytest.mark.gpu\n"
        "def test_gpu_only():\n"
        "    pass\n"
    )

    default_run = subprocess.run(
        [
            sys.executable,
            "-m",
            "pytest",
            "-c",
            str(PROJECT_ROOT / "pyproject.toml"),
            "-W",
            "error::pytest.PytestUnknownMarkWarning",
            "-v",
            str(sample),
        ],
        capture_output=True,
        text=True,
    )
    assert default_run.returncode == 0, default_run.stdout + default_run.stderr
    assert "test_fast_default PASSED" in default_run.stdout
    assert "test_slow_only" not in default_run.stdout
    assert "test_gpu_only" not in default_run.stdout
    assert "2 deselected" in default_run.stdout

    slow_run = subprocess.run(
        [
            sys.executable,
            "-m",
            "pytest",
            "-c",
            str(PROJECT_ROOT / "pyproject.toml"),
            "-W",
            "error::pytest.PytestUnknownMarkWarning",
            "-v",
            "-m",
            "slow",
            str(sample),
        ],
        capture_output=True,
        text=True,
    )
    assert slow_run.returncode == 0, slow_run.stdout + slow_run.stderr
    assert "test_slow_only PASSED" in slow_run.stdout
    assert "test_fast_default" not in slow_run.stdout
    assert "test_gpu_only" not in slow_run.stdout
    assert "2 deselected" in slow_run.stdout


def test_leftover_backups_and_work_dirs_cleaned_while_skipping_finished_videos(
    processed_tree: Path, stub_detector, stub_embedder, monkeypatch
):
    from data_processing.cli import main
    from data_processing.phase2_feature_extraction.models import MODELS

    monkeypatch.setitem(MODELS, "clip", (stub_embedder, stub_embedder.DEFAULT_MODEL))

    p1_root = processed_tree / "data_processed/transnetv2"
    kf_root = processed_tree / "data_processed/keyframes"
    p1_backup_file = p1_root / ".X.backup-token"
    p1_backup_dir = p1_root / "Videos_L21/.L21_V001.backup-token"
    kf_backup_dir = kf_root / "Videos_L21/.L21_V001.backup-token"
    p1_work = p1_root / ".work/leftover"
    p1_backup_file.write_bytes(b"stale")
    p1_backup_dir.mkdir(parents=True)
    (p1_backup_dir / "old.json").write_text("{}")
    kf_backup_dir.mkdir(parents=True)
    (kf_backup_dir / "00000.jpg").write_bytes(b"stale")
    p1_work.mkdir(parents=True)
    (p1_work / "tmp.txt").write_text("stale")

    assert main(["phase1", "--all"]) == 0
    assert stub_detector.calls == 0
    assert not p1_backup_file.exists()
    assert not p1_backup_dir.exists()
    assert not kf_backup_dir.exists()
    assert not (p1_root / ".work").exists()

    p2_dir = processed_tree / "data_processed/features/clip"
    p2_backup_file = p2_dir / ".X.backup-token"
    p2_backup_dir = p2_dir / ".Y.backup-token"
    p2_work = p2_dir / ".work/leftover"
    p2_backup_file.write_bytes(b"stale")
    p2_backup_dir.mkdir(parents=True)
    (p2_backup_dir / "old.npy").write_bytes(b"stale")
    p2_work.mkdir(parents=True)
    (p2_work / "tmp.npy").write_bytes(b"stale")

    assert main(["phase2", "--all"]) == 0
    assert stub_embedder.calls == 0
    assert not p2_backup_file.exists()
    assert not p2_backup_dir.exists()
    assert not (p2_dir / ".work").exists()
