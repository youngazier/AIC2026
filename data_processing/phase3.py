"""Phase 3: migrate and populate the SQLite catalog, and build FAISS indexes."""

from argparse import Namespace
import logging
from pathlib import Path
import sys

from .phase1_keyframe_extraction import pipeline as phase1_pipeline
from .phase2_feature_extraction import pipeline as phase2_pipeline
from .phase2_feature_extraction.build_index import build_for_model
from .phase2_feature_extraction.models import resolve_model

LOGGER = logging.getLogger(__name__)
REPO_ROOT = Path(__file__).resolve().parents[1]


def run_catalog(project_root: Path, database_path: Path | None = None) -> bool:
    """Run Alembic migrations and import every Phase 1 result into SQLite."""
    from alembic import command
    from alembic.config import Config

    from backend.app.catalog.cli import discover_manifests
    from backend.app.catalog.importer import import_video
    from backend.app.database import create_catalog_engine, create_session_factory

    db_path = (database_path or project_root / "data_processed/database/catalog.sqlite3").resolve()
    config = Config(str(REPO_ROOT / "alembic.ini"))
    config.cmd_opts = Namespace(x=[f"database_path={db_path}"])
    command.upgrade(config, "head")

    manifests = discover_manifests(Path("data_processed/transnetv2"), project_root)
    engine = create_catalog_engine(db_path)
    failed = 0
    try:
        sessions = create_session_factory(engine)
        for manifest in manifests:
            try:
                result = import_video(manifest, sessions, project_root=project_root)
                LOGGER.info("IMPORTED %s/%s: %d shots", result.batch, result.video_name, result.shot_count)
            except (ValueError, OSError) as exc:
                failed += 1
                print(f"FAILED {manifest}: {exc}", file=sys.stderr)
    finally:
        engine.dispose()
    print(f"Catalog: imported {len(manifests) - failed}, failed {failed}")
    return failed == 0


def run_phase3(
    *,
    model_key: str = "clip",
    no_playlists: bool = False,
    force: bool = False,
    no_index: bool = False,
    no_catalog: bool = False,
    project_root: Path | None = None,
) -> int:
    """Run catalog import and FAISS index construction; failures in one don't skip the other."""
    if no_index and no_catalog:
        raise ValueError("Cannot specify both --no-index and --no-catalog")
    if not no_index:
        resolve_model(model_key)

    root = (project_root or phase1_pipeline.PROJECT_ROOT).resolve()
    catalog_ok = True
    index_ok = True

    if not no_catalog:
        try:
            catalog_ok = run_catalog(root)
        except Exception as exc:
            catalog_ok = False
            LOGGER.error("Catalog import failed: %s", exc)
            print(f"Catalog import failed: {exc}", file=sys.stderr)

    if not no_index:
        try:
            features_root = (
                root / "data_processed/features"
                if project_root is not None
                else phase2_pipeline.DEFAULT_OUTPUT_ROOT
            ).resolve()
            phase1_root = (
                root / "data_processed/transnetv2"
                if project_root is not None
                else phase1_pipeline.DEFAULT_OUTPUT_ROOT
            ).resolve()
            index_root = (root / "data_processed/indexes").resolve()
            playlist_mapping = None if no_playlists else (root / "data/video_to_playlist.json").resolve()
            summary = build_for_model(
                model_key,
                features_root=features_root,
                index_root=index_root,
                playlist_mapping=playlist_mapping,
                phase1_root=phase1_root,
                force=force,
            )
            print(f"Index: built {summary['built']}, skipped {summary['skipped']}")
        except Exception as exc:
            index_ok = False
            LOGGER.error("Index build failed: %s", exc)
            print(f"Index build failed: {exc}", file=sys.stderr)

    return 0 if (catalog_ok and index_ok) else 1
