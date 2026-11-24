"""Atomic imports from published Phase 1 artifacts into the SQLite catalog."""

from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from pathlib import Path

from sqlalchemy import delete, insert, select, text
from sqlalchemy.dialects.sqlite import insert as sqlite_insert
from sqlalchemy.exc import SQLAlchemyError
from sqlalchemy.orm import Session, sessionmaker

from backend.app.database import PROJECT_ROOT
from backend.app.models import Keyframe, Shot, Video
from data_processing.phase1_keyframe_extraction.pipeline import processing_lock
from .validation import PreparedCatalogVideo, prepare_catalog_video


class CatalogImportError(ValueError):
    """An import cannot commit safely."""


class DuplicateVideoNameError(CatalogImportError):
    """Phase 2 cannot distinguish this name across batches yet."""


@dataclass(frozen=True)
class ImportResult:
    batch: str
    video_name: str
    shot_count: int
    imported_at: str


def _write_video(session: Session, prepared: PreparedCatalogVideo, imported_at: str) -> None:
    video = prepared.video
    duplicate = session.scalar(select(Video.batch).where(
        Video.video_name == video.video_name, Video.batch != video.batch,
    ).limit(1))
    if duplicate is not None:
        raise DuplicateVideoNameError(
            f'Video name {video.video_name} already belongs to batch {duplicate}'
        )
    values = asdict(video) | {'imported_at': imported_at}
    statement = sqlite_insert(Video).values(**values)
    session.execute(statement.on_conflict_do_update(
        index_elements=['batch', 'video_name'],
        set_={key: value for key, value in values.items() if key not in ('batch', 'video_name')},
    ))
    session.execute(delete(Shot).where(Shot.batch == video.batch, Shot.video_name == video.video_name))
    session.execute(insert(Shot), [asdict(shot) for shot in prepared.shots])
    session.execute(insert(Keyframe), [asdict(keyframe) for keyframe in prepared.keyframes])


def import_video(
    manifest_path: str | Path, sessions: sessionmaker[Session], *, project_root: Path = PROJECT_ROOT,
) -> ImportResult:
    """Validate first, then atomically replace one video's catalog rows.

    The existing Phase 1 exclusive lock is held through commit. A no-op write
    reserves the SQLite writer before duplicate-name checks, including writers
    importing different artifact roots into the same SQLite file.
    """
    root = project_root.resolve()
    output = root / 'data_processed/transnetv2'
    if not output.resolve().is_relative_to(root):
        raise CatalogImportError('Phase 1 root escapes the project')
    try:
        with processing_lock(output):
            prepared = prepare_catalog_video(manifest_path, project_root=root)
            imported_at = datetime.now(timezone.utc).isoformat()
            with sessions.begin() as session:
                # Python's non-legacy SQLite mode already opens a transaction.
                session.execute(text('UPDATE videos SET imported_at = imported_at WHERE 0'))
                _write_video(session, prepared, imported_at)
            return ImportResult(prepared.video.batch, prepared.video.video_name,
                                prepared.video.shot_count, imported_at)
    except SQLAlchemyError as exc:
        raise CatalogImportError(f'Catalog write failed for {manifest_path}: {exc}') from exc
