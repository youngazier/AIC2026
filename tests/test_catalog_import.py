"""Atomic catalog import, locking, queries, and CLI contracts."""

from argparse import Namespace
from dataclasses import asdict
import json
from pathlib import Path

from alembic import command
from alembic.config import Config
import pytest
from sqlalchemy import event, insert, select

from backend.app.database import PROJECT_ROOT, create_catalog_engine, create_session_factory
from backend.app.catalog.importer import CatalogImportError, DuplicateVideoNameError, import_video
from backend.app.catalog.queries import CatalogQueries
from backend.app.catalog.cli import main, discover_manifests
from backend.app.models import Video, Shot, Keyframe
from data_processing.phase1_keyframe_extraction.pipeline import processing_lock
from test_catalog_validation import published


@pytest.fixture
def catalog(published):
    root, manifest, image, result = published
    config = Config(str(PROJECT_ROOT / 'alembic.ini'))
    database = root / 'data_processed/database/catalog.sqlite3'
    config.cmd_opts = Namespace(x=[f'database_path={database}'])
    command.upgrade(config, 'head')
    engine = create_catalog_engine(database)
    sessions = create_session_factory(engine)
    yield root, manifest, image, result, engine, sessions
    engine.dispose()


def test_rerun_update_and_invalid_preserve_rows(catalog):
    root, manifest, image, result, engine, sessions = catalog
    import_video(manifest, sessions, project_root=root)
    import_video(manifest, sessions, project_root=root)
    queries = CatalogQueries(sessions)
    assert queries.list_videos().total == 1
    assert queries.list_keyframes('Videos_L21', 'L21_V001').total == 1
    result['shots'][0]['keyframe']['timestamp'] = 0.08
    manifest.write_text(json.dumps(result))
    import_video(manifest, sessions, project_root=root)
    assert queries.list_keyframes('Videos_L21', 'L21_V001').items[0].timestamp == 0.08
    before = queries.get_video('Videos_L21', 'L21_V001')
    image.write_bytes(b'invalid')
    with pytest.raises(ValueError):
        import_video(manifest, sessions, project_root=root)
    assert queries.get_video('Videos_L21', 'L21_V001') == before
    assert queries.list_keyframes('Videos_L21', 'L21_V001').items[0].timestamp == 0.08


def test_failure_after_delete_rolls_back(catalog):
    root, manifest, _, result, engine, sessions = catalog
    import_video(manifest, sessions, project_root=root)
    queries = CatalogQueries(sessions)
    before = queries.get_video('Videos_L21', 'L21_V001')
    result['shots'][0]['keyframe']['timestamp'] = 0.08
    manifest.write_text(json.dumps(result))

    def fail(connection, cursor, statement, parameters, context, executemany):
        if statement.startswith('INSERT INTO keyframes'):
            from sqlalchemy.exc import OperationalError
            raise OperationalError(statement, parameters, RuntimeError('injected failure'))

    event.listen(engine, 'before_cursor_execute', fail)
    try:
        with pytest.raises(CatalogImportError):
            import_video(manifest, sessions, project_root=root)
    finally:
        event.remove(engine, 'before_cursor_execute', fail)
    assert queries.get_video('Videos_L21', 'L21_V001') == before
    assert queries.list_keyframes('Videos_L21', 'L21_V001').items[0].timestamp == 0.04
    with sessions() as session:
        assert len(session.scalars(select(Shot)).all()) == 1


def test_duplicate_name_rejected(catalog):
    root, manifest, _, _, _, sessions = catalog
    import_video(manifest, sessions, project_root=root)
    with sessions.begin() as session:
        existing = session.get(Video, ('Videos_L21', 'L21_V001'))
        values = {column.name: getattr(existing, column.name) for column in Video.__table__.columns}
        session.delete(existing)
        session.flush()
        session.execute(insert(Video), values | {'batch': 'Videos_L22'})
    with pytest.raises(DuplicateVideoNameError):
        import_video(manifest, sessions, project_root=root)
    assert CatalogQueries(sessions).list_videos().items[0].batch == 'Videos_L22'


def test_publication_lock_blocks_import(catalog):
    root, manifest, _, _, _, sessions = catalog
    with processing_lock(root / 'data_processed/transnetv2'):
        with pytest.raises(ValueError, match='Another processor'):
            import_video(manifest, sessions, project_root=root)
    assert CatalogQueries(sessions).list_videos().total == 0


def test_queries_pagination_and_missing(catalog):
    root, manifest, _, _, _, sessions = catalog
    import_video(manifest, sessions, project_root=root)
    queries = CatalogQueries(sessions)
    original = queries.get_video('Videos_L21', 'L21_V001')
    with sessions.begin() as session:
        for batch, name in [('Videos_L20', 'L20_V002'), ('Videos_L20', 'L20_V001')]:
            values = asdict(original) | {'batch': batch, 'video_name': name, 'video_path': f'data/videos/{batch}/{name}.mp4'}
            session.execute(insert(Video), values)
    assert queries.list_videos(limit=1, offset=1).items[0].video_name == 'L20_V002'
    assert queries.list_videos(offset=3).items == ()
    assert queries.list_videos().total == 3
    assert queries.get_video('missing', 'missing') is None
    assert queries.list_keyframes('missing', 'missing') is None
    assert queries.list_keyframes('Videos_L21', 'L21_V001', offset=1).items == ()
    for limit, offset in [(0, 0), (True, 0), (1001, 0), (1, -1)]:
        with pytest.raises(ValueError):
            queries.list_videos(limit=limit, offset=offset)
        with pytest.raises(ValueError):
            queries.list_keyframes('Videos_L21', 'L21_V001', limit=limit, offset=offset)


def test_cli_reports_invalid_and_continues(catalog, capsys):
    root, manifest, _, _, _, sessions = catalog
    invalid = root / 'data_processed/transnetv2/Videos_L20/L20_V001/shots.json'
    invalid.parent.mkdir(parents=True)
    invalid.write_text('{')
    hidden = root / 'data_processed/transnetv2/.work/temp/shots.json'
    hidden.parent.mkdir(parents=True)
    hidden.write_text('{')
    assert discover_manifests(Path('data_processed/transnetv2'), root) == [invalid, manifest]
    assert main(['import', '--project-root', str(root)]) == 1
    output = capsys.readouterr()
    assert 'Imported 1; failed 1' in output.out
    assert str(invalid) in output.err
    assert CatalogQueries(sessions).list_videos().total == 1
    assert main(['import', '--project-root', str(root), '--input', str(manifest)]) == 0


def test_changed_shot_set_removes_old_rows(catalog):
    from PIL import Image

    root, manifest, image, result, _, sessions = catalog
    result['shots'] = [
        dict(shot_index=0, start_frame=0, end_frame=1, start_timestamp=0.0, end_timestamp=0.04,
             keyframe=dict(role='middle', frame_index=0, timestamp=0.0,
                           image_path=result['shots'][0]['keyframe']['image_path'])),
        dict(shot_index=1, start_frame=2, end_frame=2, start_timestamp=0.12, end_timestamp=0.12,
             keyframe=dict(role='middle', frame_index=2, timestamp=0.12,
                           image_path='keyframes/Videos_L21/L21_V001/00001.jpg')),
    ]
    second_image = image.with_name('00001.jpg')
    Image.new('RGB', (8, 6)).save(second_image)
    predictions = manifest.parent / 'L21_V001.predictions.txt'
    scenes = manifest.parent / 'L21_V001.scenes.txt'
    predictions.write_text('0 0\n0.9 0.9\n0 0\n')
    scenes.write_text('0 1\n2 2\n')
    manifest.write_text(json.dumps(result))
    import_video(manifest, sessions, project_root=root)
    queries = CatalogQueries(sessions)
    assert [row.shot_index for row in queries.list_keyframes('Videos_L21', 'L21_V001').items] == [0, 1]
    assert queries.list_keyframes('Videos_L21', 'L21_V001', limit=1, offset=1).items[0].timestamp == 0.12
    result['shots'] = [dict(shot_index=0, start_frame=0, end_frame=2,
                           start_timestamp=0, end_timestamp=0.12,
                           keyframe=dict(role='middle', frame_index=1, timestamp=0.04,
                                         image_path='keyframes/Videos_L21/L21_V001/00000.jpg'))]
    second_image.unlink()
    predictions.write_text('0 0\n0 0\n0 0\n')
    scenes.write_text('0 2\n')
    manifest.write_text(json.dumps(result))
    import_video(manifest, sessions, project_root=root)
    assert queries.list_keyframes('Videos_L21', 'L21_V001').total == 1
    with sessions() as session:
        assert len(session.scalars(select(Shot)).all()) == 1


def test_cli_empty_and_unmigrated_catalog(catalog, capsys):
    root, manifest, _, _, _, _ = catalog
    empty = root / 'data_processed/transnetv2/empty'
    empty.mkdir()
    assert main(['import', '--project-root', str(root), '--input', str(empty)]) == 1
    assert main(['import', '--project-root', str(root), '--input', str(manifest),
                 '--database-path', 'data_processed/database/unmigrated.sqlite3']) == 1
    assert 'Catalog write failed' in capsys.readouterr().err
