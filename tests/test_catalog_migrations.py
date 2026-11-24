"""Exercise the catalog migration rather than creating tables from ORM metadata."""

from argparse import Namespace
from pathlib import Path

from alembic import command
from alembic.config import Config
from alembic.autogenerate import compare_metadata
from alembic.migration import MigrationContext
import pytest
from sqlalchemy import inspect, insert, text
from sqlalchemy.exc import IntegrityError

from backend.app.database import PROJECT_ROOT, create_catalog_engine
from backend.app.models import Base, Video, Shot, Keyframe


@pytest.fixture
def migrated(tmp_path):
    path = tmp_path / 'catalog.sqlite3'
    config = Config(str(PROJECT_ROOT / 'alembic.ini'))
    config.cmd_opts = Namespace(x=[f'database_path={path}'])
    command.upgrade(config, 'head')
    engine = create_catalog_engine(path)
    yield engine, config
    engine.dispose()


def test_schema_upgrade_rerun_and_downgrade(migrated):
    engine, config = migrated
    command.upgrade(config, 'head')
    with engine.connect() as connection:
        assert connection.scalar(text('SELECT version_num FROM alembic_version')) == '0001'
        assert compare_metadata(MigrationContext.configure(connection), Base.metadata) == []
        schema = inspect(connection)
        assert schema.get_pk_constraint('keyframes')['constrained_columns'] == ['batch', 'video_name', 'shot_index']
        assert schema.get_indexes('videos')[0]['name'] == 'ix_videos_video_name'
        for table in ('videos', 'shots', 'keyframes'):
            assert schema.get_check_constraints(table)
        for table in ('shots', 'keyframes'):
            assert schema.get_foreign_keys(table)[0]['options']['ondelete'] == 'CASCADE'
    command.downgrade(config, 'base')
    assert inspect(engine).get_table_names() == ['alembic_version']
    command.upgrade(config, 'head')
    assert 'videos' in inspect(engine).get_table_names()


def test_constraints_and_cascade(migrated):
    engine, _ = migrated
    video = dict(batch='Videos_L21', video_name='L21_V001', video_path='data/videos/Videos_L21/L21_V001.mp4',
                 manifest_path='transnetv2/Videos_L21/L21_V001/shots.json', manifest_mtime_ns=1,
                 stream_index=0, width=8, height=6, duration_seconds=1, timeline_origin_seconds=0,
                 frame_count=3, shot_count=1, imported_at='2026-10-09T00:00:00Z')
    shot = dict(batch=video['batch'], video_name=video['video_name'], shot_index=0,
                start_frame=0, end_frame=2, start_timestamp=0, end_timestamp=0.1)
    keyframe = dict(batch=video['batch'], video_name=video['video_name'], shot_index=0,
                    role='middle', frame_index=1, timestamp=0.04, image_path='keyframes/test.jpg')
    with pytest.raises(IntegrityError), engine.begin() as connection:
        connection.execute(insert(Shot), shot)
    with pytest.raises(IntegrityError), engine.begin() as connection:
        connection.execute(insert(Video), dict(video, width=0))
    with engine.begin() as connection:
        connection.execute(insert(Video), video)
        connection.execute(insert(Shot), shot)
    with pytest.raises(IntegrityError), engine.begin() as connection:
        connection.execute(insert(Keyframe), dict(keyframe, role='first'))
    with engine.begin() as connection:
        connection.execute(insert(Keyframe), keyframe)
    with pytest.raises(IntegrityError), engine.begin() as connection:
        connection.execute(insert(Keyframe), keyframe)
    with engine.begin() as connection:
        connection.execute(text('DELETE FROM videos'))
        assert connection.scalar(text('SELECT COUNT(*) FROM shots')) == 0
        assert connection.scalar(text('SELECT COUNT(*) FROM keyframes')) == 0
