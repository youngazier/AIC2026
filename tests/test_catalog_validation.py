"""Catalog preparation rejects invalid artifacts before any database writes."""

from dataclasses import FrozenInstanceError
import json

from PIL import Image
import pytest

from backend.app.catalog.validation import CatalogValidationError, prepare_catalog_video


@pytest.fixture
def published(tmp_path):
    source = tmp_path / 'data/videos/Videos_L21/L21_V001.mp4'
    source.parent.mkdir(parents=True)
    source.write_bytes(b'source fixture; validation does not decode video')
    directory = tmp_path / 'data_processed/transnetv2/Videos_L21/L21_V001'
    directory.mkdir(parents=True)
    image = tmp_path / 'data_processed/keyframes/Videos_L21/L21_V001/00000.jpg'
    image.parent.mkdir(parents=True)
    Image.new('RGB', (8, 6)).save(image)
    (directory / 'L21_V001.predictions.txt').write_text('0 0\n0 0\n0 0\n')
    (directory / 'L21_V001.scenes.txt').write_text('0 2\n')
    result = dict(
        batch='Videos_L21', video_name='L21_V001', mode='phase1',
        source_key=source.relative_to(tmp_path).as_posix(),
        source_size=source.stat().st_size, source_mtime_ns=source.stat().st_mtime_ns,
        checkpoint_mtime_ns=1, pipeline_version='phase1-middle-v3', upstream_revision='test',
        threshold=0.5, visualize=False, frame_count=3,
        metadata=dict(stream_index=0, codec='h264', width=8, height=6,
                      duration_seconds=1.0, average_fps=25.0, timeline_origin_seconds=-0.2),
        shots=[dict(shot_index=0, start_frame=0, end_frame=2,
                    start_timestamp=0.0, end_timestamp=0.12,
                    keyframe=dict(role='middle', frame_index=1, timestamp=0.04,
                                  image_path=image.relative_to(tmp_path / 'data_processed').as_posix()))],
    )
    manifest = directory / 'shots.json'
    manifest.write_text(json.dumps(result))
    return tmp_path, manifest, image, result


def test_prepares_immutable_rows_with_original_timestamps(published):
    root, manifest, _, _ = published
    prepared = prepare_catalog_video(manifest, project_root=root)
    assert prepared.video.shot_count == 1
    assert prepared.video.manifest_path == 'transnetv2/Videos_L21/L21_V001/shots.json'
    assert prepared.keyframes[0].timestamp == 0.04
    assert prepared.video.timeline_origin_seconds == -0.2
    with pytest.raises(FrozenInstanceError):
        prepared.video.width = 9


@pytest.mark.parametrize('case', [
    'nan_duration', 'bool_width', 'bad_settings', 'missing_provenance',
    'wrong_identity', 'bad_frame', 'bad_timestamp', 'traversal', 'absolute',
    'bad_role', 'shot_count', 'source_changed', 'wrong_mode',
])
def test_rejects_invalid_manifest(published, case):
    root, manifest, _, result = published
    shot = result['shots'][0]
    if case == 'nan_duration':
        result['metadata']['duration_seconds'] = float('nan')
    elif case == 'bool_width':
        result['metadata']['width'] = True
    elif case == 'bad_settings':
        result['threshold'] = 2
    elif case == 'missing_provenance':
        del result['pipeline_version']
    elif case == 'wrong_identity':
        result['video_name'] = 'L21_V002'
    elif case == 'bad_frame':
        shot['keyframe']['frame_index'] = True
    elif case == 'bad_timestamp':
        shot['keyframe']['timestamp'] = 1
    elif case == 'traversal':
        result['source_key'] = 'data/videos/../outside.mp4'
    elif case == 'absolute':
        result['source_key'] = '/tmp/source.mp4'
    elif case == 'bad_role':
        shot['keyframe']['role'] = 'first'
    elif case == 'shot_count':
        result['shot_count'] = 2
    elif case == 'source_changed':
        result['source_size'] += 1
    elif case == 'wrong_mode':
        result['mode'] = 'transnetv2'
    manifest.write_text(json.dumps(result))
    with pytest.raises(CatalogValidationError):
        prepare_catalog_video(manifest, project_root=root)


def test_rejects_unreadable_image(published):
    root, manifest, image, _ = published
    image.write_bytes(b'not an image')
    with pytest.raises(CatalogValidationError):
        prepare_catalog_video(manifest, project_root=root)


@pytest.mark.parametrize('target', ['image', 'predictions'])
def test_rejects_escaping_symlink(published, tmp_path, target):
    root, manifest, image, _ = published
    path = image if target == 'image' else manifest.parent / 'L21_V001.predictions.txt'
    outside = root / 'outside'
    outside.write_bytes(path.read_bytes())
    path.unlink()
    path.symlink_to(outside)
    with pytest.raises(CatalogValidationError):
        prepare_catalog_video(manifest, project_root=root)


def test_rejects_malformed_json(published):
    root, manifest, _, _ = published
    manifest.write_text('{')
    with pytest.raises(CatalogValidationError):
        prepare_catalog_video(manifest, project_root=root)


def test_rejects_change_during_validation(published, monkeypatch):
    from backend.app.catalog import validation

    root, manifest, _, _ = published
    original = validation.validate_result

    def change_after_validation(*args):
        original(*args)
        manifest.write_text(manifest.read_text() + '\n')

    monkeypatch.setattr(validation, 'validate_result', change_after_validation)
    with pytest.raises(CatalogValidationError, match='changed during validation'):
        prepare_catalog_video(manifest, project_root=root)
