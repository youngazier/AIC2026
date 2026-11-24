"""Import published artifacts: python -m backend.app.catalog.cli import."""

import argparse
from pathlib import Path
import sys

from backend.app.database import PROJECT_ROOT, create_catalog_engine, create_session_factory
from .importer import import_video


def discover_manifests(input_path: Path, project_root: Path) -> list[Path]:
    if not input_path.is_absolute():
        input_path = project_root / input_path
    approved = (project_root / 'data_processed/transnetv2').resolve()
    if not input_path.resolve().is_relative_to(approved):
        raise ValueError('Import input must be under data_processed/transnetv2')
    if input_path.is_file() and input_path.name == 'shots.json':
        candidates = [input_path]
    elif input_path.is_dir():
        candidates = sorted(input_path.rglob('shots.json'))
    else:
        raise ValueError(f'No Phase 1 input at {input_path}')
    manifests = [path for path in candidates if not any(
        part.startswith('.') for part in path.relative_to(project_root / 'data_processed/transnetv2').parts
    )]
    if not manifests:
        raise ValueError(f'No published shots.json files found at {input_path}')
    return manifests


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    subparsers = parser.add_subparsers(dest='command', required=True)
    importer = subparsers.add_parser('import', help='Import completed Phase 1 artifacts')
    importer.add_argument('--input', type=Path, default=Path('data_processed/transnetv2'))
    importer.add_argument('--database-path', type=Path)
    importer.add_argument('--project-root', type=Path, default=PROJECT_ROOT)
    args = parser.parse_args(argv)
    root = args.project_root.resolve()
    path = args.database_path or root / 'data_processed/database/catalog.sqlite3'
    if not path.is_absolute():
        path = root / path
    engine = None
    try:
        manifests = discover_manifests(args.input, root)
        engine = create_catalog_engine(path)
        sessions = create_session_factory(engine)
        failed = 0
        for manifest in manifests:
            try:
                result = import_video(manifest, sessions, project_root=root)
                print(f'IMPORTED {result.batch}/{result.video_name}: {result.shot_count} shots')
            except (ValueError, OSError) as exc:
                failed += 1
                print(f'FAILED {manifest}: {exc}', file=sys.stderr)
        print(f'Imported {len(manifests) - failed}; failed {failed}')
        return 1 if failed else 0
    except (ValueError, OSError) as exc:
        print(f'Catalog import failed: {exc}', file=sys.stderr)
        return 1
    finally:
        if engine is not None:
            engine.dispose()


if __name__ == '__main__':
    raise SystemExit(main())
