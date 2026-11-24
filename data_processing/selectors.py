"""Shared CLI input selectors (--video, --batch, --all) for Phase 1 and Phase 2."""

from collections.abc import Sequence
from pathlib import Path
import re

from .phase1_keyframe_extraction import pipeline as phase1_pipeline
from .phase1_keyframe_extraction.metadata import VIDEO_EXTENSIONS

BATCH_PATTERN = re.compile(r"Videos_L\d+")


def add_selector_arguments(parser, input_help: str) -> None:
    """Register mutually exclusive input selectors on a CLI subparser."""
    parser.add_argument("input", nargs="?", type=Path, help=input_help)
    parser.add_argument(
        "--video",
        nargs="+",
        metavar="VIDEO",
        help="One or more video names (e.g. L21_V001) or .mp4 paths",
    )
    parser.add_argument(
        "--batch",
        nargs="+",
        metavar="BATCH",
        help="One or more batch names (e.g. Videos_L21 Videos_L22)",
    )
    parser.add_argument(
        "--all",
        dest="all_videos",
        action="store_true",
        help="Select every Videos_L* batch under data/videos/",
    )


def _batch_videos(batch_dir: Path) -> list[Path]:
    return sorted(
        path.resolve()
        for path in batch_dir.iterdir()
        if path.is_file() and path.suffix.lower() in VIDEO_EXTENSIONS
    )


def _dedupe(paths: list[Path]) -> list[Path]:
    seen: set[Path] = set()
    unique: list[Path] = []
    for path in paths:
        if path not in seen:
            seen.add(path)
            unique.append(path)
    return unique


def select_videos(
    *,
    input_path: Path | None = None,
    video: Sequence[str] | None = None,
    batch: Sequence[str] | None = None,
    all_videos: bool = False,
    videos_root: Path | None = None,
) -> list[Path]:
    """Resolve exactly one selector (--video, --batch, --all, or path) to video paths."""
    kinds = sum(bool(item) for item in (input_path is not None, video, batch, all_videos))
    if kinds == 0:
        raise ValueError("Select what to process using --video, --batch, --all, or an input path")
    if kinds > 1:
        raise ValueError("Specify exactly one of --video, --batch, --all, or an input path")

    if input_path is not None:
        return phase1_pipeline.discover_videos(input_path)

    root = (videos_root or phase1_pipeline.PROJECT_ROOT / "data" / "videos").resolve()

    if all_videos:
        if not root.is_dir():
            raise ValueError(f"Videos root does not exist: {root}")
        selected: list[Path] = []
        for entry in sorted(root.iterdir()):
            if entry.is_dir() and BATCH_PATTERN.fullmatch(entry.name):
                selected.extend(_batch_videos(entry))
        if not selected:
            raise ValueError(f"No supported videos found in Videos_L* batches under {root}")
        return _dedupe(selected)

    if batch:
        for batch_name in batch:
            if not BATCH_PATTERN.fullmatch(batch_name):
                raise ValueError(f"--batch must match Videos_L<number>, got {batch_name!r}")
        selected = []
        for batch_name in batch:
            batch_dir = root / batch_name
            if not batch_dir.is_dir():
                raise ValueError(f"Batch directory does not exist: {batch_dir}")
            found = _batch_videos(batch_dir)
            if not found:
                raise ValueError(f"No supported videos found in batch {batch_name}: {batch_dir}")
            selected.extend(found)
        return _dedupe(selected)

    assert video is not None
    by_name: dict[str, Path] = {}
    if root.is_dir():
        for entry in sorted(root.iterdir()):
            if entry.is_dir() and BATCH_PATTERN.fullmatch(entry.name):
                for path in _batch_videos(entry):
                    by_name[path.stem] = path
            elif entry.is_file() and entry.suffix.lower() in VIDEO_EXTENSIONS:
                by_name[entry.stem] = entry.resolve()

    selected = []
    for item in video:
        candidate = Path(item)
        if "/" in item or "\\" in item or candidate.suffix.lower() in VIDEO_EXTENSIONS:
            if not candidate.is_absolute() and not candidate.exists():
                rel_candidate = phase1_pipeline.PROJECT_ROOT / candidate
                if rel_candidate.exists():
                    candidate = rel_candidate
            if not candidate.is_file() or candidate.suffix.lower() not in VIDEO_EXTENSIONS:
                raise ValueError(f"Video path does not exist or has unsupported extension: {item}")
            selected.append(candidate.resolve())
        else:
            if item not in by_name:
                raise ValueError(f"Unknown video {item!r} under {root}")
            selected.append(by_name[item])
    return _dedupe(selected)
