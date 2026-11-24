"""Phase 1 entry point, also usable from this directory's Docker image."""

import argparse
import logging
from pathlib import Path
import re


def add_commands(commands) -> None:
    for name, help_text in (("transnetv2", "Detect shots only"),
                            ("phase1", "Detect shots and extract one middle keyframe per shot")):
        command = commands.add_parser(name, help=help_text)
        command.add_argument("input", nargs="?", type=Path, help="Video file or directory; defaults to data/videos")
        command.add_argument("--batch", help="Limit default input to a batch, e.g. Videos_L21")
        command.add_argument("--output-root", type=Path, help="Result root; JPEGs use its sibling keyframes/ directory")
        command.add_argument("--weights", type=Path, help="Override the checkpoint path")
        command.add_argument("--device", choices=("auto", "cpu", "gpu", "cuda"), default="auto")
        command.add_argument("--threshold", type=float, default=0.5)
        command.add_argument("--force", action="store_true", help="Rebuild even if outputs validate")
        command.add_argument("--visualize", action="store_true", help="Save a detector contact sheet")


def run(args) -> int:
    from .pipeline import DEFAULT_OUTPUT_ROOT, DEFAULT_WEIGHTS, PROJECT_ROOT, run_pipeline

    input_path = args.input or PROJECT_ROOT / "data" / "videos"
    if args.batch:
        input_path /= args.batch
    try:
        summary = run_pipeline(
            input_path, output_root=args.output_root or DEFAULT_OUTPUT_ROOT,
            weights=args.weights or DEFAULT_WEIGHTS, mode=args.command,
            threshold=args.threshold, device=args.device,
            force=args.force, visualize=args.visualize,
        )
    except (ValueError, OSError) as exc:
        logging.error("%s", exc)
        return 1
    except KeyboardInterrupt:
        logging.error("Processing interrupted; rerun the command to retry")
        return 130
    print("Processed: {processed}; skipped: {skipped}; failed: {failed}".format(**summary))
    return 1 if summary["failed"] else 0


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description="Video shot detection and keyframe extraction")
    add_commands(parser.add_subparsers(dest="command", required=True))
    args = parser.parse_args(argv)
    if args.input is not None and args.batch is not None:
        parser.error("Use either an input path or --batch")
    if args.batch is not None and not re.fullmatch(r"Videos_L\d+", args.batch):
        parser.error("--batch must look like Videos_L21")
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    return run(args)


if __name__ == "__main__":
    raise SystemExit(main())
