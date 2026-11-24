"""Command-line entry point for offline preprocessing phases."""

import argparse
import logging
from pathlib import Path
import re


def main(argv=None) -> int:
    from .phase1_keyframe_extraction.cli import add_commands, run as run_phase1

    parser = argparse.ArgumentParser(description="Offline video preprocessing")
    commands = parser.add_subparsers(dest="command", required=True)
    add_commands(commands)
    phase2 = commands.add_parser("phase2", help="Encode completed Phase 1 keyframes with CLIP")
    phase2.add_argument(
        "input", nargs="?", type=Path,
        help="Phase 1 result directory; defaults to data_processed/transnetv2",
    )
    phase2.add_argument("--batch", help="Limit default input to a batch, e.g. Videos_L21")
    phase2.add_argument("--keyframe-root", type=Path, help="Root containing Phase 1 keyframe batches")
    phase2.add_argument("--output-root", type=Path, help="Embedding output root")
    phase2.add_argument("--model", default="clip",
                        help="Feature/index directory key, for example clip")
    phase2.add_argument("--checkpoint", help="Hugging Face model name or local checkpoint path")
    phase2.add_argument("--batch-size", type=int, default=64, help="Images per inference batch")
    phase2.add_argument(
        "--device", choices=("auto", "cpu", "gpu", "cuda"), default="auto",
        help="Inference device: auto selects GPU when available; gpu is an alias for cuda",
    )
    phase2.add_argument("--force", action="store_true", help="Rebuild even when outputs validate")
    args = parser.parse_args(argv)
    if args.input is not None and args.batch is not None:
        parser.error("Use either an input path or --batch")
    if args.batch is not None and not re.fullmatch(r"Videos_L\d+", args.batch):
        parser.error("--batch must look like Videos_L21")
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    if args.command == "phase2":
        from .phase2_feature_extraction.clip_embedding import CLIPEmbedding
        from .phase2_feature_extraction.pipeline import (
            DEFAULT_KEYFRAME_ROOT, DEFAULT_OUTPUT_ROOT, DEFAULT_PHASE1_ROOT, run_pipeline,
        )
        input_path = args.input or DEFAULT_PHASE1_ROOT
        if args.batch:
            input_path /= args.batch
        try:
            summary = run_pipeline(
                input_path, keyframe_root=args.keyframe_root or DEFAULT_KEYFRAME_ROOT,
                output_root=args.output_root or DEFAULT_OUTPUT_ROOT,
                model_name=args.checkpoint or CLIPEmbedding.DEFAULT_MODEL,
                model_key=args.model,
                device="cuda" if args.device == "gpu" else args.device,
                batch_size=args.batch_size, force=args.force,
            )
        except (ValueError, OSError) as exc:
            logging.error("%s", exc)
            return 1
        except KeyboardInterrupt:
            logging.error("Processing interrupted; rerun the command to retry")
            return 130
        print("Processed: {processed}; skipped: {skipped}; failed: {failed}".format(**summary))
        return 1 if summary["failed"] else 0

    return run_phase1(args)


if __name__ == "__main__":
    raise SystemExit(main())
