"""Command-line entry point for offline preprocessing phases."""

import argparse
import logging
from pathlib import Path

from .selectors import add_selector_arguments, select_videos


def main(argv=None) -> int:
    from .phase1_keyframe_extraction.cli import add_commands, run as run_phase1

    parser = argparse.ArgumentParser(description="Offline video preprocessing")
    commands = parser.add_subparsers(dest="command", required=True)
    add_commands(commands)
    phase2 = commands.add_parser("phase2", help="Encode completed Phase 1 keyframes")
    add_selector_arguments(phase2, "Phase 1 result directory or video path")
    phase2.add_argument("--keyframe-root", type=Path, help="Root containing Phase 1 keyframe batches")
    phase2.add_argument("--output-root", type=Path, help="Embedding output root")
    phase2.add_argument("--model", default="clip",
                        help="Registered model key, for example clip")
    phase2.add_argument("--checkpoint", help="Hugging Face model name or local checkpoint path")
    phase2.add_argument("--batch-size", type=int, default=64, help="Images per inference batch")
    phase2.add_argument(
        "--device", choices=("auto", "cpu", "gpu", "cuda"), default="auto",
        help="Inference device: auto selects GPU when available; gpu is an alias for cuda",
    )
    phase2.add_argument("--force", action="store_true", help="Rebuild even when outputs validate")

    phase3 = commands.add_parser("phase3", help="Build FAISS indexes and import the SQLite catalog")
    phase3.add_argument("--model", default="clip", help="Registered model key, for example clip")
    phase3.add_argument("--no-playlists", action="store_true", help="Skip playlist indexes")
    phase3.add_argument("--force", action="store_true", help="Rebuild indexes even when valid")
    phase3.add_argument("--no-index", action="store_true", help="Skip FAISS index construction")
    phase3.add_argument("--no-catalog", action="store_true", help="Skip SQLite catalog import")

    args = parser.parse_args(argv)
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    if args.command == "phase3":
        from .phase3 import run_phase3

        try:
            return run_phase3(
                model_key=args.model,
                no_playlists=args.no_playlists,
                force=args.force,
                no_index=args.no_index,
                no_catalog=args.no_catalog,
            )
        except (ValueError, OSError) as exc:
            logging.error("%s", exc)
            return 1
        except KeyboardInterrupt:
            logging.error("Processing interrupted; rerun the command to retry")
            return 130

    if args.command == "phase2":
        from .phase2_feature_extraction import pipeline as phase2_pipeline
        from .phase2_feature_extraction.models import resolve_model

        try:
            embedder_cls, model_name = resolve_model(args.model, args.checkpoint)
            if (args.input is not None and not args.video and not args.batch
                    and not args.all_videos
                    and (args.input.is_dir() or args.input.name in {"shots.json", "manifest.json"})):
                selected: Path | list[Path] = args.input
            else:
                selected = select_videos(
                    input_path=args.input,
                    video=args.video,
                    batch=args.batch,
                    all_videos=args.all_videos,
                )
            summary = phase2_pipeline.run_pipeline(
                selected,
                phase1_root=phase2_pipeline.DEFAULT_PHASE1_ROOT,
                keyframe_root=args.keyframe_root or phase2_pipeline.DEFAULT_KEYFRAME_ROOT,
                output_root=args.output_root or phase2_pipeline.DEFAULT_OUTPUT_ROOT,
                model_name=model_name,
                model_key=args.model,
                device="cuda" if args.device == "gpu" else args.device,
                batch_size=args.batch_size,
                force=args.force,
                embedder_factory=embedder_cls,
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
