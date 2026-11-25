"""Command-line entry point for offline preprocessing phases."""

import argparse
import logging
from pathlib import Path

from .selectors import add_selector_arguments, select_phase1_results, select_videos


def _add_phase2_options(parser, *, include_io_roots: bool = True) -> None:
    if include_io_roots:
        parser.add_argument("--keyframe-root", type=Path, help="Root containing Phase 1 keyframe batches")
        parser.add_argument("--output-root", type=Path, help="Embedding output root")
    parser.add_argument("--model", default="clip", help="Registered model key, for example clip")
    parser.add_argument("--checkpoint", help="Hugging Face model name or local checkpoint path")
    parser.add_argument("--batch-size", type=int, default=512, help="Images per inference batch")
    parser.add_argument(
        "--workers",
        type=int,
        default=None,
        help="DataLoader worker processes (default: min(4, cpu_count - 1) on GPU, 0 on CPU)",
    )
    parser.add_argument(
        "--precision",
        choices=("fp32", "fp16"),
        default="fp32",
        help="Inference precision (always normalized and stored as float32)",
    )
    parser.add_argument(
        "--device",
        choices=("auto", "cpu", "gpu", "cuda"),
        default="auto",
        help="Inference device: auto selects GPU when available; gpu is an alias for cuda",
    )
    parser.add_argument("--force", action="store_true", help="Rebuild even when outputs validate")
    parser.add_argument("--verify", action="store_true", help="Deep skip check (decode every keyframe)")


def _run_all(args) -> int:
    from .phase1_keyframe_extraction import pipeline as phase1_pipeline
    from .phase2_feature_extraction import pipeline as phase2_pipeline
    from .phase2_feature_extraction.models import resolve_model
    from .phase3 import run_phase3

    embedder_cls, model_name = resolve_model(args.model, args.checkpoint)
    videos = select_videos(
        input_path=args.input,
        video=args.video,
        batch=args.batch,
        all_videos=args.all_videos,
    )
    summary1 = phase1_pipeline.run_pipeline(
        videos,
        output_root=phase1_pipeline.DEFAULT_OUTPUT_ROOT,
        weights=phase1_pipeline.DEFAULT_WEIGHTS,
        mode="phase1",
        device=args.device,
        force=args.force,
        verify=args.verify,
    )
    print("Phase 1: processed {processed}, skipped {skipped}, failed {failed}".format(**summary1))

    phase2_video = (
        [phase1_pipeline.video_identity(path)[1] for path in videos]
        if args.input is not None
        else args.video
    )
    selected, missing_videos = select_phase1_results(
        video=phase2_video,
        batch=None if args.input is not None else args.batch,
        all_videos=False if args.input is not None else args.all_videos,
        phase1_root=phase2_pipeline.DEFAULT_PHASE1_ROOT,
    )
    summary2 = phase2_pipeline.run_pipeline(
        selected,
        missing_videos=missing_videos,
        keyframe_root=phase2_pipeline.DEFAULT_KEYFRAME_ROOT,
        output_root=phase2_pipeline.DEFAULT_OUTPUT_ROOT,
        model_name=model_name,
        model_key=args.model,
        device="cuda" if args.device == "gpu" else args.device,
        batch_size=args.batch_size,
        workers=args.workers,
        precision=args.precision,
        force=args.force,
        verify=args.verify,
        embedder_factory=embedder_cls,
    )
    print("Phase 2: processed {processed}, skipped {skipped}, failed {failed}".format(**summary2))

    phase3_code = 0
    if not args.no_phase3:
        phase3_code = run_phase3(
            model_key=args.model,
            no_playlists=args.no_playlists,
            force=args.force,
        )

    if summary1["failed"] or summary2["failed"] or phase3_code != 0:
        return 1
    return 0


def main(argv=None) -> int:
    from .phase1_keyframe_extraction.cli import add_commands, run as run_phase1

    parser = argparse.ArgumentParser(description="Offline video preprocessing")
    commands = parser.add_subparsers(dest="command", required=True)
    add_commands(commands)

    phase2 = commands.add_parser("phase2", help="Encode completed Phase 1 keyframes")
    add_selector_arguments(phase2, "Phase 1 result directory or video path")
    _add_phase2_options(phase2, include_io_roots=True)

    phase3 = commands.add_parser("phase3", help="Build FAISS indexes and import the SQLite catalog")
    phase3.add_argument("--model", default="clip", help="Registered model key, for example clip")
    phase3.add_argument("--no-playlists", action="store_true", help="Skip playlist indexes")
    phase3.add_argument("--force", action="store_true", help="Rebuild indexes even when valid")
    phase3.add_argument("--no-index", action="store_true", help="Skip FAISS index construction")
    phase3.add_argument("--no-catalog", action="store_true", help="Skip SQLite catalog import")

    all_cmd = commands.add_parser("all", help="Run Phase 1, Phase 2, and Phase 3 in one go")
    add_selector_arguments(all_cmd, "Video file or directory")
    _add_phase2_options(all_cmd, include_io_roots=False)
    all_cmd.add_argument("--no-phase3", action="store_true", help="Skip index construction and catalog import")
    all_cmd.add_argument("--no-playlists", action="store_true", help="Skip playlist indexes in Phase 3")

    args = parser.parse_args(argv)
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    if args.command == "all":
        try:
            return _run_all(args)
        except (ValueError, OSError) as exc:
            logging.error("%s", exc)
            return 1
        except KeyboardInterrupt:
            logging.error("Processing interrupted; rerun the command to retry")
            return 130

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
            selected, missing_videos = select_phase1_results(
                input_path=args.input,
                video=args.video,
                batch=args.batch,
                all_videos=args.all_videos,
                phase1_root=phase2_pipeline.DEFAULT_PHASE1_ROOT,
            )
            summary = phase2_pipeline.run_pipeline(
                selected,
                missing_videos=missing_videos,
                keyframe_root=args.keyframe_root or phase2_pipeline.DEFAULT_KEYFRAME_ROOT,
                output_root=args.output_root or phase2_pipeline.DEFAULT_OUTPUT_ROOT,
                model_name=model_name,
                model_key=args.model,
                device="cuda" if args.device == "gpu" else args.device,
                batch_size=args.batch_size,
                workers=args.workers,
                precision=args.precision,
                force=args.force,
                verify=args.verify,
                embedder_factory=embedder_cls,
            )
        except (ValueError, OSError) as exc:
            logging.error("%s", exc)
            return 1
        except KeyboardInterrupt:
            logging.error("Processing interrupted; rerun the command to retry")
            return 130
        print("processed {processed}, skipped {skipped}, failed {failed}".format(**summary))
        return 1 if summary["failed"] else 0

    return run_phase1(args)


if __name__ == "__main__":
    raise SystemExit(main())
