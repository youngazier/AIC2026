"""Frame alignment, upstream parity, and filesystem recovery contracts."""

import ast
from fractions import Fraction
import json
from pathlib import Path
import shutil
import subprocess
import sys

import av
import numpy as np
from PIL import Image
import pytest
import torch

from data_processing.cli import main
from data_processing.phase1_keyframe_extraction import pipeline
from data_processing.phase1_keyframe_extraction.keyframes import extract_middle_keyframes, middle_frame
from data_processing.phase1_keyframe_extraction.metadata import decode_for_detection, probe_video
from data_processing.phase1_keyframe_extraction.transnet import validate_arrays
from data_processing.phase1_keyframe_extraction.transnetv2.inference import (
    TransNetV2Torch, predictions_to_scenes, resolve_device,
)
from data_processing.phase2_feature_extraction.pipeline import validate_phase1_input


def make_video(path, pts=(0, 40, 120, 160, 400)):
    path.parent.mkdir(parents=True, exist_ok=True)
    with av.open(str(path), "w") as container:
        stream = container.add_stream("ffv1", rate=25)
        stream.width, stream.height, stream.pix_fmt = 64, 48, "bgr0"
        stream.time_base = stream.codec_context.time_base = Fraction(1, 1000)
        for index, timestamp in enumerate(pts):
            # Distinct solid colors make wrong-frame extraction observable.
            pixels = np.full((48, 64, 3), (index * 45, 20, 200 - index * 30), dtype=np.uint8)
            frame = av.VideoFrame.from_ndarray(pixels, format="rgb24")
            frame.pts, frame.time_base = timestamp, Fraction(1, 1000)
            for packet in stream.encode(frame):
                container.mux(packet)
        for packet in stream.encode():
            container.mux(packet)
    return path


class FakeDetector:
    calls = 0
    device = "cpu"
    visualize_predictions = staticmethod(TransNetV2Torch.visualize_predictions)

    def __init__(self, *args, **kwargs):
        pass

    def predict_frames(self, frames):
        type(self).calls += 1
        single = np.zeros(len(frames), dtype=np.float32)
        single[2] = 0.9
        return single, single.copy()


@pytest.fixture
def dataset(tmp_path, monkeypatch):
    monkeypatch.setattr(pipeline, "PROJECT_ROOT", tmp_path)
    monkeypatch.setattr(pipeline, "TransNetV2Torch", FakeDetector)
    FakeDetector.calls = 0
    video = make_video(tmp_path / "data/videos/Videos_L21/L21_V001.mkv")
    organizer = tmp_path / "data/media-info/L21_V001.json"
    organizer.parent.mkdir(parents=True)
    organizer.write_text('{"title": "original"}')
    weights = tmp_path / "model.pth"
    weights.write_bytes(b"test checkpoint")
    return video, tmp_path / "data_processed/transnetv2", weights


def run(dataset, **kwargs):
    video, output, weights = dataset
    return pipeline.run_pipeline(video, output_root=output, weights=weights, **kwargs)


def result_dir(dataset):
    _, output, _ = dataset
    return output / "Videos_L21/L21_V001"


def keyframe_dir(dataset):
    _, output, _ = dataset
    return output.parent / "keyframes/Videos_L21/L21_V001"


@pytest.mark.parametrize(("start", "end", "expected"), [(0, 0, 0), (4, 5, 4), (100, 160, 130)])
def test_middle_selection(start, end, expected):
    assert middle_frame(start, end) == expected


@pytest.mark.parametrize(("probabilities", "expected"), [
    ([0, 0, 0], [[0, 2]]), ([0, 1, 1, 0, 0], [[0, 1], [3, 4]]),
    ([0, .5, 0], [[0, 2]]), ([1, 1], [[0, 1]]), ([0], [[0, 0]]),
])
def test_upstream_scene_semantics(probabilities, expected):
    assert predictions_to_scenes(np.array(probabilities)).tolist() == expected


@pytest.mark.parametrize("probabilities", [[], [np.nan], [np.inf], [-.1], [1.1]])
def test_reject_invalid_probabilities(probabilities):
    with pytest.raises(ValueError):
        predictions_to_scenes(np.array(probabilities))


def test_device_selection(monkeypatch):
    monkeypatch.setattr(torch.cuda, "is_available", lambda: True)
    assert resolve_device("auto").type == "cuda"
    assert resolve_device("gpu").type == "cuda"
    assert resolve_device("cuda").type == "cuda"
    assert resolve_device("cpu").type == "cpu"
    monkeypatch.setattr(torch.cuda, "is_available", lambda: False)
    assert resolve_device("auto").type == "cpu"
    with pytest.raises(ValueError, match="unavailable"):
        resolve_device("gpu")
    with pytest.raises(ValueError, match="Device must"):
        resolve_device("tpu")


@pytest.mark.parametrize("length", [1, 2, 49, 50, 51, 99, 100, 101])
def test_window_predictions_match_upstream(length):
    # Execute only upstream predict_frames, avoiding checkpoint/GPU dependencies.
    upstream = Path(__file__).resolve().parents[1] / "test/transnetv2_pytorch/transnetv2_pytorch/inference.py"
    if not upstream.exists():
        pytest.skip("Optional upstream checkout is unavailable")
    tree = ast.parse(upstream.read_text())
    method = next(n for n in ast.walk(tree) if isinstance(n, ast.FunctionDef) and n.name == "predict_frames")
    scope = {"np": np, "torch": torch}
    exec(compile(ast.Module(body=[method], type_ignores=[]), str(upstream), "exec"), scope)
    frames = np.random.default_rng(length).integers(0, 256, size=(length, 27, 48, 3), dtype=np.uint8)

    def predict_raw(tensor):
        # Depends on both frame content and context to reveal padding/window changes.
        values = tensor.numpy().mean(axis=(2, 3, 4)) / 255
        values = values + values.mean(axis=1, keepdims=True)
        values = values[..., None].astype(np.float32)
        return values, -values

    detector = object.__new__(TransNetV2Torch)
    detector.device, detector._input_size, detector.window_buf = "cpu", (27, 48, 3), None
    detector.predict_raw = predict_raw
    expected = scope["predict_frames"](detector, frames)
    actual = detector.predict_frames(frames)
    for left, right in zip(expected, actual, strict=True):
        np.testing.assert_allclose(left, right, rtol=1e-6, atol=1e-7)
        assert len(right) == length


@pytest.mark.parametrize("pts", [(0, 40, 80, 120, 160), (0, 40, 120, 160, 400)])
def test_decode_extract_alignment_and_full_resolution(tmp_path, pts):
    video = make_video(tmp_path / "source.mkv", pts)
    metadata = probe_video(video)
    frames, timestamps = decode_for_detection(video, metadata)
    assert frames.shape == (5, 27, 48, 3)
    np.testing.assert_allclose(timestamps, np.array(pts) / 1000, atol=1e-6)
    # Covers one-frame, two-frame, and ordinary shots.
    scenes = np.array([[0, 0], [1, 2], [3, 4]])
    images = tmp_path / "keyframes/test"
    records = extract_middle_keyframes(video, metadata, scenes, timestamps, images, "keyframes/test")
    assert [r["keyframe"]["frame_index"] for r in records] == [0, 1, 3]
    for record in records:
        index = record["keyframe"]["frame_index"]
        with Image.open(images / Path(record["keyframe"]["image_path"]).name) as image:
            assert image.size == (64, 48)
            np.testing.assert_allclose(np.asarray(image)[20, 20],
                                       [index * 45, 20, 200 - index * 30], atol=3)


def test_decode_rejects_mismatched_count_and_timing(tmp_path):
    video = make_video(tmp_path / "source.mkv")
    metadata = probe_video(video)
    _, timestamps = decode_for_detection(video, metadata)
    for bad in [timestamps[:-1], np.append(timestamps, .5), timestamps + .01]:
        with pytest.raises(ValueError, match="mismatch|differ"):
            extract_middle_keyframes(
                video, metadata, np.array([[0, 1]]), bad,
                tmp_path / "keyframes/test", "keyframes/test",
            )


def test_pipeline_rerun_does_not_copy_organizer_metadata(dataset):
    assert run(dataset) == {"processed": 1, "skipped": 0, "failed": 0}
    original = result_dir(dataset)
    video, _, _ = dataset
    assert not (original / "organizer_metadata.json").exists()
    organizer = video.parents[2] / "media-info/L21_V001.json"
    organizer.write_text('{"title": "updated"}')
    assert run(dataset) == {"processed": 0, "skipped": 1, "failed": 0}
    assert result_dir(dataset) == original
    assert FakeDetector.calls == 1
    assert not (original / "organizer_metadata.json").exists()
    organizer.write_text("malformed")
    assert run(dataset)["skipped"] == 1
    assert not (original / "organizer_metadata.json").exists()
    assert run(dataset, force=True)["processed"] == 1
    assert not (original / "organizer_metadata.json").exists()


@pytest.mark.parametrize("artifact", ["L21_V001.predictions.txt", "L21_V001.scenes.txt",
                                      "shots.json"])
def test_partial_or_corrupt_outputs_are_rebuilt(dataset, artifact):
    run(dataset)
    result = result_dir(dataset)
    (result / artifact).write_bytes(b"corrupt")
    assert run(dataset)["processed"] == 1
    assert result_dir(dataset) == result
    assert (result / artifact).read_bytes() != b"corrupt"


def test_corrupt_keyframe_is_rebuilt(dataset):
    run(dataset)
    image = keyframe_dir(dataset) / "00000.jpg"
    image.write_bytes(b"corrupt")
    assert run(dataset)["processed"] == 1
    assert image.read_bytes() != b"corrupt"


def test_only_requested_artifacts_are_written(dataset):
    run(dataset)
    result = result_dir(dataset)
    assert {path.name for path in result.iterdir()} == {
        "L21_V001.predictions.txt", "L21_V001.scenes.txt", "shots.json",
    }
    assert not (dataset[1].parent / "map-keyframes").exists()
    saved = json.loads((result / "shots.json").read_text())
    assert saved["metadata"]["width"] == 64
    assert saved["frame_count"] == 5
    assert saved["shots"][1]["end_timestamp"] == .4
    assert saved["shots"][1]["keyframe"]["frame_index"] == 3
    assert saved["shots"][1]["keyframe"]["timestamp"] == .16
    _, output, _ = dataset
    header, records, images = validate_phase1_input(result, output.parent / "keyframes")
    assert header["video_name"] == "L21_V001"
    assert records == saved["shots"]
    assert [image.name for image in images] == ["00000.jpg", "00001.jpg"]


def test_requesting_visualization_rebuilds_without_force(dataset):
    run(dataset)
    assert run(dataset, visualize=True)["processed"] == 1
    assert {path.name for path in result_dir(dataset).iterdir()} == {
        "L21_V001.predictions.txt", "L21_V001.scenes.txt", "shots.json", "L21_V001.vis.png",
    }
    assert run(dataset, visualize=True)["skipped"] == 1
    assert run(dataset, force=True)["processed"] == 1
    assert not (result_dir(dataset) / "L21_V001.vis.png").exists()


def test_legacy_extra_files_are_removed_on_rebuild(dataset):
    run(dataset)
    result = result_dir(dataset)
    for name in ("manifest.json", "metadata.json", "L21_V001.timestamps.npy"):
        (result / name).write_bytes(b"legacy")
    assert run(dataset)["processed"] == 1
    assert {path.name for path in result.iterdir()} == {
        "L21_V001.predictions.txt", "L21_V001.scenes.txt", "shots.json",
    }


@pytest.mark.parametrize("change", ["timestamp", "boundary", "image_path"])
def test_invalid_shot_records_are_rebuilt(dataset, change):
    run(dataset)
    path = result_dir(dataset) / "shots.json"
    saved = json.loads(path.read_text())
    shot = saved["shots"][0]
    if change == "timestamp":
        shot["keyframe"]["timestamp"] = -1
    elif change == "boundary":
        shot["end_frame"] += 1
    else:
        shot["keyframe"]["image_path"] = "wrong.jpg"
    path.write_text(json.dumps(saved))
    assert run(dataset)["processed"] == 1


def test_detector_only_then_phase1_and_force(dataset):
    assert run(dataset, mode="transnetv2")["processed"] == 1
    assert not keyframe_dir(dataset).exists()
    assert run(dataset, mode="phase1")["processed"] == 1
    result = result_dir(dataset)
    assert len(list(keyframe_dir(dataset).glob("*.jpg"))) == 2
    assert run(dataset, mode="transnetv2")["skipped"] == 1
    assert run(dataset, force=True, visualize=True)["processed"] == 1
    assert (result / "L21_V001.vis.png").is_file()


def test_failed_rebuild_preserves_previous_result(dataset, monkeypatch):
    run(dataset)
    result = result_dir(dataset)
    previous_shots = (result / "shots.json").read_bytes()

    def fail(*args, **kwargs):
        raise RuntimeError("extraction failed")

    monkeypatch.setattr(pipeline, "extract_middle_keyframes", fail)
    assert run(dataset, force=True)["failed"] == 1
    assert (result / "shots.json").read_bytes() == previous_shots
    assert not (dataset[1] / ".work").exists()


def test_failed_publication_restores_results_and_images(dataset, monkeypatch):
    run(dataset)
    result = result_dir(dataset)
    old_shots = (result / "shots.json").read_bytes()
    old_image = (keyframe_dir(dataset) / "00000.jpg").read_bytes()
    original_rename = Path.rename

    def fail_keyframe_publication(path, target):
        if path.name == "keyframes" and ".work" in path.parts:
            raise OSError("publication failed")
        return original_rename(path, target)

    monkeypatch.setattr(Path, "rename", fail_keyframe_publication)
    assert run(dataset, force=True, threshold=.8)["failed"] == 1
    assert (result / "shots.json").read_bytes() == old_shots
    assert (keyframe_dir(dataset) / "00000.jpg").read_bytes() == old_image
    assert not (dataset[1] / ".work").exists()


def test_docker_entrypoint_works_without_parent_package(tmp_path):
    source = Path(pipeline.__file__).parent
    target = tmp_path / "data_processing/phase1_keyframe_extraction"
    shutil.copytree(source, target, ignore=shutil.ignore_patterns("__pycache__", "*.pth"))
    result = subprocess.run(
        [sys.executable, "-S", "-m", "data_processing.phase1_keyframe_extraction.cli", "phase1", "--help"],
        cwd=tmp_path, capture_output=True, text=True,
    )
    assert result.returncode == 0, result.stderr
    assert "--visualize" in result.stdout


def test_source_and_configuration_changes_invalidate_run(dataset):
    run(dataset)
    previous_shots = (result_dir(dataset) / "shots.json").read_bytes()
    assert run(dataset, threshold=.8)["processed"] == 1
    assert (result_dir(dataset) / "shots.json").read_bytes() != previous_shots
    video, _, _ = dataset
    make_video(video, pts=(0, 40, 80, 120, 160))
    assert run(dataset, threshold=.8)["processed"] == 1


def test_threshold_comparison_survives_float32_serialization(dataset):
    # float32(0.9) must retain the original converter's equality semantics.
    assert run(dataset, threshold=.9)["processed"] == 1
    assert run(dataset, threshold=.9)["skipped"] == 1


def test_batch_continues_after_bad_video(dataset):
    video, output, weights = dataset
    bad = video.with_name("L21_V002.mp4")
    bad.write_bytes(b"not a video")
    summary = pipeline.run_pipeline(video.parent, output_root=output, weights=weights)
    assert summary == {"processed": 1, "skipped": 0, "failed": 1}


def test_cli_exit_codes_and_data_protection(dataset):
    video, output, weights = dataset
    assert main(["phase1", str(video), "--output-root", str(output), "--weights", str(weights)]) == 0
    assert main(["phase1", str(video), "--output-root", str(video.parent), "--weights", str(weights)]) == 1
    assert main(["phase1", str(video), "--output-root", str(output), "--weights", str(weights), "--threshold", "nan"]) == 1
    assert main(["phase1", str(video.with_name("absent.mp4")), "--weights", str(weights)]) == 1


def test_validation_rejects_invalid_bounds_and_probabilities():
    with pytest.raises(ValueError):
        validate_arrays(np.zeros((2, 2)), np.array([[0, 2]]), np.array([0., .04]), .5)
    with pytest.raises(ValueError):
        validate_arrays(np.array([[np.nan, 0]]), np.array([[0, 0]]), np.array([0.]), .5)
