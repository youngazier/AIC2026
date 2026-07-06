"""
Kiểm tra pipeline keyframe extraction (TransNetV2), gồm 3 phần:

1. ĐẾM KEYFRAME DETECT ĐƯỢC: mỗi dòng trong data/transnetv2/scenes/{video}.scenes.txt
   là 1 scene -> sinh đúng 1 keyframe. Tổng hợp số video / số scene theo từng
   folder batch (L21, L23, L26, ...).

2. KIỂM TRA ĐÃ TRÍCH ĐỦ KEYFRAME CHƯA: với từng video, folder
   data/keyframes/Keyframes_{Lxx}/{video}/ phải có đúng N ảnh .jpg
   và tên file phải đủ bộ 0001.jpg .. NNNN.jpg (không thiếu, không thừa).

3. KIỂM TRA MAP_KEYFRAMES: data/map_keyframes/{video}.csv phải:
   - đủ N dòng, cột `n` chạy đúng 1..N
   - `frame_idx` nằm trong khoảng [start, end] của scene tương ứng và tăng dần
   - `pts_time` khớp công thức frame_idx / fps

Exit code 0 nếu mọi video đều nhất quán, 1 nếu có lỗi.
"""
import os
import sys
from collections import defaultdict
from glob import glob

import pandas as pd

# Append project root to sys.path
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if project_root not in sys.path:
    sys.path.append(project_root)

from utils.config_loader import load_config

PTS_TIME_TOLERANCE = 0.05  # giây — dung sai làm tròn khi ghi pts_time vào CSV


def read_scenes(scenes_file):
    """Đọc file .scenes.txt -> list [(start_frame, end_frame)]."""
    scenes = []
    with open(scenes_file, "r") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            start, end = line.split()
            scenes.append((int(start), int(end)))
    return scenes


def check_keyframe_images(kf_dir, expected_count):
    """Phần 2: đủ số ảnh và đủ bộ tên 0001.jpg..NNNN.jpg. Trả về list lỗi."""
    if not os.path.isdir(kf_dir):
        return 0, ["Keyframe folder missing"]

    jpg_names = {os.path.basename(p) for p in glob(os.path.join(kf_dir, "*.jpg"))}
    errors = []
    if len(jpg_names) != expected_count:
        errors.append(f"Image count mismatch (expected {expected_count}, got {len(jpg_names)})")

    expected_names = {f"{i:04d}.jpg" for i in range(1, expected_count + 1)}
    missing = sorted(expected_names - jpg_names)
    extra = sorted(jpg_names - expected_names)
    if missing:
        errors.append(f"{len(missing)} missing image(s), e.g: {missing[:3]}")
    if extra:
        errors.append(f"{len(extra)} unexpected image name(s), e.g: {extra[:3]}")
    return len(jpg_names), errors


def check_mapping_csv(map_csv, scenes):
    """Phần 3: CSV đủ dòng, n = 1..N, frame_idx trong scene + tăng dần, pts_time = frame_idx/fps."""
    expected_count = len(scenes)
    if not os.path.exists(map_csv):
        return 0, ["Mapping CSV missing"]
    try:
        df = pd.read_csv(map_csv)
    except Exception as e:
        return 0, [f"Failed to read CSV: {e}"]

    errors = []
    if len(df) != expected_count:
        errors.append(f"CSV row count mismatch (expected {expected_count}, got {len(df)})")

    required_cols = {"n", "pts_time", "fps", "frame_idx"}
    missing_cols = required_cols - set(df.columns)
    if missing_cols:
        errors.append(f"CSV missing column(s): {sorted(missing_cols)}")
        return len(df), errors

    # Cột n phải chạy đúng 1..N
    if list(df["n"]) != list(range(1, len(df) + 1)):
        errors.append("Column 'n' is not a clean 1..N sequence")

    # frame_idx phải tăng dần nghiêm ngặt
    frame_idx = df["frame_idx"].tolist()
    if any(b <= a for a, b in zip(frame_idx, frame_idx[1:])):
        errors.append("Column 'frame_idx' is not strictly increasing")

    # Từng frame_idx phải nằm trong scene tương ứng, pts_time khớp frame_idx/fps
    out_of_scene, bad_pts = [], []
    for i, row in enumerate(df.itertuples(index=False)):
        if i < expected_count:
            start, end = scenes[i]
            if not (start <= row.frame_idx <= end):
                out_of_scene.append(f"row {i + 1}: frame_idx {row.frame_idx} not in scene [{start}, {end}]")
        if row.fps > 0 and abs(row.pts_time - row.frame_idx / row.fps) > PTS_TIME_TOLERANCE:
            bad_pts.append(f"row {i + 1}: pts_time {row.pts_time} != frame_idx/fps {row.frame_idx / row.fps:.4f}")
    if out_of_scene:
        errors.append(f"{len(out_of_scene)} frame_idx outside their scene, e.g: {out_of_scene[:3]}")
    if bad_pts:
        errors.append(f"{len(bad_pts)} pts_time mismatch(es), e.g: {bad_pts[:3]}")

    return len(df), errors


def main():
    config = load_config()
    data_dir = config["paths"]["data_dir"]
    scenes_dir = os.path.join(data_dir, "transnetv2", "scenes")
    keyframes_dir = config["paths"]["keyframes_dir"]
    map_dir = os.path.join(data_dir, "map_keyframes")
    videos_dir = config["paths"]["videos_dir"]

    print(f"=== Testing Keyframe Extraction (TransNetV2) in: {data_dir} ===")
    for path, name in [(scenes_dir, "transnetv2/scenes"), (keyframes_dir, "keyframes"), (map_dir, "map_keyframes")]:
        if not os.path.exists(path):
            print(f"ERROR: {name} directory does not exist at {path}!")
            sys.exit(1)

    # ------ PHẦN 1: đếm scene detect được, tổng hợp theo folder batch ------
    scene_files = sorted(glob(os.path.join(scenes_dir, "*.scenes.txt")))
    video_scenes = {}  # video_id -> list [(start, end)]
    for sf in scene_files:
        video_id = os.path.basename(sf).replace(".scenes.txt", "")
        try:
            video_scenes[video_id] = read_scenes(sf)
        except Exception as e:
            print(f"WARNING: cannot parse {sf}: {e}")

    # batch (L21, L23...) -> thống kê
    stats = defaultdict(lambda: {"videos": 0, "scenes": 0, "images": 0, "csv_rows": 0})
    for v, scenes in video_scenes.items():
        prefix = v.split("_")[0]
        stats[prefix]["videos"] += 1
        stats[prefix]["scenes"] += len(scenes)

    # ------ PHẦN 2 + 3: kiểm tra từng video ------
    failures = []
    for v in sorted(video_scenes):
        prefix = v.split("_")[0]
        scenes = video_scenes[v]
        expected = len(scenes)

        kf_dir = os.path.join(keyframes_dir, f"Keyframes_{prefix}", v)
        image_count, img_errors = check_keyframe_images(kf_dir, expected)
        stats[prefix]["images"] += image_count

        map_csv = os.path.join(map_dir, f"{v}.csv")
        csv_count, csv_errors = check_mapping_csv(map_csv, scenes)
        stats[prefix]["csv_rows"] += csv_count

        reasons = img_errors + csv_errors
        if reasons:
            failures.append({
                "video": v,
                "expected": expected,
                "images": image_count,
                "csv": csv_count,
                "reason": "; ".join(reasons),
            })

    # Video có file .mp4 nhưng TransNetV2 chưa xử lý
    video_files = glob(os.path.join(videos_dir, "*", "*.mp4")) + glob(os.path.join(videos_dir, "*.mp4"))
    all_videos = {os.path.basename(p)[:-4] for p in video_files}
    unprocessed = sorted(all_videos - set(video_scenes))

    # ------ BÁO CÁO ------
    print("\nSCENES DETECTED PER BATCH FOLDER:")
    print("-" * 72)
    print(f"{'Batch':<10} | {'Videos':>8} | {'Scenes':>10} | {'Images':>10} | {'CSV rows':>10}")
    print("-" * 72)
    total = {"videos": 0, "scenes": 0, "images": 0, "csv_rows": 0}
    for prefix in sorted(stats):
        s = stats[prefix]
        print(f"{prefix:<10} | {s['videos']:>8} | {s['scenes']:>10} | {s['images']:>10} | {s['csv_rows']:>10}")
        for k in total:
            total[k] += s[k]
    print("-" * 72)
    print(f"{'TOTAL':<10} | {total['videos']:>8} | {total['scenes']:>10} | {total['images']:>10} | {total['csv_rows']:>10}")

    print("\n" + "=" * 72)
    print("KEYFRAME EXTRACTION TEST SUMMARY:")
    print(f"Total videos with scene detection: {len(video_scenes)}")
    print(f"Verified successfully: {len(video_scenes) - len(failures)}")
    print(f"Failed/Mismatched: {len(failures)}")
    print(f"Unprocessed/Pending (has .mp4 but no scenes file): {len(unprocessed)}")
    if unprocessed:
        print(f"  e.g: {unprocessed[:5]}")
    print("=" * 72)

    if failures:
        print("\nERRORS & MISMATCHES:")
        print("-" * 110)
        print(f"{'Video ID':<15} | {'Expected':<10} | {'Images':<10} | {'CSV Rows':<10} | {'Reason'}")
        print("-" * 110)
        for f in failures:
            print(f"{f['video']:<15} | {f['expected']:<10} | {f['images']:<10} | {f['csv']:<10} | {f['reason']}")
        print("-" * 110)
        sys.exit(1)

    print("\nAll processed videos are fully consistent: scene count == image count == CSV mapping!")
    sys.exit(0)


if __name__ == "__main__":
    main()
