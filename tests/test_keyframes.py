import os
import sys
from glob import glob
import pandas as pd

# Append project root to sys.path
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if project_root not in sys.path:
    sys.path.append(project_root)

from utils.config_loader import load_config
from utils.load_all_video_keyframes_info import load_all_video_keyframes_info

def main():
    config = load_config()
    
    # Sử dụng config thay vì hardcode thư mục cũ
    DATASET_DIR = config["paths"]["data_dir"]
    print(f"=== Testing Keyframe Extraction from TransNetV2 in: {DATASET_DIR} ===")
    
    # Check key folders
    scenes_dir = os.path.join(DATASET_DIR, "transnetv2", "scenes")
    keyframes_dir = config["paths"]["keyframes_dir"]
    map_dir = os.path.join(DATASET_DIR, "map_keyframes")
    
    for path, name in [(scenes_dir, "scenes"), (keyframes_dir, "keyframes_transnetv2"), (map_dir, "map_keyframes")]:
        if not os.path.exists(path):
            print(f"ERROR: {name} directory does not exist at {path}!")
            sys.exit(1)
            
    # Find all videos in the dataset based on videos folder
    videos_dir = config["paths"]["videos_dir"]
    video_files = glob(os.path.join(videos_dir, "*", "*.mp4")) + glob(os.path.join(videos_dir, "*.mp4"))
    all_videos = sorted([os.path.basename(v)[:-4] for v in video_files])
    
    # Scan processed videos (those having scenes.txt)
    scene_files = glob(os.path.join(scenes_dir, "*.scenes.txt"))
    processed_videos = sorted([os.path.basename(s).replace(".scenes.txt", "") for s in scene_files])
    
    print(f"Total videos in dataset: {len(all_videos)}")
    print(f"Videos processed by TransNetV2: {len(processed_videos)}")
    
    failures = []
    successes = []
    
    for v in processed_videos:
        prefix = v.split("_")[0]
        scenes_file = os.path.join(scenes_dir, f"{v}.scenes.txt")
        kf_dir = os.path.join(keyframes_dir, f"Keyframes_{prefix}", v)
        map_csv = os.path.join(map_dir, f"{v}.csv")
        
        # 1. Count expected keyframes from scene prediction file
        try:
            with open(scenes_file, "r") as f:
                lines = [line.strip() for line in f.readlines() if line.strip()]
            expected_count = len(lines)
        except Exception as e:
            failures.append({
                "video": v,
                "reason": f"Failed to read scenes file: {e}",
                "expected": 0,
                "images": 0,
                "csv": 0
            })
            continue

        # 2. Count actual extracted images
        if not os.path.exists(kf_dir):
            image_count = 0
            reasons = ["Keyframe folder missing"]
        else:
            image_count = len(glob(os.path.join(kf_dir, "*.jpg")))
            reasons = []

        # 3. Count rows in mapping CSV
        csv_count = 0
        if not os.path.exists(map_csv):
            reasons.append("Mapping CSV missing")
        else:
            try:
                df = pd.read_csv(map_csv)
                csv_count = len(df)
            except Exception as e:
                reasons.append(f"Failed to read CSV: {e}")

        # 4. Check for mismatches
        if image_count != expected_count:
            reasons.append(f"Image count mismatch (expected {expected_count}, got {image_count})")
        if csv_count != expected_count:
            reasons.append(f"CSV row count mismatch (expected {expected_count}, got {csv_count})")
            
        if reasons:
            failures.append({
                "video": v,
                "reason": "; ".join(reasons),
                "expected": expected_count,
                "images": image_count,
                "csv": csv_count
            })
        else:
            successes.append(v)
            
    # Unprocessed videos list
    unprocessed = sorted(list(set(all_videos) - set(processed_videos)))
    
    # Print detailed report
    print("\n" + "="*80)
    print("KEYFRAME EXTRACTION TEST SUMMARY:")
    print(f"Total Videos Checked: {len(processed_videos)}")
    print(f"Verified successfully: {len(successes)}")
    print(f"Failed/Mismatched: {len(failures)}")
    print(f"Unprocessed/Pending: {len(unprocessed)}")
    print("="*80)
    
    if failures:
        print("\nERRORS & MISMATCHES:")
        print("-" * 100)
        print(f"{'Video ID':<15} | {'Expected':<10} | {'Images':<10} | {'CSV Rows':<10} | {'Reason'}")
        print("-" * 100)
        for f in failures:
            print(f"{f['video']:<15} | {f['expected']:<10} | {f['images']:<10} | {f['csv']:<10} | {f['reason']}")
        print("-" * 100)
        sys.exit(1)
    else:
        print("\nAll processed videos are fully consistent and validated successfully!")
        sys.exit(0)

if __name__ == "__main__":
    main()
