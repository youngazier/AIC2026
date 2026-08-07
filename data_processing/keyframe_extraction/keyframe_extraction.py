import csv
import hashlib
import json
import os
import sys
import subprocess
import cv2
import time
import shutil
import logging
from collections import Counter
from datetime import datetime

# Set up logging
def get_logger():
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(levelname)s - %(message)s',
        handlers=[
            logging.StreamHandler(sys.stdout)
        ]
    )
    return logging.getLogger("keyframe_extractor")

logger = get_logger()

# Append project root to import load_all_video_keyframes_info
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), "../.."))
if project_root not in sys.path:
    sys.path.append(project_root)

from utils.config_loader import load_config
from utils.load_all_video_keyframes_info import load_all_video_keyframes_info
from data_processing.keyframe_extraction.shot_sampling import expected_keyframes

# Mọi đường dẫn lấy từ config.yaml — đừng hardcode, xem ghi chú trong file đó.
PATHS = load_config()["paths"]
VIDEOS_DIR = PATHS["videos_dir"]
SCENES_DIR = PATHS["scenes_dir"]
KEYFRAMES_DIR = PATHS["keyframes_dir"]
MAP_KEYFRAMES_DIR = PATHS["map_keyframes_dir"]

all_video, video_keyframe_dict = load_all_video_keyframes_info(VIDEOS_DIR, KEYFRAMES_DIR)

def detect_video_codec(video_path):
    """Detect video codec using ffprobe"""
    try:
        cmd = ["ffprobe", "-v", "quiet", "-select_streams", "v:0", 
               "-show_entries", "stream=codec_name", "-of", "csv=p=0", video_path]
        result = subprocess.run(cmd, capture_output=True, text=True)
        if result.returncode == 0:
            return result.stdout.strip()
    except Exception as e:
        logger.warning(f"ffprobe failed for {video_path}: {e}")
    return "unknown"

def get_video_fps(video_path):
    """Get video FPS using ffprobe"""
    try:
        cmd = ["ffprobe", "-v", "quiet", "-select_streams", "v:0", 
               "-show_entries", "stream=r_frame_rate", "-of", "csv=p=0", video_path]
        result = subprocess.run(cmd, capture_output=True, text=True)
        if result.returncode == 0:
            fps_str = result.stdout.strip()
            if '/' in fps_str:
                num, den = fps_str.split('/')
                fps = float(num) / float(den) if float(den) != 0 else 25
            else:
                fps = float(fps_str) if fps_str else 25
            return fps
    except Exception as e:
        logger.warning(f"Failed to get FPS for {video_path}: {e}")
        return 25  # default fps
    
SENTINEL_NAME = "_SUCCESS"


def code_version():
    try:
        out = subprocess.run(["git", "-C", project_root, "rev-parse", "--short", "HEAD"],
                             capture_output=True, text=True, timeout=5)
        if out.returncode == 0:
            return out.stdout.strip()
    except Exception:
        pass
    return "unknown"


def read_sentinel(kf_path):
    """Đọc file _SUCCESS trong thư mục keyframe, None nếu không có/không hợp lệ."""
    try:
        with open(os.path.join(kf_path, SENTINEL_NAME), encoding="utf-8") as f:
            return json.load(f)
    except Exception:
        return None


def already_done(kf_path, map_path, expected, scenes_digest):
    """Thư mục CHỈ được coi là xong khi có _SUCCESS khớp với đầu vào hiện tại.

    Quy ước _SUCCESS lấy từ Hadoop/Spark: file này được ghi SAU CÙNG, nên sự tồn
    tại của nó là bằng chứng lần chạy trước đã đi hết đường. Đếm số .jpg thì
    không: job bị kill lúc đang ghi frame thứ 300/571 vẫn để lại một thư mục đầy
    ảnh trông rất bình thường.

    Ngoài số lượng còn so cả digest của scenes.txt — nếu TransNetV2 chạy lại và
    ranh giới shot đổi, keyframe cũ phải bị coi là hết hạn chứ không phải "đã có".
    """
    s = read_sentinel(kf_path)
    return bool(
        s
        and s.get("n_keyframes") == len(expected)
        and s.get("scenes_sha256") == scenes_digest
        and os.path.exists(map_path)
        and os.path.getsize(map_path) > 0
    )


def commit_atomically(staging_kf, staging_map, kf_path, map_path):
    """Đưa kết quả từ vùng staging sang vị trí chính thức bằng rename.

    rename trên cùng filesystem là thao tác nguyên tử: hoặc thư mục cuối cùng
    chưa tồn tại, hoặc nó đã đầy đủ — không bao giờ có trạng thái ghi dở mang
    tên chính thức. Mọi thứ ghi trực tiếp vào tên cuối cùng đều đánh mất tính
    chất này.
    """
    os.makedirs(os.path.dirname(kf_path), exist_ok=True)
    os.makedirs(os.path.dirname(map_path), exist_ok=True)
    if os.path.exists(kf_path):
        shutil.rmtree(kf_path)
    os.replace(staging_map, map_path)
    os.rename(staging_kf, kf_path)


def keyframe_extractor(v):
    """-> 'skipped' | 'ok' | 'incomplete' | 'error'"""
    prefix = v.split("_")[0]

    # Paths configured according to instructions
    video_path = os.path.join(VIDEOS_DIR, f"Videos_{prefix}", f"{v}.mp4")
    file_path = os.path.join(SCENES_DIR, f"{v}.scenes.txt")

    kf_path = os.path.join(KEYFRAMES_DIR, f"Keyframes_{prefix}", v)
    map_path = os.path.join(MAP_KEYFRAMES_DIR, f"{v}.csv")
    staging_kf = kf_path + ".partial"
    staging_map = map_path + ".partial"

    if not os.path.exists(video_path):
        logger.error(f"Video file {video_path} does not exist!")
        return "error"

    if not os.path.exists(file_path):
        logger.error(f"Transnetv2 scenes file {file_path} does not exist!")
        return "error"

    with open(file_path, "rb") as f:
        scenes_bytes = f.read()
    scenes_digest = hashlib.sha256(scenes_bytes).hexdigest()
    lines = scenes_bytes.decode("utf-8", "replace").splitlines(keepends=True)

    # Luật chọn frame nằm ở shot_sampling.py — check_integrity.py dùng CHUNG hàm
    # đó để dựng tập kỳ vọng, nên tuyệt đối đừng chép luật vào đây: hai bản lệch
    # nhau là trình kiểm tra sẽ báo "đủ" trên dữ liệu thiếu.
    expected, _shots, bad_lines = expected_keyframes(lines)
    if bad_lines:
        logger.warning(
            f"{v}: {len(bad_lines)} dòng trong scenes file không parse được, đã bỏ qua "
            f"(dòng {bad_lines[:5]}{'...' if len(bad_lines) > 5 else ''})"
        )

    if already_done(kf_path, map_path, expected, scenes_digest):
        logger.info(f"Keyframes for {v} already complete ({len(expected)} frames). Skipping.")
        return "skipped"

    if not expected:
        logger.warning(f"Scenes file {file_path} không sinh ra keyframe nào.")
        return "error"

    # Dọn vùng staging của lần chạy trước (nếu job cũ bị kill) rồi làm lại từ đầu.
    if os.path.exists(staging_kf):
        shutil.rmtree(staging_kf)
    if os.path.exists(staging_map):
        os.remove(staging_map)
    os.makedirs(staging_kf, exist_ok=True)

    # Check codec
    codec = detect_video_codec(video_path)
    logger.info(f"Video {v} codec: {codec}")
    
    # Default to OpenCV with optimized smart-seeking to avoid FFmpeg command parsing & driver issues
    use_ffmpeg = False
    fps = get_video_fps(video_path)

    cap = None
    if not use_ffmpeg:
        cap = cv2.VideoCapture(video_path)

    # Ghi vào vùng staging, chỉ rename sang tên chính thức khi đã đủ.
    with open(staging_map, "w") as mapping_file:
        mapping = csv.writer(mapping_file)
        mapping.writerow(["n", "shot_idx", "pts_time", "fps", "frame_idx"])

        frame_numbers = [frame_idx for _, _, frame_idx in expected]
        shot_indices = [shot_idx for _, shot_idx, _ in expected]
        output_paths = [os.path.join(staging_kf, f"{n:04d}.jpg") for n, _, _ in expected]

        if use_ffmpeg:
            logger.info(f"Using single-call batch extraction for {codec} codec")
            start_time = time.time()

            # Create a single filter that extracts all frames at once
            frame_selects = "+".join([f"eq(n\\,{fn})" for fn in frame_numbers])
            
            # Build the GPU-accelerated ffmpeg command
            if codec == "av1":
                cmd = [
                    "ffmpeg",
                    "-c:v", "av1_cuvid",
                    "-i", video_path,
                    "-vf", f"select={frame_selects}",
                    "-vsync", "0",
                    "-y", "-loglevel", "quiet"
                ]
            elif codec == "h264":
                cmd = [
                    "ffmpeg", 
                    "-c:v", "h264_cuvid",
                    "-i", video_path,
                    "-vf", f"select={frame_selects}",
                    "-vsync", "0",
                    "-y", "-loglevel", "quiet"
                ]
            elif codec == "hevc":
                cmd = [
                    "ffmpeg",
                    "-c:v", "hevc_cuvid", 
                    "-i", video_path,
                    "-vf", f"select={frame_selects}",
                    "-vsync", "0",
                    "-y", "-loglevel", "quiet"
                ]
            else:
                cmd = [
                    "ffmpeg",
                    "-hwaccel", "cuda",
                    "-i", video_path,
                    "-vf", f"select={frame_selects}", 
                    "-vsync", "0",
                    "-y", "-loglevel", "quiet"
                ]
            
            # Use a temporary directory for batch output
            temp_dir = f"/tmp/batch_extract_{v}"
            os.makedirs(temp_dir, exist_ok=True)
            cmd.append(f"{temp_dir}/frame_%04d.jpg")
            
            logger.info(f"Extracting {len(frame_numbers)} frames in single GPU call...")
            
            try:
                result = subprocess.run(cmd, capture_output=True, timeout=300)  # 5 min timeout
                
                all_results = []
                if result.returncode == 0:
                    # Move files to correct locations
                    for i, (frame_num, output_path, shot_idx) in enumerate(zip(frame_numbers, output_paths, shot_indices)):
                        temp_file = f"{temp_dir}/frame_{i+1:04d}.jpg"
                        if os.path.exists(temp_file):
                            try:
                                os.rename(temp_file, output_path)
                                all_results.append((i + 1, shot_idx, frame_num, True, output_path))
                            except Exception as e:
                                logger.warning(f"Failed to move {temp_file}: {e}")
                                all_results.append((i + 1, shot_idx, frame_num, False, output_path))
                        else:
                            all_results.append((i + 1, shot_idx, frame_num, False, output_path))
                else:
                    # Fallback to single-call CPU batch extraction (much faster than sequential calls)
                    logger.error(f"Batch GPU extraction failed, falling back to CPU batch extraction")
                    cpu_cmd = [
                        "ffmpeg",
                        "-i", video_path,
                        "-vf", f"select={frame_selects}",
                        "-vsync", "0",
                        "-y", "-loglevel", "quiet",
                        f"{temp_dir}/frame_%04d.jpg"
                    ]
                    cpu_result = subprocess.run(cpu_cmd, capture_output=True)
                    
                    all_results = []
                    for i, (frame_num, output_path, shot_idx) in enumerate(zip(frame_numbers, output_paths, shot_indices)):
                        temp_file = f"{temp_dir}/frame_{i+1:04d}.jpg"
                        if cpu_result.returncode == 0 and os.path.exists(temp_file):
                            try:
                                os.rename(temp_file, output_path)
                                all_results.append((i + 1, shot_idx, frame_num, True, output_path))
                            except Exception as e:
                                logger.warning(f"Failed to move {temp_file}: {e}")
                                all_results.append((i + 1, shot_idx, frame_num, False, output_path))
                        else:
                            all_results.append((i + 1, shot_idx, frame_num, False, output_path))
                
                if os.path.exists(temp_dir):
                    shutil.rmtree(temp_dir)
                    
            except subprocess.TimeoutExpired:
                logger.error("Batch extraction timed out")
                all_results = [(i + 1, s_idx, fn, False, op) for i, (fn, op, s_idx) in enumerate(zip(frame_numbers, output_paths, shot_indices))]
            
            total_time = time.time() - start_time
            successful_frames = sum(1 for r in all_results if r[3])
            
            # Write results to mapping file in order
            for idx, shot_idx, frame_num, success, keyframe_path in all_results:
                if success:
                    mapping.writerow([idx, shot_idx, round(frame_num / fps, 4), fps, frame_num])
                    print(f"Saved keyframe {frame_num} of video {v} using single-call ffmpeg")
                else:
                    print(f"Failed to extract frame {frame_num} from video {v} using ffmpeg.", file=sys.stderr)
            
            logger.info(f"{successful_frames}/{len(frame_numbers)} frames in {total_time:.2f}s ({successful_frames/total_time:.1f} fps)")
            
        else:
            # Use OpenCV optimized single-pass sequential reading
            logger.info("Using optimized single-pass OpenCV extraction")
            
            sorted_indices = sorted(range(len(frame_numbers)), key=lambda k: frame_numbers[k])
            
            fps = cap.get(cv2.CAP_PROP_FPS)
            if fps <= 0:
                fps = 25.0
                
            current_frame = 0
            for target_idx in sorted_indices:
                target_frame = frame_numbers[target_idx]
                target_shot = shot_indices[target_idx]
                while current_frame <= target_frame:
                    ret, frame = cap.read()
                    if not ret:
                        break
                    if current_frame == target_frame:
                        output_path = output_paths[target_idx]
                        cv2.imwrite(output_path, frame)
                        mapping.writerow([target_idx + 1, target_shot, round(target_frame / fps, 4), fps, target_frame])
                        print(f"Saved keyframe {target_frame} of video {v} using OpenCV (single-pass)")
                    current_frame += 1

    if cap is not None:
        cap.release()

    # --- Cổng nghiệm thu: chỉ công nhận khi đủ đúng tập kỳ vọng ---
    written = {f[:-4] for f in os.listdir(staging_kf) if f.endswith(".jpg")}
    wanted = {f"{n:04d}" for n, _, _ in expected}
    missing = sorted(wanted - written)

    commit_atomically(staging_kf, staging_map, kf_path, map_path)

    if missing:
        # Vẫn commit phần đã trích để check_integrity.py chỉ ra được frame nào
        # hỏng, nhưng KHÔNG ghi _SUCCESS: thiếu sentinel nghĩa là chưa xong, nên
        # lần chạy sau sẽ tự làm lại video này.
        logger.error(
            f"{v}: chỉ trích được {len(written)}/{len(wanted)} keyframe, thiếu "
            f"{missing[:5]}{'...' if len(missing) > 5 else ''}. Không ghi {SENTINEL_NAME}."
        )
        return "incomplete"

    # _SUCCESS ghi SAU CÙNG, và cũng ghi nguyên tử để không bao giờ tồn tại một
    # sentinel dở dang trông như thật.
    sentinel = {
        "n_keyframes": len(expected),
        "n_shots": len(_shots),
        "scenes_sha256": scenes_digest,
        "code_version": code_version(),
        "created_at": datetime.now().isoformat(timespec="seconds"),
    }
    tmp_sentinel = os.path.join(kf_path, SENTINEL_NAME + ".tmp")
    with open(tmp_sentinel, "w", encoding="utf-8") as f:
        json.dump(sentinel, f, ensure_ascii=False)
    os.replace(tmp_sentinel, os.path.join(kf_path, SENTINEL_NAME))
    return "ok"


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="Extract keyframes from TransNetV2 scene bounds.")
    parser.add_argument("-v", "--video", type=str, help="Process a single video (name or path)")
    parser.add_argument("-f", "--folder", type=str, help="Process a single video folder (e.g., L01 or Videos_L01)")
    parser.add_argument("-a", "--all", action="store_true", help="Process all videos in dataset")
    args = parser.parse_args()

    os.makedirs(MAP_KEYFRAMES_DIR, exist_ok=True)
    os.makedirs(KEYFRAMES_DIR, exist_ok=True)

    targets = []
    if args.video:
        v_name = os.path.splitext(os.path.basename(args.video))[0]
        targets = [v_name]
    elif args.folder:
        raw_folders = args.folder.replace(",", " ").split()
        prefixes = [f.strip().replace("Videos_", "") for f in raw_folders if f.strip()]
        targets = [v for v in all_video if any(v.startswith(p + "_") for p in prefixes)]
    else:
        targets = all_video

    logger.info(f"Starting keyframe extraction process for {len(targets)} video(s) using multiprocessing...")

    from concurrent.futures import ProcessPoolExecutor, as_completed

    def process_video(v):
        start = time.time()
        status = keyframe_extractor(v)
        end = time.time()
        return v, status, end - start

    # Chạy hết rồi báo cáo, KHÔNG dừng ở video lỗi đầu tiên: xử lý theo lô thì
    # điều cần biết là toàn bộ thiệt hại của một lần chạy, chứ không phải khám
    # phá từng lỗi một qua nhiều lần chạy lại.
    results = Counter()
    failed = []
    with ProcessPoolExecutor(max_workers=16) as executor:
        futures = {executor.submit(process_video, v): v for v in targets}
        for future in as_completed(futures):
            v = futures[future]
            try:
                v_name, status, duration = future.result()
                results[status] += 1
                if status in ("error", "incomplete"):
                    failed.append(f"{v_name} ({status})")
                elif status == "ok":
                    logger.info(f"Processed video {v_name} in {duration:.2f} seconds")
            except Exception as e:
                results["crashed"] += 1
                failed.append(f"{v} (crashed: {type(e).__name__}: {e})")
                logger.error(f"Error processing video {v}: {e}")

    logger.info(f"Xong {len(targets)} video: " +
                ", ".join(f"{k}={v}" for k, v in sorted(results.items())))
    if failed:
        logger.error(f"{len(failed)} video KHÔNG hoàn tất: {failed[:10]}"
                     f"{'...' if len(failed) > 10 else ''}")
        logger.error("Chạy lại lệnh này để thử lại (video đã xong sẽ được bỏ qua nhờ _SUCCESS).")
        sys.exit(1)

