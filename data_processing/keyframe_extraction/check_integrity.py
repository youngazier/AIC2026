#!/usr/bin/env python3
"""[5] Kiểm tra tính TOÀN VẸN và ĐẦY ĐỦ của toàn bộ pipeline tiền xử lý.

    manifest -> transnet -> keyframe -> features -> index

NGUYÊN TẮC: tập kỳ vọng (expected set) phải độc lập với đầu ra được kiểm tra.
Bản cũ của script này lấy danh sách công việc từ `os.listdir(map_keyframes/)`,
tức là từ chính đầu ra — video nào chết hẳn ở TransNetV2 sẽ không sinh ra csv,
nên không xuất hiện trong vòng lặp, và report vẫn in SUCCESS. Ở đây danh sách
công việc đến từ manifest (dựng từ video gốc bằng ffprobe), còn tập keyframe kỳ
vọng đến từ scenes.txt qua utils/shot_sampling.py — cùng hàm mà bên trích xuất
dùng để ghi.

Vì vậy phép so sánh là SET DIFFERENCE chứ không phải so số đếm: một frame thiếu
cộng một file rác vẫn cho ra count bằng nhau, và trong pipeline này thì count
còn vô dụng hơn nữa — nhánh OpenCV của keyframe_extraction.py khi `cap.read()`
trả về False sẽ bỏ qua cả file .jpg lẫn dòng csv, nên csv và jpg vẫn khớp nhau
trong khi dữ liệu đã mất.

CÁCH DÙNG
    python data_processing/keyframe_extraction/check_integrity.py            # kiểm tất cả
    python .../check_integrity.py -f L21,L22                                 # vài folder
    python .../check_integrity.py -v L21_V001 --verbose                      # một video
    python .../check_integrity.py --stage keyframe --stage features          # chọn stage
    python .../check_integrity.py --fast                                     # bỏ qua mở từng .jpg
    python .../check_integrity.py --hash                                     # + sha256 video gốc
    python .../check_integrity.py --deep-decode                              # + decode toàn bộ video (rất chậm)

Stage `index` cần faiss nên phải chạy bằng python của venv_embedding:
    data_processing/feature_extraction/venv_embedding/bin/python \
        data_processing/keyframe_extraction/check_integrity.py --stage index --roundtrip 20

EXIT CODE: 0 = sạch, 1 = có ERROR (hoặc có WARN khi bật --strict). Nhờ vậy dùng
được trong script tự động: stage sau chỉ chạy khi stage trước exit 0.
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
import os
import random
import re
import statistics
import subprocess
import sys
from collections import Counter, defaultdict
from concurrent.futures import ProcessPoolExecutor, as_completed
from datetime import datetime
from glob import glob

import numpy as np

project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), "../.."))
if project_root not in sys.path:
    sys.path.append(project_root)

from utils.config_loader import load_config
from data_processing.keyframe_extraction.shot_sampling import expected_keyframes

# Mọi đường dẫn lấy từ config.yaml — đừng hardcode, xem ghi chú trong file đó.
PATHS = load_config()["paths"]
VIDEOS_DIR = PATHS["videos_dir"]
SCENES_DIR = PATHS["scenes_dir"]
PREDICTIONS_DIR = PATHS["predictions_dir"]
KEYFRAMES_DIR = PATHS["keyframes_dir"]
MAP_DIR = PATHS["map_keyframes_dir"]
FEATURES_DIR = PATHS["features_dir"]
INDEX_DIR = PATHS["index_dir"]
DATA_DIR = PATHS["data_dir"]
PROCESSED_DIR = PATHS["processed_dir"]

MANIFEST_PATH = os.path.join(PROCESSED_DIR, "_manifest", "videos.json")
REPORT_DIR = os.path.join(PROCESSED_DIR, "_reports")

ALL_STAGES = ["manifest", "transnet", "keyframe", "features", "index", "anomaly"]

ERROR = "ERROR"
WARN = "WARN"

MAP_CSV_HEADER = ["n", "shot_idx", "pts_time", "fps", "frame_idx"]
SENTINEL_NAME = "_SUCCESS"     # phải khớp keyframe_extraction.SENTINEL_NAME
MIN_JPG_BYTES = 1024          # dưới ngưỡng này gần như chắc chắn là file ghi dở
BLANK_STDDEV = 1.0            # độ lệch chuẩn thấp hơn = ảnh gần như một màu
KEYFRAME_NAME_RE = re.compile(r"^\d{4}$")
ROBUST_Z_CUTOFF = 5.0         # ngưỡng phát hiện outlier theo median/MAD
# TransNetV2 loại vài frame ngay chỗ chuyển cảnh nên ranh giới các shot thường
# hở 1-2 frame — đo trên dataset hiện tại: hở 1 frame xuất hiện ~1000 lần, hở 2
# frame ~20 lần, không bao giờ hở hơn. Chỉ khoảng trống LỚN mới là mất dữ liệu.
SHOT_GAP_TOLERANCE = 5


# ----------------------------------------------------------------------------
# Finding: một phát hiện đơn lẻ. Giữ ở dạng dict thuần để pickle qua
# ProcessPoolExecutor mà không cần khai báo lớp ở cả hai phía.
# ----------------------------------------------------------------------------
def finding(stage, code, severity, video_id, detail):
    return {
        "stage": stage,
        "code": code,
        "severity": severity,
        "video_id": video_id,
        "detail": detail,
    }


def code_version() -> str:
    try:
        out = subprocess.run(
            ["git", "-C", project_root, "rev-parse", "--short", "HEAD"],
            capture_output=True, text=True, timeout=5,
        )
        if out.returncode == 0:
            return out.stdout.strip()
    except Exception:
        pass
    return "unknown"


# ============================================================================
# STAGE 0 — MANIFEST
# Danh sách nguồn duy nhất cho mọi stage sau. Dựng một lần từ video gốc, cache
# lại; các lần chạy sau chỉ đối chiếu để phát hiện video mới / mất / bị đổi.
# ============================================================================
def scan_video_files() -> dict:
    """Quét đĩa -> {video_id: {rel_path, bytes, mtime}} (rẻ, không gọi ffprobe)."""
    found = {}
    for path in sorted(glob(os.path.join(VIDEOS_DIR, "*", "*.mp4"))):
        stat = os.stat(path)
        vid = os.path.basename(path)[:-4]
        found[vid] = {
            "rel_path": os.path.relpath(path, VIDEOS_DIR),
            "bytes": stat.st_size,
            "mtime": int(stat.st_mtime),
        }
    return found


def ffprobe_video(args):
    """Đọc metadata một video. Chạy trong worker process."""
    vid, abs_path, want_hash, want_decode = args
    entry = {"video_id": vid}

    cmd = [
        "ffprobe", "-v", "error", "-select_streams", "v:0",
        "-show_entries", "stream=codec_name,r_frame_rate,nb_frames,width,height",
        "-show_entries", "format=duration,size",
        "-of", "json", abs_path,
    ]
    try:
        out = subprocess.run(cmd, capture_output=True, text=True, timeout=120)
        if out.returncode != 0:
            entry["probe_ok"] = False
            entry["probe_error"] = (out.stderr or "ffprobe returned non-zero").strip()[:300]
            return entry
        info = json.loads(out.stdout or "{}")
    except Exception as exc:
        entry["probe_ok"] = False
        entry["probe_error"] = f"{type(exc).__name__}: {exc}"[:300]
        return entry

    streams = info.get("streams") or []
    if not streams:
        entry["probe_ok"] = False
        entry["probe_error"] = "không có video stream"
        return entry

    st, fmt = streams[0], info.get("format") or {}
    entry["probe_ok"] = True
    entry["codec"] = st.get("codec_name")
    entry["width"] = st.get("width")
    entry["height"] = st.get("height")

    fps = 0.0
    raw_fps = st.get("r_frame_rate") or "0/1"
    try:
        num, den = raw_fps.split("/")
        fps = float(num) / float(den) if float(den) else 0.0
    except Exception:
        pass
    entry["fps"] = round(fps, 6)

    try:
        entry["duration"] = float(fmt.get("duration"))
    except (TypeError, ValueError):
        entry["duration"] = None

    # nb_frames trong header MP4 rất hay thiếu hoặc sai với file bị cắt cụt;
    # ghi rõ nguồn để stage sau biết con số này đáng tin đến đâu.
    nb = st.get("nb_frames")
    if nb not in (None, "N/A"):
        entry["nb_frames"] = int(nb)
        entry["nb_frames_source"] = "header"
    elif entry["duration"] and fps:
        entry["nb_frames"] = int(round(entry["duration"] * fps))
        entry["nb_frames_source"] = "estimated"
    else:
        entry["nb_frames"] = None
        entry["nb_frames_source"] = "unknown"

    if want_hash:
        h = hashlib.sha256()
        with open(abs_path, "rb") as f:
            for chunk in iter(lambda: f.read(8 << 20), b""):
                h.update(chunk)
        entry["sha256"] = h.hexdigest()

    if want_decode:
        # Checksum chỉ chứng minh file bit-identical, KHÔNG chứng minh nó decode
        # được. Đây là phép kiểm duy nhất trả lời "file này có dùng được không".
        dec = subprocess.run(
            ["ffmpeg", "-v", "error", "-i", abs_path, "-f", "null", "-"],
            capture_output=True, text=True,
        )
        err = (dec.stderr or "").strip()
        entry["decode_ok"] = dec.returncode == 0 and not err
        if not entry["decode_ok"]:
            entry["decode_error"] = err[:500] or f"exit={dec.returncode}"

    return entry


def load_or_build_manifest(args, findings):
    """Trả về {video_id: entry}. Tự bổ sung video mới, báo động video mất/đổi."""
    on_disk = scan_video_files()
    if not on_disk:
        findings.append(finding("manifest", "NO_VIDEO", ERROR, None,
                                f"Không tìm thấy .mp4 nào trong {VIDEOS_DIR}"))
        return {}

    cached = {}
    if os.path.exists(MANIFEST_PATH) and not args.refresh_manifest:
        try:
            with open(MANIFEST_PATH, encoding="utf-8") as f:
                cached = json.load(f).get("videos", {})
        except Exception as exc:
            findings.append(finding("manifest", "MANIFEST_UNREADABLE", WARN, None,
                                    f"{MANIFEST_PATH}: {exc} — dựng lại từ đầu"))

    # Video biến mất khỏi đĩa sau khi đã vào manifest: dữ liệu bị xoá nhầm.
    for vid in sorted(set(cached) - set(on_disk)):
        findings.append(finding("manifest", "VIDEO_DISAPPEARED", ERROR, vid,
                                "có trong manifest nhưng không còn trên đĩa"))

    need_probe, reprobe_reason = [], {}
    for vid, disk in sorted(on_disk.items()):
        old = cached.get(vid)
        if old is None:
            need_probe.append(vid)
            reprobe_reason[vid] = "mới"
        elif old.get("bytes") != disk["bytes"]:
            # Kích thước đổi = nội dung đổi. Đây là báo động thật, không phải nhiễu.
            findings.append(finding(
                "manifest", "VIDEO_SIZE_CHANGED", ERROR, vid,
                f"{old.get('bytes')} -> {disk['bytes']} byte kể từ lần dựng manifest"))
            need_probe.append(vid)
            reprobe_reason[vid] = "đổi kích thước"
        elif old.get("mtime") != disk["mtime"]:
            # mtime lệch mà size không đổi thường chỉ là rsync/cp không giữ timestamp.
            findings.append(finding("manifest", "VIDEO_MTIME_CHANGED", WARN, vid,
                                    "mtime thay đổi nhưng kích thước giữ nguyên"))
        elif args.hash and "sha256" not in old:
            need_probe.append(vid)
            reprobe_reason[vid] = "chưa có sha256"
        elif args.deep_decode and "decode_ok" not in old:
            need_probe.append(vid)
            reprobe_reason[vid] = "chưa decode-check"

    manifest = {vid: dict(cached[vid], **on_disk[vid]) for vid in on_disk if vid in cached}

    if need_probe:
        log(f"[manifest] ffprobe {len(need_probe)} video "
            f"({Counter(reprobe_reason.values()).most_common()})...")
        tasks = [
            (vid,
             os.path.join(VIDEOS_DIR, on_disk[vid]["rel_path"]),
             args.hash, args.deep_decode)
            for vid in need_probe
        ]
        with ProcessPoolExecutor(max_workers=args.workers) as ex:
            for entry in ex.map(ffprobe_video, tasks, chunksize=4):
                vid = entry["video_id"]
                manifest[vid] = dict(entry, **on_disk[vid])

    for vid in sorted(manifest):
        e = manifest[vid]
        if not e.get("probe_ok", False):
            findings.append(finding("manifest", "PROBE_FAILED", ERROR, vid,
                                    e.get("probe_error", "ffprobe thất bại")))
            continue
        if e.get("nb_frames") in (None, 0):
            findings.append(finding("manifest", "NO_FRAME_COUNT", WARN, vid,
                                    "không xác định được số frame — bỏ qua kiểm tra biên frame"))
        if not e.get("fps"):
            findings.append(finding("manifest", "NO_FPS", WARN, vid, "không đọc được fps"))
        if e.get("duration") in (None, 0):
            findings.append(finding("manifest", "NO_DURATION", WARN, vid, "không đọc được duration"))
        if args.deep_decode and e.get("decode_ok") is False:
            findings.append(finding("manifest", "DECODE_ERROR", ERROR, vid,
                                    f"ffmpeg decode lỗi: {e.get('decode_error', '')[:200]}"))

    os.makedirs(os.path.dirname(MANIFEST_PATH), exist_ok=True)
    tmp = MANIFEST_PATH + ".tmp"
    with open(tmp, "w", encoding="utf-8") as f:
        json.dump({
            "updated_at": datetime.now().isoformat(timespec="seconds"),
            "code_version": code_version(),
            "n_videos": len(manifest),
            "videos": manifest,
        }, f, ensure_ascii=False, indent=1)
    os.replace(tmp, MANIFEST_PATH)   # atomic: không bao giờ để lại manifest ghi dở
    return manifest


# ============================================================================
# STAGE 1+2+3 — kiểm mỗi video: transnet, keyframe, features
# Gộp vào một worker để chỉ đọc scenes.txt / stat thư mục một lần.
# ============================================================================
def keyframe_dir(vid):
    return os.path.join(KEYFRAMES_DIR, f"Keyframes_{vid.split('_')[0]}", vid)


def read_scenes(vid):
    """-> (lines, sha256) hoặc (None, None) nếu thiếu/rỗng.

    Đọc dạng bytes rồi mới decode để digest khớp đúng với digest mà
    keyframe_extraction.py ghi vào _SUCCESS.
    """
    path = os.path.join(SCENES_DIR, f"{vid}.scenes.txt")
    if not os.path.exists(path) or os.path.getsize(path) == 0:
        return None, None
    with open(path, "rb") as f:
        raw = f.read()
    return raw.decode("utf-8", "replace").splitlines(keepends=True), hashlib.sha256(raw).hexdigest()


def read_sentinel(kf_dir):
    """Nội dung _SUCCESS, hoặc None nếu chưa có/hỏng."""
    try:
        with open(os.path.join(kf_dir, SENTINEL_NAME), encoding="utf-8") as f:
            return json.load(f)
    except Exception:
        return None


def read_jpg_dir(kf_dir):
    """-> ({stem: size}, [tên file không đúng quy ước])"""
    sizes, malformed = {}, []
    try:
        with os.scandir(kf_dir) as it:
            for de in it:
                if not de.name.endswith(".jpg"):
                    continue
                stem = de.name[:-4]
                if not KEYFRAME_NAME_RE.match(stem):
                    malformed.append(de.name)
                    continue
                sizes[stem] = de.stat().st_size
    except FileNotFoundError:
        return None, []
    return sizes, malformed


def preview(items, k=6):
    items = sorted(items)
    head = ", ".join(str(x) for x in items[:k])
    return head + (f", ... (+{len(items) - k})" if len(items) > k else "")


def check_transnet(vid, entry, lines, out):
    """`lines` = nội dung scenes.txt đã đọc sẵn (None nếu thiếu/rỗng)."""
    scenes_path = os.path.join(SCENES_DIR, f"{vid}.scenes.txt")
    pred_path = os.path.join(PREDICTIONS_DIR, f"{vid}.predictions.txt")

    for path, code in ((pred_path, "PRED"), (scenes_path, "SCENES")):
        if not os.path.exists(path):
            out.append(finding("transnet", f"{code}_MISSING", ERROR, vid,
                               f"thiếu {os.path.basename(path)}"))
        elif os.path.getsize(path) == 0:
            out.append(finding("transnet", f"{code}_EMPTY", ERROR, vid,
                               f"{os.path.basename(path)} rỗng (0 byte)"))

    if lines is None:
        return

    _exp, shots, bad_lines = expected_keyframes(lines)
    if bad_lines:
        out.append(finding("transnet", "SCENES_BAD_LINE", ERROR, vid,
                           f"{len(bad_lines)} dòng không parse được: {preview(bad_lines)}"))
    if not shots:
        out.append(finding("transnet", "SCENES_NO_SHOT", ERROR, vid,
                           "không có shot hợp lệ nào"))
        return

    # Ranh giới phải hợp lệ và phủ kín video. Hở vài frame ở chỗ chuyển cảnh là
    # bình thường (xem SHOT_GAP_TOLERANCE); hở cả đoạn dài mới là mất dữ liệu,
    # còn chồng lấn thì luôn sai vì một frame sẽ thuộc hai shot.
    prev_right = None
    inverted, overlaps, gaps = [], [], []
    for line_no, left, right in shots:
        if right < left:
            inverted.append(line_no)
        if prev_right is not None:
            gap = left - prev_right - 1
            if gap < 0:
                overlaps.append(f"dòng {line_no} (chồng {-gap} frame)")
            elif gap > SHOT_GAP_TOLERANCE:
                gaps.append(f"dòng {line_no} (hở {gap} frame)")
        prev_right = right
    if inverted:
        out.append(finding("transnet", "SHOT_INVERTED", ERROR, vid,
                           f"{len(inverted)} shot có right < left: dòng {preview(inverted)}"))
    if overlaps:
        out.append(finding("transnet", "SHOT_OVERLAP", ERROR, vid,
                           f"{len(overlaps)} chỗ chồng lấn giữa các shot: {preview(overlaps)}"))
    if gaps:
        out.append(finding("transnet", "SHOT_GAP", WARN, vid,
                           f"{len(gaps)} khoảng trống > {SHOT_GAP_TOLERANCE} frame giữa các shot: "
                           f"{preview(gaps)}"))

    # Shot cuối phải chạm gần cuối video. Lệch nhiều = TransNetV2 dừng giữa
    # chừng nhưng vẫn ghi ra file — lỗi im lặng mà đếm số shot không thấy được.
    nb_frames = entry.get("nb_frames")
    fps = entry.get("fps") or 25.0
    if nb_frames:
        last_right = shots[-1][2]
        missing_frames = nb_frames - 1 - last_right
        if missing_frames > max(fps, 25):   # bỏ qua sai lệch dưới ~1 giây
            out.append(finding(
                "transnet", "SCENES_TRUNCATED", ERROR, vid,
                f"shot cuối kết thúc ở frame {last_right} nhưng video có {nb_frames} frame "
                f"(thiếu ~{missing_frames / fps:.1f}s cuối)"))
        if shots[0][1] > SHOT_GAP_TOLERANCE:
            out.append(finding("transnet", "SCENES_HEAD_MISSING", WARN, vid,
                               f"shot đầu bắt đầu ở frame {shots[0][1]} — bỏ sót "
                               f"~{shots[0][1] / fps:.1f}s đầu video"))


def gather_stats(vid, lines, scenes_digest):
    """Thu thập DỮ KIỆN (không phải phép kiểm) mà mọi stage sau đều cần.

    Chạy vô điều kiện: stage features đối chiếu số vector với số .jpg, stage
    index truy ngược keyframe -> hàng trong .npy. Nếu để việc liệt kê này nằm
    trong stage keyframe thì `--stage features` sẽ so với danh sách rỗng và báo
    sai toàn bộ dataset.
    """
    expected, shots, _ = expected_keyframes(lines) if lines else ([], [], [])
    kf_dir = keyframe_dir(vid)
    sizes, malformed = read_jpg_dir(kf_dir)
    sentinel = read_sentinel(kf_dir)
    stats = {
        "n_shots": len(shots),
        "n_expected": len(expected),
        "n_jpg": 0 if sizes is None else len(sizes),
        "jpg_bytes": 0 if sizes is None else sum(sizes.values()),
        "kf_names": [] if sizes is None else sorted(sizes),
        "sealed": sentinel is not None,
        "scenes_sha256": scenes_digest,
    }
    return stats, sizes, malformed, sentinel


def gather_feature_shapes(vid, models):
    """{model: {rows, dim}} — chỉ đọc header .npy (mmap), không nạp dữ liệu.

    Cũng là dữ kiện: stage index cần biết mỗi video có bao nhiêu vector kể cả
    khi người dùng không chọn stage features. Mọi lỗi ở đây im lặng — việc báo
    lỗi thuộc về check_features().
    """
    info = {}
    for model in models:
        path = os.path.join(FEATURES_DIR, model, f"{vid}.npy")
        if not os.path.exists(path):
            continue
        try:
            arr = np.load(path, mmap_mode="r")
        except Exception:
            continue
        if arr.ndim == 2:
            info[model] = {"rows": int(arr.shape[0]), "dim": int(arr.shape[1])}
    return info


def check_keyframe(vid, entry, lines, scenes_digest, sizes, malformed_names, sentinel, args, out):
    if lines is None:
        return
    expected, _shots, _ = expected_keyframes(lines)
    if not expected:
        return

    # Sentinel không có thì chỉ là dữ liệu trích trước khi có cơ chế này (báo gộp
    # một dòng ở cuối report). Sentinel CÓ mà lệch thực tế mới là báo động: nghĩa
    # là scenes.txt đã đổi sau khi trích, nên keyframe hiện tại đã hết hạn dù
    # thư mục trông vẫn đầy đủ.
    if sentinel:
        if sentinel.get("n_keyframes") != len(expected):
            out.append(finding("keyframe", "SENTINEL_STALE", ERROR, vid,
                               f"_SUCCESS ghi {sentinel.get('n_keyframes')} keyframe nhưng "
                               f"scenes.txt hiện tại sinh ra {len(expected)}"))
        elif sentinel.get("scenes_sha256") not in (None, scenes_digest):
            out.append(finding("keyframe", "SENTINEL_STALE", ERROR, vid,
                               "scenes.txt đã thay đổi sau khi trích keyframe "
                               "(digest trong _SUCCESS không khớp) — keyframe đã hết hạn"))

    exp_by_n = {n: (shot_idx, frame_idx) for n, shot_idx, frame_idx in expected}
    exp_ids = set(exp_by_n)

    # --- map csv ---
    map_path = os.path.join(MAP_DIR, f"{vid}.csv")
    if not os.path.exists(map_path):
        out.append(finding("keyframe", "MAP_MISSING", ERROR, vid, f"thiếu {vid}.csv"))
    elif os.path.getsize(map_path) == 0:
        out.append(finding("keyframe", "MAP_EMPTY", ERROR, vid, f"{vid}.csv rỗng"))
    else:
        with open(map_path, newline="", encoding="utf-8") as f:
            reader = csv.reader(f)
            try:
                header = next(reader)
            except StopIteration:
                header = []
            rows = list(reader)

        if header != MAP_CSV_HEADER:
            out.append(finding("keyframe", "MAP_HEADER", ERROR, vid,
                               f"header {header} != {MAP_CSV_HEADER}"))

        csv_by_n, dup_n, malformed = {}, [], 0
        for row in rows:
            if len(row) != len(MAP_CSV_HEADER):
                malformed += 1
                continue
            try:
                n, shot_idx, fps_val, frame_idx = int(row[0]), int(row[1]), float(row[3]), int(row[4])
            except ValueError:
                malformed += 1
                continue
            if n in csv_by_n:
                dup_n.append(n)
            csv_by_n[n] = (shot_idx, frame_idx, fps_val)

        if malformed:
            out.append(finding("keyframe", "MAP_MALFORMED_ROW", ERROR, vid,
                               f"{malformed} dòng csv sai định dạng"))
        if dup_n:
            out.append(finding("keyframe", "MAP_DUPLICATE_N", ERROR, vid,
                               f"cột n trùng lặp: {preview(dup_n)}"))

        # SET DIFFERENCE, không phải so count. `extra` quan trọng ngang `missing`:
        # nó thường là tàn dư của lần chạy trước với tham số khác.
        csv_ids = set(csv_by_n)
        if exp_ids - csv_ids:
            out.append(finding("keyframe", "MAP_ROW_MISSING", ERROR, vid,
                               f"{len(exp_ids - csv_ids)}/{len(exp_ids)} dòng thiếu: "
                               f"n={preview(exp_ids - csv_ids)}"))
        if csv_ids - exp_ids:
            out.append(finding("keyframe", "MAP_ROW_EXTRA", ERROR, vid,
                               f"{len(csv_ids - exp_ids)} dòng thừa không sinh ra từ scenes.txt: "
                               f"n={preview(csv_ids - exp_ids)}"))

        # Nội dung, không chỉ sự tồn tại: frame_idx phải đúng bằng frame mà luật
        # lấy mẫu tính ra, không chỉ "có một con số nào đó".
        mismatch_frame, mismatch_shot, out_of_range = [], [], []
        nb_frames = entry.get("nb_frames")
        for n in sorted(exp_ids & csv_ids):
            exp_shot, exp_frame = exp_by_n[n]
            got_shot, got_frame, _ = csv_by_n[n]
            if got_frame != exp_frame:
                mismatch_frame.append(n)
            if got_shot != exp_shot:
                mismatch_shot.append(n)
            if nb_frames and not (0 <= got_frame < nb_frames):
                out_of_range.append(n)
        if mismatch_frame:
            out.append(finding("keyframe", "MAP_FRAME_IDX_MISMATCH", ERROR, vid,
                               f"{len(mismatch_frame)} dòng có frame_idx khác luật lấy mẫu: "
                               f"n={preview(mismatch_frame)}"))
        if mismatch_shot:
            out.append(finding("keyframe", "MAP_SHOT_IDX_MISMATCH", ERROR, vid,
                               f"{len(mismatch_shot)} dòng có shot_idx sai: n={preview(mismatch_shot)}"))
        if out_of_range:
            out.append(finding("keyframe", "MAP_FRAME_OUT_OF_RANGE", ERROR, vid,
                               f"{len(out_of_range)} dòng trỏ tới frame ngoài video "
                               f"(nb_frames={nb_frames}): n={preview(out_of_range)}"))

        fps_manifest = entry.get("fps")
        fps_seen = {v[2] for v in csv_by_n.values()}
        if len(fps_seen) > 1:
            out.append(finding("keyframe", "MAP_FPS_INCONSISTENT", WARN, vid,
                               f"nhiều giá trị fps trong một file: {sorted(fps_seen)}"))
        elif fps_seen and fps_manifest and abs(next(iter(fps_seen)) - fps_manifest) > 0.01:
            out.append(finding("keyframe", "MAP_FPS_MISMATCH", WARN, vid,
                               f"fps trong csv {next(iter(fps_seen))} != ffprobe {fps_manifest:.4f} "
                               f"— pts_time sẽ lệch"))

    # --- ảnh .jpg ---
    kf_dir = keyframe_dir(vid)
    if sizes is None:
        out.append(finding("keyframe", "KF_DIR_MISSING", ERROR, vid,
                           f"thiếu thư mục {os.path.relpath(kf_dir, PROCESSED_DIR)}"))
        return

    if malformed_names:
        out.append(finding("keyframe", "KF_BAD_NAME", WARN, vid,
                           f"{len(malformed_names)} file không theo dạng NNNN.jpg: "
                           f"{preview(malformed_names)}"))

    jpg_ids = {int(stem) for stem in sizes}
    if exp_ids - jpg_ids:
        out.append(finding("keyframe", "KF_MISSING", ERROR, vid,
                           f"{len(exp_ids - jpg_ids)}/{len(exp_ids)} ảnh thiếu: "
                           f"{preview(f'{n:04d}.jpg' for n in (exp_ids - jpg_ids))}"))
    if jpg_ids - exp_ids:
        out.append(finding("keyframe", "KF_EXTRA", ERROR, vid,
                           f"{len(jpg_ids - exp_ids)} ảnh thừa (tàn dư lần chạy trước?): "
                           f"{preview(f'{n:04d}.jpg' for n in (jpg_ids - exp_ids))}"))

    truncated = sorted(s for s, b in sizes.items() if b < MIN_JPG_BYTES)
    if truncated:
        out.append(finding("keyframe", "KF_TRUNCATED", ERROR, vid,
                           f"{len(truncated)} ảnh nhỏ hơn {MIN_JPG_BYTES}B (ghi dở?): "
                           f"{preview(truncated)}"))

    if args.fast or not sizes:
        return

    # File tồn tại và đủ lớn vẫn có thể hỏng. verify() chỉ quét cấu trúc JPEG nên
    # rẻ, chạy cho toàn bộ ảnh; còn phát hiện ảnh trắng/đen cần decode đầy đủ nên
    # chỉ lấy mẫu (seed theo video_id để kết quả tái lập được).
    from PIL import Image, ImageStat

    corrupt = []
    for stem in sorted(sizes):
        path = os.path.join(kf_dir, f"{stem}.jpg")
        try:
            with Image.open(path) as im:
                im.verify()
        except Exception as exc:
            corrupt.append(f"{stem} ({type(exc).__name__})")
    if corrupt:
        out.append(finding("keyframe", "KF_CORRUPT", ERROR, vid,
                           f"{len(corrupt)} ảnh không giải mã được: {preview(corrupt, 4)}"))

    rng = random.Random(vid)
    pool = sorted(sizes)
    sample = pool if args.deep_jpg else rng.sample(pool, min(args.sample_jpg, len(pool)))
    blank = []
    for stem in sample:
        try:
            with Image.open(os.path.join(kf_dir, f"{stem}.jpg")) as im:
                small = im.convert("L").resize((32, 32))
                if ImageStat.Stat(small).stddev[0] < BLANK_STDDEV:
                    blank.append(stem)
        except Exception:
            pass       # đã báo ở KF_CORRUPT
    if blank:
        out.append(finding("keyframe", "KF_BLANK", WARN, vid,
                           f"{len(blank)}/{len(sample)} ảnh lấy mẫu gần như một màu "
                           f"(đen/trắng hoàn toàn): {preview(blank)}"))


def check_features(vid, models, n_jpg, cascade, out):
    """`cascade` = stage transnet/keyframe của video này đã lỗi."""
    for model in models:
        npy_path = os.path.join(FEATURES_DIR, model, f"{vid}.npy")
        # Lỗi hệ quả thì hạ xuống WARN: dồn ERROR vào nguyên nhân gốc, nếu không
        # một video hỏng ở stage 2 sẽ đẻ ra ERROR ở cả 3 model và làm loãng report.
        sev = WARN if cascade else ERROR
        note = " (hệ quả của lỗi keyframe)" if cascade else ""

        if not os.path.exists(npy_path):
            out.append(finding("features", "NPY_MISSING", sev, vid,
                               f"[{model}] chưa có {vid}.npy{note}"))
            continue
        try:
            arr = np.load(npy_path, mmap_mode="r")
        except Exception as exc:
            out.append(finding("features", "NPY_UNREADABLE", ERROR, vid,
                               f"[{model}] không đọc được: {type(exc).__name__}: {exc}"))
            continue

        if arr.ndim != 2:
            out.append(finding("features", "NPY_SHAPE", ERROR, vid,
                               f"[{model}] shape {arr.shape}, cần mảng 2 chiều"))
            continue

        # Bất biến then chốt: run_embedding.py encode đúng những .jpg tồn tại.
        # Lệch = có ảnh hỏng bị collate_fn lặng lẽ loại khỏi batch, hoặc .npy cũ
        # sinh trước khi keyframe được trích lại.
        if arr.shape[0] != n_jpg:
            out.append(finding("features", "NPY_COUNT_MISMATCH", sev, vid,
                               f"[{model}] {arr.shape[0]} vector nhưng có {n_jpg} ảnh .jpg{note}"))

        if arr.shape[0] == 0:
            # run_embedding.py bỏ qua video đã có .npy, nên file rỗng này sẽ
            # KHÔNG tự khỏi khi chạy lại — phải xoá đi trước.
            out.append(finding("features", "NPY_EMPTY", sev, vid,
                               f"[{model}] mảng rỗng, {n_jpg} keyframe không nằm trong index; "
                               f"phải xoá .npy trước khi encode lại{note}"))
            continue

        if not np.issubdtype(arr.dtype, np.floating):
            out.append(finding("features", "NPY_DTYPE", WARN, vid,
                               f"[{model}] dtype {arr.dtype}, mong đợi float"))

        dense = np.asarray(arr, dtype=np.float64)
        bad_rows = np.where(~np.isfinite(dense).all(axis=1))[0]
        if bad_rows.size:
            out.append(finding("features", "NPY_NOT_FINITE", ERROR, vid,
                               f"[{model}] {bad_rows.size} vector chứa NaN/Inf: "
                               f"hàng {preview(bad_rows.tolist())}"))
        norms = np.linalg.norm(dense, axis=1)
        zero_rows = np.where(norms == 0)[0]
        if zero_rows.size:
            out.append(finding("features", "NPY_ZERO_VECTOR", ERROR, vid,
                               f"[{model}] {zero_rows.size} vector toàn 0 — ảnh nguồn hỏng "
                               f"mà model vẫn trả về output: hàng {preview(zero_rows.tolist())}"))
        # Nhiều vector giống hệt nhau từng bit = encoder bị kẹt hoặc lặp ảnh.
        if arr.shape[0] >= 8:
            uniq = len({row.tobytes() for row in np.ascontiguousarray(arr)})
            if uniq < arr.shape[0] * 0.7:
                out.append(finding("features", "NPY_DUPLICATE_ROWS", WARN, vid,
                                   f"[{model}] chỉ {uniq}/{arr.shape[0]} vector là duy nhất"))


def model_dims(model, stats_by_video):
    return {s["features"][model]["dim"] for s in stats_by_video.values()
            if model in s.get("features", {})}


def check_feature_dims(models, stats_by_video, out):
    """Mọi .npy của cùng một model phải cùng số chiều.

    Chạy ở tiến trình chính vì đây là phép so CHÉO giữa các video — worker chỉ
    thấy một video nên không thể phát hiện. Cố ý KHÔNG đặt trong stage index:
    đây là dữ kiện thuần của features, ràng buộc nó vào faiss nghĩa là máy không
    cài faiss sẽ lặng lẽ bỏ qua một lỗi nghiêm trọng.
    """
    for model in models:
        dims = model_dims(model, stats_by_video)
        if len(dims) > 1:
            offenders = defaultdict(list)
            for vid, s in stats_by_video.items():
                if model in s.get("features", {}):
                    offenders[s["features"][model]["dim"]].append(vid)
            detail = "; ".join(f"dim={d}: {len(v)} video ({preview(v, 3)})"
                               for d, v in sorted(offenders.items()))
            out.append(finding("features", "DIM_INCONSISTENT", ERROR, None,
                               f"[{model}] các .npy có số chiều khác nhau — trộn nhiều phiên "
                               f"bản model trong cùng thư mục. {detail}"))


def inspect_video(task):
    """Worker: thu thập dữ kiện rồi chạy các stage được chọn cho một video."""
    vid, entry, models, args, stages = task
    out = []

    lines, scenes_digest = read_scenes(vid)
    stats, sizes, malformed_names, sentinel = gather_stats(vid, lines, scenes_digest)

    if "transnet" in stages:
        check_transnet(vid, entry, lines, out)
    if "keyframe" in stages:
        check_keyframe(vid, entry, lines, scenes_digest, sizes, malformed_names,
                       sentinel, args, out)

    stats["features"] = gather_feature_shapes(vid, models)
    if "features" in stages:
        cascade = any(f["severity"] == ERROR and f["stage"] in ("transnet", "keyframe") for f in out)
        check_features(vid, models, stats["n_jpg"], cascade, out)

    stats["video_id"] = vid
    stats["failed_stages"] = sorted({f["stage"] for f in out if f["severity"] == ERROR})
    return out, stats


# ============================================================================
# STAGE 4 — INDEX
# ============================================================================
def meta_pairs(meta):
    """np.load(metadata.npy) -> [(video_id, keyframe_name)]"""
    return [(str(row[0]), str(row[1])) for row in np.asarray(meta).reshape(-1, 2)]


def check_index(models, stats_by_video, manifest, args, out):
    try:
        import faiss
    except ImportError:
        out.append(finding("index", "FAISS_UNAVAILABLE", WARN, None,
                           "không import được faiss — bỏ qua stage index. Chạy lại bằng "
                           "data_processing/feature_extraction/venv_embedding/bin/python"))
        return

    playlist_map = {}
    vp_path = os.path.join(DATA_DIR, "video_to_playlist.json")
    if os.path.exists(vp_path):
        with open(vp_path, encoding="utf-8") as f:
            playlist_map = json.load(f)
    else:
        out.append(finding("index", "PLAYLIST_MAP_MISSING", WARN, None,
                           f"thiếu {vp_path} — bỏ qua kiểm tra index theo playlist"))

    for model in models:
        idx_path = os.path.join(INDEX_DIR, f"{model}_keyframe_embedding.index")
        meta_path = os.path.join(INDEX_DIR, f"{model}_keyframe_metadata.npy")
        if not (os.path.exists(idx_path) and os.path.exists(meta_path)):
            out.append(finding("index", "INDEX_MISSING", ERROR, None,
                               f"[{model}] thiếu index hoặc metadata trong {INDEX_DIR}"))
            continue

        index = faiss.read_index(idx_path)
        pairs = meta_pairs(np.load(meta_path, allow_pickle=True))

        if index.ntotal != len(pairs):
            out.append(finding("index", "INDEX_META_DESYNC", ERROR, None,
                               f"[{model}] index có {index.ntotal} vector nhưng metadata có "
                               f"{len(pairs)} dòng — mọi kết quả tra cứu đều lệch"))

        dup = [p for p, c in Counter(pairs).items() if c > 1]
        if dup:
            out.append(finding("index", "INDEX_DUPLICATE", ERROR, None,
                               f"[{model}] {len(dup)} cặp (video, keyframe) xuất hiện nhiều lần: "
                               f"{preview([f'{v}/{k}' for v, k in dup], 4)}"))

        in_index = defaultdict(list)
        for pos, (v, k) in enumerate(pairs):
            in_index[v].append((pos, k))

        expected_videos = {v for v, s in stats_by_video.items()
                           if s.get("features", {}).get(model, {}).get("rows", 0) > 0}
        not_indexed = expected_videos - set(in_index)
        # So với MANIFEST chứ không phải tập vừa kiểm: chạy `-f L21` không có
        # nghĩa là mọi video L22 trong index đều là lạ.
        unknown = set(in_index) - set(manifest)
        if not_indexed:
            out.append(finding("index", "INDEX_VIDEO_NOT_COVERED", ERROR, None,
                               f"[{model}] {len(not_indexed)}/{len(expected_videos)} video có .npy "
                               f"nhưng không nằm trong index: {preview(not_indexed)} "
                               f"— chạy lại build_index.py"))
        if unknown:
            out.append(finding("index", "INDEX_UNKNOWN_VIDEO", ERROR, None,
                               f"[{model}] {len(unknown)} video trong index không có trong manifest: "
                               f"{preview(unknown)}"))

        dim_mismatch = []
        for v, entries in in_index.items():
            info = stats_by_video.get(v, {}).get("features", {}).get(model)
            if info and len(entries) != info["rows"]:
                dim_mismatch.append(f"{v}({len(entries)}!={info['rows']})")
        if dim_mismatch:
            out.append(finding("index", "INDEX_VIDEO_COUNT", ERROR, None,
                               f"[{model}] {len(dim_mismatch)} video có số vector trong index khác "
                               f"số dòng .npy: {preview(dim_mismatch, 4)}"))

        dims = model_dims(model, stats_by_video)
        if len(dims) == 1 and index.d != next(iter(dims)):
            out.append(finding("index", "INDEX_DIM", ERROR, None,
                               f"[{model}] index.d={index.d} != dim của .npy={next(iter(dims))}"))

        check_index_alignment(index, pairs, model, stats_by_video, args, out)

        if playlist_map:
            check_playlist_indexes(faiss, model, pairs, playlist_map, out)


def check_index_alignment(index, pairs, model, stats_by_video, args, out):
    """Đối chiếu TỪNG vector trong index với đúng hàng .npy mà metadata trỏ tới.

    Đây là phép kiểm quan trọng nhất của stage index, vì lệch ID <-> vector là
    lỗi IM LẶNG: hệ thống vẫn chạy, FAISS vẫn trả kết quả, chỉ là trả sai
    keyframe. IndexFlatL2 lưu nguyên vector nên reconstruct() cho phép so trực
    tiếp, không cần GPU và không cần nạp model.

    Quét TOÀN BỘ chứ không lấy mẫu. Một lần hoán vị hai keyframe cạnh nhau chỉ
    ảnh hưởng 2/150.000 vị trí — lấy mẫu 300 vị trí sẽ bỏ lọt với xác suất 99,6%
    rồi in ra chữ PASS, tức là tệ hơn không kiểm gì. Mỗi video được đối chiếu
    theo lô nên chi phí không đáng kể.
    """
    if not pairs or index.ntotal != len(pairs):
        return
    try:
        index.reconstruct(0)
    except Exception:
        out.append(finding("index", "ALIGN_UNSUPPORTED", WARN, None,
                           f"[{model}] index không hỗ trợ reconstruct() — bỏ qua kiểm tra alignment"))
        return

    by_video = defaultdict(list)
    for pos, (vid, kf_name) in enumerate(pairs):
        by_video[vid].append((pos, kf_name))

    if args.align_sample:
        rng = random.Random(f"align-{model}")
        keep = set(rng.sample(sorted(by_video), min(args.align_sample, len(by_video))))
        by_video = {v: e for v, e in by_video.items() if v in keep}

    misaligned, unresolved, compared = [], 0, 0
    for vid, entries in sorted(by_video.items()):
        npy_path = os.path.join(FEATURES_DIR, model, f"{vid}.npy")
        name_to_row = {name: i for i, name
                       in enumerate(stats_by_video.get(vid, {}).get("kf_names", []))}
        if not os.path.exists(npy_path) or not name_to_row:
            unresolved += len(entries)
            continue
        arr = np.load(npy_path, mmap_mode="r")

        positions = [p for p, _ in entries]
        # build_index.py ghi các vector của cùng một video liền nhau, nên hầu như
        # luôn lấy được bằng một lời gọi reconstruct_n thay vì từng vector một.
        if positions == list(range(positions[0], positions[0] + len(positions))):
            stored = index.reconstruct_n(positions[0], len(positions))
        else:
            stored = np.vstack([index.reconstruct(p) for p in positions])

        rows = [name_to_row.get(kf_name, -1) for _, kf_name in entries]
        usable = [i for i, r in enumerate(rows) if 0 <= r < arr.shape[0]]
        unresolved += len(entries) - len(usable)
        if not usable:
            continue

        expected = np.asarray(arr[[rows[i] for i in usable]], dtype=np.float32)
        differs = ~np.isclose(stored[usable], expected, rtol=1e-4, atol=1e-5).all(axis=1)
        compared += len(usable)
        for j in np.flatnonzero(differs):
            pos, kf_name = entries[usable[j]]
            misaligned.append(f"{vid}/{kf_name}@{pos}")

    if misaligned:
        out.append(finding("index", "INDEX_MISALIGNED", ERROR, None,
                           f"[{model}] {len(misaligned)}/{compared} vector trong index KHÁC vector "
                           f"trong .npy mà metadata trỏ tới — tra cứu sẽ trả về sai keyframe: "
                           f"{preview(misaligned, 4)}"))
    if unresolved:
        out.append(finding("index", "ALIGN_UNRESOLVED", WARN, None,
                           f"[{model}] {unresolved} vị trí không truy ngược được về .npy để đối "
                           f"chiếu (thiếu .npy hoặc thiếu ảnh .jpg tương ứng)"))
    log(f"[index] alignment {model}: đã đối chiếu {compared}/{len(pairs)} vector, "
        f"{len(misaligned)} lệch.")


def check_playlist_indexes(faiss, model, global_pairs, playlist_map, out):
    playlist_dir = os.path.join(INDEX_DIR, "playlist")
    if not os.path.isdir(playlist_dir):
        out.append(finding("index", "PLAYLIST_DIR_MISSING", WARN, None,
                           f"[{model}] chưa có {playlist_dir}"))
        return

    global_set = set(global_pairs)
    covered = set()
    for idx_path in sorted(glob(os.path.join(playlist_dir, f"{model}_*.index"))):
        name = os.path.basename(idx_path)[len(model) + 1:-len(".index")]
        meta_path = os.path.join(playlist_dir, f"{model}_{name}_keyframe_metadata.npy")
        if not os.path.exists(meta_path):
            out.append(finding("index", "PLAYLIST_META_MISSING", ERROR, None,
                               f"[{model}/{name}] có .index nhưng thiếu metadata"))
            continue
        index = faiss.read_index(idx_path)
        pairs = meta_pairs(np.load(meta_path, allow_pickle=True))
        if index.ntotal != len(pairs):
            out.append(finding("index", "PLAYLIST_META_DESYNC", ERROR, None,
                               f"[{model}/{name}] {index.ntotal} vector != {len(pairs)} metadata"))
        stray = set(pairs) - global_set
        if stray:
            out.append(finding("index", "PLAYLIST_NOT_IN_GLOBAL", ERROR, None,
                               f"[{model}/{name}] {len(stray)} keyframe không có trong index global "
                               f"— hai index được build từ hai lần khác nhau"))
        wrong = {v for v, _ in pairs if playlist_map.get(v) != name}
        if wrong:
            out.append(finding("index", "PLAYLIST_WRONG_VIDEO", ERROR, None,
                               f"[{model}/{name}] {len(wrong)} video không thuộc playlist này: "
                               f"{preview(wrong)}"))
        covered |= set(pairs)

    mapped_missing = {(v, k) for v, k in global_set if playlist_map.get(v)} - covered
    if mapped_missing:
        videos = {v for v, _ in mapped_missing}
        out.append(finding("index", "PLAYLIST_INCOMPLETE", WARN, None,
                           f"[{model}] {len(mapped_missing)} keyframe của {len(videos)} video đã có "
                           f"playlist nhưng chưa vào index playlist nào: {preview(videos)}"))


def check_roundtrip(models, stats_by_video, args, out):
    """Encode lại ảnh thật rồi tra ngược index — phép kiểm end-to-end mạnh nhất.

    Khác với check_index_alignment (chỉ chứng minh .npy <-> index khớp nhau),
    phép này đi hết chuỗi .jpg -> model -> index nên bắt được cả trường hợp thứ
    tự ảnh lúc encode khác thứ tự lúc build index. Đổi lại nó cần GPU và model.
    """
    try:
        import faiss  # noqa: F401
        import torch  # noqa: F401
        from PIL import Image
        from data_processing.feature_extraction.run_embedding import create_extractor
        import faiss as _faiss
    except Exception as exc:
        out.append(finding("index", "ROUNDTRIP_UNAVAILABLE", WARN, None,
                           f"không nạp được model/faiss ({type(exc).__name__}: {exc}) — bỏ qua"))
        return

    candidates = [(v, s) for v, s in stats_by_video.items() if s.get("kf_names")]
    if not candidates:
        return
    rng = random.Random("roundtrip")

    for model in models:
        idx_path = os.path.join(INDEX_DIR, f"{model}_keyframe_embedding.index")
        meta_path = os.path.join(INDEX_DIR, f"{model}_keyframe_metadata.npy")
        if not (os.path.exists(idx_path) and os.path.exists(meta_path)):
            continue
        index = _faiss.read_index(idx_path)
        pairs = meta_pairs(np.load(meta_path, allow_pickle=True))

        extractor = create_extractor(model)
        preprocess = getattr(extractor, "preprocess", None) or getattr(extractor, "image_processor", None)

        samples = []
        while len(samples) < args.roundtrip and candidates:
            vid, st = rng.choice(candidates)
            samples.append((vid, rng.choice(st["kf_names"])))

        failures = []
        for vid, kf_name in samples:
            path = os.path.join(KEYFRAMES_DIR, f"Keyframes_{vid.split('_')[0]}", vid, f"{kf_name}.jpg")
            if not os.path.exists(path):
                continue
            with Image.open(path) as im:
                tensor = preprocess(im.convert("RGB"))
            vec = extractor.encode_images_batch(torch.stack([tensor]))
            vec = np.asarray(vec, dtype=np.float32).reshape(1, -1)
            _, ids = index.search(vec, 1)
            got = pairs[ids[0][0]] if 0 <= ids[0][0] < len(pairs) else ("?", "?")
            if got != (vid, kf_name):
                failures.append(f"{vid}/{kf_name} -> {got[0]}/{got[1]}")

        if failures:
            out.append(finding("index", "ROUNDTRIP_FAILED", ERROR, None,
                               f"[{model}] {len(failures)}/{len(samples)} ảnh không tự tìm được chính "
                               f"mình ở rank 1 — chuỗi jpg->encode->index đã lệch: "
                               f"{preview(failures, 4)}"))
        else:
            log(f"[index] roundtrip {model}: {len(samples)}/{len(samples)} ảnh tự khớp ở rank 1.")


# ============================================================================
# STAGE 5 — ANOMALY
# Bắt loại lỗi mà mọi phép đếm đều bỏ lọt: video "thành công" nhưng dữ liệu vô
# nghĩa. Một video ra đúng 1 shot thì expected == actual == 1, mọi check ở trên
# đều PASS, nhưng TransNetV2 thực chất đã hỏng.
# ============================================================================
def robust_outliers(values: dict, cutoff=ROBUST_Z_CUTOFF):
    """-> [(key, value, z)] theo z-score bền vững (median / MAD)."""
    if len(values) < 20:
        return []
    nums = list(values.values())
    med = statistics.median(nums)
    mad = statistics.median([abs(x - med) for x in nums])
    if mad == 0:
        return []
    scale = 1.4826 * mad
    return sorted(
        ((k, v, (v - med) / scale) for k, v in values.items() if abs((v - med) / scale) > cutoff),
        key=lambda t: -abs(t[2]),
    )


def check_anomaly(manifest, stats_by_video, out):
    shots_per_min, bytes_per_jpg = {}, {}
    for vid, st in stats_by_video.items():
        entry = manifest.get(vid, {})
        duration = entry.get("duration")
        if st["n_shots"] == 1:
            out.append(finding("anomaly", "SINGLE_SHOT", WARN, vid,
                               "TransNetV2 chỉ tìm thấy 1 shot cho cả video — nhiều khả năng "
                               "detector hỏng ngầm hoặc video không decode được"))
        if st["n_shots"] and duration and duration > 30:
            shots_per_min[vid] = st["n_shots"] / (duration / 60.0)
        if st["n_jpg"]:
            bytes_per_jpg[vid] = st["jpg_bytes"] / st["n_jpg"]

    for vid, val, z in robust_outliers(shots_per_min):
        out.append(finding("anomaly", "SHOT_RATE_OUTLIER", WARN, vid,
                           f"{val:.1f} shot/phút, lệch {z:+.1f}σ so với phần còn lại của dataset"))
    for vid, val, z in robust_outliers(bytes_per_jpg):
        if z < 0:   # chỉ quan tâm phía nhẹ bất thường: ảnh đơn sắc / gần như trống
            out.append(finding("anomaly", "JPG_SIZE_OUTLIER", WARN, vid,
                               f"ảnh trung bình chỉ {val / 1024:.1f}KB, lệch {z:+.1f}σ "
                               f"— nghi ngờ keyframe trống"))

    if shots_per_min:
        vals = sorted(shots_per_min.values())
        log(f"[anomaly] shot/phút: median={statistics.median(vals):.1f} "
            f"p5={vals[len(vals) // 20]:.1f} p95={vals[-max(1, len(vals) // 20)]:.1f}")


# ============================================================================
# BÁO CÁO
# ============================================================================
_QUIET = False


def log(msg):
    if not _QUIET:
        print(msg, flush=True)


STAGE_FIX = {
    "manifest": "kiểm tra lại nguồn tải; chạy lại với --refresh-manifest sau khi sửa",
    "transnet": "./data_processing/keyframe_extraction/run_pipeline.sh -m transnet -f {folders}",
    "keyframe": "./data_processing/keyframe_extraction/run_pipeline.sh -m keyframe -f {folders}",
    # run_embedding.py bỏ qua video đã có .npy nên chỉ chạy lại là chưa đủ:
    # phải xoá file hỏng trước, nếu không lần chạy sau vẫn "skipped" y như cũ.
    "features": ("xoá .npy hỏng rồi encode lại, ví dụ:\n"
                 "              while read v; do rm -f data_processed/features/<model>/$v.npy; done "
                 "< {worklist}\n"
                 "              python data_processing/feature_extraction/run_embedding.py "
                 "-f {folders} --model <model>"),
    "index": "python data_processing/feature_extraction/build_index.py --force",
    "anomaly": "kiểm tra thủ công các video được liệt kê",
}


def write_report(findings, stats_by_video, manifest, args, stages):
    os.makedirs(REPORT_DIR, exist_ok=True)
    errors = [f for f in findings if f["severity"] == ERROR]
    warns = [f for f in findings if f["severity"] == WARN]

    print()
    print("=" * 78)
    print(" VERIFY REPORT — pipeline tiền xử lý AIC2026")
    print("=" * 78)
    print(f" Thời điểm     : {datetime.now().isoformat(timespec='seconds')}")
    print(f" Code version  : {code_version()}")
    print(f" Manifest      : {len(manifest)} video ({MANIFEST_PATH})")
    print(f" Đã kiểm       : {len(stats_by_video)} video | stage: {', '.join(stages)}")
    total_expected = sum(s["n_expected"] for s in stats_by_video.values())
    total_jpg = sum(s["n_jpg"] for s in stats_by_video.values())
    print(f" Keyframe      : kỳ vọng {total_expected} / có thật {total_jpg}"
          f" ({total_expected - total_jpg:+d})")

    # --- tổng hợp theo stage ---
    # Liệt kê cả stage không được chọn nhưng vẫn có finding (vd. stage index phát
    # hiện các .npy lệch số chiều): bảng tổng hợp không được bỏ sót mục nào,
    # nếu không nó lại thành một dạng "báo đủ trên dữ liệu thiếu".
    report_stages = [s for s in ALL_STAGES
                     if s in stages or any(f["stage"] == s for f in findings)]
    print()
    print(f" {'STAGE':<10} {'VIDEO LỖI':>10} {'ERROR':>8} {'WARN':>8}   TRẠNG THÁI")
    print(" " + "-" * 74)
    for stage in report_stages:
        s_err = [f for f in errors if f["stage"] == stage]
        s_warn = [f for f in warns if f["stage"] == stage]
        bad_videos = len({f["video_id"] for f in s_err if f["video_id"]})
        status = "FAIL" if s_err else ("PASS (có cảnh báo)" if s_warn else "PASS")
        print(f" {stage:<10} {bad_videos:>10} {len(s_err):>8} {len(s_warn):>8}   {status}")

    # --- chi tiết theo mã lỗi ---
    for severity, bucket in ((ERROR, errors), (WARN, warns)):
        if not bucket:
            continue
        print()
        print(f" ---- {severity} ({len(bucket)}) " + "-" * (60 - len(severity)))
        grouped = defaultdict(list)
        for f in bucket:
            grouped[(f["stage"], f["code"])].append(f)
        for (stage, code), items in sorted(grouped.items(), key=lambda kv: -len(kv[1])):
            n_vid = len({f["video_id"] for f in items if f["video_id"]})
            scope = f"{n_vid} video" if n_vid else "toàn cục"
            print(f" [{stage}/{code}] x{len(items)} ({scope})")
            limit = len(items) if args.verbose else 3
            for f in items[:limit]:
                who = f["video_id"] or "-"
                print(f"     {who:<12} {f['detail']}")
            if len(items) > limit:
                print(f"     ... còn {len(items) - limit} mục nữa (dùng --verbose để xem hết)")

    # --- worklist: video cần chạy lại, theo stage ---
    worklists = {}
    for stage in report_stages:
        vids = sorted({f["video_id"] for f in errors if f["stage"] == stage and f["video_id"]})
        if not vids:
            continue
        path = os.path.join(REPORT_DIR, f"worklist_{stage}.txt")
        with open(path, "w", encoding="utf-8") as fh:
            fh.write("\n".join(vids) + "\n")
        worklists[stage] = (path, vids)

    if worklists:
        print()
        print(" ---- CẦN CHẠY LẠI " + "-" * 55)
        for stage, (path, vids) in worklists.items():
            folders = ",".join(sorted({v.split("_")[0] for v in vids}))
            rel = os.path.relpath(path, project_root)
            print(f" {stage:<10} {len(vids)} video -> {rel}")
            print(f"            {STAGE_FIX[stage].format(folders=folders, worklist=rel)}")

    report_path = os.path.join(REPORT_DIR, args.json or "verify.json")
    if not os.path.isabs(report_path):
        report_path = os.path.join(REPORT_DIR, os.path.basename(report_path))
    tmp = report_path + ".tmp"
    with open(tmp, "w", encoding="utf-8") as fh:
        json.dump({
            "generated_at": datetime.now().isoformat(timespec="seconds"),
            "code_version": code_version(),
            "stages": stages,
            "n_videos_checked": len(stats_by_video),
            "n_errors": len(errors),
            "n_warnings": len(warns),
            "totals": {"expected_keyframes": total_expected, "actual_keyframes": total_jpg},
            "findings": findings,
            "per_video": {v: {k: s[k] for k in ("n_shots", "n_expected", "n_jpg", "failed_stages")}
                          for v, s in sorted(stats_by_video.items())},
        }, fh, ensure_ascii=False, indent=1)
    os.replace(tmp, report_path)

    print()
    print(f" Báo cáo đầy đủ: {os.path.relpath(report_path, project_root)}")
    failed = bool(errors) or (bool(warns) and args.strict)
    print("=" * 78)
    print(" KẾT LUẬN: " + (
        f"FAIL — {len(errors)} lỗi, {len(warns)} cảnh báo." if errors else
        f"FAIL (--strict) — {len(warns)} cảnh báo được tính là lỗi." if failed else
        f"PASS — dữ liệu toàn vẹn và đầy đủ ({len(warns)} cảnh báo)." if warns else
        "PASS — dữ liệu toàn vẹn và đầy đủ."
    ))
    print("=" * 78)
    return 1 if failed else 0


# ============================================================================
def resolve_targets(args, manifest):
    """Quy đổi -v/-f thành danh sách video_id. Quy ước khớp run_pipeline.sh."""
    if args.video:
        name = os.path.splitext(os.path.basename(args.video))[0]
        if name not in manifest:
            sys.exit(f"Không tìm thấy video '{name}' trong manifest ({len(manifest)} video).")
        return [name]
    if args.folder:
        prefixes = []
        for raw in args.folder.replace(",", " ").split():
            p = raw.strip().replace("Videos_", "").replace("Keyframes_", "")
            if p and p not in prefixes:
                prefixes.append(p)
        targets = sorted(v for v in manifest if any(v.startswith(p + "_") for p in prefixes))
        if not targets:
            available = sorted({v.split("_")[0] for v in manifest})
            sys.exit(f"Không có video nào thuộc folder {prefixes}.\n"
                     f"Các folder hiện có: {', '.join(available)}")
        return targets
    return sorted(manifest)


def discover_models(args):
    if args.model:
        return [m.strip() for m in args.model.replace(",", " ").split() if m.strip()]
    if not os.path.isdir(FEATURES_DIR):
        return []
    return sorted(name for name in os.listdir(FEATURES_DIR)
                  if os.path.isdir(os.path.join(FEATURES_DIR, name))
                  and glob(os.path.join(FEATURES_DIR, name, "*.npy")))


def parse_args():
    p = argparse.ArgumentParser(
        description="[5] Kiểm tra tính toàn vẹn & đầy đủ của pipeline tiền xử lý.",
        epilog=(
            "Ví dụ:\n"
            "  %(prog)s                          # kiểm toàn bộ, mọi stage\n"
            "  %(prog)s -f L21,L22 --verbose     # vài folder, in đủ chi tiết\n"
            "  %(prog)s --stage keyframe         # chỉ một stage\n"
            "  %(prog)s --fast                   # bỏ qua việc mở từng .jpg\n"
            "  %(prog)s --hash --deep-decode     # kiểm tra tải về ở mức chặt nhất (chậm)\n"
            "Exit code 1 khi có ERROR, dùng được để chặn stage sau trong script."
        ),
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    target = p.add_mutually_exclusive_group()
    target.add_argument("-v", "--video", help="Một video ('L21_V001' hoặc đường dẫn .mp4)")
    target.add_argument("-f", "--folder", help="Một/nhiều folder ('L21' | 'L21,L22' | 'Videos_L21')")
    target.add_argument("-a", "--all", action="store_true", help="Toàn bộ dataset (mặc định)")

    p.add_argument("--stage", action="append", choices=ALL_STAGES,
                   help="Chỉ chạy stage này (lặp lại được). Mặc định: tất cả")
    p.add_argument("--model", help="Giới hạn model kiểm tra ('siglip2' | 'siglip2,dinov3')")

    p.add_argument("--refresh-manifest", action="store_true",
                   help="Dựng lại manifest từ đầu bằng ffprobe")
    p.add_argument("--hash", action="store_true",
                   help="Tính sha256 video gốc và lưu vào manifest (chậm, làm một lần)")
    p.add_argument("--deep-decode", action="store_true",
                   help="Decode toàn bộ video bằng ffmpeg để phát hiện file hỏng (rất chậm)")

    p.add_argument("--fast", action="store_true",
                   help="Chỉ stat file .jpg, không mở ảnh (bỏ qua KF_CORRUPT/KF_BLANK)")
    p.add_argument("--deep-jpg", action="store_true",
                   help="Giải mã đầy đủ TOÀN BỘ .jpg thay vì lấy mẫu (bắt cả file cụt lẫn ảnh đơn sắc)")
    p.add_argument("--sample-jpg", type=int, default=3,
                   help="Số ảnh lấy mẫu mỗi video cho phép kiểm đơn sắc (mặc định 3)")
    p.add_argument("--align-sample", type=int, default=0, metavar="N",
                   help="Chỉ đối chiếu index<->.npy trên N video ngẫu nhiên thay vì toàn bộ "
                        "(mặc định 0 = toàn bộ; chỉ dùng khi index quá lớn)")
    p.add_argument("--roundtrip", type=int, default=0, metavar="N",
                   help="Encode lại N ảnh và tra ngược index (cần GPU + venv_embedding)")

    p.add_argument("-j", "--workers", type=int, default=min(16, (os.cpu_count() or 4)),
                   help="Số process song song")
    p.add_argument("--json", help="Tên file báo cáo JSON trong _reports/ (mặc định verify.json)")
    p.add_argument("--strict", action="store_true", help="Coi WARN là lỗi (exit 1)")
    p.add_argument("--verbose", action="store_true", help="In đầy đủ mọi finding")
    p.add_argument("--quiet", action="store_true", help="Chỉ in báo cáo cuối")
    return p.parse_args()


def main():
    global _QUIET
    args = parse_args()
    _QUIET = args.quiet
    stages = args.stage or list(ALL_STAGES)

    findings = []
    manifest = load_or_build_manifest(args, findings)
    if not manifest:
        write_report(findings, {}, manifest, args, stages)
        return 1

    targets = resolve_targets(args, manifest)
    models = discover_models(args)
    if "features" in stages and not models:
        findings.append(finding("features", "NO_MODEL", WARN, None,
                                f"Không có model nào có .npy trong {FEATURES_DIR}"))

    # Lượt quét theo video luôn chạy khi có bất kỳ stage nào cần đến nó: stage
    # index và anomaly không tự kiểm từng video nhưng vẫn cần dữ kiện (danh sách
    # .jpg, số shot) mà lượt quét này thu thập.
    per_video_stages = [s for s in stages if s in ("transnet", "keyframe", "features")]
    stats_by_video = {}
    if per_video_stages or {"index", "anomaly"} & set(stages):
        log(f"[check] {len(targets)} video x stage {per_video_stages or ['(chỉ thu thập dữ kiện)']} "
            f"(model: {', '.join(models) or 'không có'}, {args.workers} worker)...")
        tasks = [(vid, manifest[vid], models, args, per_video_stages) for vid in targets]
        done = 0
        with ProcessPoolExecutor(max_workers=args.workers) as ex:
            futures = {ex.submit(inspect_video, t): t[0] for t in tasks}
            for fut in as_completed(futures):
                vid = futures[fut]
                try:
                    out, stats = fut.result()
                except Exception as exc:
                    findings.append(finding("keyframe", "CHECK_CRASHED", ERROR, vid,
                                            f"{type(exc).__name__}: {exc}"))
                    continue
                findings.extend(out)
                stats_by_video[vid] = stats
                done += 1
                if not _QUIET and done % 200 == 0:
                    log(f"        {done}/{len(tasks)}...")

    if "features" in stages and models:
        check_feature_dims(models, stats_by_video, findings)

    if "index" in stages and models:
        log("[check] index...")
        check_index(models, stats_by_video, manifest, args, findings)
        if args.roundtrip:
            check_roundtrip(models, stats_by_video, args, findings)

    if "anomaly" in stages and stats_by_video:
        check_anomaly(manifest, stats_by_video, findings)

    # Gộp thành MỘT dòng thay vì một cảnh báo mỗi video: thiếu sentinel là tình
    # trạng chung của dữ liệu trích trước khi có cơ chế này, in 873 dòng giống
    # nhau chỉ làm trôi mất các finding thật.
    if "keyframe" in stages:
        unsealed = sorted(v for v, s in stats_by_video.items()
                          if s["n_expected"] and not s["sealed"])
        if unsealed:
            findings.append(finding(
                "keyframe", "SENTINEL_MISSING", WARN, None,
                f"{len(unsealed)}/{len(stats_by_video)} thư mục keyframe chưa có "
                f"{SENTINEL_NAME} — được trích trước khi có cơ chế sentinel nên chưa được "
                f"chứng nhận là chạy trọn vẹn. Chạy lại run_pipeline.sh -m keyframe để "
                f"chứng nhận (video đã đủ sẽ được trích lại một lần rồi thôi). "
                f"VD: {preview(unsealed)}"))

    return write_report(findings, stats_by_video, manifest, args, stages)


if __name__ == "__main__":
    sys.exit(main())
