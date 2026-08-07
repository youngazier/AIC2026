#!/usr/bin/env python3
"""[4b] Gom các file .npy do run_embedding.py sinh ra thành FAISS index.

Hai tầng index cho mỗi model:
  - global   : toàn bộ keyframe của mọi video đã encode
  - playlist : mỗi chương trình truyền hình một index riêng, để truy vấn chỉ
               quét subset thay vì toàn bộ (mapping từ build_playlist_mapping.py)

Script này KHÔNG nạp model, không đụng GPU để encode — chỉ đọc .npy, ghép lại
và ghi index, nên chạy rất nhanh và không cần chờ encode xong toàn bộ dataset.

Phải activate venv_embedding trước khi chạy (faiss chỉ có trong đó):

    source data_processing/feature_extraction/venv_embedding/bin/activate
    python data_processing/feature_extraction/build_index.py                 # mọi model có .npy
    python data_processing/feature_extraction/build_index.py --model siglip2 # chỉ một model

CỐ Ý không có cờ -v/-f như run_embedding.py: index phải phủ TẤT CẢ .npy hiện
có, build từ một tập con sẽ ghi đè index cũ thành bản thiếu video.
"""
import os
import sys
import argparse
import json
import logging
from collections import defaultdict
from glob import glob

import faiss
import numpy as np
import torch

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "../../")))

from utils.config_loader import load_config
from utils.load_all_video_keyframes_info import load_all_video_keyframes_info

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

KNOWN_MODELS = ["siglip2", "beit3", "dinov3"]


def build_flat_index_if_changed(embedding_array, infos, index_path, meta_path, label, use_gpu=False, force=False):
    """
    Build IndexFlatL2 + metadata song hành, nhưng BỎ QUA nếu metadata trên đĩa
    đã y hệt metadata mới — tức playlist/global đó không có video mới thì
    không ghi lại gì cả (yêu cầu cập nhật incremental).
    """
    info_array = np.array(infos)
    if not force and os.path.exists(index_path) and os.path.exists(meta_path):
        old_info = np.load(meta_path, allow_pickle=True)
        if old_info.shape == info_array.shape and np.array_equal(old_info, info_array):
            logger.info(f"[{label}] {len(info_array)} vector, không có video mới — giữ nguyên index.")
            return False

    dimension = embedding_array.shape[1]
    index = faiss.IndexFlatL2(dimension)
    if use_gpu and torch.cuda.is_available():
        res = faiss.StandardGpuResources()
        gpu_index = faiss.index_cpu_to_gpu(res, 0, index)
        gpu_index.add(embedding_array)
        index = faiss.index_gpu_to_cpu(gpu_index)
    else:
        index.add(embedding_array)

    faiss.write_index(index, index_path)
    np.save(meta_path, info_array)
    logger.info(f"[{label}] Đã ghi {index.ntotal} vector (dim={dimension}) -> {os.path.basename(index_path)}")
    return True


def discover_models(features_dir, requested):
    """Chọn danh sách model cần build.

    `--model all` (mặc định) quét thư mục features và lấy mọi model đang có
    .npy — kể cả model mới thêm sau này, không cần sửa script.
    """
    if requested != "all":
        model_dir = os.path.join(features_dir, requested)
        if not glob(os.path.join(model_dir, "*.npy")):
            sys.exit(
                f"Không có file .npy nào trong {model_dir}\n"
                f"Chạy run_embedding.py --model {requested} trước."
            )
        return [requested]

    found = sorted(
        name for name in os.listdir(features_dir)
        if os.path.isdir(os.path.join(features_dir, name))
        and glob(os.path.join(features_dir, name, "*.npy"))
    ) if os.path.isdir(features_dir) else []

    if not found:
        sys.exit(
            f"Không tìm thấy features nào trong {features_dir}\n"
            f"Chạy run_embedding.py trước để sinh .npy."
        )
    return found


def load_embedding_dict(model_dir, all_video, video_keyframe_dict, keyframes_dir):
    """Đọc mọi .npy của một model -> {video_id: {keyframe_name: vector}}.

    Hàng thứ i trong .npy ứng với keyframe thứ i trong danh sách keyframe HỢP LỆ
    (ảnh .jpg thực sự tồn tại) đã sắp xếp của video đó — đúng thứ tự mà
    run_embedding.py đã encode. Vì vậy phải lọc lại theo file .jpg trên đĩa chứ
    không dùng thẳng video_keyframe_dict.
    """
    embedding_dict = {}
    missing = []

    for v in all_video:
        if v not in video_keyframe_dict:
            continue
        clip_path = os.path.join(model_dir, f"{v}.npy")
        if not os.path.exists(clip_path):
            missing.append(v)
            continue

        a = np.load(clip_path)
        # Bỏ qua các array rỗng để không bị lỗi lúc map dictionary
        if a.size == 0:
            continue

        # Tìm thư mục chứa keyframe thực tế của video v
        video_dir_candidates = glob(os.path.join(keyframes_dir, "*", v))
        video_dir = video_dir_candidates[0] if video_dir_candidates else os.path.join(keyframes_dir, v)

        valid_keyframes = []
        for k in video_keyframe_dict[v]:
            if os.path.exists(os.path.join(video_dir, f"{k}.jpg")):
                valid_keyframes.append(k)

        if len(valid_keyframes) != a.shape[0]:
            logger.warning(
                f"{v}: .npy có {a.shape[0]} vector nhưng đếm được {len(valid_keyframes)} keyframe. "
                f"Chỉ map được {min(len(valid_keyframes), a.shape[0])} vector đầu — "
                f"xoá {clip_path} rồi encode lại nếu số liệu lệch nhiều."
            )

        embedding_dict[v] = {}
        for i, k in enumerate(valid_keyframes):
            if i < a.shape[0]:
                embedding_dict[v][k] = a[i]

    return embedding_dict, missing


def build_for_model(model, config, all_video, video_keyframe_dict, force=False):
    features_dir = config["paths"]["features_dir"]
    index_dir = config["paths"]["index_dir"]
    model_dir = os.path.join(features_dir, model)

    logger.info(f"===== {model} =====")
    embedding_dict, missing = load_embedding_dict(
        model_dir, all_video, video_keyframe_dict, config["paths"]["keyframes_dir"]
    )
    if missing:
        logger.warning(
            f"[{model}] {len(missing)}/{len(all_video)} video chưa có .npy, không nằm trong index. "
            f"VD: {missing[:5]}"
        )

    # Gom vector theo thứ tự all_video -> keyframe đã sắp xếp, để index ổn định
    # giữa các lần build.
    embedding_list = []
    embedding_info = []
    for v in all_video:
        if v not in embedding_dict:
            continue
        for k in video_keyframe_dict[v]:
            if k in embedding_dict[v]:
                embedding_list.append(embedding_dict[v][k])
                embedding_info.append((v, k))

    if not embedding_list:
        logger.error(f"[{model}] Không có vector nào trong {model_dir} — bỏ qua model này.")
        return

    embedding_array = np.array(embedding_list, dtype=np.float32)
    os.makedirs(index_dir, exist_ok=True)

    # --- 1. GLOBAL INDEX (toàn bộ video đã encode) ---
    build_flat_index_if_changed(
        embedding_array,
        embedding_info,
        os.path.join(index_dir, f"{model}_keyframe_embedding.index"),
        os.path.join(index_dir, f"{model}_keyframe_metadata.npy"),
        label=f"{model}/global",
        use_gpu=True,
        force=force,
    )

    # --- 2. PLAYLIST INDEXES (chỉ build lại playlist có video mới) ---
    vp_path = os.path.join(config["paths"]["data_dir"], "video_to_playlist.json")
    if not os.path.exists(vp_path):
        logger.warning(
            f"Không tìm thấy {vp_path} — bỏ qua bước index theo playlist. "
            f"Chạy data_processing/build_playlist_mapping.py trước rồi chạy lại script này."
        )
        return
    with open(vp_path, encoding="utf-8") as f:
        video_playlist = json.load(f)

    playlist_dir = os.path.join(index_dir, "playlist")
    os.makedirs(playlist_dir, exist_ok=True)

    # Gom video theo playlist, giữ nguyên thứ tự all_video để index ổn định
    playlist_videos = defaultdict(list)
    unmapped = []
    for v in all_video:
        if v not in embedding_dict:
            continue
        playlist = video_playlist.get(v)
        if playlist:
            playlist_videos[playlist].append(v)
        else:
            unmapped.append(v)
    if unmapped:
        logger.warning(
            f"{len(unmapped)} video chưa có trong video_to_playlist.json — "
            f"chỉ nằm trong index global. VD: {unmapped[:5]}"
        )

    for playlist in sorted(playlist_videos):
        vectors, infos = [], []
        for v in playlist_videos[playlist]:
            for k in video_keyframe_dict[v]:
                if k in embedding_dict[v]:
                    vectors.append(embedding_dict[v][k])
                    infos.append((v, k))
        if not vectors:
            continue
        build_flat_index_if_changed(
            np.array(vectors, dtype=np.float32),
            infos,
            os.path.join(playlist_dir, f"{model}_{playlist}.index"),
            os.path.join(playlist_dir, f"{model}_{playlist}_keyframe_metadata.npy"),
            label=f"{model}/{playlist}",
            force=force,
        )


def main():
    parser = argparse.ArgumentParser(
        description="[4b] Build FAISS index (global + playlist) từ các .npy của run_embedding.py.",
        epilog=(
            "Ví dụ:\n"
            "  %(prog)s                       # build cho mọi model đang có .npy\n"
            "  %(prog)s --model siglip2       # chỉ một model\n"
            "  %(prog)s --force               # build lại kể cả khi metadata không đổi\n"
            "Index luôn phủ toàn bộ .npy hiện có nên không có cờ chọn video/folder."
        ),
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument(
        "--model", type=str, default="all", choices=[*KNOWN_MODELS, "all"],
        help="Model cần build index; 'all' (mặc định) = mọi model đang có .npy",
    )
    parser.add_argument(
        "--force", action="store_true",
        help="Build lại kể cả khi metadata không đổi (mặc định bỏ qua để tiết kiệm)",
    )
    args = parser.parse_args()

    config = load_config()
    all_video, video_keyframe_dict = load_all_video_keyframes_info(
        config["paths"]["videos_dir"], config["paths"]["keyframes_dir"]
    )

    models = discover_models(config["paths"]["features_dir"], args.model)
    logger.info(f"Build index cho model: {', '.join(models)}")

    for model in models:
        build_for_model(model, config, all_video, video_keyframe_dict, force=args.force)

    logger.info("Done.")


if __name__ == "__main__":
    main()
