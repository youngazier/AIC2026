#!/usr/bin/env python3
"""[4a] Encode keyframe thành vector, ghi ra .npy theo từng video.

Script này CHỈ trích xuất đặc trưng. Việc gom .npy lại thành FAISS index nằm ở
`build_index.py` — chạy riêng sau khi encode xong. Tách đôi như vậy vì hai bước
có đặc tính rất khác nhau: encode tốn GPU và hàng giờ đồng hồ, build index chỉ
tốn CPU/RAM và vài giây, lại phải quét TOÀN BỘ .npy nên không chạy chung với
cơ chế lọc -v/-f ở đây được.

Phải activate venv_embedding trước khi chạy (torch/open_clip chỉ có trong đó):

    source data_processing/feature_extraction/venv_embedding/bin/activate
    python data_processing/feature_extraction/run_embedding.py -a --model siglip2
    python data_processing/feature_extraction/build_index.py --model siglip2
"""
import os
import sys
import argparse
import logging
from glob import glob

import numpy as np
import torch
from PIL import Image
from torch.utils.data import DataLoader, Dataset
from tqdm import tqdm

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "../../")))

from utils.load_all_video_keyframes_info import load_all_video_keyframes_info

from data_processing.feature_extraction.feature_store import is_complete, save_atomic
from data_processing.feature_extraction.siglip2_embedding import SigLIP2Embedding
from data_processing.feature_extraction.beit3_embedding import Beit3Embedding
from data_processing.feature_extraction.dinov3_embedding import DinoV3Embedding

from utils.config_loader import load_config

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class KeyframeDataset(Dataset):
    def __init__(self, keyframe_paths, keyframe_names, preprocess_fn):
        self.keyframe_paths = keyframe_paths
        self.keyframe_names = keyframe_names
        self.preprocess_fn = preprocess_fn

    def __len__(self):
        return len(self.keyframe_paths)

    def __getitem__(self, idx):
        path = self.keyframe_paths[idx]
        name = self.keyframe_names[idx]
        try:
            image = Image.open(path).convert("RGB")
            tensor = self.preprocess_fn(image)
            return tensor, name
        except Exception as e:
            logger.error(f"Error loading image {path}: {e}")
            return None, name

def collate_fn(batch):
    batch = [item for item in batch if item[0] is not None]
    if not batch:
        return None
    tensors, names = zip(*batch)
    return torch.stack(tensors), names


def create_extractor(model_name):
    if model_name == "siglip2":
        return SigLIP2Embedding()
    if model_name == "beit3":
        return Beit3Embedding()
    if model_name == "dinov3":
        return DinoV3Embedding()
    raise ValueError("Unsupported model")


def resolve_targets(args, all_video):
    """Quy đổi cờ -v/-f/-a thành danh sách video_id cần encode.

    Quy ước đặt tên khớp với run_pipeline.sh / keyframe_extraction.py:
      -v  nhận cả "L21_V001" lẫn đường dẫn "/.../L21_V001.mp4"
      -f  nhận cả "L21" lẫn "Videos_L21"/"Keyframes_L21", ngăn bằng dấu phẩy
          hoặc khoảng trắng ("L21,L22" hay "L21 L22" đều được)
    """
    if args.video:
        name = os.path.splitext(os.path.basename(args.video))[0]
        if name not in all_video:
            sys.exit(f"Không tìm thấy video '{name}' trong dataset ({len(all_video)} video).")
        return [name]

    if args.folder:
        prefixes = []
        for raw in args.folder.replace(",", " ").split():
            p = raw.strip().replace("Videos_", "").replace("Keyframes_", "")
            if p and p not in prefixes:
                prefixes.append(p)
        targets = [v for v in all_video if any(v.startswith(p + "_") for p in prefixes)]
        if not targets:
            available = sorted({v.split("_")[0] for v in all_video})
            sys.exit(
                f"Không có video nào thuộc folder {prefixes}.\n"
                f"Các folder hiện có: {', '.join(available)}"
            )
        # Báo riêng folder nào rỗng, tránh gõ nhầm 1 trong nhiều folder mà không hay
        empty = [p for p in prefixes if not any(v.startswith(p + "_") for v in all_video)]
        if empty:
            logger.warning(f"Các folder sau không khớp video nào, bỏ qua: {empty}")
        return targets

    return list(all_video)


def main():
    parser = argparse.ArgumentParser(
        description="[4a] Encode keyframe thành vector .npy (chưa build index — xem build_index.py).",
        epilog=(
            "Ví dụ:\n"
            "  %(prog)s -a --model siglip2 --batch_size 64   # toàn bộ dataset\n"
            "  %(prog)s -v L21_V001                          # một video\n"
            "  %(prog)s -f L21                               # một folder\n"
            "  %(prog)s -f L21,L22   |   %(prog)s -f 'L21 L22'   # nhiều folder\n"
            "Không truyền -v/-f/-a thì mặc định chạy toàn bộ dataset.\n"
            "Encode xong nhớ chạy ./build_index.py để cập nhật FAISS index."
        ),
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument("--model", type=str, default="siglip2", choices=["siglip2", "beit3", "dinov3"], help="Model dùng để embedding")
    parser.add_argument("--batch_size", type=int, default=64, help="Batch size khi encode")

    # Chọn tối đa MỘT target; không chọn gì = toàn bộ dataset (giữ tương thích
    # với cách gọi cũ chỉ có --model/--batch_size).
    target = parser.add_mutually_exclusive_group()
    target.add_argument("-v", "--video", type=str, help="Một video (tên 'L21_V001' hoặc đường dẫn .mp4)")
    target.add_argument("-f", "--folder", type=str, help="Một hoặc nhiều folder, ngăn bằng dấu phẩy/khoảng trắng ('L21' | 'L21,L22' | 'Videos_L21')")
    target.add_argument("-a", "--all", action="store_true", help="Toàn bộ dataset (mặc định)")
    args = parser.parse_args()

    config = load_config()
    all_video, video_keyframe_dict = load_all_video_keyframes_info(
        config["paths"]["videos_dir"], config["paths"]["keyframes_dir"]
    )

    targets = resolve_targets(args, all_video)

    # Model được nạp LAZY: chỉ khởi tạo khi thực sự có video cần encode.
    # Nhờ đó chạy lại lúc mọi video đã có .npy sẽ không tốn thời gian nạp
    # model lên GPU.
    extractor = None

    out_dir = os.path.join(config["paths"]["features_dir"], f"{args.model}")
    os.makedirs(out_dir, exist_ok=True)

    logger.info(f"Encode {len(targets)}/{len(all_video)} video với model {args.model} -> {out_dir}")

    preprocess_fn = None

    # Biến lưu trữ số chiều vector động (Dynamic Dimension)
    feature_dim = None

    encoded = 0
    skipped = 0

    failed = []

    for v in tqdm(targets, desc=f"Processing videos with {args.model}"):
        if v not in video_keyframe_dict or not video_keyframe_dict[v]:
            logger.warning(f"Video {v} has no keyframes in video_keyframe_dict. Skipping.")
            continue

        np_out = f"{out_dir}/{v}.npy"

        # Tìm thư mục chứa keyframe thực tế của video v
        video_dir_candidates = glob(os.path.join(config["paths"]["keyframes_dir"], "*", v))
        video_dir = video_dir_candidates[0] if video_dir_candidates else os.path.join(config["paths"]["keyframes_dir"], v)

        keyframe_paths = []
        keyframe_names = []
        for k in video_keyframe_dict[v]:
            keyframe_path = os.path.join(video_dir, f"{k}.jpg")
            if os.path.exists(keyframe_path):
                keyframe_paths.append(keyframe_path)
                keyframe_names.append(k)
            else:
                logger.warning(f"Image not found: {keyframe_path}")

        # Bỏ qua chỉ khi .npy hiện có ĐÚNG là kết quả đầy đủ của số keyframe đang
        # có trên đĩa (luật ở feature_store.is_complete, dùng chung với bên kiểm).
        if is_complete(np_out, len(keyframe_paths)):
            skipped += 1
            continue
        if os.path.exists(np_out):
            logger.warning(
                f"{v}: .npy hiện có không khớp {len(keyframe_paths)} keyframe trên đĩa "
                f"— encode lại và ghi đè."
            )

        keyframe_list = []
        if keyframe_paths:
            if extractor is None:
                extractor = create_extractor(args.model)
                preprocess_fn = getattr(extractor, "preprocess", None) or getattr(extractor, "image_processor", None)
            dataset = KeyframeDataset(keyframe_paths, keyframe_names, preprocess_fn)
            dataloader = DataLoader(
                dataset,
                batch_size=args.batch_size,
                shuffle=False,
                num_workers=4,
                collate_fn=collate_fn
            )

            for batch in dataloader:
                if batch is None:
                    continue
                image_tensors, _ = batch
                embeddings = extractor.encode_images_batch(image_tensors)
                if embeddings is not None:
                    if feature_dim is None:
                        feature_dim = embeddings.shape[-1]
                        logger.info(f"Dynamically inferred feature dimension: {feature_dim}")
                    keyframe_list.append(embeddings)

        keyframe_array = np.vstack(keyframe_list) if keyframe_list else None

        # Cổng nghiệm thu: số vector phải bằng đúng số ảnh đưa vào encode.
        # collate_fn loại IM LẶNG mọi ảnh không mở được ra khỏi batch, nên thiếu
        # vector là chuyện xảy ra được mà không có exception nào.
        if keyframe_array is None or keyframe_array.shape[0] != len(keyframe_paths):
            got = 0 if keyframe_array is None else keyframe_array.shape[0]
            if not keyframe_paths:
                logger.error(
                    f"{v}: không tìm thấy ảnh .jpg nào để encode. Không ghi .npy — "
                    f"một .npy rỗng sẽ khiến video này biến mất khỏi index mà vẫn được "
                    f"coi là đã xong. Trích keyframe trước rồi chạy lại."
                )
            else:
                logger.error(
                    f"{v}: encode ra {got}/{len(keyframe_paths)} vector — có ảnh hỏng bị "
                    f"loại khỏi batch. Không ghi .npy."
                )
            failed.append(v)
            continue

        save_atomic(np_out, keyframe_array)
        encoded += 1

    total_npy = len(glob(os.path.join(out_dir, "*.npy")))
    logger.info(
        f"Xong. Encode mới {encoded} video, bỏ qua {skipped} video đã đủ .npy. "
        f"Tổng cộng {total_npy} file .npy trong {out_dir}"
    )
    if encoded:
        logger.info(
            f"Có .npy mới — chạy tiếp để cập nhật FAISS index:\n"
            f"    {os.path.join(os.path.dirname(os.path.abspath(__file__)), 'build_index.py')} --model {args.model}"
        )

    # Chạy hết mọi video rồi mới báo lỗi, và exit != 0 để script gọi bên ngoài
    # biết mà dừng — im lặng exit 0 với dữ liệu thiếu là cách lỗi lan xuống index.
    if failed:
        logger.error(
            f"{len(failed)}/{len(targets)} video KHÔNG encode được: "
            f"{failed[:10]}{'...' if len(failed) > 10 else ''}"
        )
        sys.exit(1)

if __name__ == "__main__":
    main()
