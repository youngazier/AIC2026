from glob import glob
from typing import Tuple


def load_all_video_keyframes_info(videos_dir: str, keyframes_dir: str) -> Tuple[list[str], dict[str, list]]:
    """Liệt kê toàn bộ video gốc và keyframe đã trích xuất.

    Video gốc và keyframe nằm ở HAI gốc thư mục khác nhau nên phải truyền riêng:
        videos_dir    = data/videos                          -> Videos_LXX/LXX_VXXX.mp4
        keyframes_dir = data_processed/keyframes_transnetv2  -> Keyframes_LXX/LXX_VXXX/*.jpg
    Cả hai đều lấy từ config.yaml, đừng hardcode ở nơi gọi.

    Trả về:
        all_video           : danh sách video_id đã sắp xếp, VD ["L21_V001", ...]
        video_keyframe_dict : {video_id: [tên keyframe đã sắp xếp]}
    """
    print("loading all videos and keyframes information")
    all_keyframe = glob(f"{keyframes_dir}/*/*/*.jpg")
    video_keyframe_dict = {}
    all_video = glob(f"{videos_dir}/*/*.mp4")
    all_video = [v.rsplit("/", 1)[-1][:-4] for v in all_video]
    all_video = sorted(all_video)

    print(f"loaded {len(all_keyframe)} keyframes")
    print(f"loaded {len(all_video)} videos")

    for kf in all_keyframe:
        _, vid, kf = kf[:-4].rsplit("/", 2)
        if vid not in video_keyframe_dict.keys():
            video_keyframe_dict[vid] = [kf]
        else:
            video_keyframe_dict[vid].append(kf)

    for k, v in video_keyframe_dict.items():
        video_keyframe_dict[k] = sorted(v)

    return all_video, video_keyframe_dict