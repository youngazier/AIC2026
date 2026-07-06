"""
Sinh file ánh xạ video_id -> playlist bằng cách ĐỌC METADATA TỪNG VIDEO
(media-info/*.json của BTC), KHÔNG dựa vào tên folder Lxx — nên BTC có trộn
nhiều thể loại video vào cùng folder cũng không ảnh hưởng.

Cơ chế phân loại 2 tầng (cascade):
  1. CHANNEL MAPPING (data/playlist_mapping.json — gán tay channel_id -> playlist):
     tín hiệu tin cậy nhất, vì mỗi kênh YouTube thường là một chương trình/thể loại.
  2. KEYWORD RULES (data/playlist_rules.json — luật từ khóa trên title/keywords/author):
     lưới đỡ cho các trường hợp channel mapping bó tay: kênh mới chưa gán tay,
     hoặc khi bạn nghi một kênh trộn nhiều thể loại (dùng --prefer rules).

Video không tầng nào phân loại được sẽ được liệt kê cuối script để xử lý tay.
Nếu 2 tầng cho kết quả KHÁC NHAU, script in báo cáo "disagreement" — đây chính là
dấu hiệu kênh trộn thể loại, đáng để bạn mở video ra kiểm tra.

Đầu ra:
  - data/video_to_playlist.json: {"L21_V001": "60-Giay", ...}
    (đầu vào của run_embedding.py và cột videos.playlist trong SQLite sau này)

Cách chạy:
  python data_processing/build_playlist_mapping.py                  # channel ưu tiên (mặc định)
  python data_processing/build_playlist_mapping.py --prefer rules   # rules ưu tiên
"""
import argparse
import json
import os
import sys
import unicodedata
from collections import Counter
from glob import glob

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
from utils.config_loader import load_config


def normalize(text: str) -> str:
    """lowercase + bỏ dấu tiếng Việt để so khớp từ khóa bất chấp cách gõ."""
    text = text.lower().replace("đ", "d")
    text = unicodedata.normalize("NFD", text)
    return "".join(c for c in text if unicodedata.category(c) != "Mn")


def classify_by_rules(info: dict, rules: list, use_description: bool = False) -> str | None:
    """Chấm điểm từng playlist theo số pattern khớp trên title/keywords/author.
    Trả về playlist thắng rõ ràng; hòa điểm hoặc không khớp gì -> None."""
    parts = [info.get("title") or "", info.get("author") or ""]
    parts += info.get("keywords") or []
    if use_description:
        parts.append(info.get("description") or "")
    text = normalize(" | ".join(p for p in parts if p))

    scores = Counter()
    for rule in rules:
        hits = sum(1 for p in rule["patterns"] if normalize(p) in text)
        if hits:
            scores[rule["playlist"]] += hits

    ranked = scores.most_common(2)
    if not ranked:
        return None
    if len(ranked) == 2 and ranked[0][1] == ranked[1][1]:
        return None  # hòa điểm = không đủ chắc chắn
    return ranked[0][0]


def main():
    config = load_config()

    parser = argparse.ArgumentParser(description="Build video_id -> playlist mapping")
    parser.add_argument(
        "--media-info-dir",
        default="/tmp2/maitanha/vgu/ttn/data/AIC2025/media-info-aic25-b1/media-info",
        help="Thư mục chứa các file media-info *.json của BTC",
    )
    parser.add_argument(
        "--channel-mapping",
        default=os.path.join(config["paths"]["data_dir"], "playlist_mapping.json"),
        help="File gán tay channel_id -> playlist (tầng 1)",
    )
    parser.add_argument(
        "--rules",
        default=os.path.join(config["paths"]["data_dir"], "playlist_rules.json"),
        help="File luật từ khóa (tầng 2)",
    )
    parser.add_argument(
        "--prefer",
        choices=["channel", "rules"],
        default="channel",
        help="Tầng nào thắng khi cả hai đều cho kết quả (mặc định: channel)",
    )
    parser.add_argument(
        "--use-description",
        action="store_true",
        help="Cho luật từ khóa quét cả description (dễ nhiễu vì chứa boilerplate kênh)",
    )
    parser.add_argument(
        "--output",
        default=os.path.join(config["paths"]["data_dir"], "video_to_playlist.json"),
    )
    args = parser.parse_args()

    channel_map = {}
    if os.path.exists(args.channel_mapping):
        with open(args.channel_mapping, encoding="utf-8") as f:
            channel_map = json.load(f)
    else:
        print(f"(Không có {args.channel_mapping} — chạy thuần bằng keyword rules)")

    with open(args.rules, encoding="utf-8") as f:
        rules = json.load(f)

    files = sorted(glob(os.path.join(args.media_info_dir, "*.json")))
    if not files:
        sys.exit(f"Không tìm thấy file media-info nào trong {args.media_info_dir}")

    video_to_playlist = {}
    source_counts = Counter()
    disagreements = []  # (video_id, theo_channel, theo_rules)
    unmapped = []       # (video_id, channel_id, author, title)

    for fp in files:
        video_id = os.path.splitext(os.path.basename(fp))[0]
        with open(fp, encoding="utf-8") as f:
            info = json.load(f)

        entry = channel_map.get(info.get("channel_id"))
        by_channel = entry["playlist"] if entry else None
        by_rules = classify_by_rules(info, rules, args.use_description)

        if by_channel and by_rules and by_channel != by_rules:
            disagreements.append((video_id, by_channel, by_rules))

        if args.prefer == "channel":
            final, source = (by_channel, "channel") if by_channel else (by_rules, "rules")
        else:
            final, source = (by_rules, "rules") if by_rules else (by_channel, "channel")

        if final:
            video_to_playlist[video_id] = final
            source_counts[source] += 1
        else:
            unmapped.append((video_id, info.get("channel_id"), info.get("author"), info.get("title")))

    os.makedirs(os.path.dirname(args.output), exist_ok=True)
    with open(args.output, "w", encoding="utf-8") as f:
        json.dump(video_to_playlist, f, ensure_ascii=False, indent=1, sort_keys=True)

    # ---- Báo cáo ----
    print(f"Đã ghi {len(video_to_playlist)}/{len(files)} video -> {args.output}")
    for pl, n in Counter(video_to_playlist.values()).most_common():
        print(f"  {pl}: {n} video")
    print(f"Nguồn quyết định: {dict(source_counts)}")

    if disagreements:
        print(f"\nDISAGREEMENT — 2 tầng cho kết quả khác nhau ({len(disagreements)} video, "
              f"đang lấy theo --prefer {args.prefer}). Dấu hiệu kênh trộn thể loại, nên kiểm tra:")
        for vid, ch, ru in disagreements[:20]:
            print(f"  {vid}: channel->{ch} | rules->{ru}")

    if unmapped:
        print(f"\nCẢNH BÁO — {len(unmapped)} video KHÔNG phân loại được, cần bổ sung "
              f"playlist_mapping.json hoặc playlist_rules.json:")
        for vid, cid, author, title in unmapped[:20]:
            print(f"  {vid} | {cid} | {author} | {title}")
        sys.exit(1)


if __name__ == "__main__":
    main()
