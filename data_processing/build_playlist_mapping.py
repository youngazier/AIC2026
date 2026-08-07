"""
Sinh file ánh xạ video_id -> playlist cho dự án AIC2026.

  Input:  data/media-info-aic25-b*/media-info/*.json  (metadata gốc của BTC, tự dò tìm)
          data/playlist_rules.json                    (bộ luật channel/title -> playlist)
  Output: data/video_to_playlist.json                 ({video_id: playlist})

Playlist = tên chương trình truyền hình (đúng danh sách BTC công bố). Mỗi playlist được
build thành 1 FAISS index riêng (data_processed/index/playlist/{model}_{playlist}.index) để khi truy
vấn chỉ quét subset thay vì index global.

--- TÍN HIỆU PHÂN LOẠI (thiết kế để dùng lại được cho b2/b3) ---
KHÔNG dùng folder prefix (Lxx): số hiệu folder đổi theo từng batch, không đoán trước được.
KHÔNG dùng keywords: keywords của L24 (múa lân) chứa "Serie A", "ONE championship" - là
keywords chung của cả kênh, không phải của video.

Dùng 2 tầng:
  1. channel_id -> khoanh vùng các rule ứng viên (tín hiệu ổn định nhất giữa các batch)
  2. regex text -> phân biệt khi 1 kênh phát nhiều chương trình. Bắt buộc phải có, vì:
       - HTV Sports        = Cúp Truyền hình (xe đạp) + Lân Sư Rồng Chợ Lớn
       - HTV Entertainment = Tản mạn Mê Kông + Đôi mắt Mê Kông
     Khớp title trước; title không đủ thông tin (VD tiêu đề chỉ có tên vận động viên,
     hoặc chỉ là một chuỗi hashtag) thì mới xét tiếp description.

Video của kênh lạ (chương trình mới ở b2/b3) không bị bỏ rơi: nó được cấp playlist tự sinh
từ tên kênh (`Khac-<author>`) và được liệt kê ở cuối báo cáo để bổ sung rule.

--- KHI CÓ BATCH MỚI ---
Giải nén media-info của batch mới vào data/ rồi chạy lại (không cần truyền tham số):
    python3 data_processing/build_playlist_mapping.py --strict
Script tự dò mọi thư mục data/media-info* và gộp lại. --strict trả về mã lỗi 1 nếu có
kênh lạ chưa có rule, kèm channel_id để chép thẳng vào data/playlist_rules.json.
"""
import argparse
import json
import os
import re
import sys
import unicodedata
from collections import Counter, defaultdict
from glob import glob

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
from utils.config_loader import load_config

META_FIELDS = ("title", "description", "author", "channel_id")
PLAYLIST_NAME_RE = re.compile(r"^[A-Za-z0-9-]+$")  # playlist được ghép vào tên file index


# ---------------------------------------------------------------------------
# Tiện ích
# ---------------------------------------------------------------------------
def normalize(text):
    """Chuẩn hoá về chữ thường, bỏ dấu tiếng Việt, gom khoảng trắng.

    'Đôi Mắt MeKong  Tập 3' -> 'doi mat mekong tap 3'
    """
    if not text:
        return ""
    text = text.replace("Đ", "D").replace("đ", "d")
    text = unicodedata.normalize("NFD", text)
    text = "".join(c for c in text if unicodedata.category(c) != "Mn")
    return re.sub(r"\s+", " ", text.lower()).strip()


def slugify(text):
    """Sinh nhãn an toàn để ghép vào tên file index."""
    return re.sub(r"[^a-z0-9]+", "-", normalize(text)).strip("-") or "Unknown"


def load_rules(path):
    """Đọc + validate data/playlist_rules.json. Regex được compile sẵn 1 lần.

    Đây là nguồn duy nhất của bộ luật: thiếu file thì dừng hẳn, không có bản dự phòng
    trong code — 2 nguồn luật song song chính là thứ đã làm playlist_mapping.json và
    playlist_rules.json (bản cũ) lệch nhau lúc trước.
    """
    if not os.path.exists(path):
        sys.exit(f"LỖI: Không tìm thấy bộ luật {path}")
    try:
        with open(path, encoding="utf-8") as f:
            rules = json.load(f)["rules"]
    except (json.JSONDecodeError, KeyError, TypeError) as e:
        sys.exit(f"LỖI: {path} sai định dạng (cần object có khoá 'rules'): {e}")

    seen_playlists = set()
    for i, rule in enumerate(rules):
        where = f"{path} rule #{i}"
        playlist = rule.get("playlist")
        if not playlist or not PLAYLIST_NAME_RE.match(playlist):
            sys.exit(f"LỖI: {where}: 'playlist' phải là chuỗi chỉ gồm [A-Za-z0-9-], đang là {playlist!r}")
        if playlist in seen_playlists:
            sys.exit(f"LỖI: {where}: playlist {playlist!r} bị khai báo trùng")
        seen_playlists.add(playlist)

        if not isinstance(rule.get("channels"), list) or not rule["channels"]:
            sys.exit(f"LỖI: {where} ({playlist}): 'channels' phải là list không rỗng")

        pattern = rule.get("title")
        if pattern is None:
            rule["_regex"] = None
        elif isinstance(pattern, str):
            try:
                rule["_regex"] = re.compile(pattern)
            except re.error as e:
                sys.exit(f"LỖI: {where} ({playlist}): regex 'title' không hợp lệ — {e}")
        else:
            sys.exit(f"LỖI: {where} ({playlist}): 'title' phải là chuỗi regex hoặc null")

    # Kênh phát nhiều chương trình thì bắt buộc mọi rule của kênh đó phải có regex,
    # nếu không việc phân biệt sẽ phụ thuộc vào thứ tự khai báo -> gán sai âm thầm.
    per_channel = defaultdict(list)
    for rule in rules:
        for channel_id in rule["channels"]:
            per_channel[channel_id].append(rule)
    for channel_id, shared in per_channel.items():
        if len(shared) > 1:
            no_regex = [r["playlist"] for r in shared if r["_regex"] is None]
            if no_regex:
                sys.exit(
                    f"LỖI: channel {channel_id} được {len(shared)} playlist dùng chung, "
                    f"nên mọi rule của nó phải có 'title' để phân biệt. Thiếu ở: {no_regex}"
                )
    return rules


def discover_media_info_dirs(data_dir, explicit):
    """Tìm mọi thư mục media-info của tất cả các batch (b1, b2, b3...).

    Chấp nhận: đường dẫn tới chính thư mục chứa *.json, hoặc thư mục cha của nó.
    """
    roots = explicit or sorted(glob(os.path.join(data_dir, "media-info*")))
    dirs = []
    for root in roots:
        if not os.path.isdir(root):
            continue
        if glob(os.path.join(root, "*.json")):
            dirs.append(root)
        # Cấu trúc BTC: media-info-aic25-b1/media-info/*.json
        dirs.extend(d for d in sorted(glob(os.path.join(root, "*", ""))) if glob(os.path.join(d, "*.json")))
    return sorted({os.path.normpath(d) for d in dirs})


def load_videos(dirs):
    """Đọc metadata mọi video -> {video_id: {title, description, author, channel_id}}."""
    videos, broken = {}, []
    for d in dirs:
        for fp in sorted(glob(os.path.join(d, "*.json"))):
            video_id = os.path.splitext(os.path.basename(fp))[0]
            try:
                with open(fp, encoding="utf-8") as f:
                    info = json.load(f)
            except (OSError, json.JSONDecodeError) as e:
                broken.append((video_id, str(e)))
                continue
            if video_id in videos:
                continue  # trùng video_id giữa các batch: giữ bản gặp trước
            videos[video_id] = {k: info.get(k) for k in META_FIELDS}
    return videos, broken


# ---------------------------------------------------------------------------
# Phân loại
# ---------------------------------------------------------------------------
def classify(meta, rules):
    """Trả về (playlist, đã_khớp_rule, nguồn_tín_hiệu)."""
    channel_id = meta.get("channel_id")
    title = normalize(meta.get("title"))
    description = normalize(meta.get("description"))

    # Tầng 1: khoanh vùng ứng viên theo channel_id
    scoped = [r for r in rules if channel_id and channel_id in r["channels"]]
    if len(scoped) == 1:
        rule = scoped[0]
        matched = rule["_regex"] is None or rule["_regex"].search(f"{title} {description}")
        return rule["playlist"], True, "channel" if matched else "channel (text không khớp)"

    # Tầng 2: 1 kênh nhiều chương trình (hoặc kênh lạ) -> phân biệt bằng text.
    # Title trước cho cả bộ rule, hết mới tới description: tránh việc description
    # (thường chứa quảng cáo chéo các chương trình khác cùng kênh) lấn át title.
    for field, text in (("title", title), ("description", description)):
        for rule in scoped or rules:
            if rule["_regex"] and rule["_regex"].search(text):
                return rule["playlist"], True, field

    # Chương trình mới chưa có rule: vẫn cấp playlist riêng theo kênh để không rơi
    # khỏi tầng index playlist. Được cảnh báo ở cuối báo cáo để bổ sung rule.
    return f"Khac-{slugify(meta.get('author') or channel_id)}", False, "không khớp rule nào"


# ---------------------------------------------------------------------------
def main():
    config = load_config()
    data_dir = config["paths"]["data_dir"]

    parser = argparse.ArgumentParser(
        description="Sinh data/video_to_playlist.json từ metadata BTC (mọi batch)",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "--media-info-dir", nargs="*", default=None, metavar="DIR",
        help="Thư mục media-info. Bỏ trống = tự dò mọi batch trong data/media-info*",
    )
    parser.add_argument(
        "--rules", default=os.path.join(data_dir, "playlist_rules.json"),
        help="Bộ luật channel/title -> playlist",
    )
    parser.add_argument("--output", default=os.path.join(data_dir, "video_to_playlist.json"))
    parser.add_argument(
        "--strict", action="store_true",
        help="Thoát với mã lỗi 1 nếu còn video chưa khớp rule nào",
    )
    args = parser.parse_args()

    rules = load_rules(args.rules)
    dirs = discover_media_info_dirs(data_dir, args.media_info_dir)
    if not dirs:
        sys.exit(f"LỖI: Không tìm thấy thư mục media-info nào trong {data_dir}")
    print(f"📜 Bộ luật: {args.rules} ({len(rules)} playlist)")
    print("📂 Nguồn metadata:")
    for d in dirs:
        print(f"   - {d} ({len(glob(os.path.join(d, '*.json')))} file)")

    videos, broken = load_videos(dirs)
    if not videos:
        sys.exit("LỖI: Không đọc được video nào")

    video_to_playlist = {}
    unmatched, suspicious, by_description = [], [], []
    for video_id, meta in videos.items():
        playlist, matched, source = classify(meta, rules)
        video_to_playlist[video_id] = playlist
        if not matched:
            unmatched.append((video_id, meta))
        elif "không khớp" in source:
            suspicious.append((video_id, meta, playlist))
        elif source == "description":
            by_description.append((video_id, meta, playlist))

    os.makedirs(os.path.dirname(args.output), exist_ok=True)
    with open(args.output, "w", encoding="utf-8") as f:
        json.dump(video_to_playlist, f, ensure_ascii=False, indent=1, sort_keys=True)

    # ---- Báo cáo ----
    counts = Counter(video_to_playlist.values())
    total = len(videos)
    print(f"\n✅ Đã ghi {total} video / {len(counts)} playlist -> {args.output}")
    print("\n--- PHÂN BỐ VIDEO THEO PLAYLIST ---")
    for playlist, n in counts.most_common():
        print(f"  📌 {playlist}: {n} video ({n / total:.1%})")

    largest, largest_n = counts.most_common(1)[0]
    print(f"\n🔍 Playlist lớn nhất: {largest} = {largest_n / total:.1%} dataset "
          f"(query lọc đúng playlist chỉ quét ~{largest_n / total:.0%} index global)")

    if by_description:
        print(f"\nℹ️  {len(by_description)} video phân loại nhờ description (title không đủ tín hiệu):")
        for video_id, meta, playlist in by_description[:5]:
            print(f"   - {video_id} -> {playlist} | {(meta.get('title') or '')[:60]}")

    if broken:
        print(f"\n⚠️  {len(broken)} file metadata hỏng, đã bỏ qua:")
        for video_id, err in broken[:10]:
            print(f"   - {video_id}: {err}")

    if suspicious:
        print(f"\n⚠️  {len(suspicious)} video khớp channel nhưng text không khớp pattern "
              f"(vẫn được gán, nên rà lại rule):")
        for video_id, meta, playlist in suspicious[:10]:
            print(f"   - {video_id} -> {playlist} | {(meta.get('title') or '')[:70]}")

    if unmatched:
        print(f"\n⚠️  {len(unmatched)} video chưa có rule (đã tự cấp playlist 'Khac-<kênh>'). "
              f"Thêm rule vào {args.rules} nếu muốn gom đúng chương trình:")
        seen = Counter((m.get("author"), m.get("channel_id")) for _, m in unmatched)
        for (author, channel_id), n in seen.most_common(10):
            example = next(v for v, m in unmatched if m.get("channel_id") == channel_id)
            print(f"   - {author} ({channel_id}): {n} video | VD {example}: "
                  f"{(videos[example].get('title') or '')[:60]}")
        if args.strict:
            sys.exit(1)


if __name__ == "__main__":
    main()
