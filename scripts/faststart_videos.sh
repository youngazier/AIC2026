#!/usr/bin/env bash
#
# Đưa atom `moov` của mọi file .mp4 lên ĐẦU file (còn gọi là "faststart").
#
# VÌ SAO CẦN: mp4 xuất ra mặc định để `moov` — bảng mục lục ánh xạ thời gian -> vị trí
# byte — ở CUỐI file. Thẻ <video> muốn phát hay tua bất cứ đâu đều phải có bảng này
# trước, nên với file 130 MB trình duyệt buộc phải: xin đoạn đầu, phát hiện thiếu moov,
# ngắt kết nối, xin lại ~1.3 MB ở cuối. Hai vòng lượt đi lượt về đó xảy ra MỖI LẦN mở
# modal, và nếu lúc đó 6 khe kết nối đang bận tải thumbnail thì video treo hoặc bắn lỗi.
# Sau khi faststart, request đầu tiên đã đủ để phát và tua tức thì.
#
# KHÔNG mã hoá lại: `-c copy` chỉ chép nguyên luồng hình/tiếng rồi xếp lại thứ tự atom,
# nên chất lượng giữ nguyên tuyệt đối và chạy ~1-2 giây/file thay vì hàng phút.
#
#   ./scripts/faststart_videos.sh              # chạy thật, ghi đè tại chỗ
#   DRY_RUN=1 ./scripts/faststart_videos.sh    # chỉ liệt kê file sẽ sửa
#
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
VIDEOS_DIR="${VIDEOS_DIR:-$ROOT/data/videos}"
DRY_RUN="${DRY_RUN:-0}"

command -v ffmpeg >/dev/null || { echo "Cần ffmpeg trong PATH."; exit 1; }
[ -d "$VIDEOS_DIR" ] || { echo "Không thấy thư mục video: $VIDEOS_DIR"; exit 1; }

# moov đã ở đầu chưa: đọc 4 byte tên atom ngay sau atom `ftyp` đầu file. Rẻ hơn nhiều so
# với gọi ffprobe cho từng file, mà 873 file thì chênh lệch đó là vài phút.
is_faststart() {
  python3 - "$1" <<'PY'
import struct, sys
with open(sys.argv[1], 'rb') as f:
    off = 0
    while True:
        f.seek(off); h = f.read(8)
        if len(h) < 8: sys.exit(1)
        size = struct.unpack('>I', h[:4])[0]
        atom = h[4:8]
        if size == 1: size = struct.unpack('>Q', f.read(8))[0]
        if atom == b'moov': sys.exit(0)
        if atom == b'mdat' or size <= 0: sys.exit(1)
        off += size
PY
}

total=0; fixed=0; skipped=0; failed=0
while IFS= read -r -d '' src; do
  total=$((total + 1))
  if is_faststart "$src"; then
    skipped=$((skipped + 1)); continue
  fi
  if [ "$DRY_RUN" = "1" ]; then
    echo "SE SUA: $src"; fixed=$((fixed + 1)); continue
  fi

  # Ghi ra file tạm CẠNH bản gốc rồi mới `mv` đè: mv trong cùng filesystem là thao tác
  # nguyên tử, nên máy sập giữa chừng cũng không để lại file .mp4 cụt đầu cụt đuôi.
  tmp="${src}.faststart.tmp.mp4"
  if ffmpeg -nostdin -v error -y -i "$src" -c copy -movflags +faststart "$tmp" \
     && [ -s "$tmp" ]; then
    mv -f "$tmp" "$src"
    fixed=$((fixed + 1))
    echo "OK: $src"
  else
    rm -f "$tmp"
    failed=$((failed + 1))
    echo "LOI (giu nguyen ban goc): $src" >&2
  fi
done < <(find "$VIDEOS_DIR" -name '*.mp4' -not -name '*.faststart.tmp.mp4' -print0 | sort -z)

echo "---"
echo "Tong $total file · da sua $fixed · da faststart san $skipped · loi $failed"
