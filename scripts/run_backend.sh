#!/usr/bin/env bash
# Khởi chạy backend tìm kiếm.
#
#   ./scripts/run_backend.sh                    # cổng 8000
#   PORT=8080 ./scripts/run_backend.sh          # đổi cổng
#   AIC_BACKEND_MODELS=siglip2 ./scripts/run_backend.sh   # chỉ nạp 1 model cho nhanh
#   AIC_SKIP_ENCODERS=1 ./scripts/run_backend.sh          # không nạp model AI (dev tầng DB)
#
# Script này tồn tại để không ai phải nhớ đường dẫn venv. Venv nằm trong
# data_processing/feature_extraction/ vì lý do lịch sử; ĐỪNG `mv` nó — venv chứa
# đường dẫn tuyệt đối trong bin/ và pyvenv.cfg, di chuyển là hỏng.
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
VENV="$ROOT_DIR/data_processing/feature_extraction/venv_embedding"
PORT="${PORT:-8000}"
HOST="${HOST:-0.0.0.0}"

if [[ ! -x "$VENV/bin/python" ]]; then
    echo "Không tìm thấy venv tại $VENV" >&2
    exit 1
fi

if [[ ! -f "$ROOT_DIR/data_processed/metadata.db" ]]; then
    echo "Chưa có metadata.db. Chạy trước:" >&2
    echo "  $VENV/bin/python scripts/init_db.py" >&2
    echo "  $VENV/bin/python scripts/ingest_db.py" >&2
    exit 1
fi

cd "$ROOT_DIR"
# --workers 1 là BẮT BUỘC: mỗi worker là một tiến trình riêng, sẽ nạp lại toàn bộ
# FAISS index (vài GB RAM) và một bản model trên GPU. Xử lý song song đã có threadpool
# của FastAPI lo (FAISS/PyTorch nhả GIL khi tính toán).
exec "$VENV/bin/python" -m uvicorn backend.main:app \
    --host "$HOST" --port "$PORT" --workers 1 "$@"
