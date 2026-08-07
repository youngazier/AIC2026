#!/usr/bin/env bash

set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
VENV_ACTIVATE="$SCRIPT_DIR/transnetv2/venv_transnetv2/bin/activate"

if [[ -f "$VENV_ACTIVATE" ]]; then
    echo "[VENV] Activating virtual environment: $VENV_ACTIVATE"
    source "$VENV_ACTIVATE" || true
else
    echo "[WARNING] Virtual environment not found at $VENV_ACTIVATE. Continuing with current Python."
fi

set -e

# Đọc đường dẫn từ config.yaml thay vì hardcode: config là nơi khai báo duy nhất,
# nếu bash và python tự khai báo riêng thì sớm muộn cũng lệch nhau (nơi ghi keyframe
# khác nơi đọc keyframe).
PROJECT_ROOT="$(cd "$SCRIPT_DIR/../.." && pwd)"
read -r VIDEOS_DIR TRANSNET_OUTPUT_DIR < <(
    python3 -c "
import sys
sys.path.insert(0, '$PROJECT_ROOT')
from utils.config_loader import load_config
p = load_config()['paths']
print(p['videos_dir'], p['transnetv2_dir'])
"
) || { echo "[ERROR] Không đọc được config.yaml. Kiểm tra pyyaml đã cài trong venv chưa."; exit 1; }

if [[ -z "$VIDEOS_DIR" || -z "$TRANSNET_OUTPUT_DIR" ]]; then
    echo "[ERROR] config.yaml thiếu paths.videos_dir hoặc paths.transnetv2_dir"
    exit 1
fi
echo "[CONFIG] videos_dir    = $VIDEOS_DIR"
echo "[CONFIG] transnetv2_dir = $TRANSNET_OUTPUT_DIR"

VIDEO=""
FOLDER=""
RUN_ALL=false
MODE="all"
GPU=""

usage() {
    echo "Usage: $0 [TARGET OPTION] [MODE OPTION] [GPU OPTION]"
    echo ""
    echo "Target Options (Choose EXACTLY ONE):"
    echo "  -v, --video <name|path>        Run for 1 video (e.g. L01_V001 or /path/to/L01_V001.mp4)"
    echo "  -f, --folder <names|paths>     Run for 1 or MULTIPLE folders (e.g. 'L21 L22 L23' or 'L21,L22')"
    echo "  -a, --all                      Run for all videos in dataset"
    echo ""
    echo "Mode Options:"
    echo "  -m, --mode <mode>              Mode: transnet | keyframe | all (default: all)"
    echo ""
    echo "GPU Options:"
    echo "  -g, --gpu <id>                 Specify CUDA GPU device ID (e.g. 0 or 0,1)"
    echo "  -h, --help                     Show this help message"
    exit 1
}

# Parse flags
while [[ $# -gt 0 ]]; do
    case "$1" in
        -v|--video)
            VIDEO="$2"
            shift 2
            ;;
        -f|--folder)
            FOLDER="$2"
            shift 2
            ;;
        -a|--all)
            RUN_ALL=true
            shift
            ;;
        -m|--mode)
            MODE="$2"
            shift 2
            ;;
        -g|--gpu)
            GPU="$2"
            shift 2
            ;;
        -h|--help)
            usage
            ;;
        *)
            echo "Error: Unknown option '$1'"
            usage
            ;;
    esac
done

# Validate target selection
TARGET_COUNT=0
if [[ -n "$VIDEO" ]]; then TARGET_COUNT=$((TARGET_COUNT+1)); fi
if [[ -n "$FOLDER" ]]; then TARGET_COUNT=$((TARGET_COUNT+1)); fi
if [[ "$RUN_ALL" = true ]]; then TARGET_COUNT=$((TARGET_COUNT+1)); fi

if [[ $TARGET_COUNT -eq 0 ]]; then
    echo "[ERROR] You must specify a target option: -v, -f, or -a"
    usage
elif [[ $TARGET_COUNT -gt 1 ]]; then
    echo "[ERROR] Please specify ONLY ONE target option (-v, -f, or -a)"
    usage
fi

if [[ "$MODE" != "transnet" && "$MODE" != "keyframe" && "$MODE" != "all" ]]; then
    echo "[ERROR] Invalid mode '$MODE'. Allowed modes: transnet, keyframe, all"
    exit 1
fi

if [[ -n "$GPU" ]]; then
    export CUDA_VISIBLE_DEVICES="$GPU"
    echo "[GPU] Set CUDA_VISIBLE_DEVICES=$CUDA_VISIBLE_DEVICES"
fi

# Resolve list of video files to process
VIDEO_FILES=()
TARGET_PATHS=()

if [[ -n "$VIDEO" ]]; then
    if [[ -f "$VIDEO" ]]; then
        VIDEO_FILES+=("$VIDEO")
        TARGET_PATHS+=("$VIDEO")
    else
        V_NAME=$(basename "$VIDEO" .mp4)
        FOUND_PATH=$(find "$VIDEOS_DIR" -name "${V_NAME}.mp4" | head -n 1)
        if [[ -n "$FOUND_PATH" && -f "$FOUND_PATH" ]]; then
            VIDEO_FILES+=("$FOUND_PATH")
            TARGET_PATHS+=("$FOUND_PATH")
        else
            echo "[ERROR] Video '$VIDEO' not found under $VIDEOS_DIR"
            exit 1
        fi
    fi

elif [[ -n "$FOLDER" ]]; then
    # Support space-separated or comma-separated folder list, e.g. "L21 L22 L23" or "L21,L22"
    CLEAN_FOLDERS=$(echo "$FOLDER" | tr ',' ' ')
    for F_ITEM in $CLEAN_FOLDERS; do
        F_NAME=$(basename "$F_ITEM")
        if [[ ! "$F_NAME" =~ ^Videos_ ]]; then
            F_NAME="Videos_${F_NAME}"
        fi
        
        TARGET_PATH="$VIDEOS_DIR/$F_NAME"
        if [[ ! -d "$TARGET_PATH" ]]; then
            ALT_PATH="$VIDEOS_DIR/$(basename "$F_ITEM")"
            if [[ -d "$ALT_PATH" ]]; then
                TARGET_PATH="$ALT_PATH"
            else
                echo "[WARNING] Folder '$F_ITEM' not found in $VIDEOS_DIR, skipping."
                continue
            fi
        fi
        
        TARGET_PATHS+=("$TARGET_PATH")
        while IFS= read -r v_file; do
            if [[ -n "$v_file" ]]; then
                VIDEO_FILES+=("$v_file")
            fi
        done < <(find "$TARGET_PATH" -maxdepth 1 -name "*.mp4" | sort)
    done

elif [[ "$RUN_ALL" = true ]]; then
    TARGET_PATHS+=("$VIDEOS_DIR")
    while IFS= read -r v_file; do
        if [[ -n "$v_file" ]]; then
            VIDEO_FILES+=("$v_file")
        fi
    done < <(find "$VIDEOS_DIR" -name "*.mp4" | sort)
fi

TOTAL_VIDEOS=${#VIDEO_FILES[@]}
if [[ $TOTAL_VIDEOS -eq 0 ]]; then
    echo "[ERROR] No valid videos found to process."
    exit 1
fi

echo "=================================================="
echo " Starting Pipeline Execution"
echo " Target Mode : $MODE"
echo " Total Videos: $TOTAL_VIDEOS"
if [[ -n "$GPU" ]]; then
    echo " GPU ID      : $GPU"
fi
echo "=================================================="

TRANSNET_SCRIPT="$SCRIPT_DIR/transnetv2/transnetv2_pytorch/inference.py"
KEYFRAME_SCRIPT="$SCRIPT_DIR/keyframe_extraction.py"

# Từ đây trở đi TẮT set -e: phần trên là kiểm tra cấu hình nên fail-fast là đúng,
# còn phần dưới là xử lý dữ liệu theo lô. Với 800+ video, dừng ở lỗi đầu tiên
# nghĩa là bạn phát hiện từng lỗi một qua từng lần chạy lại, thay vì biết toàn bộ
# thiệt hại sau một lần chạy. Lỗi được gom lại và báo ở cuối, kèm exit code.
set +e
FAILURES=()

# 1. RUN TRANSNETV2
if [[ "$MODE" == "transnet" || "$MODE" == "all" ]]; then
    echo ""
    echo ">>> STEP 1: Running TransNetV2 Scene Boundary Detection..."
    
    IDX=0
    for V_FILE in "${VIDEO_FILES[@]}"; do
        IDX=$((IDX + 1))
        V_NAME=$(basename "$V_FILE" .mp4)
        SCENES_TXT="$TRANSNET_OUTPUT_DIR/scenes/${V_NAME}.scenes.txt"
        PRED_TXT="$TRANSNET_OUTPUT_DIR/predictions/${V_NAME}.predictions.txt"
        
        # Thorough check: both files must exist and be non-empty (>0 bytes)
        if [[ -f "$SCENES_TXT" && -s "$SCENES_TXT" && -f "$PRED_TXT" && -s "$PRED_TXT" ]]; then
            # Valid output exists
            continue
        else
            # Remove incomplete/corrupted results if any
            if [[ -f "$SCENES_TXT" || -f "$PRED_TXT" ]]; then
                echo "[$IDX/$TOTAL_VIDEOS] [CLEANUP] Removing incomplete/corrupted TransNetV2 output for $V_NAME..."
                rm -f "$SCENES_TXT" "$PRED_TXT"
            fi
        fi
    done

    # Run batch inference per target path
    mkdir -p "$TRANSNET_OUTPUT_DIR"
    for T_PATH in "${TARGET_PATHS[@]}"; do
        echo "[TransNetV2] Batch processing path: $T_PATH"
        if [[ "$RUN_ALL" = true ]]; then
            python3 "$TRANSNET_SCRIPT" "$T_PATH" -o "$TRANSNET_OUTPUT_DIR/" --all_videos
        else
            python3 "$TRANSNET_SCRIPT" "$T_PATH" -o "$TRANSNET_OUTPUT_DIR/"
        fi
        if [[ $? -ne 0 ]]; then
            echo "[ERROR] TransNetV2 thất bại ở $T_PATH — tiếp tục các path còn lại."
            FAILURES+=("transnet:$T_PATH")
        fi
    done
fi

# 2. RUN KEYFRAME EXTRACTION
if [[ "$MODE" == "keyframe" || "$MODE" == "all" ]]; then
    echo ""
    echo ">>> STEP 2: Running Keyframe Extraction..."
    
    if [[ -n "$VIDEO" ]]; then
        V_NAME=$(basename "$VIDEO" .mp4)
        python3 "$KEYFRAME_SCRIPT" -v "$V_NAME"
    elif [[ -n "$FOLDER" ]]; then
        python3 "$KEYFRAME_SCRIPT" -f "$FOLDER"
    elif [[ "$RUN_ALL" = true ]]; then
        python3 "$KEYFRAME_SCRIPT" -a
    fi
    if [[ $? -ne 0 ]]; then
        FAILURES+=("keyframe")
    fi
fi

echo ""
echo "=================================================="
if [[ ${#FAILURES[@]} -eq 0 ]]; then
    echo " Pipeline Finished Successfully!"
    echo "=================================================="
    echo " Bước tiếp theo — xác nhận dữ liệu đã đủ trước khi encode:"
    echo "   python $SCRIPT_DIR/check_integrity.py"
else
    echo " Pipeline Finished WITH FAILURES (${#FAILURES[@]}):"
    for f in "${FAILURES[@]}"; do echo "   - $f"; done
    echo "=================================================="
    echo " Chạy lại chính lệnh này để thử lại phần chưa xong"
    echo " (video đã hoàn tất được bỏ qua nhờ file _SUCCESS)."
    echo " Xem chi tiết còn thiếu gì: python $SCRIPT_DIR/check_integrity.py"
    exit 1
fi
