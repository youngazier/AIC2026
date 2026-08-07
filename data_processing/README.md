# Video Data Processing Pipeline (AIC 2026)

Tài liệu hướng dẫn vận hành toàn bộ khâu xử lý dữ liệu offline: từ video thô của BTC đến FAISS index sẵn sàng cho truy vấn.

## 🔗 Luồng dữ liệu tổng quan

Bước [1] và [2] gộp trong một script `run_pipeline.sh`. Bước [3] độc lập, chạy lúc nào cũng được. Bước [4] cần cả hai nhánh xong.

```text
        data/videos/Videos_LXX/*.mp4                  data/media-info-*/media-info/*.json
                    │                                                │
    ┌───────────────┴────────────────┐                                │
    │        run_pipeline.sh         │                                │
    │  ┌──────────────────────────┐  │                                │
    │  │ [1] TransNetV2           │  │  phân cảnh                     │
    │  │     → *.scenes.txt       │  │                                │
    │  └────────────┬─────────────┘  │                    ┌───────────┴────────────┐
    │  ┌────────────┴─────────────┐  │                    │  [3] Build playlist    │
    │  │ [2] Keyframe extraction  │  │  cắt mid-frame     │      → video_to_       │
    │  │     → *.jpg + *.csv      │  │                    │        playlist.json   │
    │  └────────────┬─────────────┘  │                    └───────────┬────────────┘
    └───────────────┼────────────────┘                                │
                    └────────────────────┬───────────────────────────-┘
                                         │
                          ┌──────────────┴───────────────┐
                          │  [4a] run_embedding.py       │
                          │       → .npy từng video      │
                          │  [4b] build_index.py         │
                          │       → FAISS index          │
                          │         (global + playlist)  │
                          └──────────────────────────────┘
```

---

## 📂 Cấu trúc thư mục

```text
data_processing/
├── build_playlist_mapping.py            # [3] Phân loại video -> playlist từ metadata BTC
├── keyframe_extraction/
│   ├── run_pipeline.sh                  # [1+2] ĐIỂM VÀO CHÍNH: chạy TransNetV2 rồi cắt keyframe
│   ├── keyframe_extraction.py           # [2] Cắt ảnh keyframe từ kết quả TransNetV2
│   ├── check_integrity.py               # [2] Đối chiếu số ảnh .jpg với số dòng trong .csv
│   └── transnetv2/                      # [1] Mô hình phân cảnh TransNetV2 (PyTorch)
│       ├── transnetv2_pytorch/          #     Mã nguồn model + inference.py
│       ├── venv_transnetv2/             #     Môi trường ảo RIÊNG của TransNetV2
│       └── setup.py
└── feature_extraction/
    ├── run_embedding.py                 # [4a] Encode keyframe -> .npy từng video
    ├── build_index.py                   # [4b] Gom .npy -> FAISS index (global + playlist)
    ├── base_embedding.py                #      Lớp cơ sở dùng chung
    ├── siglip2_embedding.py             #     Cài đặt SigLIP 2
    ├── beit3_embedding.py               #     Cài đặt BEiT-3
    ├── dinov3_embedding.py              #     Cài đặt DINOv3
    ├── test_retrieval.py                #     Kiểm thử truy vấn nhanh
    ├── app.py                           #     API phục vụ embedding
    └── venv_embedding/                  #     Môi trường ảo RIÊNG của feature extraction
```

> **Hai môi trường ảo tách biệt.** TransNetV2 và feature extraction có phụ thuộc xung đột nhau nên mỗi bên một venv riêng, đặt ngay trong thư mục của nó. Đừng gộp.

---

## [1] + [2] 🎬 Phân cảnh & Trích xuất Keyframes — `run_pipeline.sh`

Dùng `run_pipeline.sh` cho cả hai bước. Script tự kích hoạt `venv_transnetv2`, tự resolve danh sách video, và gọi lần lượt TransNetV2 rồi keyframe extraction. **Không cần activate venv hay export `CUDA_VISIBLE_DEVICES` thủ công.**

### Cài đặt phụ thuộc (chỉ lần đầu)

```bash
cd /tmp2/maitanha/vgu/ttn/AIC2026/data_processing/keyframe_extraction/transnetv2
source venv_transnetv2/bin/activate

# Đăng ký package cục bộ ở chế độ editable
uv pip install -e .

# Gỡ bản ffmpeg sai tên rồi cài đúng gói ffmpeg-python
uv pip uninstall ffmpeg
uv pip install ffmpeg-python

# PyTorch tương thích CUDA 12.4
uv pip install torch --index-url https://download.pytorch.org/whl/cu124 --force-reinstall
```

### Chạy

```bash
cd /tmp2/maitanha/vgu/ttn/AIC2026/data_processing/keyframe_extraction

# Toàn bộ dataset, cả 2 bước, trên GPU 7
./run_pipeline.sh -a -g 7

# Một video
./run_pipeline.sh -v L21_V001 -g 7

# Nhiều folder (chấp nhận "L21" lẫn "Videos_L21", ngăn bằng khoảng trắng hoặc dấu phẩy)
./run_pipeline.sh -f "L21 L22 L23" -g 7
./run_pipeline.sh -f L21,L22 -g 7

# Chỉ chạy một bước
./run_pipeline.sh -a -m transnet -g 7     # chỉ phân cảnh
./run_pipeline.sh -a -m keyframe          # chỉ cắt keyframe (scenes đã có sẵn)
```

**Bắt buộc chọn đúng một** target trong `-v` / `-f` / `-a`; script báo lỗi nếu thiếu hoặc chọn nhiều hơn một.

| Nhóm | Cờ | Ý nghĩa |
|---|---|---|
| Target | `-v, --video <name\|path>` | Một video. Nhận cả `L21_V001` lẫn đường dẫn `.mp4` đầy đủ |
| | `-f, --folder <names>` | Một hoặc nhiều folder, ngăn bằng khoảng trắng hoặc dấu phẩy |
| | `-a, --all` | Toàn bộ dataset |
| Mode | `-m, --mode <mode>` | `transnet` \| `keyframe` \| `all` (mặc định `all`) |
| GPU | `-g, --gpu <id>` | Set `CUDA_VISIBLE_DEVICES`, nhận cả `0` lẫn `0,1` |
| | `-h, --help` | Hiện trợ giúp |

### Kết quả

```text
data_processed/transnetv2/scenes/LXX_VXXX.scenes.txt            # [1] cặp frame đầu–cuối mỗi shot
data_processed/transnetv2/predictions/LXX_VXXX.predictions.txt  # [1] xác suất thô
data_processed/keyframes_transnetv2/Keyframes_LXX/LXX_VXXX/*.jpg # [2] ảnh keyframe
data_processed/map_keyframes/LXX_VXXX.csv                        # [2] keyframe_name -> frame_idx, timestamp
```

### Cơ chế cache — chạy lại an toàn sau khi bị gián đoạn

Cả hai bước đều resume được, nhưng theo hai cách khác nhau:

- **Bước [1]**: `inference.py` bỏ qua video nào đã có sẵn file `.scenes.txt` hoặc `.predictions.txt`. Nó chỉ kiểm tra *file có tồn tại không*, nên nếu job chết giữa lúc ghi thì file rỗng vẫn bị coi là xong. Vì vậy `run_pipeline.sh` quét trước một lượt và **xoá các cặp file dở dang** (thiếu một trong hai, hoặc 0 byte) để chúng được xử lý lại.
- **Bước [2]**: đối chiếu số ảnh `.jpg` với số dòng trong `.csv`. Đủ thì bỏ qua; thiếu thì xoá sạch kết quả cũ của video đó rồi cắt lại từ đầu.

### Kiểm tra tính toàn vẹn

Đối chiếu số ảnh `.jpg` thực tế với số dòng `.csv` của từng video, liệt kê video nào thiếu:

```bash
python check_integrity.py
```

### Chạy thủ công từng script (khi cần debug)

`run_pipeline.sh` gọi lại đúng hai script này, nên chỉ dùng khi muốn tách riêng để gỡ lỗi:

```bash
source transnetv2/venv_transnetv2/bin/activate

# [1] TransNetV2 — thêm --visualize để xuất ảnh trực quan hoá kiểm tra bằng mắt
CUDA_VISIBLE_DEVICES=7 python transnetv2/transnetv2_pytorch/inference.py \
    /tmp2/maitanha/vgu/ttn/data/AIC2025/videos \
    -o /tmp2/maitanha/vgu/ttn/AIC2026/data_processed/transnetv2/ --all_videos

# [2] Keyframe extraction — không truyền tham số = chạy toàn bộ dataset
python keyframe_extraction.py -f L21,L22
```

`--all_videos` quét đệ quy toàn bộ cây thư mục; bỏ cờ này thì chỉ lấy video nằm trực tiếp trong thư mục đầu vào.

---

## [3] 🎯 Xây dựng Playlist mapping

Đọc metadata BTC (`media-info/*.json`), phân loại từng video vào đúng chương trình truyền hình. Mục đích: mỗi playlist được build thành một FAISS index riêng ở bước [4], khi truy vấn chỉ quét subset thay vì toàn bộ index global.

### Chạy

```bash
cd /tmp2/maitanha/vgu/ttn/AIC2026
python data_processing/build_playlist_mapping.py --strict
```

Không cần truyền tham số: script **tự dò mọi thư mục `data/media-info*`** và gộp tất cả batch lại. Khi BTC thả b2/b3, chỉ cần giải nén vào `data/` rồi chạy lại đúng lệnh trên.

| Tham số | Ý nghĩa |
|---|---|
| `--strict` | Thoát với mã lỗi 1 nếu còn video chưa khớp rule nào. **Nên luôn bật** khi chạy batch mới |
| `--rules` | Đường dẫn bộ luật (mặc định `data/playlist_rules.json`) |
| `--media-info-dir` | Chỉ định thủ công thư mục metadata, ghi đè cơ chế tự dò |
| `--output` | Đường dẫn file kết quả (mặc định `data/video_to_playlist.json`) |

**Input:** `data/playlist_rules.json` — bộ luật, **nguồn duy nhất**, không có bản dự phòng trong code.
**Output:** `data/video_to_playlist.json` — `{video_id: playlist}`, đọc bởi `build_index.py` ở bước [4b].

### Cách phân loại hoạt động

Hai tầng, **cố ý không dùng folder prefix `Lxx`** vì số hiệu folder đổi theo từng batch nên không đoán trước được:

1. **`channel_id`** khoanh vùng các rule ứng viên — tín hiệu ổn định nhất giữa các batch.
2. **regex trên title** phân biệt khi một kênh phát nhiều chương trình. Bắt buộc phải có, vì:
   - *HTV Sports* phát cả Cúp Truyền hình (xe đạp) lẫn Lân Sư Rồng Chợ Lớn
   - *HTV Entertainment* phát cả Tản mạn Mê Kông lẫn Đôi mắt Mê Kông

   Khớp title trước cho toàn bộ rule, hết mới xét tới `description`. **Không đảo thứ tự này** — description YouTube thường quảng cáo chéo các chương trình khác cùng kênh, để nó lấn title là gán sai ngay.

`keywords` **không** được dùng: đó là keywords chung của cả kênh chứ không phải của video (keywords các clip múa lân chứa "Serie A", "ONE championship").

Video của kênh lạ không bị bỏ rơi — được cấp playlist tự sinh `Khac-<tên-kênh>` và liệt kê ở cuối báo cáo kèm `channel_id` để chép thẳng vào `playlist_rules.json`.

### Thêm chương trình mới

Sửa `data/playlist_rules.json`, không cần đụng Python. Phần `_readme` ngay đầu file có hướng dẫn đầy đủ.

```json
{
  "playlist": "Tan-Man-Me-Kong",
  "channels": ["UCbq8aOyj9ZtIqcD-0MwG1fQ"],
  "title": "tan man me ?kong",
  "note": "HTV Entertainment — kênh này phát 2 chương trình, phải có title để phân biệt."
}
```

- `playlist` là tên file FAISS index nên **chỉ dùng `[A-Za-z0-9-]`**.
- Regex khớp trên text đã **bỏ dấu tiếng Việt + chuyển chữ thường**: `"Cúp Truyền Hình"` → `cup truyen hinh`. Nhờ vậy `me ?kong` bắt được cả "Mê Kông", "MeKong" lẫn "mekong".
- Trong JSON dấu `\` phải viết đôi: regex `\s*` ghi thành `"\\s*"`.

Script validate bộ luật trước khi chạy và dừng ngay khi phát hiện: tên playlist chứa ký tự không hợp lệ, regex sai cú pháp, playlist trùng tên, hoặc **nhiều playlist dùng chung một channel mà thiếu `title`** — trường hợp cuối nguy hiểm nhất vì nó khiến kết quả phụ thuộc thứ tự khai báo và gán sai âm thầm.

> ⚠️ Sửa playlist mapping xong thì các FAISS index playlist cũ đã lỗi thời, phải chạy lại bước [4]. Không tốn kém: encode có cache theo `.npy` từng video nên video cũ không phải encode lại, chỉ tốn thời gian concatenate.

---

## [4] 🧬 Trích xuất đặc trưng & Build FAISS Index

Hai script tách rời, chạy nối tiếp nhau:

| | Script | Việc | Chi phí |
|---|---|---|---|
| **[4a]** | `run_embedding.py` | Encode keyframe → `.npy` từng video | GPU, hàng giờ |
| **[4b]** | `build_index.py` | Gom `.npy` → FAISS index (global + playlist) | CPU/RAM, vài giây |

Tách đôi vì hai bước có tính chất trái ngược: encode nặng và cần lọc theo video/folder để chia máy, còn build index nhẹ nhưng **bắt buộc phải quét toàn bộ `.npy`** — build từ một tập con sẽ ghi đè index cũ thành bản thiếu video. Nhốt chung một script thì mỗi lần encode thử một video lại kéo theo một lượt build index thừa.

```bash
source data_processing/feature_extraction/venv_embedding/bin/activate

python data_processing/feature_extraction/run_embedding.py -f L21   # [4a] encode folder L21
python data_processing/feature_extraction/run_embedding.py -f L22   # [4a] encode tiếp folder L22
python data_processing/feature_extraction/build_index.py            # [4b] một lượt, index gồm cả hai
```

### Điều kiện tiên quyết

| Cần có | Từ bước | Kiểm tra nhanh |
|---|---|---|
| Keyframe `.jpg` | [2] | `ls data_processed/keyframes_transnetv2/` |
| `video_to_playlist.json` | [3] | `ls data/video_to_playlist.json` |
| `venv_embedding` đã activate | — | `which python` phải trỏ vào `venv_embedding/bin/` |

Thiếu playlist mapping thì script **vẫn chạy**: build index global rồi thoát kèm cảnh báo, bỏ qua tầng playlist.

### [4a] Encode — `run_embedding.py`

`run_embedding.py` đọc mọi đường dẫn từ `config.yaml` nên không nhận tham số đường dẫn. Hai việc phải tự làm trước khi chạy: **activate `venv_embedding`** (không phải `venv_transnetv2`) và chọn GPU bằng `CUDA_VISIBLE_DEVICES`.

```bash
cd /tmp2/maitanha/vgu/ttn/AIC2026
source data_processing/feature_extraction/venv_embedding/bin/activate
RUN=data_processing/feature_extraction/run_embedding.py

# Toàn bộ dataset, SigLIP 2, batch 64, GPU 7
CUDA_VISIBLE_DEVICES=7 python $RUN -a --model siglip2 --batch_size 64

# Một video — nhận cả tên lẫn đường dẫn .mp4 đầy đủ
CUDA_VISIBLE_DEVICES=7 python $RUN -v L21_V001
CUDA_VISIBLE_DEVICES=7 python $RUN -v data/videos/Videos_L21/L21_V001.mp4

# Một folder — nhận cả "L21" lẫn "Videos_L21"
CUDA_VISIBLE_DEVICES=7 python $RUN -f L21

# Nhiều folder — ngăn bằng dấu phẩy hoặc khoảng trắng
CUDA_VISIBLE_DEVICES=7 python $RUN -f L21,L22,L23
CUDA_VISIBLE_DEVICES=7 python $RUN -f "L21 L22 L23"

# Model khác — mỗi model ghi ra thư mục .npy và bộ index riêng, không đè lên nhau
CUDA_VISIBLE_DEVICES=7 python $RUN -a --model beit3  --batch_size 32
CUDA_VISIBLE_DEVICES=7 python $RUN -a --model dinov3 --batch_size 64
```

> Quên activate venv thì script chết ngay lúc import với `ModuleNotFoundError` (`open_clip` ở `run_embedding.py`, `faiss` ở `build_index.py`) — không có cơ chế tự dò venv, đây là chủ ý.

Cùng quy ước đặt tên với `run_pipeline.sh` ở bước [1+2], nhưng **khác một điểm**: ở đây `-v/-f/-a` là tuỳ chọn, không truyền gì thì mặc định chạy toàn bộ dataset (giữ tương thích với cách gọi cũ chỉ có `--model`/`--batch_size`). Chọn nhiều hơn một target thì argparse báo lỗi ngay.

| Nhóm | Cờ | Ý nghĩa |
|---|---|---|
| Target | `-v, --video <name\|path>` | Một video. Nhận cả `L21_V001` lẫn đường dẫn `.mp4` |
| | `-f, --folder <names>` | Một hoặc nhiều folder, ngăn bằng dấu phẩy hoặc khoảng trắng. Chấp nhận `L21`, `Videos_L21`, `Keyframes_L21` |
| | `-a, --all` | Toàn bộ dataset (mặc định khi không truyền target nào) |
| Model | `--model` | `siglip2` \| `beit3` \| `dinov3` (mặc định `siglip2`) |
| | `--batch_size` | Batch size khi encode (mặc định `64`) |

Gõ sai tên thì script dừng ngay thay vì âm thầm encode 0 video: `-v` không khớp video nào → thoát kèm số video hiện có; `-f` không khớp folder nào → thoát kèm danh sách folder hợp lệ. Truyền nhiều folder mà chỉ một cái sai thì vẫn chạy, kèm warning liệt kê folder bị bỏ.

Chạy xong script tổng kết số video encode mới / bỏ qua, và chỉ khi có `.npy` mới thì mới nhắc chạy `build_index.py`:

```text
Xong. Encode mới 1 video, bỏ qua 12 video đã có .npy. Tổng cộng 13 file .npy trong .../features/siglip2
Có .npy mới — chạy tiếp để cập nhật FAISS index:
    .../feature_extraction/build_index.py --model siglip2
```

Chạy toàn bộ rất lâu (873 video / ~150k keyframe) nên đưa vào `nohup` hoặc `tmux`:

```bash
CUDA_VISIBLE_DEVICES=7 nohup python $RUN -a --model siglip2 --batch_size 64 \
    > /tmp/embed_siglip2.log 2>&1 &
tail -f /tmp/embed_siglip2.log
```

Muốn chia việc cho nhiều GPU thì cắt theo folder, mỗi GPU một nhóm. Các tiến trình ghi ra `.npy` riêng biệt theo từng video nên không tranh nhau file, và vì encode **không còn đụng tới index** nên cũng không có chuyện tiến trình này ghi đè index của tiến trình kia — đợi tất cả xong rồi chạy `build_index.py` đúng một lần:

```bash
CUDA_VISIBLE_DEVICES=5 nohup python $RUN -f "L21 L22 L23"     > /tmp/embed_g5.log 2>&1 &
CUDA_VISIBLE_DEVICES=6 nohup python $RUN -f "L24 L25 L26"     > /tmp/embed_g6.log 2>&1 &
CUDA_VISIBLE_DEVICES=7 nohup python $RUN -f "L27 L28 L29 L30" > /tmp/embed_g7.log 2>&1 &
wait
python data_processing/feature_extraction/build_index.py --model siglip2
```

### [4b] Build index — `build_index.py`

Đọc `.npy`, ghép theo thứ tự ổn định rồi ghi FAISS index. Không nạp model, không encode gì cả.

```bash
cd /tmp2/maitanha/vgu/ttn/AIC2026
source data_processing/feature_extraction/venv_embedding/bin/activate
BUILD=data_processing/feature_extraction/build_index.py

python $BUILD                      # mọi model đang có .npy
python $BUILD --model siglip2      # chỉ một model
python $BUILD --force              # build lại kể cả khi metadata không đổi
```

| Tham số | Ý nghĩa | Mặc định |
|---|---|---|
| `--model` | `siglip2` \| `beit3` \| `dinov3` \| `all` | `all` |
| `--force` | Bỏ qua cơ chế so metadata, luôn ghi lại index | tắt |

`--model all` **tự dò** thư mục nào trong `data_processed/features/` đang có `.npy`, nên thêm model mới sau này không phải sửa script. Chỉ định `--model X` mà chưa có `.npy` thì thoát kèm lệnh cần chạy trước.

Không có cờ `-v`/`-f` ở đây là **cố ý**: index phải phủ toàn bộ `.npy` hiện có. Đổi lại, script tự báo phần còn thiếu để biết index đang phủ tới đâu:

```text
===== siglip2 =====
[siglip2] 860/873 video chưa có .npy, không nằm trong index. VD: ['L21_V004', ...]
[siglip2/global] Đã ghi 5845 vector (dim=1536) -> siglip2_keyframe_embedding.index
[siglip2/60-Giay] Đã ghi 5845 vector (dim=1536) -> siglip2_60-Giay.index
```

Cơ chế so metadata vẫn giữ nguyên: playlist nào không có video mới thì in `giữ nguyên index` và không ghi lại file. Nhờ vậy chạy `build_index.py` nhiều lần rất rẻ.

Ngoài ra script đối chiếu số dòng trong `.npy` với số ảnh `.jpg` của từng video; lệch thì cảnh báo kèm tên file cần xoá để encode lại — bắt được trường hợp `.npy` ghi dở do job bị kill giữa chừng, vốn sẽ bị `run_embedding.py` coi là "đã xong" và bỏ qua mãi mãi.

Batch size chọn theo VRAM: SigLIP 2 ViT-gopt-16-384 và DINOv3 ViT-H+ chịu được `64`, BEiT-3 large nặng hơn nên hạ về `32`. Gặp OOM thì giảm một nửa; DataLoader cố định `num_workers=4`.

### Kết quả

```text
data_processed/features/{model}/LXX_VXXX.npy                        # vector từng video (cache)
data_processed/index/{model}_keyframe_embedding.index               # index global
data_processed/index/{model}_keyframe_metadata.npy                  # metadata song hành
data_processed/index/playlist/{model}_{playlist}.index              # index từng playlist
data_processed/index/playlist/{model}_{playlist}_keyframe_metadata.npy
```

Metadata là mảng `(N, 2)` gồm `[video_id, keyframe_name]`, khớp 1-1 theo vị trí với vector trong `.index`: FAISS trả về id `i` thì thông tin nằm ở `metadata[i]`.

### Kiểm tra sau khi chạy

```bash
# Số video đã encode (phải bằng số video có keyframe)
ls data_processed/features/siglip2/*.npy | wc -l

# Web app truy vấn thử: nhập text -> trả top-k keyframe kèm ảnh thật
python data_processing/feature_extraction/test_retrieval.py
# rồi mở http://localhost:8000/ ; API JSON: /search?query=...&model=siglip2&top_k=10
```

`test_retrieval.py` tự nạp cả 3 model, model nào chưa có index thì bỏ qua kèm cảnh báo. Script **hardcode `CUDA_VISIBLE_DEVICES=7`** và cổng `8000` ngay trong file — sửa trực tiếp nếu GPU/cổng đó đang bận.

### Cơ chế cache — chạy lại rất rẻ

- Video đã có `.npy` thì **bỏ qua encode**. Model chỉ được nạp lên GPU khi thực sự có video mới cần encode, nên lần chạy lại chỉ để cập nhật index playlist gần như tức thì.
- Index chỉ được ghi lại khi metadata thay đổi. Playlist không có video mới thì giữ nguyên file cũ.

Nhờ kiến trúc per-video `.npy`, việc gom nhóm lại theo bất kỳ tiêu chí nào (playlist, thể loại, vòng thi...) chỉ là một phép concatenate — **không bao giờ phải chạy lại model encode**. Muốn encode lại một video thì xoá đúng file `data_processed/features/{model}/LXX_VXXX.npy` của nó rồi chạy lại.

---

## 🗺️ Quy ước đường dẫn

`config.yaml` là **nơi duy nhất khai báo đường dẫn**. Mọi script đều đọc từ đó — kể cả `run_pipeline.sh` (bash gọi `config_loader` qua `python3 -c`). Đừng hardcode lại ở bất kỳ đâu.

Hai gốc thư mục tách bạch theo vai trò:

| Gốc | Vai trò | Khoá config |
|---|---|---|
| `data/` | Dữ liệu **gốc** BTC cấp, chỉ đọc | `data_dir`, `videos_dir` |
| `data_processed/` | Dữ liệu **sinh ra** bởi pipeline | `transnetv2_dir`, `scenes_dir`, `predictions_dir`, `keyframes_dir`, `map_keyframes_dir`, `features_dir`, `index_dir` |

```yaml
videos_dir:        "{root_dir}/data/videos"                            # Videos_LXX/LXX_VXXX.mp4
scenes_dir:        "{root_dir}/data_processed/transnetv2/scenes"       # LXX_VXXX.scenes.txt
keyframes_dir:     "{root_dir}/data_processed/keyframes_transnetv2"    # Keyframes_LXX/LXX_VXXX/*.jpg
map_keyframes_dir: "{root_dir}/data_processed/map_keyframes"           # LXX_VXXX.csv
features_dir:      "{root_dir}/data_processed/features"                # {model}/LXX_VXXX.npy
index_dir:         "{root_dir}/data_processed/index"                   # {model}_*.index + playlist/
```

Vì video gốc và keyframe nằm ở hai gốc khác nhau, `load_all_video_keyframes_info()` nhận **hai tham số riêng** chứ không suy ra cả hai từ một `data_path`:

```python
all_video, video_keyframe_dict = load_all_video_keyframes_info(
    config["paths"]["videos_dir"], config["paths"]["keyframes_dir"]
)
```

> **Lưu ý:** `venv_transnetv2` cần `pyyaml` để đọc config. Đã cài sẵn; nếu tạo lại venv thì nhớ `uv pip install pyyaml`.
>
> `scripts/fix_dataset.sh` vẫn hardcode đường dẫn cũ, nhưng đó là script tái cấu trúc dataset chạy một lần lúc mới tải về, không nằm trong pipeline này.

---

## 📝 Lộ trình công việc (Todo List)

### 1. Tiền xử lý dữ liệu (Preprocessing — TransNetV2)
- [x] Triển khai chạy suy diễn mô hình TransNetV2 tìm phân cảnh.
- [x] Triển khai trích xuất Keyframe từ kết quả phân cảnh của TransNetV2.
- [x] Tối ưu hoá thuật toán cắt keyframe bằng OpenCV chế độ 1 lượt đọc (Single-pass).
- [x] Script `check_integrity.py` đối chiếu số ảnh trích xuất với số dòng CSV.
- [x] Thống nhất đường dẫn về `config.yaml`, gỡ toàn bộ hardcode trong `run_pipeline.sh`, `keyframe_extraction.py`, `check_integrity.py`. Bước [4] nhận đủ 150.242 keyframe / 873 video (trước đó là 0).
- [ ] Viết test kiểm tra số lượng video tải về, số keyframe so với kết quả TransNetV2.

### 2. Phân loại Playlist (Metadata Processing)
- [x] Đọc metadata BTC, tự dò mọi batch (b1/b2/b3).
- [x] Phân loại 873 video vào 9 playlist theo `channel_id` + regex title, khớp đúng số liệu BTC công bố.
- [x] Tách bộ luật ra `data/playlist_rules.json` kèm validate schema.
- [ ] Cân nhắc chia nhỏ 2 playlist lớn nhất (`Mon-Ngon-Moi-Ngay`, `60-Giay`) nếu lọc theo chương trình chưa đủ giảm không gian tìm kiếm.

### 3. Trích xuất đặc trưng ảnh (Feature Extraction)
- [/] Implement OpenCLIP (SigLIP 2, ViT-H/14, CLIP-ViT-L/14,...)
- [ ] Implement BEiT-3 (Multimodal Foundation Model)
- [ ] Implement InternVL2 (Vision-Language Model)
- [ ] Implement DINOv3 / DINOv2 (Self-supervised Vision Features)

### 4. Tìm kiếm và các tính năng bổ trợ (Indexing & Downstream Tasks)
- [x] Xây dựng FAISS index hai tầng (global + playlist), tách riêng khỏi encode trong `build_index.py`.
- [ ] Trích xuất văn bản trong ảnh bằng OCR.
- [ ] Nhận diện vật thể (Object Detection).
- [ ] Tích hợp tính năng Metadata-filtering và Text-search.
