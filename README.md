# AI Challenge 2026 (AIC 2026) - Video Processing & Semantic Retrieval Pipeline

Dự án này là hệ thống xử lý dữ liệu video đầu-cuối (end-to-end pipeline) nhằm phục vụ cuộc thi **AI Challenge 2026**. Hệ thống thực hiện việc phân tích cảnh video, trích xuất keyframes, tạo vector nhúng (embeddings) đa mô hình AI (SigLIP 2, BEiT-3, DINOv3), xây dựng chỉ mục tìm kiếm vector FAISS siêu tốc và cung cấp API truy vấn ngữ nghĩa (Semantic Search).

---

## 📂 Cấu trúc thư mục dự án

```text
AIC2026/
├── config.yaml                  # File cấu hình tập trung các đường dẫn dữ liệu và thông số mô hình
├── main.py                      # Script chính khởi chạy kiểm tra dự án
├── pyproject.toml               # File quản lý package, dependency dự án
├── uv.lock                      # Khóa phiên bản dependency của công cụ uv
├── data/                        # [Được tạo tự động] Thư mục chứa dữ liệu trung gian và kết quả
│   ├── videos/                  # Chứa các video gốc chia theo nhóm (.mp4)
│   ├── keyframes/               # Các ảnh keyframe trích xuất từ phân cảnh
│   ├── features/                # Vector embedding thô dạng npy cho từng video
│   └── index/                   # File FAISS Index (.index) và Metadata (.npy) của các mô hình
├── data_processing/             # Pipeline xử lý dữ liệu chính
│   ├── keyframe_extraction/     # Trích xuất shot cảnh và ảnh keyframes
│   │   ├── transnetv2/          # Mô hình phát hiện chuyển cảnh TransNetV2
│   │   └── keyframe_extraction.py # Trích xuất keyframes tối ưu qua OpenCV/FFmpeg
│   └── feature_extraction/      # Trích xuất đặc trưng AI và lập chỉ mục
│       ├── base_embedding.py    # Lớp cơ sở (Abstract Class) định nghĩa cấu trúc Model Embedding
│       ├── siglip2_embedding.py # Tích hợp mô hình SigLIP 2 (Google / OpenCLIP)
│       ├── beit3_embedding.py   # Tích hợp mô hình BEiT-3 (Microsoft / UniLM)
│       ├── dinov3_embedding.py  # Tích hợp mô hình DINOv3 (Meta / HuggingFace)
│       ├── run_embedding.py     # Script chạy batch trích xuất đặc trưng & dựng FAISS index
│       └── test_retrieval.py    # FastAPI service hỗ trợ tìm kiếm ngữ nghĩa thời gian thực
├── support_models/              # Chứa checkpoint và cấu hình mô hình tải về cục bộ (BEiT-3, v.v.)
├── utils/                       # Các module tiện ích dùng chung
│   ├── config_loader.py         # Trình nạp file cấu hình config.yaml
│   └── load_all_video_keyframes_info.py # Đọc thông tin video và keyframes từ dataset
├── scripts/                     # Chứa các bash script hỗ trợ vận hành hệ thống
│   └── fix_dataset.sh           # Script chuẩn hóa cấu trúc dữ liệu dataset AIC
└── tests/                       # Hệ thống các test case kiểm tra tính nhất quán dữ liệu
    ├── test_keyframes.py        # Kiểm tra tính khớp số lượng keyframe trích xuất và CSV ánh xạ
    └── test_embeddings.py       # Kiểm tra tính khớp của các vector embedding và truy vấn FAISS
```

---

## 🛠️ Yêu cầu môi trường & Cài đặt

Dự án được quản lý bằng bộ công cụ **uv** hiện đại cho hiệu năng cực cao.

### 1. Khởi tạo môi trường ảo Python 3.10
```bash
# Di chuyển vào thư mục dự án
cd /tmp2/maitanha/vgu/ttn/AIC2026

# Tạo môi trường ảo
uv venv --python 3.10
source .venv/bin/activate
```

### 2. Cài đặt các thư viện cần thiết
```bash
# Cài đặt toàn bộ dependencies trong file pyproject.toml
uv pip install -e .

# Cài đặt PyTorch hỗ trợ CUDA 12.4 (hoặc phiên bản phù hợp với GPU của bạn)
uv pip install torch torchvision --index-url https://download.pytorch.org/whl/cu124 --force-reinstall
```

### 3. Cài đặt môi trường ảo riêng cho TransNetV2 (Khuyên dùng)
TransNetV2 sử dụng các phiên bản thư viện cũ hơn (hoặc TensorFlow/PyTorch tùy chỉnh), khuyến khích tạo môi trường ảo riêng tại `data_processing/keyframe_extraction/venv_transnetv2`:
```bash
cd data_processing/keyframe_extraction/transnetv2
uv venv ../../venv_transnetv2 --python 3.10
source ../../venv_transnetv2/bin/activate
uv pip install -e .
uv pip install torch --index-url https://download.pytorch.org/whl/cu124
```

---

## 🚀 Quy trình vận hành chi tiết (Step-by-Step)

### Bước 1: Chuẩn hóa thư mục Dataset (AIC Format)
Nếu tập dữ liệu thô có cấu trúc lồng nhau phức tạp, chạy script chuẩn hóa cấu trúc thư mục:
```bash
bash scripts/fix_dataset.sh
```

### Bước 2: Phát hiện phân cảnh video (TransNetV2 Inference)
Chạy suy diễn mô hình TransNetV2 để sinh ra danh sách phân cảnh dưới dạng file `.scenes.txt`:
```bash
# Kích hoạt venv của transnetv2
source data_processing/venv_transnetv2/bin/activate

# Chạy inference trên toàn bộ thư mục videos
CUDA_VISIBLE_DEVICES=7 transnetv2_pytorch /path/to/videos_dir -o /path/to/output_dir --all_videos
```

### Bước 3: Trích xuất ảnh Keyframes
Sử dụng script trích xuất để cắt ảnh frame giữa (mid-frame) của từng shot cảnh được phát hiện:
```bash
# Di chuyển đến thư mục chứa script
cd data_processing/keyframe_extraction

# Kích hoạt môi trường chạy chính của hệ thống
source ../../.venv/bin/activate

# Chạy trích xuất tối ưu bằng OpenCV (Single-pass)
CUDA_VISIBLE_DEVICES=7 python keyframe_extraction.py
```
*Kết quả:* Ảnh keyframes `.jpg` được sinh ra tại `data/keyframes/` và file CSV ánh xạ thông tin thời gian thực được tạo tại `data/map_keyframes/`.

### Bước 4: Trích xuất vector đặc trưng & Xây dựng FAISS Index
Sinh các vector đặc trưng cho từng frame ảnh và đẩy vào cơ sở dữ liệu vector index FAISS:
```bash
cd ../feature_extraction

# Trích xuất với mô hình SigLIP2
python run_embedding.py --model siglip2 --batch_size 64

# Trích xuất với mô hình BEiT-3
python run_embedding.py --model beit3 --batch_size 32

# Trích xuất với mô hình DINOv3
python run_embedding.py --model dinov3 --batch_size 64
```
*Kết quả:* Tự động sinh ra tệp vector index `.index` và siêu dữ liệu song hành `.npy` trong thư mục `data/index/`.

### Bước 5: Khởi chạy API dịch vụ Tìm kiếm ngữ nghĩa
Bật API endpoint để hỗ trợ truy vấn tức thì qua văn bản:
```bash
# Chạy API server
python test_retrieval.py
```
Server chạy mặc định trên cổng `8000`. Xem tài liệu chi tiết cách truy cập API tại [feature_extraction README](./data_processing/feature_extraction/README.md).

---

## 🧪 Hệ thống kiểm tra chất lượng & Kiểm thử (Testing)

Để đảm bảo pipeline hoạt động hoàn hảo và không bị lệch dữ liệu ở bất kỳ bước nào, chúng tôi xây dựng bộ test suite tích hợp tại thư mục `tests/`:

1. **Kiểm tra trích xuất Keyframes (`test_keyframes.py`):**
   Kiểm tra chéo giữa file dự đoán của TransNetV2, số lượng ảnh `.jpg` thực tế được lưu và số lượng hàng trong file CSV ánh xạ.
   ```bash
   python tests/test_keyframes.py
   ```

2. **Kiểm tra tính nhất quán vector nhúng (`test_embeddings.py`):**
   Kiểm tra số lượng vector trong file `.npy` có khớp hoàn toàn với số ảnh keyframe và mô phỏng chạy truy vấn láng giềng gần nhất trên FAISS.
   ```bash
   python tests/test_embeddings.py
   ```

---

## 📊 So sánh các mô hình nhúng hiện có

| Mô hình | Nhà phát triển | Kiểu dữ liệu | Đặc điểm nổi bật | Kích thước vector |
| :--- | :--- | :--- | :--- | :--- |
| **SigLIP 2** | Google | Multimodal (Text-Image) | Khả năng căn chỉnh văn bản - hình ảnh mạnh mẽ, tối ưu tìm kiếm truy vấn mô tả. | 1152 chiều |
| **BEiT-3** | Microsoft | Multimodal (Text-Image) | Foundation Model mạnh mẽ, độ chính xác cao trong bài toán Retrieval. | 768 chiều |
| **DINOv3** | Meta | Vision Only | Tự học không giám sát tốt, giữ nguyên đặc trưng bố cục, góc chụp, chi tiết nhỏ của vật thể. | 1280 chiều |
