# Feature Extraction - AI Challenge 2026

## 🎯 Tổng quan (Abstract)
Thư mục này chịu trách nhiệm trích xuất đặc trưng hình ảnh (feature extraction) từ các keyframes của video. Mục đích chính là chuyển đổi dữ liệu hình ảnh thô thành các vector nhúng (embeddings) mật độ cao, phục vụ trực tiếp cho hệ thống tìm kiếm và truy xuất video thông minh trong dự án AI Challenge 2026.

## 🏗️ Kiến trúc mã nguồn

Kiến trúc của module trích xuất được thiết kế xoay quanh tính mở rộng và tái sử dụng, cho phép dễ dàng tích hợp các mô hình AI mới thông qua tư duy hướng đối tượng.

### Lớp trừu tượng `BaseEmbedding`
`BaseEmbedding` đóng vai trò là một interface cốt lõi (core interface) nhằm chuẩn hóa toàn bộ quy trình trích xuất embedding cho bất kỳ model nào. Các model khi được đưa vào hệ thống bắt buộc phải kế thừa lớp này và triển khai các phương thức bắt buộc. Việc này đảm bảo đầu vào và đầu ra luôn đồng nhất, giúp hệ thống dễ dàng chuyển đổi qua lại giữa các mô hình mà không làm phá vỡ kiến trúc chung.

### Các module kế thừa
Hiện tại, hệ thống hỗ trợ các mô hình trích xuất đặc trưng tiên tiến sau:
- **SigLIP2**: Triển khai tại [`siglip2_embedding.py`](./siglip2_embedding.py)
- **BEIT3**: Triển khai tại [`beit3_embedding.py`](./beit3_embedding.py)
- **DINOv3**: Triển khai tại [`dinov3_embedding.py`](./dinov3_embedding.py)

Mỗi file sẽ chứa logic cụ thể để nạp mô hình, tiền xử lý hình ảnh cho phù hợp với đầu vào của mạng, và tiến hành inference để lấy ra vector.

## 💾 Cấu trúc lưu trữ FAISS Index

Để đảm bảo hiệu năng tối đa cho việc tìm kiếm vector quy mô lớn, dữ liệu sau khi trích xuất được chia thành hai phần và lưu trữ dưới dạng cặp file: `.index` và `.npy`. Việc tách biệt này là vô cùng quan trọng: nó giúp FAISS chỉ nạp các khối lượng vector bằng số thực vào bộ nhớ (RAM/VRAM) để tìm kiếm khoảng cách siêu tốc, trong khi siêu dữ liệu (metadata) chỉ được truy xuất khi cần thiết.

### 1. File Vector (`index_path`)
- **Định dạng**: `.index` (Định dạng chuẩn do thư viện FAISS cung cấp).
- **Mục đích**: Chứa thuần túy các vector embedding đã được trích xuất (thường ở định dạng float32 hoặc float16). Tệp này được lập chỉ mục (indexed) để tối ưu hóa việc tính toán khoảng cách (L2, Cosine/Inner Product) giữa các vector.
- **Cấu trúc**: Là một cấu trúc nhị phân của cấu trúc dữ liệu index trong FAISS (VD: `IndexFlatL2`, `IndexIVFPQ`). Trong file này, mỗi vector được liên kết ngầm định với một ID kiểu số nguyên (bắt đầu từ 0).

### 2. File Metadata (`meta_path`)
- **Định dạng**: `.npy` (Numpy array).
- **Mục đích**: Chứa thông tin ngữ cảnh (metadata) tương ứng 1-1 với từng vector trong file `.index`.
- **Cấu trúc**: Là một mảng (array) của thư viện Numpy, trong đó chứa các dictionary (hoặc structured array) lưu trữ thông tin. Thứ tự của các phần tử trong mảng này tương ứng chính xác với thứ tự vector trong file `.index`. Nghĩa là metadata tại phần tử thứ `i` trong tệp Numpy là của vector thứ `i` trong FAISS.
- **Ví dụ về dữ liệu của một phần tử**:
  ```python
  {
      "video_id": "L01_V001",
      "frame_idx": 1250,
      "image_path": "/data/keyframes/L01_V001/frame_1250.jpg",
      "timestamp": "00:00:50.00"
  }
  ```

### 3. Cơ chế Mapping và Cách trích xuất kết quả
Sự liên kết giữa file Vector (`.index`) và file Metadata (`.npy`) dựa trên **vị trí tương đối (implicit 1:1 mapping)**:
- FAISS lưu trữ các vector một cách tuần tự và ngầm định gán cho mỗi vector một ID (chỉ số) bắt đầu từ `0`.
- File Metadata là một mảng Numpy có số lượng phần tử khớp chính xác với số lượng vector.
- **Quy tắc mapping:** ID của một vector do FAISS trả về chính là chỉ số (index) để truy xuất thẳng vào mảng metadata. Nghĩa là Vector `ID = i` trong FAISS sẽ chứa thông tin hiển thị tại phần tử `metadata[i]`.

Khi bạn tìm kiếm bằng một query_vector, FAISS sẽ trả về các ID này. Bạn chỉ việc dùng chúng để lấy thông tin chi tiết từ Numpy array.

**Ví dụ minh họa:**
```python
import faiss
import numpy as np

# 1. Nạp dữ liệu
index = faiss.read_index("/path/to/features/siglip2_features.index")
metadata = np.load("/path/to/features/siglip2_meta.npy", allow_pickle=True)

# 2. Tìm kiếm vector
# Giả sử query_vector là vector đã trích xuất từ câu truy vấn (shape: [1, D])
k_neighbors = 5 
distances, indices = index.search(query_vector, k_neighbors)

# 3. Lấy thông tin hiển thị
print("Top kết quả tìm kiếm:")
for i, idx in enumerate(indices[0]):
    if idx != -1:  # -1 nghĩa là không tìm thấy
        info = metadata[idx]
        print(f"Top {i+1} - Khoảng cách: {distances[0][i]:.4f}")
        print(f"   Video: {info['video_id']}, Frame: {info['frame_idx']}")
        print(f"   Ảnh: {info['image_path']}")
```

## 🚀 Hướng dẫn nhanh (Quick Start)

Dưới đây là một ví dụ nhanh để bạn chạy script trích xuất đặc trưng thông qua file `run_embedding.py`.

**Chạy lệnh qua Terminal:**
```bash
# Khởi chạy trích xuất với mô hình SigLIP2
python run_embedding.py \
    --model siglip2 \
    --data_dir /path/to/AIC2026/keyframes \
    --output_dir /path/to/AIC2026/features \
    --batch_size 64
```

**Mã nguồn minh họa (cách sử dụng bên trong mã):**
```python
from siglip2_embedding import SigLIP2Embedding

# 1. Khởi tạo mô hình
extractor = SigLIP2Embedding(device='cuda')

# 2. Thiết lập đường dẫn lưu trữ
index_path = "/path/to/features/siglip2_features.index"
meta_path = "/path/to/features/siglip2_meta.npy"

# 3. Tiến hành trích xuất và lưu trữ
extractor.extract_and_save(
    data_dir="/path/to/keyframes",
    index_save_path=index_path,
    meta_save_path=meta_path,
    batch_size=64
)

print("Trích xuất thành công và đã lưu file Index + Meta!")
```
