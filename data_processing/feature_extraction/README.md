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
- **Mục đích**: Chứa thông tin định danh tương ứng 1-1 với từng vector trong file `.index`.
- **Cấu trúc**: Là một mảng Numpy 2 chiều có kích thước `(N, 2)`, trong đó mỗi dòng chứa cặp thông tin `[video_id, keyframe_id]` tương ứng với vector ở vị trí tương đương trong index.
- **Ví dụ về dữ liệu của một phần tử**:
  ```python
  # metadata[i] trả về:
  array(["L01_V001", "0052"], dtype='<U6')  # video_id và keyframe_id (tên ảnh)
  ```

### 3. Cơ chế Mapping và Cách trích xuất kết quả
Sự liên kết giữa file Vector (`.index`) và file Metadata (`.npy`) dựa trên **vị trí tương đối (implicit 1:1 mapping)**:
- FAISS lưu trữ các vector một cách tuần tự và ngầm định gán cho mỗi vector một ID (chỉ số dòng) bắt đầu từ `0`.
- File Metadata là một mảng Numpy có số lượng dòng khớp chính xác với số lượng vector trong file `.index`.
- **Quy tắc mapping:** ID của một vector do FAISS trả về chính là chỉ số (index) để truy xuất thẳng vào dòng tương ứng trong mảng metadata. Nghĩa là Vector `ID = i` trong FAISS tương ứng với thông tin lưu trữ tại `metadata[i]`.

Khi bạn tìm kiếm bằng một query_vector, FAISS sẽ trả về các ID này. Bạn chỉ việc dùng chúng để lấy thông tin chi tiết từ Numpy array.

**Ví dụ minh họa:**
```python
import faiss
import numpy as np

# 1. Nạp dữ liệu
index = faiss.read_index("/path/to/features/siglip2_keyframe_embedding.index")
metadata = np.load("/path/to/features/siglip2_keyframe_metadata.npy", allow_pickle=True)

# 2. Tìm kiếm vector
# Giả sử query_vector là vector đã trích xuất từ câu truy vấn (shape: [1, D])
k_neighbors = 5 
distances, indices = index.search(query_vector, k_neighbors)

# 3. Lấy thông tin hiển thị
print("Top kết quả tìm kiếm:")
for i, idx in enumerate(indices[0]):
    if idx != -1:  # -1 nghĩa là không tìm thấy
        video_id, keyframe_id = metadata[idx]
        print(f"Top {i+1} - Khoảng cách L2: {distances[0][i]:.4f}")
        print(f"   Video: {video_id}, Keyframe: {keyframe_id}")
```

## 🚀 Hướng dẫn nhanh (Quick Start)

Dưới đây là các ví dụ cụ thể về cách vận hành trích xuất đặc trưng hình ảnh và tích hợp chúng vào mã nguồn.

### 1. Khởi chạy thông qua Terminal
Script `run_embedding.py` tự động đọc cấu hình thư mục đầu vào/đầu ra từ `config.yaml` và hỗ trợ các tham số CLI sau:

```bash
# Trích xuất đặc trưng với mô hình SigLIP 2
python run_embedding.py --model siglip2 --batch_size 64

# Trích xuất đặc trưng với mô hình BEiT-3
python run_embedding.py --model beit3 --batch_size 32

# Trích xuất đặc trưng với mô hình DINOv3
python run_embedding.py --model dinov3 --batch_size 64
```

### 2. Sử dụng trong mã nguồn Python
Mỗi class mô hình (`SigLIP2Embedding`, `Beit3Embedding`, `DinoV3Embedding`) kế thừa từ `BaseEmbedding` và cung cấp các phương thức chuẩn hóa để mã hóa ảnh hoặc văn bản:

```python
import torch
from siglip2_embedding import SigLIP2Embedding

# 1. Khởi tạo extractor (tự động sử dụng GPU nếu khả dụng)
extractor = SigLIP2Embedding()

# 2. Mã hóa một hình ảnh đơn lẻ
image_feature = extractor.encode_image("/path/to/keyframe.jpg")
print("Single image feature shape:", image_feature.shape)

# 3. Mã hóa một batch hình ảnh (dạng torch.Tensor đã tiền xử lý)
# Tiền xử lý có sẵn qua extractor.preprocess (cho DinoV3) hoặc extractor.preprocess/image_processor (cho SigLIP2, BEiT-3)
image_processor = getattr(extractor, "preprocess", None) or getattr(extractor, "image_processor", None)
# Giả sử có một list PIL Images
# image_tensors = torch.stack([image_processor(img) for img in pil_images])
# batch_features = extractor.encode_images_batch(image_tensors)

# 4. Mã hóa một câu truy vấn văn bản (chỉ khả dụng với mô hình hỗ trợ Multimodal như SigLIP2 và BEiT-3)
text_feature = extractor.encode_text("a drone view of a city")
print("Text feature shape:", text_feature.shape)
```

### 3. API Dịch vụ Tìm kiếm (FastAPI)
Để chạy và thử nghiệm tìm kiếm nhanh qua HTTP API, bạn có thể khởi động server:
```bash
python test_retrieval.py
```
API này sẽ tải toàn bộ model lên GPU và lưu FAISS Index + Metadata của cả 3 mô hình vào bộ nhớ để tìm kiếm tức thì.

