# Data Processing Pipeline (AIC 2026)

Tài liệu này mô tả toàn cảnh quy trình xử lý dữ liệu (Data Processing) của hệ thống. Hệ thống kết hợp việc xử lý hình ảnh, văn bản (OCR), vật thể (Object Detection), âm thanh (Audio) và tạo mô tả tổng hợp (Image Captioning).

---

## 1. Các thành phần đã hoàn thiện (Production-Ready)
*Tham khảo thêm tại: `data_processing/README.md`*

- **Scene & Keyframe Extraction:** Phân rã video thành các shot cảnh bằng `TransNetV2` và trích xuất mid-frame làm ảnh đại diện (keyframe) thông qua `OpenCV/FFmpeg`.
- **Feature Embedding:** Cung cấp mã hóa đa mô hình (`SigLIP2`, `BEiT-3`, `DINOv3`) cho keyframes, lập chỉ mục `FAISS` và sinh siêu dữ liệu (Metadata) dưới dạng Numpy arrays.

---

## 2. Các thành phần đang thử nghiệm (Testing Phase) & Gợi ý cải tiến

### A. Object Detection (RAM++ & GroundingDINO)
**Mục đích:** Tự động sinh tags (RAM++) và phát hiện hộp bao quanh vật thể (GroundingDINO).
* **Cấu trúc Output hiện tại:**
  ```json
  {"frame_id": "L21_V001_001", "objects": [{"label": "PERSON", "score": 0.82, "box": [97.5, 182.0, 519.5, 718.4]}], "object_summary": ["PERSON", "FISH"]}
  ```
* **Gợi ý chỉnh sửa trước khi tích hợp chính thức:**
  1. **Chuẩn hóa Bounding Box:** Tọa độ `box` hiện đang lưu dưới dạng pixel tuyệt đối `[x1, y1, x2, y2]`. Nên chuyển đổi sang dạng tỷ lệ (normalized) `[0.0 -> 1.0]` bằng cách chia cho chiều rộng/chiều cao ảnh. Điều này giúp frontend vẽ hộp box chính xác dù resize video/ảnh.
  2. **Trích xuất chuỗi thô cho FTS5:** Trường `object_summary` đang là mảng `["PERSON", "FISH"]`. Khi nạp vào SQLite, nên nối chúng thành chuỗi `"person, fish"` để module FTS5 (Full-Text Search) dễ dàng lập chỉ mục từ khóa.

### B. Nhận diện chữ viết (OCR - PPOCR + VietOCR + VLM)
**Mục đích:** Đọc văn bản tiếng Việt xuất hiện trong video (slide học tập, bảng hiệu, phụ đề thời sự).
* **Gợi ý chỉnh sửa trước khi tích hợp chính thức:**
  1. **Lọc văn bản rác:** Thường xuyên có các ký tự rác hoặc nhiễu từ bối cảnh. Cần có bước *Normalization* (bỏ dấu câu dư thừa, lowercase) trước khi nạp vào FTS5.
  2. **Liên kết thời gian (Temporal mapping):** OCR text tồn tại trên màn hình nhiều giây. Thay vì lưu OCR riêng rẽ cho từng frame gây phình to Database, hãy nhóm các frame liên tiếp có chung văn bản lại với nhau.

---

## 3. Định hướng phát triển bổ sung (Future Integrations)

### A. Xử lý âm thanh (Audio Processing / ASR)
Dữ liệu thi (tin tức, ôn thi) có rất nhiều thông tin nằm ở lời thoại. Tìm kiếm "đoạn MC nói về chứng khoán" sẽ rất khó nếu chỉ dùng hình ảnh.
* **Gợi ý triển khai:**
  1. Dùng **FFmpeg** tách file `.wav` (16kHz, mono) từ video gốc.
  2. Sử dụng mô hình **Whisper** (hoặc `PhoWhisper` ưu tiên tiếng Việt) để dịch giọng nói thành văn bản (Speech-to-Text). Output của Whisper là các phân đoạn: `{"start": 12.0, "end": 15.5, "text": "Thị trường chứng khoán hôm nay"}`.
  3. Ánh xạ các mốc thời gian này vào các `frame_id` tương ứng và lưu đoạn hội thoại (transcript) vào SQLite FTS5.

### B. Mô tả hình ảnh (Image Captioning / Dense Video Captioning)
Để tìm kiếm ngữ nghĩa tốt nhất, các vector nhúng thô của SigLIP2 là chưa đủ. Các câu hỏi truy vấn KIS đôi khi cực kỳ phức tạp (ví dụ: *"người phụ nữ mặc áo đỏ đang cầm cốc nước đứng cạnh một chiếc xe đạp màu xanh"*).
* **Gợi ý triển khai:** Sử dụng một **VLM (Vision-Language Model)** như `InternVL2-8B`, `Qwen2-VL`, hoặc `LLaVA` chạy offline.
* **Quy trình:**
  Tạo một Prompt tổng hợp cung cấp cả bối cảnh (context):
  > "Dưới đây là một khung hình từ video. Các vật thể phát hiện được bao gồm: {object_summary}. Các chữ viết trong ảnh: {ocr_text}. Lời thoại ở thời điểm này: {asr_text}. Hãy viết một đoạn mô tả chi tiết bằng tiếng Việt về toàn cảnh khung hình này."
* **Lưu trữ:** Lưu câu caption này vào SQLite. Ngoài ra, bạn có thể mã hóa (encode) câu caption này bằng một mô hình Text Embedding (như `BGE-M3` hoặc `E5-multilingual`) để tạo thêm một Vector Database (FAISS) phụ, chuyên dành cho tìm kiếm văn bản $\leftrightarrow$ văn bản.
