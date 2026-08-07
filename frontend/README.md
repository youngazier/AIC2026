# Frontend tìm kiếm AIC 2026

Giao diện thi đấu: gõ truy vấn → duyệt keyframe → tick chọn → sinh file CSV nộp cho BTC.
React 18 + Vite, không dùng thư viện UI ngoài (mỗi dependency là một thứ có thể hỏng
lúc đang thi).

## Chạy

```bash
# 1. Backend phải chạy trước
cd .. && ./scripts/run_backend.sh          # http://localhost:8000

# 2. Frontend
cd frontend
npm install
npm run dev                                # http://localhost:5173
```

Backend chạy ở máy khác hoặc cổng khác:

```bash
BACKEND_URL=http://192.168.1.50:8000 npm run dev
```

Vite proxy mọi request `/api` sang backend, nên trong code chỉ có đường dẫn tương đối —
ảnh keyframe và video cũng đi qua đúng đường đó, không phải cấu hình host ở hai nơi.

## Ba dạng truy vấn

| Dạng | Ô nhập | Một dòng CSV |
|---|---|---|
| **KIS** | một ô mô tả | `L21_V001,1000` |
| **Q&A** | mô tả + ô câu trả lời | `L21_V001,1000,6` |
| **TRAKE** | cả đoạn, tự tách theo dấu `.` | `L21_V010,101,156,203,251` |

TRAKE tách câu thành các sự kiện và hiện chúng thành chip bấm được — bấm một chip là
tìm riêng khoảnh khắc của sự kiện đó, tick frame tìm được thì nó nối vào đúng dòng của
video đó. Backend chưa có thuật toán xếp hạng riêng cho TRAKE (chưa căn chỉnh thời gian
giữa các sự kiện), nên phần này là quy trình thủ công có hỗ trợ, không phải tự động.

## Luồng dùng

1. Chọn dạng truy vấn, model, `top-k`, playlist (tùy chọn).
2. Gõ mô tả → **Tìm kiếm** (hoặc Enter; Shift+Enter để xuống dòng).
3. Ở mỗi ô kết quả:
   - **tick / bấm vào ảnh** → thêm frame vào danh sách nộp;
   - **◎ Tương tự** → DINOv3 tìm frame giống nó, kết quả mở ở cột bên phải;
   - **▶ Video** → xem video đúng tại frame đó.
4. Trong cửa sổ video: `←` `→` lùi/tiến 1 frame (Shift = 10 frame), `Space` phát/dừng,
   rồi **+ Lấy frame** để đưa frame tại con trỏ vào danh sách nộp.
5. Gõ tên file → **Tạo file CSV**. File được ghi vào `submissions/<tên>.csv` trên máy
   chạy backend. **⭳ Tải về** lưu thêm một bản về máy đang mở trình duyệt.

## Những chỗ dễ mất điểm mà UI cố gắng chặn

- **Thứ tự dòng là điểm số.** Final Score = trung bình `R@k` với k ∈ {1,5,20,50,100},
  nên frame tự tin nhất phải nằm ở dòng đầu. Danh sách nộp giữ đúng thứ tự tick và có
  nút ↑ ↓ để xếp lại — không có chỗ nào tự sắp xếp lại sau lưng bạn.
- **Trần 100 dòng** của BTC được chặn ngay khi tick, kèm lời nhắc bỏ bớt dòng cuối.
- **`frame` chứ không phải `keyframe`.** Số nộp là `frame_idx` (khung hình thật trong
  video), không phải tên file keyframe `0052`. Ô kết quả hiện `frame 1000` to và rõ,
  còn `kf 0052` để nhỏ bên dưới.
- **Quy đổi giây ↔ frame** trong trình xem video dùng `frame = floor(t × fps)` và tua
  tới giữa frame `t = (f + 0.5) / fps`, khớp với quy ước `timestamp = frame_idx / fps`
  của pipeline. `fps` lấy từ `GET /api/v1/videos/{id}` chứ không đoán 25 hay 30.
- **Model không dùng được thì bị làm mờ**, kèm lý do khi rê chuột, thay vì cho bấm rồi
  báo lỗi.

## Cấu trúc

```text
src/
├── api.js                      # Nơi DUY NHẤT gọi backend
├── constants.js                # Dạng truy vấn, lựa chọn model, tách TRAKE, format giờ
├── App.jsx                     # State chung + luồng chọn frame
└── components/
    ├── SearchPanel.jsx         # Thanh truy vấn, toggle dạng/model, top-k, playlist
    ├── ResultGrid.jsx          # Lưới kết quả chính
    ├── ResultCard.jsx          # Một ô keyframe (dùng chung cho lưới và cột tương tự)
    ├── SimilarPanel.jsx        # Cột kết quả tìm-ảnh-giống-ảnh
    ├── VideoModal.jsx          # Xem video, bước từng frame, lấy frame tại con trỏ
    └── SubmissionPanel.jsx     # Danh sách nộp, xếp hạng, xem trước và ghi CSV
```

Một nguồn sự thật duy nhất cho việc chọn frame: `rows` (danh sách nộp). Ô nào đang được
tick là suy ra từ `rows`, nên bỏ tick ở lưới và xóa dòng ở bảng nộp luôn khớp nhau.

## Build production

```bash
npm run build      # -> dist/
```

`dist/` là file tĩnh; serve nó cùng origin với backend thì không cần proxy nữa vì mọi
URL trong code đều tương đối.
