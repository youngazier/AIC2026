# Thông tin Cuộc thi AI Challenge 2026 (AIC 2026)

Tài liệu này tổng hợp cấu trúc tập dữ liệu (dataset), các dạng truy vấn/câu hỏi từ Ban tổ chức (BTC) và các yêu cầu kỹ thuật đặc thù của cuộc thi nhằm định hình cho việc thiết kế kiến trúc hệ thống Backend và cơ sở dữ liệu.

---

## 📂 1. Đặc thù của Tập dữ liệu (Dataset)

Dataset của cuộc thi bao gồm nhiều thể loại video đa dạng và phong phú như:
- Tin tức HTV (News)
- Giải đua xe đạp (Sports)
- Múa lân (Culture/Events)
- Ôn thi THPT (Slide bài giảng, bài giảng online)
- Nấu ăn (Cooking/Tutorials)
- Và các thể loại đời sống, giải trí khác.

### Chiến lược Mã hóa (Encoding Strategy)
Để tối ưu hóa tìm kiếm, hệ thống cần hỗ trợ tìm kiếm song hành qua hai luồng dữ liệu:
1. **Global Encoding (Mã hóa toàn cục):** Trích xuất và lập chỉ mục (FAISS Index) toàn bộ keyframe của tất cả video trong hệ thống để tìm kiếm không giới hạn thể loại.
2. **Category/Playlist Encoding (Mã hóa theo thể loại):** Nhóm các video theo playlist hoặc phân loại thể loại trước khi xây dựng các FAISS Index riêng biệt. Điều này giúp tăng tốc độ tìm kiếm và giảm thiểu nhiễu từ các thể loại không liên quan khi người dùng đã xác định được nhóm nội dung.

---

## 🎯 2. Các dạng câu hỏi/truy vấn của cuộc thi

Hệ thống Backend cần giải quyết 4 tác vụ cốt lõi sau:

### A. Known-Item Search (KIS) - Tìm kiếm vật thể/phân cảnh xác định
- **Mục tiêu:** Đầu vào là mô tả một phân cảnh cụ thể, hệ thống phải tìm ra đúng frame của phân cảnh đó trong video cơ sở dữ liệu. Kết quả trả về phải chứa tên video và frame index chính xác nằm trong khoảng Ground-Truth (GT) của BTC.
- **Phân loại:**
  1. **Video Query (Truy vấn qua video mẫu):** Thí sinh được xem một video trình chiếu mẫu từ BTC, sau đó tự viết mô tả bằng văn bản (không được chụp ảnh màn hình) và đưa vào công cụ tìm kiếm.
  2. **Text Query (Truy vấn bằng văn bản):** BTC chỉ đưa ra mô tả bằng chữ và thí sinh không được xem video trước.
  - *Ví dụ:* *"Tìm phân cảnh tôi vô tình đánh rơi chiếc chìa khóa có móc khóa hình gấu bông màu hồng khi đang đi bộ qua một quầy bán hoa quả chín."*

### B. Ad-hoc Video Search (AVS) - Tìm kiếm ngữ nghĩa tổng quát
- **Mục tiêu:** Tìm kiếm tất cả các phân cảnh trong cơ sở dữ liệu khớp với một mô tả ngữ nghĩa chung mà không nhắm đến một khoảnh khắc duy nhất. Hệ thống phải trả về danh sách các phân đoạn video xếp hạng theo điểm tương đồng ngữ nghĩa từ cao xuống thấp.
- **Đặc trưng:** Mang tính bao quát, không xác định trước liệu phân cảnh đó có tồn tại trong dữ liệu hay không.
  - *Ví dụ:* *"Tìm tất cả phân cảnh có người lớn đang hướng dẫn trẻ em trồng cây hoặc tưới hoa trong vườn."*

### C. Video QA (VQA) - Hỏi đáp nội dung Video
- **Mục tiêu:** Đầu vào gồm một đoạn video dài và một câu hỏi truy vấn. Hệ thống cần trả về hình ảnh chính xác định vị câu trả lời cùng với câu trả lời dạng văn bản ngắn gọn.
- **Yêu cầu kỹ thuật:** AI phải thực hiện suy luận thời gian (Temporal reasoning) và đếm đối tượng (Counting) thông qua việc:
  1. Xác định và theo dõi nhân vật/vật thể được đề cập.
  2. Định vị khoảng thời gian xảy ra hành động.
  3. Đếm số lượng hoặc phân tích chuỗi sự kiện tiếp theo (ví dụ: chuỗi hành động ai tặng quà sau khi thổi nến).
  - *Ví dụ:* Clip sinh nhật 2 phút với câu hỏi: *"Trong buổi tiệc, cậu bé mặc chiếc áo hình siêu nhân màu đỏ đã thổi tổng cộng bao nhiêu cây nến trên bánh kem và sau đó ai là người đã tặng quà cho cậu bé?"*

### D. Conversational KIS (KISC) - Tìm kiếm qua hội thoại đa lượt
- **Mục tiêu:** Tìm kiếm thông qua hội thoại tương tác với trợ lý ảo (Virtual Assistant) để làm rõ ý định của người dùng khi truy vấn ban đầu quá mơ hồ.
- **Luồng xử lý điển hình:**
  * *Lượt 1 (User):* "Tìm giúp tôi đoạn video tôi gặp một người bạn cũ vào tuần trước." *(Mơ hồ)*
  * *Lượt 2 (Agent):* "Bạn có nhớ cuộc gặp đó diễn ra ở trong nhà hay ngoài trời không? Người bạn đó là nam hay nữ?" *(Thu hẹp phạm vi)*
  * *Lượt 3 (User):* "Chúng tôi gặp nhau ở một quán cà phê ngoài trời, anh ấy mặc áo sơ mi màu xanh dương." *(Bổ sung chi tiết)*
  * *Lượt 4 (Agent):* Lọc metadata (thời gian: tuần trước) + không gian (cà phê ngoài trời) + nhận diện vật thể/màu sắc (nam, áo sơ mi xanh) $\rightarrow$ Trả về kết quả chính xác nhất.

---

## 🔒 3. Giới hạn & Ràng buộc Hệ thống
- **Không sử dụng Docker:** Phải cài đặt, cấu hình môi trường và vận hành trực tiếp trên máy chủ thông qua Python environment hoặc các dịch vụ local chạy dưới quyền User.
- **Cơ sở dữ liệu Vector:** Hiện tại đang sử dụng **FAISS** để tìm kiếm vector embedding nhanh.
- **Cơ sở dữ liệu text/metadata:** Cần tích hợp thêm giải pháp lưu trữ metadata văn bản (kết quả OCR từ slide bài giảng/tin tức, kết quả Object Detection/nhận diện hành động) mà không phụ thuộc vào Docker.
