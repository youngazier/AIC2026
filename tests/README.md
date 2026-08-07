# Kiểm thử pipeline tiền xử lý

Hai loại kiểm tra khác nhau, đừng nhầm lẫn:

| | Kiểm tra **CODE** (thư mục này) | Kiểm tra **DỮ LIỆU** (`check_integrity.py`) |
|---|---|---|
| Trả lời câu hỏi | Pipeline có viết đúng không? | Dữ liệu hiện có đủ và toàn vẹn không? |
| Chạy trên | Dataset giả trong thư mục tạm | `data/` + `data_processed/` thật |
| Thời gian | ~30 giây | ~10 giây (873 video) |
| Khi nào chạy | Sau mỗi lần sửa code | Sau mỗi lần chạy pipeline |

Cả hai đều trả về **exit code 0 khi sạch, 1 khi có lỗi**, nên dùng được trong script tự động.

---

## 1. Chạy test code

```bash
cd /tmp2/maitanha/vgu/ttn/AIC2026

pytest                                    # toàn bộ (~30s)
pytest -v                                 # xem tên từng test
pytest tests/test_check_integrity.py      # một file
pytest -k "sentinel"                      # lọc theo tên
pytest -x                                 # dừng ngay ở lỗi đầu tiên
pytest --lf                               # chỉ chạy lại các test đã fail lần trước
```

Dùng **python gốc** (đã có `pytest`, `numpy`, `PIL`, `cv2`), **không** cần `venv_embedding`.
Các test cần `faiss` tự gọi sang `venv_embedding/bin/python` bằng subprocess, nên không phải cài thêm gì.

Bộ test **không bao giờ đụng vào `data/` hay `data_processed/` thật** — nó dựng một dataset giả (video `.mp4` do `ffmpeg` sinh ra) trong thư mục tạm của pytest và xoá sau khi xong.

### Có gì trong đây

| File | Kiểm cái gì | Số test |
|---|---|---|
| `test_shot_sampling.py` | Luật chọn keyframe từ ranh giới shot | 17 |
| `test_feature_store.py` | Quy tắc `.npy` đã hoàn chỉnh chưa + ghi nguyên tử | 13 |
| `test_check_integrity.py` | **Tiêm lỗi**: làm hỏng dữ liệu rồi xem trình kiểm tra có bắt được không | 44 |
| `test_keyframe_extraction.py` | Phía sinh dữ liệu: staging, sentinel `_SUCCESS`, cổng nghiệm thu | 8 |

`test_check_integrity.py` là phần đáng học nhất. Một trình kiểm tra không bao giờ báo lỗi thì
vô dụng, mà điều đó **không nhìn ra được** từ việc chạy nó trên dữ liệu đang PASS. Cách duy nhất
biết nó có tác dụng là cố tình làm hỏng dữ liệu rồi xem nó có kêu không. Ví dụ:

```python
def test_thieu_va_thua_bu_tru_nhau_van_bi_bat(dataset):
    """Chính là lỗ hổng của bản cũ: so số ĐẾM thì hai lỗi này triệt tiêu nhau."""
    jpg_path(dataset, VID, 2).unlink()          # xoá 1 ảnh
    write_jpg(jpg_path(dataset, VID, 99), seed=1)  # thêm 1 ảnh rác
    rc, report, _ = run_verify(dataset)
    assert {"KF_MISSING", "KF_EXTRA"} <= codes(report, "ERROR")
    assert rc == 1
```

### Thêm một test mới

Mọi thứ cần thiết đã có trong `conftest.py`:

```python
def test_loi_moi_cua_ban(dataset):     # `dataset` = bản sao dataset giả đang sạch
    lam_hong_cai_gi_do(dataset)        # tiêm đúng MỘT lỗi
    rc, report, _ = run_verify(dataset)
    assert "MA_LOI_MONG_DOI" in codes(report, "ERROR")
    assert rc == 1
```

Nguyên tắc: **mỗi test tiêm đúng một lỗi**. Tiêm nhiều lỗi cùng lúc thì khi test đỏ bạn không
biết cái nào gây ra.

---

## 2. Kiểm tra dữ liệu thật

```bash
# Sau mỗi lần chạy pipeline
python data_processing/keyframe_extraction/check_integrity.py

# Có faiss -> kiểm luôn cả FAISS index (nên dùng bản này)
data_processing/feature_extraction/venv_embedding/bin/python \
    data_processing/keyframe_extraction/check_integrity.py
```

Các cờ hay dùng:

```bash
-f L21,L22            # chỉ vài folder
--verbose             # in đủ mọi finding thay vì 3 dòng đầu mỗi loại
--fast                # bỏ qua việc mở từng .jpg (nhanh hơn, kém chặt hơn)
--stage keyframe      # chỉ một stage
--strict              # coi WARN là lỗi
--hash                # tính sha256 video gốc, lưu vào manifest (chạy một lần)
--deep-decode         # decode toàn bộ video để phát hiện file hỏng (rất chậm)
--roundtrip 20        # encode lại 20 ảnh và tra ngược index (cần GPU)
```

Đầu ra:

| Đường dẫn | Nội dung |
|---|---|
| `data_processed/_reports/verify.json` | Báo cáo đầy đủ (mọi finding + thống kê từng video) |
| `data_processed/_reports/worklist_<stage>.txt` | Danh sách video cần chạy lại ở stage đó |
| `data_processed/_manifest/videos.json` | Manifest video gốc (ffprobe, kích thước, sha256 nếu có) |

---

## 3. Quy trình khuyến nghị

```bash
# a. Sửa code -> chạy test code
pytest

# b. Chạy pipeline
./data_processing/keyframe_extraction/run_pipeline.sh -a

# c. Xác nhận dữ liệu đủ TRƯỚC khi encode (bước này là cái chặn lỗi lan xuống)
python data_processing/keyframe_extraction/check_integrity.py --stage keyframe || exit 1

# d. Encode + build index
source data_processing/feature_extraction/venv_embedding/bin/activate
python data_processing/feature_extraction/run_embedding.py -a --model siglip2
python data_processing/feature_extraction/build_index.py --model siglip2

# e. Xác nhận lần cuối, gồm cả alignment index <-> .npy
python data_processing/keyframe_extraction/check_integrity.py --roundtrip 20
```

Điểm mấu chốt ở bước **(c)**: đừng sang stage sau khi stage trước chưa PASS. Lỗi ở keyframe mà
cứ encode tiếp thì bạn sẽ mất hàng giờ GPU để tạo ra một index thiếu dữ liệu, và chỉ phát hiện
ra lúc đang thi.

---

## 4. Hai file cũ trong thư mục này

`test_embeddings.py` và `test_keyframes.py` là **script chạy tay trên dữ liệu thật**, không phải
pytest test. Chúng bị loại khỏi `pytest` (khai báo trong `conftest.py`) vì tên khớp mẫu thu thập
nhưng import ở mức module lại làm hỏng cả phiên chạy.

- `test_keyframes.py` — **đã hỏng**: trỏ vào layout cũ `data/transnetv2/` (nay là
  `data_processed/transnetv2/`) và dựa trên luật "1 scene = 1 keyframe", trong khi pipeline hiện
  lấy 1/2/3 keyframe tuỳ độ dài shot. Mọi thứ nó định kiểm nay đã nằm trong `check_integrity.py`
  ở dạng chặt hơn. Xoá được.
- `test_embeddings.py` — chạy bằng `venv_embedding/bin/python tests/test_embeddings.py`,
  kiểm chất lượng truy vấn của model (khác mục đích với bộ test ở trên).

---

## 5. Ghi chú kỹ thuật

**Dataset giả hoạt động thế nào.** `utils/config_loader.py` đọc biến môi trường `AIC_CONFIG`;
nếu có, `{root_dir}` được suy ra từ thư mục chứa file config đó. Bộ test ghi một `config.yaml`
tạm rồi chạy script y hệt lúc chạy thật — chỉ khác gốc dữ liệu. Nhờ vậy test chạy đúng **file
script thật**, không phải một bản mô phỏng.

**Cái gì chưa được test tự động.** `run_embedding.py` và `build_index.py` cần GPU + checkpoint
model nên không nằm trong `pytest`; phần logic tách được của chúng đã đưa sang `feature_store.py`
và có test. Muốn kiểm đầu-cuối thật thì dùng `check_integrity.py --roundtrip N`.
