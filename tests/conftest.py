"""Hạ tầng dùng chung cho bộ test kiểm thử pipeline tiền xử lý.

Ý tưởng: dựng một DATASET GIẢ nhưng THẬT — video .mp4 do ffmpeg sinh ra, ảnh
.jpg mở được, .npy đúng định dạng — rồi tiêm lỗi vào bản sao của nó và kiểm tra
xem check_integrity.py có bắt được không.

Không bao giờ đụng vào data/ hay data_processed/ thật. Toàn bộ dataset giả nằm
trong thư mục tạm của pytest và bị xoá sau khi chạy xong. Cơ chế: biến môi
trường AIC_CONFIG trỏ load_config() sang một config.yaml khác (xem
utils/config_loader.py), nên script chạy y hệt lúc chạy thật, chỉ khác gốc dữ liệu.
"""
import json
import os
import shutil
import subprocess
import sys
from pathlib import Path

import numpy as np
import pytest
from PIL import Image

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from data_processing.keyframe_extraction.shot_sampling import expected_keyframes  # noqa: E402

CHECK_SCRIPT = PROJECT_ROOT / "data_processing" / "keyframe_extraction" / "check_integrity.py"
KEYFRAME_SCRIPT = PROJECT_ROOT / "data_processing" / "keyframe_extraction" / "keyframe_extraction.py"
VENV_PY = PROJECT_ROOT / "data_processing" / "feature_extraction" / "venv_embedding" / "bin" / "python"

MODEL = "testmodel"
DIM = 8

# Hai file dưới đây là SCRIPT chạy tay trên dữ liệu thật (test_embeddings.py cần
# faiss/torch của venv_embedding; test_keyframes.py trỏ vào layout data/ cũ và
# dựa trên luật "1 scene = 1 keyframe" đã lỗi thời). Tên chúng khớp mẫu thu thập
# của pytest nên sẽ làm hỏng cả phiên chạy — loại ra ở đây thay vì sửa/xoá file.
collect_ignore = ["test_embeddings.py", "test_keyframes.py"]

# Hai video đủ để chạm cả ba nhánh của luật lấy mẫu (shot ngắn/vừa/dài).
#   L01_V001: shot dài 249 frame -> 3 keyframe, shot ngắn 49 frame -> 1 keyframe
#   L01_V002: hai shot 99 frame  -> 2 keyframe mỗi shot
VIDEOS = {
    "L01_V001": {"n_frames": 300, "scenes": "0 249\n250 299\n"},
    "L01_V002": {"n_frames": 200, "scenes": "0 99\n100 199\n"},
}
FPS = 25

CONFIG_TEMPLATE = """\
paths:
  root_dir: "{root_dir}"
  data_dir: "{root_dir}/data"
  videos_dir: "{root_dir}/data/videos"
  processed_dir: "{root_dir}/data_processed"
  transnetv2_dir: "{root_dir}/data_processed/transnetv2"
  scenes_dir: "{root_dir}/data_processed/transnetv2/scenes"
  predictions_dir: "{root_dir}/data_processed/transnetv2/predictions"
  keyframes_dir: "{root_dir}/data_processed/keyframes_transnetv2"
  map_keyframes_dir: "{root_dir}/data_processed/map_keyframes"
  features_dir: "{root_dir}/data_processed/features"
  index_dir: "{root_dir}/data_processed/index"
models: {{}}
urls: {{}}
"""

# Dựng FAISS index bằng python của venv_embedding (faiss không có ở python gốc).
# Thứ tự gom vector phải khớp build_index.py: duyệt video đã sắp xếp, trong mỗi
# video duyệt keyframe đã sắp xếp.
BUILD_INDEX_SNIPPET = """
import os, sys
import numpy as np, faiss
root, model = sys.argv[1], sys.argv[2]
feat = os.path.join(root, "data_processed", "features", model)
kfroot = os.path.join(root, "data_processed", "keyframes_transnetv2")
vecs, info = [], []
for v in sorted(x[:-4] for x in os.listdir(feat) if x.endswith(".npy")):
    a = np.load(os.path.join(feat, v + ".npy"))
    d = os.path.join(kfroot, "Keyframes_" + v.split("_")[0], v)
    names = sorted(n[:-4] for n in os.listdir(d) if n.endswith(".jpg"))
    for i, n in enumerate(names):
        vecs.append(a[i]); info.append((v, n))
arr = np.array(vecs, dtype=np.float32)
idx = faiss.IndexFlatL2(arr.shape[1]); idx.add(arr)
out = os.path.join(root, "data_processed", "index"); os.makedirs(out, exist_ok=True)
faiss.write_index(idx, os.path.join(out, model + "_keyframe_embedding.index"))
np.save(os.path.join(out, model + "_keyframe_metadata.npy"), np.array(info))
"""


def jpg_path(root: Path, vid: str, n: int) -> Path:
    return keyframe_dir(root, vid) / f"{n:04d}.jpg"


def keyframe_dir(root: Path, vid: str) -> Path:
    return root / "data_processed" / "keyframes_transnetv2" / f"Keyframes_{vid.split('_')[0]}" / vid


def scenes_path(root: Path, vid: str) -> Path:
    return root / "data_processed" / "transnetv2" / "scenes" / f"{vid}.scenes.txt"


def map_path(root: Path, vid: str) -> Path:
    return root / "data_processed" / "map_keyframes" / f"{vid}.csv"


def npy_path(root: Path, vid: str, model: str = MODEL) -> Path:
    return root / "data_processed" / "features" / model / f"{vid}.npy"


def video_path(root: Path, vid: str) -> Path:
    return root / "data" / "videos" / f"Videos_{vid.split('_')[0]}" / f"{vid}.mp4"


def write_jpg(path: Path, seed: int, blank: bool = False):
    """Ảnh nhiễu ngẫu nhiên (>1KB, độ lệch chuẩn cao) hoặc ảnh đen tuyền."""
    rng = np.random.default_rng(seed)
    data = (np.zeros((96, 128, 3), dtype="uint8") if blank
            else rng.integers(0, 256, (96, 128, 3)).astype("uint8"))
    path.parent.mkdir(parents=True, exist_ok=True)
    Image.fromarray(data).save(path, quality=90)


def build_dataset(root: Path, *, with_features=True, with_sentinel=True):
    """Dựng dataset giả nhất quán hoàn toàn: chạy check_integrity phải PASS."""
    (root / "config.yaml").write_text(CONFIG_TEMPLATE.format(root_dir=root), encoding="utf-8")

    for vid, spec in VIDEOS.items():
        # --- video gốc: dùng ffmpeg để ffprobe đọc được fps/nb_frames thật ---
        vp = video_path(root, vid)
        vp.parent.mkdir(parents=True, exist_ok=True)
        subprocess.run(
            ["ffmpeg", "-v", "error", "-f", "lavfi",
             "-i", f"testsrc=duration={spec['n_frames'] / FPS}:size=128x96:rate={FPS}",
             "-pix_fmt", "yuv420p", "-y", str(vp)],
            check=True,
        )

        # --- đầu ra TransNetV2 ---
        sp = scenes_path(root, vid)
        sp.parent.mkdir(parents=True, exist_ok=True)
        sp.write_text(spec["scenes"], encoding="utf-8")
        pp = root / "data_processed" / "transnetv2" / "predictions" / f"{vid}.predictions.txt"
        pp.parent.mkdir(parents=True, exist_ok=True)
        pp.write_text("0.0\n" * spec["n_frames"], encoding="utf-8")

        # --- keyframe + map csv, sinh từ ĐÚNG luật mà pipeline dùng ---
        expected, shots, _ = expected_keyframes(spec["scenes"].splitlines(keepends=True))
        mp = map_path(root, vid)
        mp.parent.mkdir(parents=True, exist_ok=True)
        rows = ["n,shot_idx,pts_time,fps,frame_idx"]
        for n, shot_idx, frame_idx in expected:
            rows.append(f"{n},{shot_idx},{round(frame_idx / FPS, 4)},{float(FPS)},{frame_idx}")
            write_jpg(jpg_path(root, vid, n), seed=hash((vid, n)) % (2**31))
        mp.write_text("\n".join(rows) + "\n", encoding="utf-8")

        if with_sentinel:
            seal(root, vid)

        # --- đặc trưng ---
        if with_features:
            rng = np.random.default_rng(abs(hash(vid)) % (2**31))
            arr = rng.standard_normal((len(expected), DIM)).astype("float32")
            np_p = npy_path(root, vid)
            np_p.parent.mkdir(parents=True, exist_ok=True)
            np.save(np_p, arr)

    return root


def seal(root: Path, vid: str, **overrides):
    """Ghi _SUCCESS giống hệt keyframe_extraction.py."""
    import hashlib

    raw = scenes_path(root, vid).read_bytes()
    expected, shots, _ = expected_keyframes(raw.decode().splitlines(keepends=True))
    payload = {
        "n_keyframes": len(expected),
        "n_shots": len(shots),
        "scenes_sha256": hashlib.sha256(raw).hexdigest(),
        "code_version": "test",
        "created_at": "2026-01-01T00:00:00",
    }
    payload.update(overrides)
    (keyframe_dir(root, vid) / "_SUCCESS").write_text(json.dumps(payload), encoding="utf-8")


def build_index(root: Path) -> bool:
    """Dựng FAISS index cho dataset giả. False nếu không có venv_embedding."""
    if not VENV_PY.exists():
        return False
    proc = subprocess.run([str(VENV_PY), "-c", BUILD_INDEX_SNIPPET, str(root), MODEL],
                          capture_output=True, text=True)
    if proc.returncode != 0:
        pytest.fail(f"Không dựng được index giả:\n{proc.stdout}\n{proc.stderr}")
    return True


def run_verify(root: Path, *extra, python=None):
    """Chạy check_integrity.py trên dataset giả -> (exit_code, report, output)."""
    env = dict(os.environ, AIC_CONFIG=str(root / "config.yaml"))
    cmd = [str(python or sys.executable), str(CHECK_SCRIPT), "--quiet", *extra]
    proc = subprocess.run(cmd, capture_output=True, text=True, env=env, cwd=str(PROJECT_ROOT))
    report_file = root / "data_processed" / "_reports" / "verify.json"
    if not report_file.exists():
        pytest.fail(f"check_integrity.py không sinh ra báo cáo:\n{proc.stdout}\n{proc.stderr}")
    report = json.loads(report_file.read_text(encoding="utf-8"))
    return proc.returncode, report, proc.stdout + proc.stderr


def codes(report, severity=None):
    """Tập mã lỗi trong báo cáo, lọc theo mức độ nếu cần."""
    return {f["code"] for f in report["findings"]
            if severity is None or f["severity"] == severity}


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------
@pytest.fixture(scope="session")
def dataset_template(tmp_path_factory):
    """Dựng dataset sạch MỘT lần cho cả phiên (ffmpeg khá chậm)."""
    return build_dataset(tmp_path_factory.mktemp("template"))


@pytest.fixture
def dataset(dataset_template, tmp_path):
    """Bản sao riêng cho từng test, để lỗi tiêm vào không lây sang test khác."""
    root = tmp_path / "ds"
    shutil.copytree(dataset_template, root)
    (root / "config.yaml").write_text(CONFIG_TEMPLATE.format(root_dir=root), encoding="utf-8")
    return root
