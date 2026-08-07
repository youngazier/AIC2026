# utils/config_loader.py
import os
import yaml

def load_config():
    """Tự động tìm file config.yaml từ Root và trả về dạng dictionary.

    Biến môi trường AIC_CONFIG cho phép trỏ sang một config.yaml khác; khi đó
    {root_dir} được suy ra từ thư mục chứa file config đó, nên toàn bộ pipeline
    chạy trên một gốc dữ liệu khác mà không phải sửa gì. Bộ test dùng đúng cơ
    chế này để chạy trên dataset giả thay vì dữ liệu thật.
    """
    # Đi tìm vị trí file config.yaml dựa trên vị trí file hiện tại
    current_dir = os.path.dirname(os.path.abspath(__file__))
    root_dir = os.path.abspath(os.path.join(current_dir, "../"))
    config_path = os.path.abspath(os.environ.get("AIC_CONFIG") or os.path.join(root_dir, "config.yaml"))

    with open(config_path, "r", encoding="utf-8") as f:
        config = yaml.safe_load(f)

    # Gốc để giải placeholder = thư mục chứa config.yaml. Với config mặc định
    # thì đúng bằng root_dir như trước, nên không đổi hành vi hiện tại.
    base_root = os.path.dirname(config_path)
    if "paths" in config and "root_dir" in config["paths"]:
        config["paths"]["root_dir"] = base_root

    # Đệ quy thay thế các placeholder và chuẩn hóa đường dẫn tương đối thành tuyệt đối.
    #
    # Việc chuẩn hóa CHỈ áp dụng trong section `paths` (và các khóa tên *_dir/*_path ở
    # nơi khác). Trước đây mọi chuỗi đều bị nối thành đường dẫn, nên một giá trị cấu
    # hình bình thường như backend.gpu_device: "7" sẽ biến thành ".../AIC2026/7" —
    # sai một cách âm thầm. Mọi nơi tiêu thụ path đều đọc qua config["paths"], nên
    # thu hẹp phạm vi như thế này không đổi hành vi của pipeline hiện có.
    def resolve_item(item, parent_key=None, in_paths=False):
        if isinstance(item, dict):
            return {
                k: resolve_item(v, k, in_paths=in_paths or k == "paths")
                for k, v in item.items()
            }
        elif isinstance(item, list):
            return [resolve_item(v, parent_key, in_paths) for v in item]
        elif isinstance(item, str):
            # Thay thế placeholder {root_dir} hoặc {paths.root_dir}
            val = item.replace("{root_dir}", base_root).replace("{paths.root_dir}", base_root)

            # Bỏ qua việc nối path cho các trường tên mô hình hoặc tham số cấu hình
            if parent_key in ["model_name", "pretrained"]:
                return val

            looks_like_path = in_paths or (
                parent_key is not None
                and (parent_key.endswith("_dir") or parent_key.endswith("_path"))
            )

            # Chuẩn hóa đường dẫn tương đối (không bắt đầu bằng /, http, hoặc các ký tự đặc biệt)
            if (looks_like_path and
                not val.startswith("/") and
                not val.startswith("http://") and
                not val.startswith("https://") and
                not val.startswith("{") and
                val != ""):
                val = os.path.abspath(os.path.join(base_root, val))
            return val
        return item

    return resolve_item(config)