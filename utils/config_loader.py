# utils/config_loader.py
import os
import yaml

def load_config():
    """Tự động tìm file config.yaml từ Root và trả về dạng dictionary."""
    # Đi tìm vị trí file config.yaml dựa trên vị trí file hiện tại
    current_dir = os.path.dirname(os.path.abspath(__file__))
    root_dir = os.path.abspath(os.path.join(current_dir, "../"))
    config_path = os.path.join(root_dir, "config.yaml")
    
    with open(config_path, "r", encoding="utf-8") as f:
        config = yaml.safe_load(f)
        
    base_root = root_dir
    if "paths" in config and "root_dir" in config["paths"]:
        config["paths"]["root_dir"] = root_dir

    # Đệ quy thay thế các placeholder và chuẩn hóa đường dẫn tương đối thành tuyệt đối
    def resolve_item(item, parent_key=None):
        if isinstance(item, dict):
            return {k: resolve_item(v, k) for k, v in item.items()}
        elif isinstance(item, list):
            return [resolve_item(v, parent_key) for v in item]
        elif isinstance(item, str):
            # Thay thế placeholder {root_dir} hoặc {paths.root_dir}
            val = item.replace("{root_dir}", base_root).replace("{paths.root_dir}", base_root)
            
            # Bỏ qua việc nối path cho các trường tên mô hình hoặc tham số cấu hình
            if parent_key in ["model_name", "pretrained"]:
                return val

            # Chuẩn hóa đường dẫn tương đối (không bắt đầu bằng /, http, hoặc các ký tự đặc biệt)
            if (not val.startswith("/") and 
                not val.startswith("http://") and 
                not val.startswith("https://") and 
                not val.startswith("{") and
                val != ""):
                val = os.path.abspath(os.path.join(base_root, val))
            return val
        return item

    return resolve_item(config)