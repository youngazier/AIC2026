"""Package trích xuất đặc trưng.

Các lớp model được nạp LƯỜI (PEP 562). Trước đây `__init__.py` import sẵn
SigLIP2/BEiT-3 ở mức module, nên chỉ cần chạm vào bất kỳ file nào trong package
là kéo theo open_clip + torch. Hệ quả: những module thuần numpy như
`feature_store.py` không import được ngoài venv_embedding, tức là không kiểm thử
được. Tên xuất ra giữ nguyên như cũ, chỉ khác là chi phí nạp dồn tới lúc dùng thật.
"""
_LAZY = {
    "SigLIP2Embedding": ".siglip2_embedding",
    "Beit3Embedding": ".beit3_embedding",
    "DinoV3Embedding": ".dinov3_embedding",
}

__all__ = list(_LAZY)


def __getattr__(name):
    if name in _LAZY:
        from importlib import import_module
        return getattr(import_module(_LAZY[name], __name__), name)
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")


def __dir__():
    return sorted(list(globals()) + __all__)
