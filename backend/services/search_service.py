"""Tầng nghiệp vụ: điều phối FAISS + SQLite để trả về kết quả tìm kiếm.

Đây là nơi DUY NHẤT chứa logic xếp hạng. Route chỉ nhận request và ủy quyền xuống
đây; repository chỉ biết đọc dữ liệu, không biết gì về thứ hạng.

Hai luồng hiện có:

    KIS      text  -> encode (1..n model) -> FAISS mỗi model -> [RRF nếu >1] -> hydrate
    SIMILAR  frame -> lấy lại vector đã lưu trong index -> FAISS -> hydrate

Vì sao tìm-ảnh-giống-ảnh KHÔNG encode lại ảnh: vector của mọi keyframe đã nằm sẵn
trong IndexFlatL2, `reconstruct()` lấy ra tức thì và khớp tuyệt đối với index. Nạp
JPEG rồi đẩy qua GPU chỉ để lấy lại đúng con số đó là lãng phí thuần túy.

CHẶNG C sẽ chèn thêm kênh chữ (caption/OCR/ASR qua FTS5) vào đúng chỗ RRF đang trộn
các kênh visual — cấu trúc `_fuse_rrf` dưới đây nhận danh sách kênh bất kỳ nên không
phải viết lại gì, chỉ thêm một kênh nữa vào dict truyền vào.
"""
import logging
import time

logger = logging.getLogger(__name__)


class SearchRequestError(ValueError):
    """Request sai (model chưa nạp, playlist không tồn tại...) -> HTTP 400.

    Cố ý KHÔNG dùng HTTPException ở tầng service: giữ service độc lập với FastAPI để
    unit test không phải dựng app. Route là nơi duy nhất dịch lỗi này sang HTTP.
    """


def _name_of(model) -> str:
    """Tên chuỗi của một ModelName.

    `.value` chứ không phải `str()`: Python 3.10 chưa có StrEnum, nên `str()` của một
    thành viên (str, Enum) trả về "ModelName.SIGLIP2" thay vì "siglip2".
    """
    return model.value if hasattr(model, "value") else str(model)


class SearchService:
    def __init__(self, faiss_repos: dict, encoders: dict, sqlite_repo, settings,
                 indexed_playlists=None, playlist_videos=None):
        self.faiss_repos = faiss_repos
        self.encoders = encoders
        self.sqlite_repo = sqlite_repo
        self.settings = settings
        self.indexed_playlists = indexed_playlists or set()
        self.playlist_videos = playlist_videos or {}

    # --- Kiểm tra đầu vào --------------------------------------------------- #
    def _resolve_models(self, models: list, require_text: bool = True) -> list:
        """Đổi danh sách ModelName sang tên chuỗi, sau khi chắc chắn dùng được.

        `require_text=False` cho luồng tìm-ảnh-giống-ảnh: DINOv3 không có text encoder
        nhưng vẫn là model TỐT NHẤT cho việc so ảnh với ảnh.
        """
        if not models:
            raise SearchRequestError("Phải chọn ít nhất một model.")

        names = []
        for model in models:
            name = _name_of(model)
            if name in names:
                continue
            if name not in self.faiss_repos:
                raise SearchRequestError(
                    f"Model '{name}' chưa được nạp. Đang có: {sorted(self.faiss_repos)}."
                )
            if require_text and name not in self.encoders:
                hint = (
                    " DINOv3 là model thị giác thuần, không encode được text — chỉ dùng cho "
                    "tìm ảnh-giống-ảnh." if name == "dinov3" else ""
                )
                raise SearchRequestError(
                    f"Model '{name}' không có text encoder nên không tìm bằng text được."
                    f"{hint} Model tìm được bằng text: {sorted(self.encoders)}."
                )
            names.append(name)
        return names

    def _resolve_playlist(self, playlist, models: list):
        if not playlist:
            return None
        if playlist not in self.indexed_playlists:
            raise SearchRequestError(
                f"Playlist '{playlist}' không có index FAISS. Hợp lệ: {sorted(self.indexed_playlists)}."
            )
        for name in models:
            if not self.faiss_repos[name].has_scope(playlist):
                raise SearchRequestError(
                    f"Model '{name}' chưa build index cho playlist '{playlist}'."
                )
        return playlist

    # --- Hợp nhất nhiều kênh ------------------------------------------------ #
    @staticmethod
    def _fuse_rrf(hits_by_channel: dict, k_rrf: int) -> list:
        """Reciprocal Rank Fusion: [(video, kf, distance, score, [kênh])] đã xếp hạng.

        RRF cộng 1/(k + thứ_hạng) của từng kênh thay vì cộng điểm thô, vì điểm của hai
        model KHÔNG cùng thang đo — cosine của SigLIP2 và của BEiT-3 không so sánh trực
        tiếp được, nhưng "đứng thứ 3" thì kênh nào cũng như kênh nào. Frame được nhiều
        model cùng bầu sẽ nổi lên trên, đó chính là điều ta muốn ở chế độ hybrid.
        """
        fused = {}
        for channel, hits in hits_by_channel.items():
            for rank, (video_id, keyframe_name, distance) in enumerate(hits, start=1):
                pair = (video_id, keyframe_name)
                entry = fused.get(pair)
                if entry is None:
                    entry = fused[pair] = {"score": 0.0, "distance": distance, "channels": []}
                entry["score"] += 1.0 / (k_rrf + rank)
                # Giữ khoảng cách TỐT NHẤT (nhỏ nhất) trong các kênh, chỉ để hiển thị.
                entry["distance"] = min(entry["distance"], distance)
                entry["channels"].append(channel)

        return sorted(
            (
                (pair[0], pair[1], e["distance"], e["score"], e["channels"])
                for pair, e in fused.items()
            ),
            key=lambda row: row[3],
            reverse=True,
        )

    # --- Hydrate ------------------------------------------------------------ #
    def _hydrate(self, ranked: list) -> list:
        """[(video, kf, distance, score, channels)] -> list dict đúng schema KeyframeResult.

        Thứ tự của `ranked` LÀ thứ hạng cuối cùng; SQL không đảm bảo thứ tự nên phải
        lắp lại theo đúng danh sách này, quên là mất sạch công xếp hạng.
        """
        details = self.sqlite_repo.get_details_by_pairs([(v, k) for v, k, *_ in ranked])

        results, missing = [], []
        for rank, (video_id, keyframe_name, distance, score, channels) in enumerate(ranked, 1):
            detail = details.get((video_id, keyframe_name))
            if detail is None:
                # FAISS biết frame này nhưng SQLite thì không -> index và DB lệch pha.
                missing.append(f"{video_id}/{keyframe_name}")
                continue
            results.append(
                {
                    "rank": rank,
                    "video_id": video_id,
                    "keyframe_name": keyframe_name,
                    "frame_idx": detail["frame_idx"],
                    "timestamp": detail["timestamp"],
                    "shot_idx": detail["shot_idx"],
                    "playlist": detail["playlist"],
                    "score": score,
                    "distance": distance,
                    "matched_channels": channels,
                    "image_url": f"/api/v1/keyframes/{video_id}/{keyframe_name}.jpg",
                    "title": detail["title"],
                    "author": detail["author"],
                    "watch_url": detail["watch_url"],
                    "thumbnail_url": detail["thumbnail_url"],
                    "publish_date": detail["publish_date"],
                }
            )

        if missing:
            logger.warning(
                "%d frame có trong FAISS nhưng không có trong SQLite (VD %s). "
                "Nhiều khả năng index và metadata.db được build từ 2 lần chạy pipeline khác nhau "
                "— chạy lại scripts/ingest_db.py --force.",
                len(missing), missing[:3],
            )
        return results

    # --- Luồng KIS ---------------------------------------------------------- #
    def execute_kis(self, text_query: str, models: list, playlist=None,
                    topk: int = 100, exclude_videos=None) -> dict:
        names = self._resolve_models(models, require_text=True)
        scope = self._resolve_playlist(playlist, names)
        excluded = set(exclude_videos or [])
        timings = {}

        # Lấy dư rồi mới cắt: frame bị exclude ăn mất chỗ trong danh sách, và RRF cần
        # danh sách dài hơn topk để hai model có không gian đồng thuận với nhau.
        overfetch = self.settings.overfetch if (excluded or len(names) > 1) else 1
        fetch_k = topk * overfetch

        hits_by_channel = {}
        encode_ms = faiss_ms = 0.0
        for name in names:
            started = time.perf_counter()
            query_vector = self.encoders[name].encode_text(text_query)
            encode_ms += (time.perf_counter() - started) * 1000
            if query_vector is None:
                raise SearchRequestError(f"Encoder '{name}' không encode được câu truy vấn này.")

            started = time.perf_counter()
            hits = self.faiss_repos[name].search(query_vector, top_k=fetch_k, playlist=scope)
            faiss_ms += (time.perf_counter() - started) * 1000
            hits_by_channel[f"visual_{name}"] = [h for h in hits if h[0] not in excluded]

        timings["encode"] = encode_ms
        timings["faiss"] = faiss_ms

        if len(names) == 1:
            channel = f"visual_{names[0]}"
            # Vector đã chuẩn hóa nên L2 bình phương quy đổi thẳng sang cosine. Đưa ra
            # thang "càng lớn càng tốt" để UI không phải nhớ chiều nào là tốt.
            ranked = [
                (v, k, d, 1.0 - d / 2.0, [channel]) for v, k, d in hits_by_channel[channel]
            ][:topk]
        else:
            started = time.perf_counter()
            ranked = self._fuse_rrf(hits_by_channel, self.settings.k_rrf)[:topk]
            timings["fuse"] = (time.perf_counter() - started) * 1000

        started = time.perf_counter()
        results = self._hydrate(ranked)
        timings["hydrate"] = (time.perf_counter() - started) * 1000

        return {
            "results": results,
            "models_used": names,
            "timings_ms": {k: round(v, 2) for k, v in timings.items()},
        }

    # --- Luồng tìm ảnh giống ảnh -------------------------------------------- #
    def execute_similar(self, video_id: str, keyframe_name: str, models: list,
                        playlist=None, topk: int = 100, exclude_videos=None,
                        exclude_self: bool = True) -> dict:
        """Tìm keyframe giống một keyframe cho trước (mặc định dùng DINOv3).

        Không có bước encode: vector nguồn lấy thẳng từ index, xem `_fuse_rrf` docstring
        ở trên để biết vì sao nhiều model vẫn trộn được bằng RRF.
        """
        names = self._resolve_models(models, require_text=False)
        scope = self._resolve_playlist(playlist, names)
        excluded = set(exclude_videos or [])
        timings = {}

        # +1 để bù cho chính frame nguồn, luôn đứng đầu với distance = 0.
        overfetch = self.settings.overfetch if excluded else 1
        fetch_k = topk * overfetch + 1

        hits_by_channel = {}
        lookup_ms = faiss_ms = 0.0
        for name in names:
            started = time.perf_counter()
            # Vector luôn lấy từ index GLOBAL: nó chứa mọi keyframe, còn index playlist
            # chỉ chứa frame của playlist đó nên frame nguồn có thể không nằm trong.
            query_vector = self.faiss_repos[name].get_vector(video_id, keyframe_name)
            lookup_ms += (time.perf_counter() - started) * 1000
            if query_vector is None:
                raise SearchRequestError(
                    f"Không tìm thấy '{video_id}/{keyframe_name}' trong index của model "
                    f"'{name}'. Kiểm tra lại tên video và tên keyframe."
                )

            started = time.perf_counter()
            hits = self.faiss_repos[name].search(query_vector, top_k=fetch_k, playlist=scope)
            faiss_ms += (time.perf_counter() - started) * 1000
            hits_by_channel[f"visual_{name}"] = [
                h for h in hits
                if h[0] not in excluded
                and not (exclude_self and h[0] == video_id and h[1] == keyframe_name)
            ]

        timings["lookup"] = lookup_ms
        timings["faiss"] = faiss_ms

        if len(names) == 1:
            channel = f"visual_{names[0]}"
            ranked = [
                (v, k, d, 1.0 - d / 2.0, [channel]) for v, k, d in hits_by_channel[channel]
            ][:topk]
        else:
            started = time.perf_counter()
            ranked = self._fuse_rrf(hits_by_channel, self.settings.k_rrf)[:topk]
            timings["fuse"] = (time.perf_counter() - started) * 1000

        started = time.perf_counter()
        results = self._hydrate(ranked)
        timings["hydrate"] = (time.perf_counter() - started) * 1000

        return {
            "results": results,
            "models_used": names,
            "timings_ms": {k: round(v, 2) for k, v in timings.items()},
        }
