"""Lớp truy cập SQLite (xem docs/database_architecture.md mục 3.2).

Phần ĐỌC dùng raw SQL thay vì ORM: FTS5 (MATCH/bm25) không biểu diễn được qua ORM,
và các câu hydrate ở đây chạy trên đường găng của mỗi request nên cần nhìn thấy rõ
SQL thật sự được sinh ra.
"""
from sqlalchemy import bindparam, text
from sqlalchemy.orm import Session

# Các cột trả về khi hydrate kết quả tìm kiếm. Gộp sẵn thông tin video để UI dựng
# được card kết quả mà không phải gọi thêm API.
_DETAIL_COLUMNS = """
    k.video_id, k.keyframe_name, k.frame_idx, k.timestamp, k.shot_idx, k.playlist,
    k.object_summary, k.ocr_summary, k.caption, k.asr_text,
    v.title, v.author, v.watch_url, v.thumbnail_url, v.publish_date, v.fps
"""


class SQLiteRepository:
    def __init__(self, db: Session):
        self.db = db

    # --- Hydrate kết quả tìm kiếm ------------------------------------------- #
    def get_details_by_pairs(self, pairs: list) -> dict:
        """[(video_id, keyframe_name)] -> {(video_id, keyframe_name): dict thông tin}.

        Trả về DICT chứ không phải list, vì SQL không đảm bảo thứ tự: người gọi phải
        tự sắp lại theo đúng thứ hạng đã tính. Quên bước đó là mất sạch công xếp hạng.

        Dùng row-values `(a, b) IN (VALUES ...)` (SQLite >= 3.15) — một câu SQL duy
        nhất cho cả trăm cặp, thay vì trăm mệnh đề OR.
        """
        if not pairs:
            return {}

        placeholders = ", ".join(f"(:v{i}, :k{i})" for i in range(len(pairs)))
        params = {}
        for i, (video_id, keyframe_name) in enumerate(pairs):
            params[f"v{i}"] = video_id
            params[f"k{i}"] = keyframe_name

        sql = f"""
            SELECT {_DETAIL_COLUMNS}
            FROM keyframes k
            LEFT JOIN videos v ON v.video_id = k.video_id
            WHERE (k.video_id, k.keyframe_name) IN (VALUES {placeholders})
        """
        rows = self.db.execute(text(sql), params).mappings().all()
        return {(r["video_id"], r["keyframe_name"]): dict(r) for r in rows}

    # --- Registry / thống kê ------------------------------------------------ #
    def list_playlists(self) -> list:
        sql = """
            SELECT playlist, COUNT(*) AS n_videos
            FROM videos WHERE playlist IS NOT NULL AND playlist != ''
            GROUP BY playlist ORDER BY playlist
        """
        return [dict(r) for r in self.db.execute(text(sql)).mappings().all()]

    def stats(self) -> dict:
        row = self.db.execute(
            text(
                "SELECT (SELECT COUNT(*) FROM videos) AS n_videos, "
                "       (SELECT COUNT(*) FROM keyframes) AS n_keyframes"
            )
        ).mappings().one()
        return dict(row)

    def get_video(self, video_id: str):
        row = self.db.execute(
            text("SELECT * FROM videos WHERE video_id = :vid"), {"vid": video_id}
        ).mappings().first()
        return dict(row) if row else None

    # --- Full-text search (Chặng C: chỉ có tác dụng khi các cột text đã có dữ liệu) --- #
    def search_fts(self, match_expr: str, playlist=None, limit: int = 300) -> list:
        """Trả về [(video_id, keyframe_name, bm25_score)], bm25 CÀNG NHỎ CÀNG LIÊN QUAN.

        `match_expr` PHẢI được làm sạch trước khi truyền vào (cú pháp MATCH của FTS5
        ném lỗi với câu văn thô chứa dấu nháy, dấu hai chấm, AND/OR...).
        """
        sql = """
            SELECT video_id, keyframe_name, bm25(keyframes_fts) AS score
            FROM keyframes_fts
            WHERE keyframes_fts MATCH :expr
        """
        params = {"expr": match_expr, "limit": limit}
        if playlist:
            sql += " AND playlist = :playlist"
            params["playlist"] = playlist
        sql += " ORDER BY score LIMIT :limit"

        rows = self.db.execute(text(sql), params).fetchall()
        return [(r[0], r[1], float(r[2])) for r in rows]

    def filter_keyframes_by_objects(self, labels: list, playlist=None) -> list:
        """Lọc cứng: các (video_id, keyframe_name) chứa ĐỦ TẤT CẢ nhãn trong `labels`."""
        if not labels:
            return []
        sql = text(
            """
            SELECT k.video_id, k.keyframe_name
            FROM keyframes k
            JOIN detected_objects o ON o.keyframe_id = k.id
            WHERE o.label IN :labels
              AND (:playlist IS NULL OR k.playlist = :playlist)
            GROUP BY k.id
            HAVING COUNT(DISTINCT o.label) = :n
            """
        ).bindparams(bindparam("labels", expanding=True))
        rows = self.db.execute(
            sql, {"labels": list(labels), "playlist": playlist, "n": len(set(labels))}
        ).fetchall()
        return [(r[0], r[1]) for r in rows]
