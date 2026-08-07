"""Dependency Injection cho tầng API.

Tài nguyên nặng (FAISS index, text encoder) nằm ở app.state do lifespan nạp một lần;
các hàm dưới đây chỉ lấy chúng ra và ghép với Session của request hiện tại. Nhờ vậy
unit test có thể tiêm repository giả vào SearchService mà không cần nạp index thật.
"""
from fastapi import Depends, Request
from sqlalchemy.orm import Session

from backend.core.database import get_db
from backend.repositories.sqlite_repo import SQLiteRepository
from backend.services.search_service import SearchService


def get_sqlite_repo(db: Session = Depends(get_db)) -> SQLiteRepository:
    return SQLiteRepository(db)


def get_search_service(
    request: Request,
    sqlite_repo: SQLiteRepository = Depends(get_sqlite_repo),
) -> SearchService:
    state = request.app.state
    return SearchService(
        faiss_repos=state.faiss_repos,
        encoders=state.encoders,
        sqlite_repo=sqlite_repo,
        settings=state.settings,
        indexed_playlists=state.indexed_playlists,
        playlist_videos=state.playlist_videos,
    )
