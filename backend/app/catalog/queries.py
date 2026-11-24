"""Detached, typed catalog queries with stable pagination."""

from dataclasses import dataclass
from typing import Generic, TypeVar

from sqlalchemy import func, select
from sqlalchemy.orm import Session, sessionmaker

from backend.app.models import Keyframe, Video
from .validation import PreparedKeyframe, PreparedVideo


@dataclass(frozen=True)
class VideoDetail(PreparedVideo):
    imported_at: str


T = TypeVar('T')


@dataclass(frozen=True)
class Page(Generic[T]):
    items: tuple[T, ...]
    total: int
    limit: int
    offset: int


def _pagination(limit: int, offset: int) -> None:
    if type(limit) is not int or not 1 <= limit <= 1000:
        raise ValueError('limit must be an integer between 1 and 1000')
    if type(offset) is not int or offset < 0:
        raise ValueError('offset must be a nonnegative integer')


def _video(row: Video) -> VideoDetail:
    return VideoDetail(**{name: getattr(row, name) for name in VideoDetail.__dataclass_fields__})


class CatalogQueries:
    def __init__(self, sessions: sessionmaker[Session]):
        self.sessions = sessions

    def list_videos(self, *, limit: int = 50, offset: int = 0) -> Page[VideoDetail]:
        _pagination(limit, offset)
        with self.sessions() as session:
            total = session.scalar(select(func.count()).select_from(Video))
            rows = session.scalars(select(Video).order_by(Video.batch, Video.video_name)
                                   .limit(limit).offset(offset))
            return Page(tuple(_video(row) for row in rows), total, limit, offset)

    def get_video(self, batch: str, video_name: str) -> VideoDetail | None:
        with self.sessions() as session:
            row = session.get(Video, (batch, video_name))
            return _video(row) if row is not None else None

    def list_keyframes(self, batch: str, video_name: str, *, limit: int = 50,
                       offset: int = 0) -> Page[PreparedKeyframe] | None:
        _pagination(limit, offset)
        with self.sessions() as session:
            if session.get(Video, (batch, video_name)) is None:
                return None
            condition = (Keyframe.batch == batch, Keyframe.video_name == video_name)
            total = session.scalar(select(func.count()).select_from(Keyframe).where(*condition))
            rows = session.scalars(select(Keyframe).where(*condition).order_by(Keyframe.shot_index)
                                   .limit(limit).offset(offset))
            items = tuple(PreparedKeyframe(**{
                name: getattr(row, name) for name in PreparedKeyframe.__dataclass_fields__
            }) for row in rows)
            return Page(items, total, limit, offset)
