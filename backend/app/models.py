"""SQLAlchemy models for the local SQLite catalog."""

from sqlalchemy import CheckConstraint, ForeignKeyConstraint, Index, Integer, Text
from sqlalchemy.dialects.sqlite import REAL
from sqlalchemy.orm import DeclarativeBase, Mapped, mapped_column, relationship


class Base(DeclarativeBase):
    """Shared metadata for catalog models and Alembic migrations."""


class Video(Base):
    """One completed Phase 1 result, identified by batch and video name."""

    __tablename__ = "videos"
    __table_args__ = (
        Index("ix_videos_video_name", "video_name"),
        CheckConstraint("width > 0"),
        CheckConstraint("height > 0"),
        CheckConstraint("duration_seconds > 0"),
        CheckConstraint("frame_count > 0"),
        CheckConstraint("shot_count > 0"),
    )

    batch: Mapped[str] = mapped_column(Text, primary_key=True)
    video_name: Mapped[str] = mapped_column(Text, primary_key=True)
    video_path: Mapped[str] = mapped_column(Text, nullable=False, unique=True)
    manifest_path: Mapped[str] = mapped_column(Text, nullable=False)
    manifest_mtime_ns: Mapped[int] = mapped_column(Integer, nullable=False)
    stream_index: Mapped[int] = mapped_column(Integer, nullable=False)
    codec: Mapped[str | None] = mapped_column(Text)
    width: Mapped[int] = mapped_column(Integer, nullable=False)
    height: Mapped[int] = mapped_column(Integer, nullable=False)
    duration_seconds: Mapped[float] = mapped_column(REAL, nullable=False)
    average_fps: Mapped[float | None] = mapped_column(REAL)
    timeline_origin_seconds: Mapped[float] = mapped_column(REAL, nullable=False)
    frame_count: Mapped[int] = mapped_column(Integer, nullable=False)
    shot_count: Mapped[int] = mapped_column(Integer, nullable=False)
    imported_at: Mapped[str] = mapped_column(Text, nullable=False)

    shots: Mapped[list["Shot"]] = relationship(
        back_populates="video",
        cascade="all, delete-orphan",
        passive_deletes=True,
    )


class Shot(Base):
    """One shot within a video, identified by batch, video name, and shot index."""
    __tablename__ = "shots"
    __table_args__ = (
        ForeignKeyConstraint(
            ["batch", "video_name"],
            ["videos.batch", "videos.video_name"],
            name="fk_shots_videos",
            ondelete="CASCADE",
        ),
        CheckConstraint("shot_index >= 0"),
        CheckConstraint("start_frame >= 0"),
        CheckConstraint("end_frame >= start_frame"),
        CheckConstraint("start_timestamp >= 0"),
        CheckConstraint("end_timestamp >= start_timestamp"),
    )

    batch: Mapped[str] = mapped_column(Text, primary_key=True)
    video_name: Mapped[str] = mapped_column(Text, primary_key=True)
    shot_index: Mapped[int] = mapped_column(Integer, primary_key=True)
    start_frame: Mapped[int] = mapped_column(Integer, nullable=False)
    end_frame: Mapped[int] = mapped_column(Integer, nullable=False)
    start_timestamp: Mapped[float] = mapped_column(REAL, nullable=False)
    end_timestamp: Mapped[float] = mapped_column(REAL, nullable=False)

    video: Mapped["Video"] = relationship(back_populates="shots")
    keyframes: Mapped["Keyframe | None"] = relationship(
        back_populates="shot",
        cascade="all, delete-orphan",
        passive_deletes=True,
        single_parent=True,
    )


class Keyframe(Base):
    """One keyframe within a shot, identified by batch, video name, and shot index."""
    __tablename__ = "keyframes"
    __table_args__ = (
        ForeignKeyConstraint(
            ["batch", "video_name", "shot_index"],
            ["shots.batch", "shots.video_name", "shots.shot_index"],
            name="fk_keyframes_shots",
            ondelete="CASCADE",
        ),
        CheckConstraint("shot_index >= 0"),
        CheckConstraint("frame_index >= 0"),
        CheckConstraint("role = 'middle'", name="ck_keyframes_role_middle"),
        CheckConstraint("timestamp >= 0"),
    )

    batch: Mapped[str] = mapped_column(Text, primary_key=True)
    video_name: Mapped[str] = mapped_column(Text, primary_key=True)
    shot_index: Mapped[int] = mapped_column(Integer, primary_key=True)
    role: Mapped[str] = mapped_column(Text, nullable=False, default="middle")
    frame_index: Mapped[int] = mapped_column(Integer, nullable=False)
    timestamp: Mapped[float] = mapped_column(REAL, nullable=False)
    image_path: Mapped[str] = mapped_column(Text, nullable=False, unique=True)

    shot: Mapped["Shot"] = relationship(back_populates="keyframes")
