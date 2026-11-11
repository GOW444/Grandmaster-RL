from datetime import date, datetime

from sqlalchemy import (
    Boolean,
    Date,
    DateTime,
    Float,
    ForeignKey,
    Index,
    Integer,
    String,
    Text,
    UniqueConstraint,
    func,
)
from sqlalchemy.dialects.postgresql import ARRAY
from sqlalchemy.orm import Mapped, mapped_column, relationship

from app.db import Base

# Same six themes the RL engine was trained on (env/learner_model.py).
THEMES: list[str] = ["fork", "pin", "mate", "endgame", "skewer", "discovery"]
START_RATING = 1000.0


class User(Base):
    __tablename__ = "users"

    id: Mapped[int] = mapped_column(primary_key=True)
    username: Mapped[str] = mapped_column(String(32), unique=True, index=True)
    email: Mapped[str] = mapped_column(String(255), unique=True, index=True)
    password_hash: Mapped[str] = mapped_column(String(255))
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), server_default=func.now())

    rating: Mapped[float] = mapped_column(Float, default=START_RATING)
    xp: Mapped[int] = mapped_column(Integer, default=0)
    current_streak: Mapped[int] = mapped_column(Integer, default=0)
    longest_streak: Mapped[int] = mapped_column(Integer, default=0)
    last_active_on: Mapped[date | None] = mapped_column(Date, nullable=True)

    skills: Mapped[list["ThemeSkill"]] = relationship(back_populates="user", cascade="all, delete-orphan")


class Puzzle(Base):
    """A Lichess puzzle. `moves` is UCI; moves[0] is the opponent's setup move."""

    __tablename__ = "puzzles"

    id: Mapped[str] = mapped_column(String(10), primary_key=True)  # Lichess PuzzleId
    fen: Mapped[str] = mapped_column(Text)
    moves: Mapped[str] = mapped_column(Text)
    rating: Mapped[int] = mapped_column(Integer)
    rating_deviation: Mapped[int] = mapped_column(Integer)
    popularity: Mapped[int] = mapped_column(Integer, default=0)
    nb_plays: Mapped[int] = mapped_column(Integer, default=0)
    themes: Mapped[list[str]] = mapped_column(ARRAY(String))
    primary_theme: Mapped[str] = mapped_column(String(16))
    game_url: Mapped[str | None] = mapped_column(Text, nullable=True)
    opening_tags: Mapped[str | None] = mapped_column(Text, nullable=True)

    __table_args__ = (
        # Powers "nearest puzzle to target rating within a theme" (the KD-tree query, in SQL).
        Index("ix_puzzles_theme_rating", "primary_theme", "rating"),
        Index("ix_puzzles_rating", "rating"),
    )


class ThemeSkill(Base):
    """Per-user, per-theme skill estimate (phi in the engine's learner model)."""

    __tablename__ = "theme_skills"

    user_id: Mapped[int] = mapped_column(ForeignKey("users.id", ondelete="CASCADE"), primary_key=True)
    theme: Mapped[str] = mapped_column(String(16), primary_key=True)
    rating: Mapped[float] = mapped_column(Float, default=START_RATING)
    attempts: Mapped[int] = mapped_column(Integer, default=0)
    solved: Mapped[int] = mapped_column(Integer, default=0)
    total_time_ms: Mapped[int] = mapped_column(Integer, default=0)

    user: Mapped[User] = relationship(back_populates="skills")


class Attempt(Base):
    __tablename__ = "attempts"

    id: Mapped[int] = mapped_column(primary_key=True)
    user_id: Mapped[int] = mapped_column(ForeignKey("users.id", ondelete="CASCADE"), index=True)
    puzzle_id: Mapped[str] = mapped_column(ForeignKey("puzzles.id"), index=True)
    theme: Mapped[str] = mapped_column(String(16))
    mode: Mapped[str] = mapped_column(String(16))  # coach | rated | theme | daily | review | rush
    selector: Mapped[str | None] = mapped_column(String(16), nullable=True)  # ppo | rating_match
    solved: Mapped[bool] = mapped_column(Boolean)
    time_ms: Mapped[int] = mapped_column(Integer)
    hints_used: Mapped[int] = mapped_column(Integer, default=0)
    puzzle_rating: Mapped[int] = mapped_column(Integer)
    theme_rating_before: Mapped[float] = mapped_column(Float)
    theme_rating_after: Mapped[float] = mapped_column(Float)
    rating_before: Mapped[float] = mapped_column(Float)
    rating_after: Mapped[float] = mapped_column(Float)
    xp_gained: Mapped[int] = mapped_column(Integer, default=0)
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), server_default=func.now(), index=True)

    puzzle: Mapped[Puzzle] = relationship()

    __table_args__ = (Index("ix_attempts_user_created", "user_id", "created_at"),)


class SavedPuzzle(Base):
    __tablename__ = "saved_puzzles"

    user_id: Mapped[int] = mapped_column(ForeignKey("users.id", ondelete="CASCADE"), primary_key=True)
    puzzle_id: Mapped[str] = mapped_column(ForeignKey("puzzles.id"), primary_key=True)
    note: Mapped[str | None] = mapped_column(Text, nullable=True)
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), server_default=func.now())

    puzzle: Mapped[Puzzle] = relationship()


class UserBadge(Base):
    __tablename__ = "user_badges"

    id: Mapped[int] = mapped_column(primary_key=True)
    user_id: Mapped[int] = mapped_column(ForeignKey("users.id", ondelete="CASCADE"), index=True)
    code: Mapped[str] = mapped_column(String(32))
    earned_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), server_default=func.now())

    __table_args__ = (UniqueConstraint("user_id", "code"),)
