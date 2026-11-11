from datetime import date, datetime

from pydantic import BaseModel, ConfigDict, EmailStr, Field


class RegisterIn(BaseModel):
    username: str = Field(min_length=3, max_length=32, pattern=r"^[A-Za-z0-9_]+$")
    email: EmailStr
    password: str = Field(min_length=8, max_length=128)


class TokenOut(BaseModel):
    access_token: str
    token_type: str = "bearer"


class ThemeSkillOut(BaseModel):
    theme: str
    rating: float
    attempts: int
    solved: int
    avg_time_ms: int | None


class UserOut(BaseModel):
    model_config = ConfigDict(from_attributes=True)

    id: int
    username: str
    email: str
    rating: float
    xp: int
    level: int
    xp_into_level: int
    xp_for_next: int
    current_streak: int
    longest_streak: int
    created_at: datetime


class PuzzleOut(BaseModel):
    model_config = ConfigDict(from_attributes=True)

    id: str
    fen: str
    moves: list[str]  # UCI. moves[0] is the opponent's setup move; the user plays moves[1], [3], ...
    rating: int
    themes: list[str]
    primary_theme: str
    game_url: str | None


class NextPuzzleOut(BaseModel):
    puzzle: PuzzleOut
    mode: str
    selector: str | None = None
    target_rating: float | None = None
    reason: str | None = None
    saved: bool = False


class AttemptIn(BaseModel):
    puzzle_id: str
    solved: bool
    time_ms: int = Field(ge=0, le=60 * 60 * 1000)
    hints_used: int = Field(default=0, ge=0, le=10)
    mode: str = Field(default="rated", pattern=r"^(coach|rated|theme|daily|review|rush)$")
    selector: str | None = Field(default=None, pattern=r"^(ppo|rating_match)$")


class AttemptOut(BaseModel):
    solved: bool
    theme: str
    theme_rating_before: float
    theme_rating_after: float
    rating_before: float
    rating_after: float
    xp_gained: int
    level: int
    new_badges: list[dict]
    streak: int


class AttemptHistoryOut(BaseModel):
    model_config = ConfigDict(from_attributes=True)

    id: int
    puzzle_id: str
    theme: str
    mode: str
    selector: str | None
    solved: bool
    time_ms: int
    puzzle_rating: int
    rating_after: float
    created_at: datetime


class SaveIn(BaseModel):
    note: str | None = Field(default=None, max_length=500)


class SavedOut(BaseModel):
    puzzle: PuzzleOut
    note: str | None
    created_at: datetime


class DailyActivity(BaseModel):
    day: date
    attempts: int
    solved: int
