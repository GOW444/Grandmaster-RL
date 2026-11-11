from functools import lru_cache
from pathlib import Path

from pydantic import field_validator
from pydantic_settings import BaseSettings, SettingsConfigDict

BACKEND_DIR = Path(__file__).resolve().parents[1]
REPO_ROOT = BACKEND_DIR.parent


class Settings(BaseSettings):
    model_config = SettingsConfigDict(env_file=BACKEND_DIR / ".env", extra="ignore")

    database_url: str = "postgresql+psycopg://grandmaster:grandmaster@localhost:5432/grandmaster"
    jwt_secret: str = "dev-only-insecure-secret-change-me-in-prod-0123456789"
    jwt_algorithm: str = "HS256"
    jwt_expire_minutes: int = 60 * 24 * 7
    cors_origins: list[str] = ["http://localhost:5173"]

    # RL engine artifacts (from the `engine` branch, used read-only)
    policy_npz: Path = BACKEND_DIR / "app" / "assets" / "ppo_policy.npz"  # torch-free export (preferred)
    ppo_checkpoint: Path = REPO_ROOT / "checkpoints" / "ppo" / "ppo_final.zip"
    ppo_vecnormalize: Path = REPO_ROOT / "checkpoints" / "ppo" / "vec_normalize.pkl"
    coach_use_vecnormalize: bool = False  # matches how the reported results were evaluated

    @field_validator("database_url")
    @classmethod
    def _psycopg_driver(cls, v: str) -> str:
        # Hosted Postgres (Neon, Render, Supabase) hands out postgres:// or postgresql:// URLs.
        for prefix in ("postgres://", "postgresql://"):
            if v.startswith(prefix):
                return "postgresql+psycopg://" + v[len(prefix):]
        return v


@lru_cache
def get_settings() -> Settings:
    return Settings()
