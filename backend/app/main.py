import logging
from contextlib import asynccontextmanager

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from app.config import get_settings
from app.routers import attempts, auth, puzzles, stats
from app.services.coach import coach

logging.basicConfig(level=logging.INFO)


@asynccontextmanager
async def lifespan(_: FastAPI):
    coach.load()
    yield


app = FastAPI(title="GrandmasterRL Trainer API", version="0.1.0", lifespan=lifespan)
app.add_middleware(
    CORSMiddleware,
    allow_origins=get_settings().cors_origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)
for r in (auth.router, puzzles.router, attempts.router, stats.router):
    app.include_router(r)


@app.get("/api/health")
def health() -> dict:
    return {"status": "ok", "coach": "ppo" if coach.available else "rating_match_fallback", "coach_error": coach.error}
