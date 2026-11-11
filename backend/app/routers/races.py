"""Puzzle Battle: timed, 3-strike puzzle races — solo, private (join by code) or quick match.

Real-time-ish via short polling (GET /races/{id} every ~1s), which works on any host,
including free tiers that don't keep WebSockets alive. All timing and answer checks
happen on the server."""

import random
import secrets
from datetime import datetime, timedelta, timezone

from fastapi import APIRouter, Depends, HTTPException, Query, status
from pydantic import BaseModel, Field
from sqlalchemy import func, select
from sqlalchemy.orm import Session

from app.db import get_db
from app.models import Puzzle, Race, RacePlayer, User
from app.routers.common import puzzle_out
from app.security import get_current_user
from app.services.validate import check_solution

router = APIRouter(prefix="/api/races", tags=["races"])

LADDER = 120  # more than anyone can solve in 5 minutes
COUNTDOWN_S = 4
GRACE_S = 2  # network slack on the server-side deadline
QUICK_WAIT_S = 90  # public lobbies older than this aren't matched


def _now() -> datetime:
    return datetime.now(timezone.utc)


def _ladder(db: Session) -> list[str]:
    """Escalating difficulty: ~600 -> ~2600, random themes. Uses the (rating) index."""
    ids: list[str] = []
    for i in range(LADDER):
        target = 600 + i * 17 + random.randint(-30, 30)
        offset = random.randint(0, 20)
        pid = db.scalar(
            select(Puzzle.id).where(Puzzle.rating >= target).order_by(Puzzle.rating).offset(offset).limit(1)
        )
        if pid and pid not in ids:
            ids.append(pid)
    return ids


def _code(db: Session) -> str:
    alphabet = "ABCDEFGHJKLMNPQRSTUVWXYZ23456789"
    while True:
        c = "".join(secrets.choice(alphabet) for _ in range(6))
        if not db.scalar(select(Race.id).where(Race.code == c)):
            return c


def _deadline(race: Race) -> datetime | None:
    return race.started_at + timedelta(seconds=race.duration_s) if race.started_at else None


def _settle(db: Session, race: Race) -> None:
    """Lazily advance race state: time-outs and all-players-done."""
    if race.status != "running":
        return
    now = _now()
    over = now >= _deadline(race)
    for p in race.players:
        if p.finished_at is None and (over or p.errors >= race.max_errors):
            p.finished_at = min(now, _deadline(race))
    if over or all(p.finished_at for p in race.players):
        race.status = "finished"
    db.commit()


def _start(race: Race) -> None:
    race.status = "running"
    race.started_at = _now() + timedelta(seconds=COUNTDOWN_S)


def _state(db: Session, race: Race, user: User) -> dict:
    _settle(db, race)
    me = next((p for p in race.players if p.user_id == user.id), None)
    players = sorted(race.players, key=lambda p: (-p.score, p.errors, p.joined_at))
    out = {
        "id": race.id,
        "code": race.code,
        "visibility": race.visibility,
        "status": race.status,
        "duration_s": race.duration_s,
        "max_errors": race.max_errors,
        "host": race.players[0].user.username if race.players else None,
        "is_host": race.host_id == user.id,
        "server_now": _now(),
        "started_at": race.started_at,
        "ends_at": _deadline(race),
        "players": [
            {"username": p.user.username, "score": p.score, "errors": p.errors,
             "best_rating": p.best_rating, "finished": p.finished_at is not None, "you": p.user_id == user.id}
            for p in players
        ],
        "me": None,
    }
    if me:
        out["me"] = {"score": me.score, "errors": me.errors, "index": me.current_index,
                     "finished": me.finished_at is not None}
        # Only reveal the current puzzle, and only once the race is live.
        if race.status == "running" and _now() >= race.started_at and me.finished_at is None \
                and me.current_index < len(race.puzzle_ids):
            out["puzzle"] = puzzle_out(db.get(Puzzle, race.puzzle_ids[me.current_index]))
    return out


def _get_race(db: Session, race_id: int, lock: bool = False) -> Race:
    q = select(Race).where(Race.id == race_id)
    if lock:
        q = q.with_for_update()
    race = db.scalar(q)
    if race is None:
        raise HTTPException(status.HTTP_404_NOT_FOUND, "Race not found")
    return race


class CreateIn(BaseModel):
    duration_s: int = Field(180, ge=60, le=600)
    max_errors: int = Field(3, ge=1, le=5)
    visibility: str = Field("private", pattern=r"^(solo|private)$")


class AnswerIn(BaseModel):
    index: int = Field(ge=0)
    moves: list[str] = Field(max_length=40)


@router.post("", status_code=status.HTTP_201_CREATED)
def create_race(body: CreateIn, user: User = Depends(get_current_user), db: Session = Depends(get_db)) -> dict:
    race = Race(code=_code(db), host_id=user.id, visibility=body.visibility, duration_s=body.duration_s,
                max_errors=body.max_errors, puzzle_ids=_ladder(db))
    race.players.append(RacePlayer(user_id=user.id))
    if body.visibility == "solo":
        _start(race)
    db.add(race)
    db.commit()
    return _state(db, race, user)


@router.post("/quick")
def quick_match(duration_s: int = Query(180, ge=60, le=600), user: User = Depends(get_current_user),
                db: Session = Depends(get_db)) -> dict:
    """Join the oldest open public lobby with the same time control, or open one and wait."""
    cutoff = _now() - timedelta(seconds=QUICK_WAIT_S)
    already_in = select(RacePlayer.race_id).where(RacePlayer.user_id == user.id)
    race = db.scalar(
        select(Race).where(Race.visibility == "public", Race.status == "waiting", Race.duration_s == duration_s,
                           Race.created_at >= cutoff, Race.id.not_in(already_in))
        .order_by(Race.created_at).limit(1).with_for_update(skip_locked=True)
    )
    if race:
        race.players.append(RacePlayer(user_id=user.id))
        _start(race)  # two players -> go
    else:
        race = Race(code=_code(db), host_id=user.id, visibility="public", duration_s=duration_s,
                    max_errors=3, puzzle_ids=_ladder(db))
        race.players.append(RacePlayer(user_id=user.id))
        db.add(race)
    db.commit()
    return _state(db, race, user)


@router.post("/join")
def join_by_code(code: str = Query(min_length=4, max_length=8), user: User = Depends(get_current_user),
                 db: Session = Depends(get_db)) -> dict:
    race = db.scalar(select(Race).where(Race.code == code.strip().upper()).with_for_update())
    if race is None:
        raise HTTPException(status.HTTP_404_NOT_FOUND, "No race with that code")
    if not any(p.user_id == user.id for p in race.players):
        if race.status != "waiting":
            raise HTTPException(status.HTTP_409_CONFLICT, "That race has already started")
        if len(race.players) >= 8:
            raise HTTPException(status.HTTP_409_CONFLICT, "Race is full")
        race.players.append(RacePlayer(user_id=user.id))
        db.commit()
    return _state(db, race, user)


@router.get("/leaderboard")
def rush_leaderboard(duration_s: int = Query(180), db: Session = Depends(get_db),
                     _: User = Depends(get_current_user)) -> list[dict]:
    best = (
        select(RacePlayer.user_id, func.max(RacePlayer.score).label("best"))
        .join(Race).where(Race.duration_s == duration_s, Race.status == "finished")
        .group_by(RacePlayer.user_id).subquery()
    )
    rows = db.execute(
        select(User.username, best.c.best).join(best, best.c.user_id == User.id).order_by(best.c.best.desc()).limit(25)
    ).all()
    return [{"username": u, "score": s} for u, s in rows]


@router.get("/{race_id}")
def get_race(race_id: int, user: User = Depends(get_current_user), db: Session = Depends(get_db)) -> dict:
    return _state(db, _get_race(db, race_id), user)


@router.post("/{race_id}/start")
def start_race(race_id: int, user: User = Depends(get_current_user), db: Session = Depends(get_db)) -> dict:
    race = _get_race(db, race_id, lock=True)
    if race.host_id != user.id:
        raise HTTPException(status.HTTP_403_FORBIDDEN, "Only the host can start")
    if race.status == "waiting":
        _start(race)
        db.commit()
    return _state(db, race, user)


@router.post("/{race_id}/leave", status_code=status.HTTP_204_NO_CONTENT)
def leave_race(race_id: int, user: User = Depends(get_current_user), db: Session = Depends(get_db)) -> None:
    race = _get_race(db, race_id, lock=True)
    me = next((p for p in race.players if p.user_id == user.id), None)
    if me is None:
        return
    if race.status == "waiting":
        race.players.remove(me)
        if not race.players:
            db.delete(race)
        elif race.host_id == user.id:
            race.host_id = race.players[0].user_id
    elif me.finished_at is None:
        me.finished_at = _now()  # resign
    db.commit()


@router.post("/{race_id}/answer")
def answer(race_id: int, body: AnswerIn, user: User = Depends(get_current_user), db: Session = Depends(get_db)) -> dict:
    race = _get_race(db, race_id)
    _settle(db, race)
    me = db.scalar(
        select(RacePlayer).where(RacePlayer.race_id == race.id, RacePlayer.user_id == user.id).with_for_update()
    )
    if me is None:
        raise HTTPException(status.HTTP_403_FORBIDDEN, "You're not in this race")
    now = _now()
    if race.status != "running" or me.finished_at or now < race.started_at \
            or now > _deadline(race) + timedelta(seconds=GRACE_S):
        raise HTTPException(status.HTTP_409_CONFLICT, "Race isn't accepting answers")
    if body.index != me.current_index:
        raise HTTPException(status.HTTP_409_CONFLICT, "Stale puzzle index")

    puzzle = db.get(Puzzle, race.puzzle_ids[me.current_index])
    correct = check_solution(puzzle.fen, puzzle.moves.split(), body.moves)
    if correct:
        me.score += 1
        me.best_rating = max(me.best_rating, puzzle.rating)
    else:
        me.errors += 1
    me.current_index += 1
    if me.errors >= race.max_errors or me.current_index >= len(race.puzzle_ids):
        me.finished_at = now
    db.commit()
    db.refresh(race)
    return {"correct": correct, "solution": puzzle.moves.split(), **_state(db, race, user)}
