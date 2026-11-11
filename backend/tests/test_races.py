import time
from datetime import timedelta

from app.db import SessionLocal
from app.models import Puzzle, Race
from app.services.validate import check_solution
from tests.conftest import REAL


def test_check_solution():
    sol = REAL.moves.split()  # f2g3 | e6e7 b2b1 b3c1 b1c1 h6c1
    assert check_solution(REAL.fen, sol, ["e6e7", "b3c1", "h6c1"])
    assert not check_solution(REAL.fen, sol, ["e6e7", "b3c1"])  # incomplete
    assert not check_solution(REAL.fen, sol, ["h6h7"])  # wrong
    assert not check_solution(REAL.fen, sol, ["a1a8"])  # illegal
    # back-rank mate-in-1 where an alternative mate (Re8#) differs from the stored line (Ra8#)
    fen = "6k1/1p3ppp/8/8/8/8/5PPP/R3R1K1 b - - 0 1"
    assert check_solution(fen, ["b7b6", "a1a8"], ["e1e8"])
    assert check_solution(fen, ["b7b6", "a1a8"], ["a1a8"])


def _solve_current(client, auth, race):
    p = race["puzzle"]
    moves = p["moves"][1::2]
    return client.post(f"/api/races/{race['id']}/answer", headers=auth, json={"index": race["me"]["index"], "moves": moves}).json()


def _skip_countdown(race_id):
    with SessionLocal() as db:
        r = db.get(Race, race_id)
        r.started_at = r.started_at - timedelta(seconds=10)
        db.commit()


def test_solo_race_scoring_and_strikes(client, auth):
    race = client.post("/api/races", headers=auth, json={"visibility": "solo", "duration_s": 180, "max_errors": 3}).json()
    assert race["status"] == "running" and "puzzle" not in race  # countdown hides the puzzle
    _skip_countdown(race["id"])
    race = client.get(f"/api/races/{race['id']}", headers=auth).json()
    r = _solve_current(client, auth, race)
    assert r["correct"] and r["me"]["score"] == 1 and r["me"]["index"] == 1
    # replaying an old index is rejected
    assert client.post(f"/api/races/{race['id']}/answer", headers=auth, json={"index": 0, "moves": []}).status_code == 409
    for i in range(3):
        r = client.post(f"/api/races/{race['id']}/answer", headers=auth,
                        json={"index": r["me"]["index"], "moves": ["a1a1"]}).json()
        assert not r["correct"]
    assert r["me"]["finished"] and r["me"]["errors"] == 3 and r["status"] == "finished"
    lb = client.get("/api/races/leaderboard?duration_s=180", headers=auth).json()
    assert any(row["score"] >= 1 for row in lb)


def test_private_race_by_code(client, auth):
    import uuid

    race = client.post("/api/races", headers=auth, json={"visibility": "private"}).json()
    assert race["status"] == "waiting" and race["is_host"]
    name = "f" + uuid.uuid4().hex[:8]
    tok = client.post("/api/auth/register", json={"username": name, "email": f"{name}@x.io", "password": "password123"}).json()["access_token"]
    friend = {"Authorization": f"Bearer {tok}"}
    joined = client.post(f"/api/races/join?code={race['code'].lower()}", headers=friend).json()
    assert len(joined["players"]) == 2 and not joined["is_host"]
    assert client.post(f"/api/races/{race['id']}/start", headers=friend).status_code == 403
    started = client.post(f"/api/races/{race['id']}/start", headers=auth).json()
    assert started["status"] == "running"
    assert client.post(f"/api/races/join?code={race['code']}", headers={"Authorization": "Bearer x"}).status_code == 401


def test_quick_match_pairs_two_users(client, auth):
    import uuid

    a = client.post("/api/races/quick?duration_s=300", headers=auth).json()
    assert a["status"] == "waiting" and a["visibility"] == "public"
    name = "q" + uuid.uuid4().hex[:8]
    tok = client.post("/api/auth/register", json={"username": name, "email": f"{name}@x.io", "password": "password123"}).json()["access_token"]
    b = client.post("/api/races/quick?duration_s=300", headers={"Authorization": f"Bearer {tok}"}).json()
    assert b["id"] == a["id"] and b["status"] == "running" and len(b["players"]) == 2


def test_custom_range_and_browse(client, auth):
    for _ in range(5):
        d = client.get("/api/puzzles/next?mode=custom&min_rating=1200&max_rating=1500&theme=mate", headers=auth).json()
        assert 1200 <= d["puzzle"]["rating"] <= 1500 and d["puzzle"]["primary_theme"] == "mate"
    assert client.get("/api/puzzles/next?mode=custom&min_rating=1500&max_rating=1200", headers=auth).status_code == 400

    page = client.get("/api/puzzles/browse?min_rating=1000&max_rating=1999&theme=pin&page_size=5", headers=auth).json()
    assert page["total"] == 10 and len(page["items"]) == 5
    assert [i["rating"] for i in page["items"]] == sorted(i["rating"] for i in page["items"])
    pid = page["items"][0]["id"]
    client.post("/api/attempts", headers=auth, json={"puzzle_id": pid, "solved": False, "time_ms": 1000, "mode": "custom" if False else "rated"})
    failed = client.get("/api/puzzles/browse?status=failed", headers=auth).json()
    assert [i["id"] for i in failed["items"]] == [pid] and failed["items"][0]["status"] == "failed"
