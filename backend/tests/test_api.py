def test_register_login_me(client):
    body = {"username": "alice_t", "email": "alice_t@example.com", "password": "password123"}
    assert client.post("/api/auth/register", json=body).status_code == 201
    assert client.post("/api/auth/register", json=body).status_code == 409
    assert client.post("/api/auth/login", data={"username": "alice_t", "password": "nope-nope"}).status_code == 401
    tok = client.post("/api/auth/login", data={"username": "ALICE_T", "password": "password123"}).json()["access_token"]
    me = client.get("/api/auth/me", headers={"Authorization": f"Bearer {tok}"}).json()
    assert me["username"] == "alice_t" and me["rating"] == 1000 and me["level"] == 1


def test_requires_auth(client):
    assert client.get("/api/stats/overview").status_code == 401
    assert client.get("/api/stats/overview", headers={"Authorization": "Bearer junk"}).status_code == 401


def test_theme_drill_picks_nearest_unseen(client, auth):
    seen = set()
    for _ in range(3):
        p = client.get("/api/puzzles/next?mode=theme&theme=pin", headers=auth).json()["puzzle"]
        assert p["primary_theme"] == "pin" and abs(p["rating"] - 1000) <= 400
        assert p["id"] not in seen
        seen.add(p["id"])
        client.post("/api/attempts", headers=auth, json={"puzzle_id": p["id"], "solved": True, "time_ms": 5000, "mode": "theme"})


def test_coach_ppo_and_ab(client, auth):
    assert client.get("/api/health").json()["coach"] == "ppo"
    d = client.get("/api/puzzles/next?mode=coach&selector=ppo", headers=auth).json()
    assert d["selector"] == "ppo" and 400 <= d["target_rating"] <= 3000 and "PPO" in d["reason"]
    d = client.get("/api/puzzles/next?mode=coach&selector=rating_match", headers=auth).json()
    assert d["selector"] == "rating_match"


def test_attempt_updates_rating_once(client, auth):
    first = client.post("/api/attempts", headers=auth,
                        json={"puzzle_id": "00008", "solved": True, "time_ms": 9000, "mode": "rated"}).json()
    # solving an 1819 puzzle at 1000 is a big upset -> large gain, badges
    assert first["theme"] == "fork" and first["theme_rating_after"] - first["theme_rating_before"] > 40
    codes = {b["code"] for b in first["new_badges"]}
    assert {"first_blood", "giant_slayer", "speed_demon"} <= codes
    again = client.post("/api/attempts", headers=auth,
                        json={"puzzle_id": "00008", "solved": True, "time_ms": 9000, "mode": "rated"}).json()
    assert again["rating_after"] == again["rating_before"]  # repeats are unrated
    assert again["xp_gained"] <= 3


def test_hint_counts_as_miss_for_rating(client, auth):
    r = client.post("/api/attempts", headers=auth,
                    json={"puzzle_id": "T00010", "solved": True, "time_ms": 20000, "hints_used": 1, "mode": "rated"}).json()
    assert r["theme_rating_after"] < r["theme_rating_before"]


def test_review_save_and_stats(client, auth):
    assert client.get("/api/puzzles/review", headers=auth).status_code == 404
    client.post("/api/attempts", headers=auth, json={"puzzle_id": "T00020", "solved": False, "time_ms": 30000, "mode": "rated"})
    assert client.get("/api/puzzles/review", headers=auth).json()["puzzle"]["id"] == "T00020"
    client.post("/api/attempts", headers=auth, json={"puzzle_id": "T00020", "solved": True, "time_ms": 3000, "mode": "review"})
    assert client.get("/api/puzzles/review", headers=auth).status_code == 404

    assert client.put("/api/puzzles/T00020/save", headers=auth, json={"note": "tricky"}).status_code == 204
    saved = client.get("/api/puzzles/saved", headers=auth).json()
    assert [s["puzzle"]["id"] for s in saved] == ["T00020"] and saved[0]["note"] == "tricky"
    assert client.delete("/api/puzzles/T00020/save", headers=auth).status_code == 204
    assert client.get("/api/puzzles/saved", headers=auth).json() == []

    ov = client.get("/api/stats/overview", headers=auth).json()
    assert ov["totals"]["attempts"] == 2 and len(ov["themes"]) == 6
    for path in ["rating-history", "activity", "time-buckets", "coach-ab", "badges", "leaderboard?by=xp"]:
        assert client.get(f"/api/stats/{path}", headers=auth).status_code == 200


def test_daily_is_stable(client, auth):
    a = client.get("/api/puzzles/daily", headers=auth).json()["puzzle"]["id"]
    b = client.get("/api/puzzles/daily", headers=auth).json()["puzzle"]["id"]
    assert a == b
