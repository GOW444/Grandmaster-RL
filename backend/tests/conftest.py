import os
import sys
from pathlib import Path

import pytest

BACKEND = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(BACKEND))
os.environ.setdefault(
    "DATABASE_URL", "postgresql+psycopg://grandmaster:grandmaster@localhost:5432/grandmaster_test"
)

from fastapi.testclient import TestClient  # noqa: E402

from app.db import Base, SessionLocal, engine  # noqa: E402
from app.main import app  # noqa: E402
from app.models import THEMES, Puzzle  # noqa: E402

# A real Lichess puzzle (fork) plus synthetic rows spread over themes/ratings.
REAL = Puzzle(
    id="00008", fen="r6k/pp2r2p/4Rp1Q/3p4/8/1N1P2R1/PqP2bPP/7K b - - 0 24",
    moves="f2g3 e6e7 b2b1 b3c1 b1c1 h6c1", rating=1819, rating_deviation=77, popularity=95,
    nb_plays=10446, themes=["crushing", "hangingPiece", "long", "middlegame"], primary_theme="fork",
)


@pytest.fixture(scope="session", autouse=True)
def _schema():
    Base.metadata.drop_all(engine)
    Base.metadata.create_all(engine)
    with SessionLocal() as db:
        db.add(REAL)
        n = 0
        for t in THEMES:
            for r in range(600, 2400, 100):
                n += 1
                db.add(Puzzle(id=f"T{n:05d}", fen=REAL.fen, moves=REAL.moves, rating=r, rating_deviation=80,
                              popularity=95, nb_plays=100, themes=[t], primary_theme=t))
        db.commit()
    yield
    Base.metadata.drop_all(engine)


@pytest.fixture(scope="session")
def client():
    with TestClient(app) as c:  # runs lifespan -> loads the PPO checkpoint
        yield c


@pytest.fixture
def auth(client):
    import uuid

    name = "u" + uuid.uuid4().hex[:10]
    r = client.post("/api/auth/register", json={"username": name, "email": f"{name}@example.com", "password": "password123"})
    assert r.status_code == 201
    return {"Authorization": f"Bearer {r.json()['access_token']}"}
