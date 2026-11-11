"""Import Lichess puzzles into Postgres.

Applies the same filters and primary-theme assignment as the engine's
scripts/build_dataset.py, so the app's puzzle pool is the RL agent's pool.

    python scripts/import_puzzles.py --src ../data/raw/lichess_db_puzzle.csv.zst --per-theme 40000

Uses Postgres COPY for speed; streams .zst directly (no decompress step).
"""

import argparse
import csv
import io
import random
import sys
from pathlib import Path

BACKEND = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(BACKEND))
sys.path.insert(0, str(BACKEND.parent))

from scripts_shared import open_puzzle_csv  # noqa: E402
from app.db import Base, engine  # noqa: E402
from app.models import THEMES  # noqa: E402

ALIASES = {"discovery": {"discoveredAttack", "discoveredCheck"}}
RATING_MIN, RATING_MAX, MAX_RD = 400, 3000, 150


def primary_theme(tags: set[str]) -> str | None:
    # Identical precedence to scripts/build_dataset.py::_assign_primary_theme
    for t in THEMES:
        if tags & ({t} | ALIASES.get(t, set())):
            return t
    return None


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--src", type=Path, required=True)
    ap.add_argument("--per-theme", type=int, default=40_000,
                    help="max puzzles per theme (reservoir-sampled, stratified across rating); 0 = all")
    ap.add_argument("--seed", type=int, default=7)
    args = ap.parse_args()

    rng = random.Random(args.seed)
    k = args.per_theme
    # Reservoir per (theme, 100-point rating band) keeps the full rating range covered.
    reservoirs: dict[tuple[str, int], list[list[str]]] = {}
    seen: dict[tuple[str, int], int] = {}
    cap_per_band = max(1, k // 26) if k else 0

    n = 0
    with open_puzzle_csv(args.src) as fh:
        for row in csv.DictReader(fh):
            n += 1
            rating, rd = int(row["Rating"]), int(row["RatingDeviation"])
            if rd > MAX_RD or not (RATING_MIN <= rating <= RATING_MAX):
                continue
            tags = set(row["Themes"].split())
            theme = primary_theme(tags)
            if theme is None:
                continue
            rec = [row["PuzzleId"], row["FEN"], row["Moves"], str(rating), str(rd),
                   row.get("Popularity") or "0", row.get("NbPlays") or "0",
                   "{" + ",".join(sorted(tags)) + "}", theme, row.get("GameUrl") or "",
                   row.get("OpeningTags") or ""]
            key = (theme, (rating - RATING_MIN) // 100)
            bucket = reservoirs.setdefault(key, [])
            if not cap_per_band:
                bucket.append(rec)
                continue
            seen[key] = seen.get(key, 0) + 1
            if len(bucket) < cap_per_band:
                bucket.append(rec)
            else:
                j = rng.randrange(seen[key])
                if j < cap_per_band:
                    bucket[j] = rec
            if n % 500_000 == 0:
                print(f"  scanned {n:,} rows", flush=True)

    rows = [r for b in reservoirs.values() for r in b]
    print(f"Scanned {n:,} rows; importing {len(rows):,} puzzles")

    Base.metadata.create_all(engine)
    buf = io.StringIO()
    w = csv.writer(buf)
    w.writerows(rows)
    buf.seek(0)
    cols = "id, fen, moves, rating, rating_deviation, popularity, nb_plays, themes, primary_theme, game_url, opening_tags"
    raw = engine.raw_connection()
    try:
        with raw.cursor() as cur:
            cur.execute("CREATE TEMP TABLE puzzles_stage (LIKE puzzles INCLUDING DEFAULTS) ON COMMIT DROP")
            with cur.copy(f"COPY puzzles_stage ({cols}) FROM STDIN WITH (FORMAT csv)") as cp:
                while chunk := buf.read(1 << 20):
                    cp.write(chunk)
            cur.execute(f"INSERT INTO puzzles ({cols}) SELECT {cols} FROM puzzles_stage ON CONFLICT (id) DO NOTHING")
            inserted = cur.rowcount
            cur.execute("SELECT primary_theme, count(*) FROM puzzles GROUP BY 1 ORDER BY 1")
            counts = cur.fetchall()
        raw.commit()
    finally:
        raw.close()
    print(f"Inserted {inserted:,} new puzzles. Totals per theme: {dict(counts)}")


if __name__ == "__main__":
    main()
