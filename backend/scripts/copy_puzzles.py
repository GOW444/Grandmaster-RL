"""Copy the puzzles table from one Postgres to another in resumable batches.

    python scripts/copy_puzzles.py --src <local url> --dst <remote url>

Each batch is its own transaction (COPY into a temp table, INSERT ... ON CONFLICT DO
NOTHING), so a dropped connection loses at most one batch and a re-run picks up where
it left off. Useful for seeding a hosted DB (Neon) from a local one over a flaky link.
"""

import argparse
import io
import time

import psycopg

COLS = "id, fen, moves, rating, rating_deviation, popularity, nb_plays, themes, primary_theme, game_url, opening_tags"


def connect(url: str) -> psycopg.Connection:
    for attempt in range(8):
        try:
            return psycopg.connect(url, connect_timeout=15, keepalives=1, keepalives_idle=30)
        except psycopg.OperationalError as e:
            wait = min(60, 2 ** attempt)
            print(f"  connect failed ({type(e).__name__}); retrying in {wait}s", flush=True)
            time.sleep(wait)
    raise SystemExit("could not connect")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--src", required=True)
    ap.add_argument("--dst", required=True)
    ap.add_argument("--batch", type=int, default=10_000)
    a = ap.parse_args()
    src_url = a.src.replace("postgresql+psycopg://", "postgresql://")
    dst_url = a.dst.replace("postgresql+psycopg://", "postgresql://")

    with psycopg.connect(src_url) as src:
        total = src.execute("select count(*) from puzzles").fetchone()[0]
    dst = connect(dst_url)
    have = dst.execute("select count(*) from puzzles").fetchone()[0]
    last = dst.execute("select coalesce(max(id), '') from puzzles").fetchone()[0]
    print(f"source {total:,} | destination already has {have:,}", flush=True)

    while True:
        with psycopg.connect(src_url) as src:
            buf = io.BytesIO()
            with src.cursor().copy(
                f"COPY (SELECT {COLS} FROM puzzles WHERE id > %s ORDER BY id LIMIT %s) TO STDOUT",
                (last, a.batch),
            ) as cp:
                for chunk in cp:
                    buf.write(chunk)
            ids = src.execute(
                "select max(id) from (select id from puzzles where id > %s order by id limit %s) t", (last, a.batch)
            ).fetchone()[0]
        if ids is None:
            break
        for attempt in range(6):
            try:
                with dst.transaction(), dst.cursor() as cur:
                    cur.execute("CREATE TEMP TABLE IF NOT EXISTS stage (LIKE puzzles) ON COMMIT DELETE ROWS")
                    with cur.copy(f"COPY stage ({COLS}) FROM STDIN") as cp:
                        cp.write(buf.getvalue())
                    cur.execute(f"INSERT INTO puzzles ({COLS}) SELECT {COLS} FROM stage ON CONFLICT (id) DO NOTHING")
                break
            except psycopg.OperationalError as e:
                print(f"  batch failed ({type(e).__name__}); reconnecting", flush=True)
                try:
                    dst.close()
                except Exception:
                    pass
                time.sleep(min(30, 2 ** attempt))
                dst = connect(dst_url)
        else:
            raise SystemExit("giving up after repeated failures")
        last = ids
        have = dst.execute("select count(*) from puzzles").fetchone()[0]
        print(f"  {have:,} / {total:,}", flush=True)

    dst.execute("ANALYZE puzzles")
    dst.commit()
    print("done:", dict(dst.execute("select primary_theme, count(*) from puzzles group by 1 order by 1").fetchall()))


if __name__ == "__main__":
    main()
