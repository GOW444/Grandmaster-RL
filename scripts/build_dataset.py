"""
Phase 1: Data Pipeline
======================
Reads lichess_puzzles_reduced.csv, filters by rating quality and theme,
saves one CSV per primary theme, and builds a 1D KDTree per theme.

Usage:
    python scripts/build_dataset.py --csv lichess_puzzles_reduced.csv
"""

import argparse
import logging
import pickle
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.spatial import KDTree
from tqdm import tqdm

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------
THEMES: list[str] = ["fork", "pin", "mate", "endgame", "skewer", "discovery"]
RATING_MIN: int = 400
RATING_MAX: int = 3000
MAX_RATING_DEVIATION: int = 150

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(levelname)-8s  %(message)s",
    datefmt="%H:%M:%S",
)
log = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _assign_primary_theme(themes_str: str) -> str | None:
    """Return the first matching primary theme in a space-separated tag string.

    Args:
        themes_str: Raw Themes field value, e.g. ``"fork pin crushing"``.

    Returns:
        The first matching primary theme string, or ``None`` if no match.
    """
    if not isinstance(themes_str, str):
        return None
    tags = set(themes_str.split())
    for theme in THEMES:
        if theme in tags:
            return theme
    return None


def _build_tree(df: pd.DataFrame) -> KDTree:
    """Build a 1-D KDTree on the Rating column of *df*.

    Args:
        df: DataFrame that must contain a ``Rating`` column.

    Returns:
        A :class:`scipy.spatial.KDTree` built on the ratings.
    """
    ratings = df["Rating"].values.astype(float).reshape(-1, 1)
    return KDTree(ratings)


# ---------------------------------------------------------------------------
# Main pipeline
# ---------------------------------------------------------------------------

def build_dataset(
    csv_path: Path,
    processed_dir: Path,
    indices_dir: Path,
    force: bool = False,
) -> None:
    """Filter the Lichess puzzle CSV and build per-theme KD-tree indices.

    The function is idempotent: if output files already exist and *force* is
    ``False`` it will skip re-processing.

    Args:
        csv_path:      Path to the raw ``lichess_puzzles_reduced.csv``.
        processed_dir: Directory where per-theme CSVs are written.
        indices_dir:   Directory where per-theme ``.pkl`` index files are written.
        force:         If ``True``, overwrite existing outputs.
    """
    processed_dir.mkdir(parents=True, exist_ok=True)
    indices_dir.mkdir(parents=True, exist_ok=True)

    # ------------------------------------------------------------------
    # Early-exit if everything already built
    # ------------------------------------------------------------------
    all_exist = all(
        (processed_dir / f"{t}.csv").exists() and (indices_dir / f"{t}.pkl").exists()
        for t in THEMES
    )
    if all_exist and not force:
        log.info("All output files already exist. Pass --force to rebuild.")
        return

    # ------------------------------------------------------------------
    # Load CSV
    # ------------------------------------------------------------------
    log.info("Loading %s …", csv_path)
    df = pd.read_csv(
        csv_path,
        usecols=["PuzzleId", "FEN", "Moves", "Rating", "RatingDeviation", "Themes"],
        dtype={"PuzzleId": str, "FEN": str, "Moves": str,
               "Rating": int, "RatingDeviation": int, "Themes": str},
    )
    total_raw = len(df)
    log.info("  Loaded %d rows.", total_raw)

    # ------------------------------------------------------------------
    # Global filters
    # ------------------------------------------------------------------
    df = df[df["RatingDeviation"] <= MAX_RATING_DEVIATION]
    df = df[(df["Rating"] >= RATING_MIN) & (df["Rating"] <= RATING_MAX)]
    log.info("  After rating-quality filter: %d rows (dropped %d).",
             len(df), total_raw - len(df))

    # ------------------------------------------------------------------
    # Theme assignment
    # ------------------------------------------------------------------
    log.info("Assigning primary themes …")
    df["PrimaryTheme"] = df["Themes"].apply(_assign_primary_theme)
    df_with_theme = df.dropna(subset=["PrimaryTheme"])
    total_kept = len(df_with_theme)
    total_discarded = len(df) - total_kept
    log.info(
        "  Kept %d puzzles with a primary theme; discarded %d.",
        total_kept,
        total_discarded,
    )

    # ------------------------------------------------------------------
    # Per-theme processing
    # ------------------------------------------------------------------
    stats: dict[str, int] = {}
    for theme in tqdm(THEMES, desc="Building indices"):
        out_csv = processed_dir / f"{theme}.csv"
        out_pkl = indices_dir / f"{theme}.pkl"

        if out_csv.exists() and out_pkl.exists() and not force:
            log.info("  [%s] Output already exists — skipping.", theme)
            theme_df = pd.read_csv(out_csv)
            stats[theme] = len(theme_df)
            continue

        theme_df = df_with_theme[df_with_theme["PrimaryTheme"] == theme].copy()
        theme_df = theme_df.reset_index(drop=True)
        stats[theme] = len(theme_df)

        # Save filtered CSV
        theme_df.to_csv(out_csv, index=False)

        # Build and save KD-tree
        tree = _build_tree(theme_df)
        with open(out_pkl, "wb") as fh:
            pickle.dump((tree, theme_df), fh, protocol=pickle.HIGHEST_PROTOCOL)

        log.info("  [%s] %d puzzles → saved CSV + KD-tree.", theme, len(theme_df))

    # ------------------------------------------------------------------
    # Summary
    # ------------------------------------------------------------------
    log.info("\n=== Dataset Build Summary ===")
    log.info("  Raw rows:           %d", total_raw)
    log.info("  After quality filter: %d", len(df))
    log.info("  Discarded (no theme): %d", total_discarded)
    log.info("  Total kept:          %d", total_kept)
    log.info("  Per-theme breakdown:")
    for theme, count in stats.items():
        log.info("    %-12s %d", theme, count)


# ---------------------------------------------------------------------------
# CLI entry point
# ---------------------------------------------------------------------------

def _parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Build Grandmaster-RL dataset: filter and index puzzles."
    )
    parser.add_argument(
        "--csv",
        type=Path,
        default=Path("lichess_puzzles_reduced.csv"),
        help="Path to the raw Lichess puzzle CSV (default: lichess_puzzles_reduced.csv).",
    )
    parser.add_argument(
        "--processed_dir",
        type=Path,
        default=Path("data/processed"),
        help="Directory for per-theme CSVs.",
    )
    parser.add_argument(
        "--indices_dir",
        type=Path,
        default=Path("data/indices"),
        help="Directory for per-theme KD-tree pkl files.",
    )
    parser.add_argument(
        "--force",
        action="store_true",
        help="Overwrite existing output files.",
    )
    return parser.parse_args(argv)


if __name__ == "__main__":
    args = _parse_args()
    if not args.csv.exists():
        log.error("CSV file not found: %s", args.csv)
        sys.exit(1)
    build_dataset(args.csv, args.processed_dir, args.indices_dir, force=args.force)
