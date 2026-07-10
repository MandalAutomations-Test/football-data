"""Pull NFL data from nflverse via nfl_data_py and cache it locally.

Grabs seasonal player stats, weekly stats, ADP-relevant roster info,
and injuries. Everything lands in data/ as parquet so the dataset
builder never hits the network.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import nfl_data_py as nfl
import pandas as pd

DATA_DIR = Path(__file__).resolve().parent.parent / "data"

FANTASY_POSITIONS = ["QB", "RB", "WR", "TE"]


def fetch(seasons: list[int]) -> None:
    DATA_DIR.mkdir(exist_ok=True)

    print(f"Fetching seasonal stats for {seasons}...")
    seasonal = nfl.import_seasonal_data(seasons)
    seasonal.to_parquet(DATA_DIR / "seasonal.parquet")

    print("Fetching weekly stats...")
    weekly = nfl.import_weekly_data(seasons)
    weekly.to_parquet(DATA_DIR / "weekly.parquet")

    print("Fetching player IDs / rosters...")
    rosters = nfl.import_seasonal_rosters(seasons)
    rosters = rosters[rosters["position"].isin(FANTASY_POSITIONS)]
    rosters.to_parquet(DATA_DIR / "rosters.parquet")

    print("Fetching injuries...")
    try:
        injuries = nfl.import_injuries(seasons)
        injuries.to_parquet(DATA_DIR / "injuries.parquet")
    except Exception as e:  # injuries feed is flaky for some seasons
        print(f"  skipped injuries: {e}")

    print("Fetching schedule (for SOS features)...")
    schedules = nfl.import_schedules(seasons)
    schedules.to_parquet(DATA_DIR / "schedules.parquet")

    print(f"Done. Files written to {DATA_DIR}/")


def main() -> None:
    parser = argparse.ArgumentParser(description="Fetch nflverse data")
    parser.add_argument(
        "--seasons",
        nargs="+",
        type=int,
        default=[2021, 2022, 2023, 2024, 2025],
        help="Seasons to pull",
    )
    args = parser.parse_args()
    fetch(args.seasons)


if __name__ == "__main__":
    main()
