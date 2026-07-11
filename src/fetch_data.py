"""Pull NFL data from nflverse via nflreadpy and cache it locally.

Grabs seasonal player stats, weekly stats, ADP-relevant roster info,
and injuries. Everything lands in data/ as parquet so the dataset
builder never hits the network.

nflreadpy replaced the deprecated nfl_data_py, which pulled from the
legacy nflverse `player_stats` release that stopped updating after the
2024 season. The new `stats_player` release renamed a few columns, so
we rename them back here to keep the downstream schema stable.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import nflreadpy as nfl

DATA_DIR = Path(__file__).resolve().parent.parent / "data"

FANTASY_POSITIONS = ["QB", "RB", "WR", "TE"]

# The seasonal stats now carry their own name/position columns; drop them
# so the roster merge in build_dataset.py stays collision-free.
SEASONAL_DROP = ["player_name", "player_display_name", "position", "position_group"]


def fetch(seasons: list[int]) -> None:
    DATA_DIR.mkdir(exist_ok=True)

    print(f"Fetching seasonal stats for {seasons}...")
    seasonal = nfl.load_player_stats(seasons, summary_level="reg").to_pandas()
    seasonal = seasonal.rename(columns={"passing_interceptions": "interceptions"})
    seasonal = seasonal.drop(columns=SEASONAL_DROP, errors="ignore")
    seasonal.to_parquet(DATA_DIR / "seasonal.parquet")

    print("Fetching weekly stats...")
    weekly = nfl.load_player_stats(seasons, summary_level="week").to_pandas()
    weekly = weekly.rename(columns={"passing_interceptions": "interceptions"})
    weekly.to_parquet(DATA_DIR / "weekly.parquet")

    print("Fetching player IDs / rosters...")
    rosters = nfl.load_rosters(seasons).to_pandas()
    rosters = rosters.rename(columns={"gsis_id": "player_id", "full_name": "player_name"})
    rosters = rosters[rosters["position"].isin(FANTASY_POSITIONS)]
    rosters.to_parquet(DATA_DIR / "rosters.parquet")

    print("Fetching injuries...")
    try:
        injuries = nfl.load_injuries(seasons).to_pandas()
        injuries.to_parquet(DATA_DIR / "injuries.parquet")
    except Exception as e:  # injuries feed is flaky for some seasons
        print(f"  skipped injuries: {e}")

    print("Fetching schedule (for SOS features)...")
    schedules = nfl.load_schedules(seasons).to_pandas()
    schedules = schedules[schedules["season"].isin(seasons)]
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
