"""Turn cached nflverse stats into an instruction-tuning dataset.

The model learns fantasy-draft reasoning from generated Q&A pairs:
  * player season summaries ("How did X perform in 2024?")
  * head-to-head draft comparisons ("Who should I draft, X or Y?")
  * positional tier questions ("Top 10 RBs by half-PPR last season")
  * consistency / boom-bust analysis from weekly variance
  * strategy prompts (positional scarcity, bye weeks)

Output: data/train.jsonl in chat format ready for TRL's SFTTrainer.
"""

from __future__ import annotations

import json
import random
from pathlib import Path

import pandas as pd

DATA_DIR = Path(__file__).resolve().parent.parent / "data"
OUT_FILE = DATA_DIR / "train.jsonl"

SYSTEM_PROMPT = (
    "You are a fantasy football draft assistant. You give sharp, "
    "data-driven draft advice using player stats, consistency metrics, "
    "and positional scarcity. Scoring is half-PPR unless stated otherwise."
)

random.seed(42)


# ---------------------------------------------------------------- scoring
def half_ppr(row: pd.Series) -> float:
    return round(
        row.get("passing_yards", 0) * 0.04
        + row.get("passing_tds", 0) * 4
        + row.get("interceptions", 0) * -2
        + row.get("rushing_yards", 0) * 0.1
        + row.get("rushing_tds", 0) * 6
        + row.get("receiving_yards", 0) * 0.1
        + row.get("receiving_tds", 0) * 6
        + row.get("receptions", 0) * 0.5
        + row.get("rushing_fumbles_lost", 0) * -2
        + row.get("receiving_fumbles_lost", 0) * -2,
        1,
    )


def load() -> tuple[pd.DataFrame, pd.DataFrame]:
    seasonal = pd.read_parquet(DATA_DIR / "seasonal.parquet")
    weekly = pd.read_parquet(DATA_DIR / "weekly.parquet")
    rosters = pd.read_parquet(DATA_DIR / "rosters.parquet")

    id_map = rosters[["player_id", "player_name", "position", "team"]].drop_duplicates(
        "player_id"
    )
    seasonal = seasonal.merge(id_map, on="player_id", how="inner")
    weekly = weekly[weekly["position"].isin(["QB", "RB", "WR", "TE"])]

    seasonal["fpts"] = seasonal.apply(half_ppr, axis=1)
    seasonal["fppg"] = (seasonal["fpts"] / seasonal["games"].clip(lower=1)).round(2)
    weekly["fpts"] = weekly.apply(half_ppr, axis=1)
    return seasonal, weekly


# ---------------------------------------------------------------- examples
def example(user: str, assistant: str) -> dict:
    return {
        "messages": [
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": user},
            {"role": "assistant", "content": assistant},
        ]
    }


def season_summaries(seasonal: pd.DataFrame) -> list[dict]:
    out = []
    top = seasonal[(seasonal["games"] >= 8) & (seasonal["fppg"] >= 8)]
    for _, r in top.iterrows():
        parts = [f"{r.player_name} ({r.position}, {r.team}) in {r.season}:"]
        if r.get("passing_yards", 0) > 500:
            parts.append(
                f"{int(r.passing_yards)} passing yards, {int(r.passing_tds)} TDs, "
                f"{int(r.interceptions)} INTs."
            )
        if r.get("rushing_yards", 0) > 100:
            parts.append(
                f"{int(r.rushing_yards)} rushing yards and {int(r.rushing_tds)} rushing TDs."
            )
        if r.get("receiving_yards", 0) > 100:
            parts.append(
                f"{int(r.receptions)} catches for {int(r.receiving_yards)} yards "
                f"and {int(r.receiving_tds)} TDs."
            )
        parts.append(
            f"Totaled {r.fpts} half-PPR points over {int(r.games)} games "
            f"({r.fppg} per game)."
        )
        q = random.choice(
            [
                f"How did {r.player_name} perform in the {r.season} season?",
                f"Give me a fantasy recap of {r.player_name}'s {r.season} season.",
                f"What were {r.player_name}'s {r.season} fantasy numbers?",
            ]
        )
        out.append(example(q, " ".join(parts)))
    return out


def draft_comparisons(seasonal: pd.DataFrame, n: int = 1500) -> list[dict]:
    out = []
    latest = seasonal["season"].max()
    pool = seasonal[(seasonal["season"] == latest) & (seasonal["games"] >= 8)]
    for pos in ["QB", "RB", "WR", "TE"]:
        players = pool[pool["position"] == pos].nlargest(40, "fppg")
        rows = players.to_dict("records")
        for _ in range(min(n // 4, len(rows) * 3)):
            a, b = random.sample(rows, 2)
            hi, lo = (a, b) if a["fppg"] >= b["fppg"] else (b, a)
            gap = round(hi["fppg"] - lo["fppg"], 2)
            verdict = (
                f"Lean {hi['player_name']}. He averaged {hi['fppg']} half-PPR points "
                f"per game last season vs {lo['fppg']} for {lo['player_name']} "
                f"(a {gap} ppg edge) across {int(hi['games'])} games. "
            )
            if gap < 1.5:
                verdict += (
                    "It's close though — this is effectively a coin flip, so factor "
                    "in team situation, injury history, and your roster construction."
                )
            else:
                verdict += "The production gap is meaningful at this draft slot."
            q = random.choice(
                [
                    f"Who should I draft: {a['player_name']} or {b['player_name']}?",
                    f"{a['player_name']} vs {b['player_name']} — who do you take?",
                    f"I'm on the clock. {a['player_name']} or {b['player_name']}?",
                ]
            )
            out.append(example(q, verdict))
    return out


def positional_tiers(seasonal: pd.DataFrame) -> list[dict]:
    out = []
    for season in seasonal["season"].unique():
        pool = seasonal[(seasonal["season"] == season) & (seasonal["games"] >= 8)]
        for pos, label in [("QB", "quarterbacks"), ("RB", "running backs"),
                           ("WR", "wide receivers"), ("TE", "tight ends")]:
            top = pool[pool["position"] == pos].nlargest(10, "fppg")
            if len(top) < 10:
                continue
            lines = [
                f"{i + 1}. {r.player_name} ({r.team}) — {r.fppg} ppg, {r.fpts} total"
                for i, (_, r) in enumerate(top.iterrows())
            ]
            ans = (
                f"Top 10 {label} by half-PPR points per game in {season}:\n"
                + "\n".join(lines)
            )
            out.append(
                example(f"Who were the top 10 fantasy {label} in {season}?", ans)
            )
    return out


def consistency(weekly: pd.DataFrame) -> list[dict]:
    out = []
    latest = weekly["season"].max()
    wk = weekly[weekly["season"] == latest]
    grp = (
        wk.groupby(["player_display_name", "position"])["fpts"]
        .agg(["mean", "std", "count"])
        .reset_index()
    )
    grp = grp[(grp["count"] >= 10) & (grp["mean"] >= 9)]
    for _, r in grp.iterrows():
        cv = round(r["std"] / r["mean"], 2) if r["mean"] else 0
        profile = (
            "a steady weekly floor — good for roster stability"
            if cv < 0.55
            else "a boom-bust profile — high ceiling but rough floor weeks"
        )
        ans = (
            f"{r.player_display_name} averaged {round(r['mean'], 1)} half-PPR points "
            f"per week in {latest} with a standard deviation of {round(r['std'], 1)} "
            f"(CV {cv}). That's {profile}."
        )
        out.append(
            example(
                f"Is {r.player_display_name} a consistent fantasy option or boom-bust?",
                ans,
            )
        )
    return out


def strategy() -> list[dict]:
    pairs = [
        (
            "What's the best overall strategy for a half-PPR snake draft?",
            "Anchor your first three picks with high-volume RB/WR talent, then attack "
            "positional scarcity: elite TEs and QBs are worth early picks only if they "
            "project a full tier above the field. Rounds 4-8, target volume — receivers "
            "with 120+ target projections and RBs with passing-down roles. Late rounds "
            "are for upside swings: rookies with a path to volume, handcuffs to fragile "
            "starters, and second-year breakout candidates. Never draft a kicker or "
            "defense before the last two rounds.",
        ),
        (
            "When should I draft a quarterback?",
            "In 1QB leagues, wait. QB scoring is deep — the gap between QB5 and QB12 is "
            "usually 2-3 points per game, far smaller than the RB/WR dropoff over the "
            "same rounds. Take an elite dual-threat QB early only if he projects to "
            "outscore the position by 4+ ppg. In superflex, flip this completely: QBs "
            "fly off the board and you need two startable ones by round 4.",
        ),
        (
            "Should I care about bye weeks during my draft?",
            "Barely. Prioritize best player available — a single bye-week logjam costs "
            "you at most one loss, while passing on a better player hurts every week. "
            "Only use byes as a tiebreaker between two players you have graded evenly, "
            "and avoid stacking your QB and both top receivers on the same bye.",
        ),
        (
            "What is positional scarcity and why does it matter?",
            "Scarcity is how fast production drops off within a position. Replacement-"
            "level value differs: the RB25 is far worse relative to RB1 than QB25 is to "
            "QB1. Draft value equals points above replacement, not raw points — which is "
            "why running backs and elite tight ends get drafted ahead of quarterbacks "
            "with higher raw totals.",
        ),
        (
            "How should I approach the late rounds of my draft?",
            "Swing for ceilings, not floors. A late-round pick who returns steady 6-point "
            "weeks is droppable; you want lottery tickets — ambiguous backfields, rookie "
            "receivers, players returning from injury into vacated volume. Handcuff your "
            "own RB1 if his backup would inherit a bell-cow role, and grab your defense "
            "and kicker with your final two picks.",
        ),
    ]
    return [example(q, a) for q, a in pairs]


def main() -> None:
    seasonal, weekly = load()
    examples: list[dict] = []
    examples += season_summaries(seasonal)
    examples += draft_comparisons(seasonal)
    examples += positional_tiers(seasonal)
    examples += consistency(weekly)
    examples += strategy() * 5  # upweight strategy reasoning

    random.shuffle(examples)
    with open(OUT_FILE, "w") as f:
        for ex in examples:
            f.write(json.dumps(ex) + "\n")
    print(f"Wrote {len(examples)} examples to {OUT_FILE}")


if __name__ == "__main__":
    main()
