"""Identify games where the starting QB was replaced mid-game due to injury.

Strategy:
  - "starter" = QB who threw the first pass of the game for each team
  - "finisher" = QB who threw the most passes in the 2nd half (Q3+Q4)
  - If starter != finisher AND there were meaningful 2nd-half attempts by a different QB,
    flag the game as a mid-game QB change.

We then cross-reference against our existing backup_qb_games.csv to exclude
games already tagged as "backup starts" and focus on true mid-game injuries.
"""

from pathlib import Path
import pandas as pd
from src.data_loader import get_pbp_data
from config import SEASONS

print("Loading play-by-play data...")
COLS = ["season", "week", "game_id", "posteam", "qtr", "play_type",
        "passer_player_name", "epa", "game_seconds_remaining"]
pbp = get_pbp_data(SEASONS)[COLS]

pass_plays = pbp[
    (pbp["play_type"] == "pass") &
    pbp["passer_player_name"].notna() &
    pbp["passer_player_name"].ne("")
].copy()

# --- Identify starter: QB who threw the first pass for each team in each game ---
first_pass = (
    pass_plays
    .sort_values("game_seconds_remaining", ascending=False)  # highest = earliest in game
    .groupby(["season", "week", "posteam"])
    .first()
    .reset_index()[["season", "week", "posteam", "passer_player_name"]]
    .rename(columns={"passer_player_name": "starter"})
)

# --- Identify finisher: QB with most passes in 2nd half (qtrs 3+4+OT) ---
second_half = pass_plays[pass_plays["qtr"] >= 3].copy()

finisher = (
    second_half
    .groupby(["season", "week", "posteam", "passer_player_name"])
    .size()
    .reset_index(name="attempts")
    .sort_values("attempts", ascending=False)
    .groupby(["season", "week", "posteam"])
    .first()
    .reset_index()[["season", "week", "posteam", "passer_player_name", "attempts"]]
    .rename(columns={"passer_player_name": "finisher", "attempts": "finisher_attempts"})
)

# --- Also capture starter's 2nd-half attempts ---
starter_sh = (
    second_half
    .merge(first_pass, on=["season", "week", "posteam"])
    .query("passer_player_name == starter")
    .groupby(["season", "week", "posteam"])
    .size()
    .reset_index(name="starter_sh_attempts")
)

# --- Merge and flag changes ---
df = first_pass.merge(finisher, on=["season", "week", "posteam"], how="left")
df = df.merge(starter_sh, on=["season", "week", "posteam"], how="left")
df["starter_sh_attempts"] = df["starter_sh_attempts"].fillna(0).astype(int)
df["finisher_attempts"] = df["finisher_attempts"].fillna(0).astype(int)

# Flag: different QB in 2nd half with meaningful volume (≥5 attempts)
# and starter had very few 2nd-half attempts (≤2)
df["mid_game_change"] = (
    df["starter"] != df["finisher"]
) & (
    df["finisher_attempts"] >= 5
) & (
    df["starter_sh_attempts"] <= 2
)

changes = df[df["mid_game_change"]].copy()

# --- Remove games already in backup_qb_games.csv as starter-level exclusions ---
BACKUP_PATH = Path("exports/backup_qb_games.csv")
if BACKUP_PATH.exists():
    backup = pd.read_csv(BACKUP_PATH)
    backup_starts = set(zip(backup["season"], backup["week"], backup["posteam"]))
    changes["already_excluded"] = changes.apply(
        lambda r: (r["season"], r["week"], r["posteam"]) in backup_starts, axis=1
    )
else:
    changes["already_excluded"] = False

new_changes = changes[~changes["already_excluded"]].copy()

print(f"\nTotal mid-game QB changes detected: {len(changes)}")
print(f"Already in backup_qb_games.csv (backup starts): {changes['already_excluded'].sum()}")
print(f"New mid-game injuries to evaluate: {len(new_changes)}")

print("\n--- New mid-game QB changes (not already excluded) ---")
print(f"{'Season':>6} {'Wk':>3} {'Team':>5}  {'Starter':<25} {'Finisher':<25} {'Fin Att':>7} {'Str SH Att':>10}")
print("-" * 90)
for _, r in new_changes.sort_values(["season", "week"]).iterrows():
    print(f"{r['season']:>6} {r['week']:>3} {r['posteam']:>5}  {r['starter']:<25} {r['finisher']:<25} {r['finisher_attempts']:>7} {r['starter_sh_attempts']:>10}")

# Save
out = new_changes[["season", "week", "posteam", "starter", "finisher",
                    "finisher_attempts", "starter_sh_attempts"]].copy()
out.to_csv("exports/midgame_qb_injury_games.csv", index=False)
print(f"\nSaved {len(out)} games to exports/midgame_qb_injury_games.csv")
