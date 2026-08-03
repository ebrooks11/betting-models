"""Identify games where the starting QB was replaced mid-game due to injury.

Strategy:
  - "starter" = QB who threw the first pass of the game for each team
  - "finisher" = QB who threw the most passes in the 2nd half (Q3+Q4)
  - If starter != finisher AND there were meaningful 2nd-half attempts by a different QB,
    flag the game as a mid-game QB change.
  - Garbage-time filter: if the score margin when the backup first appeared was >= 17
    points (either direction), the starter was likely pulled because the game was over,
    not because of injury. Those are excluded.

We then cross-reference against our existing backup_qb_games.csv to exclude
games already tagged as "backup starts" and focus on true mid-game injuries.
"""

from pathlib import Path
import pandas as pd
from src.data_loader import get_pbp_data
from config import SEASONS

# Blowout threshold: if score margin when backup first appeared exceeds this,
# treat it as a garbage-time pull rather than a mid-game injury.
BLOWOUT_MARGIN = 17

print("Loading play-by-play data...")
COLS = ["season", "week", "game_id", "posteam", "qtr", "play_type",
        "passer_player_name", "epa", "game_seconds_remaining",
        "score_differential"]
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

# --- Garbage-time filter: find score margin when backup first appeared ---
# For each flagged game, get the score_differential on the backup's first pass play.
# score_differential is from the posteam's perspective (positive = posteam winning).
# We want the absolute margin regardless of which team is ahead.
backup_first_play = (
    pass_plays
    .merge(
        changes[["season", "week", "posteam", "finisher"]],
        on=["season", "week", "posteam"]
    )
    .query("passer_player_name == finisher")
    .sort_values("game_seconds_remaining", ascending=False)
    .groupby(["season", "week", "posteam"])
    .first()
    .reset_index()[["season", "week", "posteam", "score_differential", "game_seconds_remaining"]]
    .rename(columns={
        "score_differential": "margin_at_entry",
        "game_seconds_remaining": "seconds_remaining_at_entry",
    })
)

changes = changes.merge(backup_first_play, on=["season", "week", "posteam"], how="left")
changes["margin_at_entry"] = changes["margin_at_entry"].fillna(0)
changes["abs_margin_at_entry"] = changes["margin_at_entry"].abs()

# Flag as garbage time if blowout when backup entered
changes["garbage_time"] = changes["abs_margin_at_entry"] >= BLOWOUT_MARGIN

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
non_garbage = new_changes[~new_changes["garbage_time"]].copy()
garbage = new_changes[new_changes["garbage_time"]].copy()

# Apply final-week and early-week filters to count actionable games
final_week = non_garbage["season"].map(lambda s: 18 if s >= 2021 else 17)
actionable = non_garbage[(non_garbage["week"] < final_week) & (non_garbage["week"] >= 4)]

print(f"\nTotal mid-game QB changes detected: {len(changes)}")
print(f"Already in backup_qb_games.csv:      {changes['already_excluded'].sum()}")
print(f"Garbage-time pulls (margin ≥{BLOWOUT_MARGIN}):   {len(garbage)}")
print(f"Remaining after garbage-time filter: {len(non_garbage)}")
print(f"Actionable (wk 4+, not final wk):   {len(actionable)}")

print(f"\n--- Garbage-time pulls excluded (margin ≥{BLOWOUT_MARGIN} at backup entry) ---")
print(f"{'Season':>6} {'Wk':>3} {'Team':>5}  {'Starter':<22} {'Finisher':<22} {'Margin':>7} {'SecRem':>7}")
print("-" * 80)
for _, r in garbage.sort_values(["season", "week"]).iterrows():
    print(f"{r['season']:>6} {r['week']:>3} {r['posteam']:>5}  {r['starter']:<22} {r['finisher']:<22} {r['margin_at_entry']:>+7.0f} {r['seconds_remaining_at_entry']:>7.0f}")

print(f"\n--- Mid-game QB changes (not garbage time, not already excluded) ---")
print(f"{'Season':>6} {'Wk':>3} {'Team':>5}  {'Starter':<22} {'Finisher':<22} {'Margin':>7} {'SecRem':>7} {'FinAtt':>7}")
print("-" * 90)
for _, r in non_garbage.sort_values(["season", "week"]).iterrows():
    flag = " *" if r["week"] < (18 if r["season"] >= 2021 else 17) and r["week"] >= 4 else ""
    print(f"{r['season']:>6} {r['week']:>3} {r['posteam']:>5}  {r['starter']:<22} {r['finisher']:<22} {r['margin_at_entry']:>+7.0f} {r['seconds_remaining_at_entry']:>7.0f} {r['finisher_attempts']:>7}{flag}")

print(f"\n* = actionable game (week 4+, not final week of season)")

# Save full non-garbage list
out = non_garbage[["season", "week", "posteam", "starter", "finisher",
                    "finisher_attempts", "starter_sh_attempts",
                    "margin_at_entry", "seconds_remaining_at_entry"]].copy()
out.to_csv("exports/midgame_qb_injury_games.csv", index=False)
print(f"\nSaved {len(out)} games to exports/midgame_qb_injury_games.csv")
