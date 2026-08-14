"""Export per-game rolling EPA features (3-game window, shift 1) to parquet.

For each game, off_epa_per_play and def_epa_per_play are the mean of the
raw EPA values from the prior 1-3 games (not averages of averages).
"""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

import numpy as np
import pandas as pd
from src.data_loader import get_pbp_data, get_schedule_data, get_qbr_data
from config import SEASONS

WINDOW = 3
WINDOW5 = 5

print("Loading data...")
PBP_COLS = [
    "season", "week", "play_type", "posteam", "defteam",
    "epa", "cpoe", "fumble_lost", "interception", "down", "ydstogo",
    "sack", "qb_hit", "yards_gained",
    "drive_time_of_possession", "fixed_drive",
    "passer_player_name", "rusher_player_name",
    "offense_personnel", "defense_personnel",
    "first_down", "air_yards", "complete_pass", "pass_attempt",
    "qb_scramble", "rush_attempt", "rushing_yards", "first_down_rush", "tackled_for_loss",
]
pbp = get_pbp_data(SEASONS)[PBP_COLS]
schedules = get_schedule_data(SEASONS)
qbr = get_qbr_data(SEASONS)

plays = pbp[pbp["play_type"].isin(["pass", "run"])].copy()
pass_plays = plays[(plays["play_type"] == "pass") & plays["cpoe"].notna()].copy()
non_to_plays = plays[plays["fumble_lost"] == 0].copy()

offense = (
    plays.groupby(["season", "week", "posteam"])
    .agg(off_epa_per_play=("epa", "mean"))
    .reset_index()
    .rename(columns={"posteam": "team"})
)

defense = (
    plays.groupby(["season", "week", "defteam"])
    .agg(def_epa_per_play=("epa", "mean"))
    .reset_index()
    .rename(columns={"defteam": "team"})
)

cpoe = (
    pass_plays.groupby(["season", "week", "posteam"])
    .agg(cpoe_per_game=("cpoe", "mean"))
    .reset_index()
    .rename(columns={"posteam": "team"})
)

# Defensive CPOE allowed: avg CPOE of passes thrown against this defense (defteam perspective)
def_cpoe_allowed = (
    pass_plays.groupby(["season", "week", "defteam"])
    .agg(def_cpoe_allowed=("cpoe", "mean"))
    .reset_index()
    .rename(columns={"defteam": "team"})
)

offense_no_to = (
    non_to_plays.groupby(["season", "week", "posteam"])
    .agg(off_epa_no_to=("epa", "mean"))
    .reset_index()
    .rename(columns={"posteam": "team"})
)

defense_no_to = (
    non_to_plays.groupby(["season", "week", "defteam"])
    .agg(def_epa_no_to=("epa", "mean"))
    .reset_index()
    .rename(columns={"defteam": "team"})
)

early_down_plays = plays[plays["down"].isin([1, 2])].copy()
first_down_plays = plays[plays["down"] == 1].copy()
second_down_plays = plays[plays["down"] == 2].copy()
second_long_plays = plays[(plays["down"] == 2) & (plays["ydstogo"] >= 7)].copy()

offense_early = (
    early_down_plays.groupby(["season", "week", "posteam"])
    .agg(off_epa_early_down=("epa", "mean"))
    .reset_index()
    .rename(columns={"posteam": "team"})
)

defense_early = (
    early_down_plays.groupby(["season", "week", "defteam"])
    .agg(def_epa_early_down=("epa", "mean"))
    .reset_index()
    .rename(columns={"defteam": "team"})
)

offense_first = (
    first_down_plays.groupby(["season", "week", "posteam"])
    .agg(off_epa_first_down=("epa", "mean"))
    .reset_index()
    .rename(columns={"posteam": "team"})
)

defense_first = (
    first_down_plays.groupby(["season", "week", "defteam"])
    .agg(def_epa_first_down=("epa", "mean"))
    .reset_index()
    .rename(columns={"defteam": "team"})
)

offense_second = (
    second_down_plays.groupby(["season", "week", "posteam"])
    .agg(off_epa_second_down=("epa", "mean"))
    .reset_index()
    .rename(columns={"posteam": "team"})
)

defense_second = (
    second_down_plays.groupby(["season", "week", "defteam"])
    .agg(def_epa_second_down=("epa", "mean"))
    .reset_index()
    .rename(columns={"defteam": "team"})
)

offense_second_long = (
    second_long_plays.groupby(["season", "week", "posteam"])
    .agg(off_epa_second_long=("epa", "mean"))
    .reset_index()
    .rename(columns={"posteam": "team"})
)

defense_second_long = (
    second_long_plays.groupby(["season", "week", "defteam"])
    .agg(def_epa_second_long=("epa", "mean"))
    .reset_index()
    .rename(columns={"defteam": "team"})
)

plays_per_game = (
    plays.groupby(["season", "week", "posteam"])
    .size()
    .reset_index(name="plays")
    .rename(columns={"posteam": "team"})
)

first_down_rate = (
    plays[plays["first_down"].notna()]
    .groupby(["season", "week", "posteam"])
    .agg(first_downs=("first_down", "sum"), total_plays=("first_down", "count"))
    .assign(first_down_rate=lambda d: d["first_downs"] / d["total_plays"])
    .reset_index()[["season", "week", "posteam", "first_down_rate"]]
    .rename(columns={"posteam": "team"})
)

# QB passing metrics
pass_att_plays = pbp[(pbp["play_type"] == "pass") & (pbp["pass_attempt"] == 1)].copy()

ypa_per_game = (
    pass_att_plays.groupby(["season", "week", "posteam"])
    .agg(pass_yards=("yards_gained", "sum"), pass_attempts_count=("pass_attempt", "sum"))
    .assign(ypa=lambda d: d["pass_yards"] / d["pass_attempts_count"].replace(0, float("nan")))
    .reset_index()[["season", "week", "posteam", "ypa"]]
    .rename(columns={"posteam": "team"})
)

# Defensive YPA allowed: yards per pass attempt allowed (defteam perspective)
def_ypa_allowed = (
    pass_att_plays.groupby(["season", "week", "defteam"])
    .agg(def_pass_yards=("yards_gained", "sum"), def_pass_atts=("pass_attempt", "sum"))
    .assign(def_ypa_allowed=lambda d: d["def_pass_yards"] / d["def_pass_atts"].replace(0, float("nan")))
    .reset_index()[["season", "week", "defteam", "def_ypa_allowed"]]
    .rename(columns={"defteam": "team"})
)

adot_per_game = (
    pass_att_plays[pass_att_plays["air_yards"].notna()]
    .groupby(["season", "week", "posteam"])
    .agg(adot=("air_yards", "mean"))
    .reset_index()
    .rename(columns={"posteam": "team"})
)

pass_attempts_per_game = (
    pass_att_plays.groupby(["season", "week", "posteam"])
    .agg(pass_attempts_pg=("pass_attempt", "sum"))
    .reset_index()
    .rename(columns={"posteam": "team"})
)

completions_per_game = (
    pass_att_plays[pass_att_plays["complete_pass"].notna()]
    .groupby(["season", "week", "posteam"])
    .agg(completions_pg=("complete_pass", "sum"))
    .reset_index()
    .rename(columns={"posteam": "team"})
)

# QB rushing yards: scrambles + designed QB runs (rusher is the passer)
qb_rush_plays = pbp[
    (pbp["play_type"] == "run") &
    (pbp["qb_scramble"] == 1)
].copy()
qb_rush_per_game = (
    qb_rush_plays.groupby(["season", "week", "posteam"])
    .agg(qb_rush_yards_pg=("yards_gained", "sum"))
    .reset_index()
    .rename(columns={"posteam": "team"})
)

# Explosive plays: any play gaining 15+ yards
explosive_plays = plays[plays["yards_gained"] >= 15].copy()
explosive_per_game = (
    explosive_plays.groupby(["season", "week", "posteam"])
    .size()
    .reset_index(name="explosive_plays_pg")
    .rename(columns={"posteam": "team"})
)

# First downs per game (raw count)
first_downs_per_game = (
    plays[plays["first_down"] == 1]
    .groupby(["season", "week", "posteam"])
    .agg(first_downs_pg=("first_down", "sum"))
    .reset_index()
    .rename(columns={"posteam": "team"})
)

# Third down rate: fraction of offensive plays that are 3rd down (lower = better, team converting earlier downs)
third_down_rate = (
    plays.groupby(["season", "week", "posteam"])
    .apply(lambda g: (g["down"] == 3).sum() / len(g), include_groups=False)
    .reset_index(name="third_down_rate")
    .rename(columns={"posteam": "team"})
)

# 4th down aggression rate: go-for-it attempts / total 4th down opportunities
all_fourth = pbp[(pbp["down"] == 4) & pbp["posteam"].notna()].copy()
fourth_down_rate = (
    all_fourth.groupby(["season", "week", "posteam"])
    .apply(lambda g: (g["play_type"].isin(["run", "pass"])).sum() / len(g), include_groups=False)
    .reset_index(name="fourth_down_attempt_rate")
    .rename(columns={"posteam": "team"})
)

# Rushing metrics (non-scramble runs only)
designed_runs = pbp[(pbp["play_type"] == "run") & (pbp["qb_scramble"] != 1) & pbp["rushing_yards"].notna()].copy()

rush_yards_pg = (
    designed_runs.groupby(["season", "week", "posteam"])
    .agg(rush_yards_pg=("rushing_yards", "sum"))
    .reset_index().rename(columns={"posteam": "team"})
)

rush_ypc = (
    designed_runs.groupby(["season", "week", "posteam"])
    .agg(rush_attempts=("rush_attempt", "sum"), rush_yards=("rushing_yards", "sum"))
    .assign(rush_ypc=lambda d: d["rush_yards"] / d["rush_attempts"].replace(0, float("nan")))
    .reset_index()[["season", "week", "posteam", "rush_ypc"]]
    .rename(columns={"posteam": "team"})
)

rush_epa = (
    designed_runs.groupby(["season", "week", "posteam"])
    .agg(rush_epa=("epa", "mean"))
    .reset_index().rename(columns={"posteam": "team"})
)

rush_first_down_rate = (
    designed_runs[designed_runs["first_down_rush"].notna()]
    .groupby(["season", "week", "posteam"])
    .agg(rush_fds=("first_down_rush", "sum"), rush_atts=("first_down_rush", "count"))
    .assign(rush_first_down_rate=lambda d: d["rush_fds"] / d["rush_atts"].replace(0, float("nan")))
    .reset_index()[["season", "week", "posteam", "rush_first_down_rate"]]
    .rename(columns={"posteam": "team"})
)

rush_explosive_rate = (
    designed_runs.groupby(["season", "week", "posteam"])
    .apply(lambda g: (g["rushing_yards"] >= 10).sum() / len(g), include_groups=False)
    .reset_index(name="rush_explosive_rate")
    .rename(columns={"posteam": "team"})
)

# Defensive rushing counterparts (defteam perspective — yards/EPA allowed per carry)
def_rush_yards_pg = (
    designed_runs.groupby(["season", "week", "defteam"])
    .agg(def_rush_yards_pg=("rushing_yards", "sum"))
    .reset_index().rename(columns={"defteam": "team"})
)

def_rush_ypc = (
    designed_runs.groupby(["season", "week", "defteam"])
    .agg(rush_attempts=("rush_attempt", "sum"), rush_yards=("rushing_yards", "sum"))
    .assign(def_rush_ypc=lambda d: d["rush_yards"] / d["rush_attempts"].replace(0, float("nan")))
    .reset_index()[["season", "week", "defteam", "def_rush_ypc"]]
    .rename(columns={"defteam": "team"})
)

def_rush_epa = (
    designed_runs.groupby(["season", "week", "defteam"])
    .agg(def_rush_epa=("epa", "mean"))
    .reset_index().rename(columns={"defteam": "team"})
)

def_rush_first_down_rate = (
    designed_runs[designed_runs["first_down_rush"].notna()]
    .groupby(["season", "week", "defteam"])
    .agg(rush_fds=("first_down_rush", "sum"), rush_atts=("first_down_rush", "count"))
    .assign(def_rush_first_down_rate=lambda d: d["rush_fds"] / d["rush_atts"].replace(0, float("nan")))
    .reset_index()[["season", "week", "defteam", "def_rush_first_down_rate"]]
    .rename(columns={"defteam": "team"})
)

def_rush_explosive_rate = (
    designed_runs.groupby(["season", "week", "defteam"])
    .apply(lambda g: (g["rushing_yards"] >= 10).sum() / len(g), include_groups=False)
    .reset_index(name="def_rush_explosive_rate")
    .rename(columns={"defteam": "team"})
)

# Defensive: TFL rate (from defteam perspective) — how often this team's defense generates a TFL per rush attempt faced
tfl_allowed = (
    designed_runs[designed_runs["tackled_for_loss"].notna()]
    .groupby(["season", "week", "defteam"])
    .agg(tfls=("tackled_for_loss", "sum"), rush_atts_def=("tackled_for_loss", "count"))
    .assign(def_tfl_rate=lambda d: d["tfls"] / d["rush_atts_def"].replace(0, float("nan")))
    .reset_index()[["season", "week", "defteam", "def_tfl_rate"]]
    .rename(columns={"defteam": "team"})
)

# Personnel package metrics (2016+ only)
import re

def parse_off_personnel(s):
    if pd.isna(s):
        return None
    rb = int(m.group(1)) if (m := re.search(r'(\d+) RB', s)) else 0
    te = int(m.group(1)) if (m := re.search(r'(\d+) TE', s)) else 0
    wr = int(m.group(1)) if (m := re.search(r'(\d+) WR', s)) else 0
    if rb == 1 and te == 1 and wr == 3:
        return "11"
    if rb == 1 and te == 2 and wr == 2:
        return "12"
    if rb == 2 and te == 1 and wr == 2:
        return "21"
    return "other"

def parse_def_personnel(s):
    if pd.isna(s):
        return None
    db = int(m.group(1)) if (m := re.search(r'(\d+) DB', s)) else None
    if db is None:
        # Try summing CB + S + FS + SS
        cbs = int(m.group(1)) if (m := re.search(r'(\d+) CB', s)) else 0
        fs  = int(m.group(1)) if (m := re.search(r'(\d+) FS', s)) else 0
        ss  = int(m.group(1)) if (m := re.search(r'(\d+) SS', s)) else 0
        db = cbs + fs + ss
    if db == 4:
        return "base"
    if db == 5:
        return "nickel"
    if db == 6:
        return "dime"
    return "other"

personnel_plays = plays[plays["offense_personnel"].notna()].copy()
personnel_plays["off_pkg"] = personnel_plays["offense_personnel"].map(parse_off_personnel)
personnel_plays["def_pkg"] = personnel_plays["defense_personnel"].map(parse_def_personnel)

# Offensive EPA by own personnel package
def epa_by_off_pkg(df, pkg_val, out_col):
    return (
        df[df["off_pkg"] == pkg_val]
        .groupby(["season", "week", "posteam"])
        .agg(**{out_col: ("epa", "mean")})
        .reset_index()
        .rename(columns={"posteam": "team"})
    )

off_11_epa = epa_by_off_pkg(personnel_plays, "11", "off_11_epa")
off_12_epa = epa_by_off_pkg(personnel_plays, "12", "off_12_epa")

# Early-down (1st & 2nd down) EPA by personnel package
early_personnel_plays = personnel_plays[personnel_plays["down"].isin([1, 2])].copy()
off_11_epa_early = epa_by_off_pkg(early_personnel_plays, "11", "off_11_epa_early_down")
off_12_epa_early = epa_by_off_pkg(early_personnel_plays, "12", "off_12_epa_early_down")

# Offensive personnel rates (posteam perspective)
def pkg_rate(df, team_col, pkg_col, pkg_val, out_col):
    total = df.groupby(["season", "week", team_col]).size().reset_index(name="total")
    pkg   = df[df[pkg_col] == pkg_val].groupby(["season", "week", team_col]).size().reset_index(name="pkg")
    merged = total.merge(pkg, on=["season", "week", team_col], how="left").fillna({"pkg": 0})
    merged[out_col] = merged["pkg"] / merged["total"]
    return merged[["season", "week", team_col, out_col]].rename(columns={team_col: "team"})

off_11_rate = pkg_rate(personnel_plays, "posteam", "off_pkg", "11", "off_11_rate")
off_12_rate = pkg_rate(personnel_plays, "posteam", "off_pkg", "12", "off_12_rate")
off_21_rate = pkg_rate(personnel_plays, "posteam", "off_pkg", "21", "off_21_rate")

# Offensive EPA by defensive package faced
def epa_vs_def_pkg(df, pkg_val, out_col):
    return (
        df[df["def_pkg"] == pkg_val]
        .groupby(["season", "week", "posteam"])
        .agg(**{out_col: ("epa", "mean")})
        .reset_index()
        .rename(columns={"posteam": "team"})
    )

off_vs_nickel_epa = epa_vs_def_pkg(personnel_plays, "nickel", "off_vs_nickel_epa")
off_vs_base_epa   = epa_vs_def_pkg(personnel_plays, "base",   "off_vs_base_epa")
off_vs_dime_epa   = epa_vs_def_pkg(personnel_plays, "dime",   "off_vs_dime_epa")

# Defensive personnel rates (defteam perspective)
def_nickel_rate = pkg_rate(personnel_plays, "defteam", "def_pkg", "nickel", "def_nickel_rate")
def_base_rate   = pkg_rate(personnel_plays, "defteam", "def_pkg", "base",   "def_base_rate")
def_dime_rate   = pkg_rate(personnel_plays, "defteam", "def_pkg", "dime",   "def_dime_rate")

# Defensive EPA allowed by offensive package faced (epa is offense-perspective, so higher = worse defense)
def epa_vs_off_pkg(df, pkg_val, out_col):
    return (
        df[df["off_pkg"] == pkg_val]
        .groupby(["season", "week", "defteam"])
        .agg(**{out_col: ("epa", "mean")})
        .reset_index()
        .rename(columns={"defteam": "team"})
    )

def_vs_11_epa = epa_vs_off_pkg(personnel_plays, "11", "def_vs_11_epa")
def_vs_12_epa = epa_vs_off_pkg(personnel_plays, "12", "def_vs_12_epa")
def_vs_21_epa = epa_vs_off_pkg(personnel_plays, "21", "def_vs_21_epa")

# OL metrics: sack rate and QB hit rate (pass protection), stuff rate (run blocking)
pass_plays_ol = pbp[(pbp["play_type"] == "pass") & pbp["sack"].notna()].copy()
sack_rate = (
    pass_plays_ol.groupby(["season", "week", "posteam"])
    .agg(sacks=("sack", "sum"), pass_attempts_ol=("sack", "count"))
    .assign(sack_rate=lambda d: d["sacks"] / d["pass_attempts_ol"])
    .reset_index()[["season", "week", "posteam", "sack_rate"]]
    .rename(columns={"posteam": "team"})
)

# Defensive sack rate: how often this team's defense generates a sack per pass play (defteam perspective)
def_sack_rate = (
    pass_plays_ol.groupby(["season", "week", "defteam"])
    .agg(def_sacks=("sack", "sum"), def_pass_atts=("sack", "count"))
    .assign(def_sack_rate=lambda d: d["def_sacks"] / d["def_pass_atts"])
    .reset_index()[["season", "week", "defteam", "def_sack_rate"]]
    .rename(columns={"defteam": "team"})
)

qb_hit_rate = (
    pass_plays_ol.groupby(["season", "week", "posteam"])
    .agg(qb_hits=("qb_hit", "sum"), pass_attempts_ol=("qb_hit", "count"))
    .assign(qb_hit_rate=lambda d: d["qb_hits"] / d["pass_attempts_ol"])
    .reset_index()[["season", "week", "posteam", "qb_hit_rate"]]
    .rename(columns={"posteam": "team"})
)

# Defensive QB hit rate: how often the defense generates a QB hit (grouped by defteam)
def_qb_hit_rate = (
    pass_plays_ol.groupby(["season", "week", "defteam"])
    .agg(def_qb_hits=("qb_hit", "sum"), def_pass_atts=("qb_hit", "count"))
    .assign(def_qb_hit_rate=lambda d: d["def_qb_hits"] / d["def_pass_atts"])
    .reset_index()[["season", "week", "defteam", "def_qb_hit_rate"]]
    .rename(columns={"defteam": "team"})
)

# Turnovers committed per team per game (interceptions thrown + fumbles lost)
all_plays = pbp[pbp["posteam"].notna()].copy()
turnovers_off = (
    all_plays.groupby(["season", "week", "posteam"])
    .agg(turnovers_committed=("interception", "sum"), fumbles_lost=("fumble_lost", "sum"))
    .assign(turnovers_committed=lambda d: d["turnovers_committed"] + d["fumbles_lost"])
    .reset_index()[["season", "week", "posteam", "turnovers_committed"]]
    .rename(columns={"posteam": "team"})
)
# Turnovers forced = opponent's turnovers committed (join on defteam)
turnovers_def = (
    all_plays.groupby(["season", "week", "defteam"])
    .agg(int_forced=("interception", "sum"), fum_forced=("fumble_lost", "sum"))
    .assign(turnovers_forced=lambda d: d["int_forced"] + d["fum_forced"])
    .reset_index()[["season", "week", "defteam", "turnovers_forced"]]
    .rename(columns={"defteam": "team"})
)

run_plays_ol = pbp[(pbp["play_type"] == "run") & pbp["yards_gained"].notna()].copy()
stuff_rate = (
    run_plays_ol.groupby(["season", "week", "posteam"])
    .agg(stuffed=("yards_gained", lambda x: (x <= 0).sum()), rush_attempts=("yards_gained", "count"))
    .assign(stuff_rate=lambda d: d["stuffed"] / d["rush_attempts"])
    .reset_index()[["season", "week", "posteam", "stuff_rate"]]
    .rename(columns={"posteam": "team"})
)

# Time of possession: sum drive_time_of_possession (MM:SS) per team per game
drives = pbp[pbp["drive_time_of_possession"].notna() & pbp["posteam"].notna()].copy()
drives = drives[["season", "week", "posteam", "fixed_drive", "drive_time_of_possession"]].drop_duplicates()

def parse_mmss(s):
    parts = str(s).split(":")
    return int(parts[0]) * 60 + int(parts[1])

drives["top_seconds"] = drives["drive_time_of_possession"].apply(parse_mmss)
top = (
    drives.groupby(["season", "week", "posteam"])["top_seconds"]
    .sum()
    .reset_index()
    .rename(columns={"posteam": "team", "top_seconds": "top_seconds_per_game"})
)

TEAM_MAP = {
    "OAK": "LV",
    "SD":  "LAC",
    "STL": "LA",
    "LAR": "LA",
    "WSH": "WAS",
}

game_epa = offense.merge(defense, on=["season", "week", "team"], how="outer")
game_epa = game_epa.merge(cpoe, on=["season", "week", "team"], how="left")
game_epa = game_epa.merge(def_cpoe_allowed, on=["season", "week", "team"], how="left")
game_epa = game_epa.merge(offense_no_to, on=["season", "week", "team"], how="left")
game_epa = game_epa.merge(defense_no_to, on=["season", "week", "team"], how="left")
game_epa = game_epa.merge(offense_early, on=["season", "week", "team"], how="left")
game_epa = game_epa.merge(defense_early, on=["season", "week", "team"], how="left")
game_epa = game_epa.merge(offense_first, on=["season", "week", "team"], how="left")
game_epa = game_epa.merge(defense_first, on=["season", "week", "team"], how="left")
game_epa = game_epa.merge(offense_second, on=["season", "week", "team"], how="left")
game_epa = game_epa.merge(defense_second, on=["season", "week", "team"], how="left")
game_epa = game_epa.merge(offense_second_long, on=["season", "week", "team"], how="left")
game_epa = game_epa.merge(defense_second_long, on=["season", "week", "team"], how="left")
game_epa = game_epa.merge(plays_per_game, on=["season", "week", "team"], how="left")
game_epa = game_epa.merge(first_down_rate, on=["season", "week", "team"], how="left")
game_epa = game_epa.merge(ypa_per_game, on=["season", "week", "team"], how="left")
game_epa = game_epa.merge(def_ypa_allowed, on=["season", "week", "team"], how="left")
game_epa = game_epa.merge(adot_per_game, on=["season", "week", "team"], how="left")
game_epa = game_epa.merge(pass_attempts_per_game, on=["season", "week", "team"], how="left")
game_epa = game_epa.merge(completions_per_game, on=["season", "week", "team"], how="left")
game_epa = game_epa.merge(qb_rush_per_game, on=["season", "week", "team"], how="left")
game_epa = game_epa.merge(explosive_per_game, on=["season", "week", "team"], how="left")
game_epa = game_epa.merge(first_downs_per_game, on=["season", "week", "team"], how="left")
game_epa = game_epa.merge(third_down_rate, on=["season", "week", "team"], how="left")
game_epa = game_epa.merge(fourth_down_rate, on=["season", "week", "team"], how="left")
game_epa = game_epa.merge(rush_yards_pg, on=["season", "week", "team"], how="left")
game_epa = game_epa.merge(rush_ypc, on=["season", "week", "team"], how="left")
game_epa = game_epa.merge(rush_epa, on=["season", "week", "team"], how="left")
game_epa = game_epa.merge(rush_first_down_rate, on=["season", "week", "team"], how="left")
game_epa = game_epa.merge(rush_explosive_rate, on=["season", "week", "team"], how="left")
game_epa = game_epa.merge(def_rush_yards_pg, on=["season", "week", "team"], how="left")
game_epa = game_epa.merge(def_rush_ypc, on=["season", "week", "team"], how="left")
game_epa = game_epa.merge(def_rush_epa, on=["season", "week", "team"], how="left")
game_epa = game_epa.merge(def_rush_first_down_rate, on=["season", "week", "team"], how="left")
game_epa = game_epa.merge(def_rush_explosive_rate, on=["season", "week", "team"], how="left")
game_epa = game_epa.merge(tfl_allowed, on=["season", "week", "team"], how="left")
game_epa = game_epa.merge(top, on=["season", "week", "team"], how="left")
game_epa = game_epa.merge(sack_rate, on=["season", "week", "team"], how="left")
game_epa = game_epa.merge(def_sack_rate, on=["season", "week", "team"], how="left")
game_epa = game_epa.merge(qb_hit_rate, on=["season", "week", "team"], how="left")
game_epa = game_epa.merge(def_qb_hit_rate, on=["season", "week", "team"], how="left")
game_epa = game_epa.merge(stuff_rate, on=["season", "week", "team"], how="left")
game_epa = game_epa.merge(off_11_epa, on=["season", "week", "team"], how="left")
game_epa = game_epa.merge(off_12_epa, on=["season", "week", "team"], how="left")
game_epa = game_epa.merge(off_11_epa_early, on=["season", "week", "team"], how="left")
game_epa = game_epa.merge(off_12_epa_early, on=["season", "week", "team"], how="left")
game_epa = game_epa.merge(off_11_rate, on=["season", "week", "team"], how="left")
game_epa = game_epa.merge(off_12_rate, on=["season", "week", "team"], how="left")
game_epa = game_epa.merge(off_21_rate, on=["season", "week", "team"], how="left")
game_epa = game_epa.merge(off_vs_nickel_epa, on=["season", "week", "team"], how="left")
game_epa = game_epa.merge(off_vs_base_epa, on=["season", "week", "team"], how="left")
game_epa = game_epa.merge(off_vs_dime_epa, on=["season", "week", "team"], how="left")
game_epa = game_epa.merge(def_nickel_rate, on=["season", "week", "team"], how="left")
game_epa = game_epa.merge(def_base_rate, on=["season", "week", "team"], how="left")
game_epa = game_epa.merge(def_dime_rate, on=["season", "week", "team"], how="left")
game_epa = game_epa.merge(def_vs_11_epa, on=["season", "week", "team"], how="left")
game_epa = game_epa.merge(def_vs_12_epa, on=["season", "week", "team"], how="left")
game_epa = game_epa.merge(turnovers_off, on=["season", "week", "team"], how="left")
game_epa = game_epa.merge(turnovers_def, on=["season", "week", "team"], how="left")
game_epa = game_epa.merge(def_vs_21_epa, on=["season", "week", "team"], how="left")
game_epa["team"] = game_epa["team"].replace(TEAM_MAP)
game_epa = game_epa.sort_values(["team", "season", "week"]).reset_index(drop=True)

game_epa["off_epa_rolling"] = (
    game_epa.groupby(["team", "season"])["off_epa_per_play"]
    .transform(lambda x: x.shift(1).rolling(WINDOW, min_periods=1).mean())
)
game_epa["def_epa_rolling"] = (
    game_epa.groupby(["team", "season"])["def_epa_per_play"]
    .transform(lambda x: x.shift(1).rolling(WINDOW, min_periods=1).mean())
)
game_epa["cpoe_rolling"] = (
    game_epa.groupby(["team", "season"])["cpoe_per_game"]
    .transform(lambda x: x.shift(1).rolling(WINDOW, min_periods=1).mean())
)
game_epa["off_epa_no_to_rolling"] = (
    game_epa.groupby(["team", "season"])["off_epa_no_to"]
    .transform(lambda x: x.shift(1).rolling(WINDOW, min_periods=1).mean())
)
game_epa["def_epa_no_to_rolling"] = (
    game_epa.groupby(["team", "season"])["def_epa_no_to"]
    .transform(lambda x: x.shift(1).rolling(WINDOW, min_periods=1).mean())
)
game_epa["plays_per_game_rolling"] = (
    game_epa.groupby(["team", "season"])["plays"]
    .transform(lambda x: x.shift(1).rolling(WINDOW, min_periods=1).mean())
)
game_epa["first_down_rate_rolling"] = (
    game_epa.groupby(["team", "season"])["first_down_rate"]
    .transform(lambda x: x.shift(1).rolling(WINDOW, min_periods=1).mean())
)
for _col in ["ypa", "adot", "pass_attempts_pg", "completions_pg",
             "qb_rush_yards_pg", "explosive_plays_pg", "first_downs_pg", "third_down_rate",
             "fourth_down_attempt_rate",
             "rush_yards_pg", "rush_ypc", "rush_epa", "rush_first_down_rate",
             "rush_explosive_rate",
             "def_rush_yards_pg", "def_rush_ypc", "def_rush_epa",
             "def_rush_first_down_rate", "def_rush_explosive_rate",
             "def_tfl_rate", "def_sack_rate"]:
    game_epa[f"{_col}_rolling"] = (
        game_epa.groupby(["team", "season"])[_col]
        .transform(lambda x: x.shift(1).rolling(WINDOW, min_periods=1).mean())
    )
game_epa["off_epa_early_down_rolling"] = (
    game_epa.groupby(["team", "season"])["off_epa_early_down"]
    .transform(lambda x: x.shift(1).rolling(WINDOW, min_periods=1).mean())
)
game_epa["def_epa_early_down_rolling"] = (
    game_epa.groupby(["team", "season"])["def_epa_early_down"]
    .transform(lambda x: x.shift(1).rolling(WINDOW, min_periods=1).mean())
)
game_epa["off_epa_first_down_rolling"] = (
    game_epa.groupby(["team", "season"])["off_epa_first_down"]
    .transform(lambda x: x.shift(1).rolling(WINDOW, min_periods=1).mean())
)
game_epa["def_epa_first_down_rolling"] = (
    game_epa.groupby(["team", "season"])["def_epa_first_down"]
    .transform(lambda x: x.shift(1).rolling(WINDOW, min_periods=1).mean())
)
game_epa["off_epa_second_down_rolling"] = (
    game_epa.groupby(["team", "season"])["off_epa_second_down"]
    .transform(lambda x: x.shift(1).rolling(WINDOW, min_periods=1).mean())
)
game_epa["def_epa_second_down_rolling"] = (
    game_epa.groupby(["team", "season"])["def_epa_second_down"]
    .transform(lambda x: x.shift(1).rolling(WINDOW, min_periods=1).mean())
)
game_epa["off_epa_second_long_rolling"] = (
    game_epa.groupby(["team", "season"])["off_epa_second_long"]
    .transform(lambda x: x.shift(1).rolling(WINDOW, min_periods=1).mean())
)
game_epa["def_epa_second_long_rolling"] = (
    game_epa.groupby(["team", "season"])["def_epa_second_long"]
    .transform(lambda x: x.shift(1).rolling(WINDOW, min_periods=1).mean())
)
game_epa["top_rolling"] = (
    game_epa.groupby(["team", "season"])["top_seconds_per_game"]
    .transform(lambda x: x.shift(1).rolling(WINDOW, min_periods=1).mean())
)
game_epa["sack_rate_rolling"] = (
    game_epa.groupby(["team", "season"])["sack_rate"]
    .transform(lambda x: x.shift(1).rolling(WINDOW, min_periods=1).mean())
)
game_epa["qb_hit_rate_rolling"] = (
    game_epa.groupby(["team", "season"])["qb_hit_rate"]
    .transform(lambda x: x.shift(1).rolling(WINDOW, min_periods=1).mean())
)
game_epa["def_qb_hit_rate_rolling"] = (
    game_epa.groupby(["team", "season"])["def_qb_hit_rate"]
    .transform(lambda x: x.shift(1).rolling(WINDOW, min_periods=1).mean())
)
game_epa["stuff_rate_rolling"] = (
    game_epa.groupby(["team", "season"])["stuff_rate"]
    .transform(lambda x: x.shift(1).rolling(WINDOW, min_periods=1).mean())
)

PERSONNEL_COLS = [
    "off_11_epa", "off_12_epa",
    "off_11_epa_early_down", "off_12_epa_early_down",
    "off_11_rate", "off_12_rate", "off_21_rate",
    "off_vs_nickel_epa", "off_vs_base_epa", "off_vs_dime_epa",
    "def_nickel_rate", "def_base_rate", "def_dime_rate",
    "def_vs_11_epa", "def_vs_12_epa", "def_vs_21_epa",
]
for col in PERSONNEL_COLS:
    game_epa[f"{col}_rolling"] = (
        game_epa.groupby(["team", "season"])[col]
        .transform(lambda x: x.shift(1).rolling(WINDOW, min_periods=1).mean())
    )

# Opponent-adjusted EPA: off_epa_per_play minus the opponent's def_epa_rolling going into that game.
# This measures how much better/worse the offense performed relative to what that defense typically allows.
# Uses the opponent's PRIOR rolling defensive EPA (no leakage — same shift-1 window).
all_games_sched = schedules[schedules["game_type"].isin(["REG", "WC", "DIV", "CON", "SB"])].copy()
all_games_sched["home_team"] = all_games_sched["home_team"].replace(TEAM_MAP)
all_games_sched["away_team"] = all_games_sched["away_team"].replace(TEAM_MAP)
opp_map_home = all_games_sched[["season", "week", "home_team", "away_team"]].rename(
    columns={"home_team": "team", "away_team": "opponent"}
)
opp_map_away = all_games_sched[["season", "week", "away_team", "home_team"]].rename(
    columns={"away_team": "team", "home_team": "opponent"}
)
opp_map = pd.concat([opp_map_home, opp_map_away], ignore_index=True)

# Look up opponent's def_epa_rolling going into this game
opp_def_rolling = game_epa[["season", "week", "team", "def_epa_rolling", "def_vs_11_epa_rolling", "def_vs_12_epa_rolling"]].rename(
    columns={"team": "opponent", "def_epa_rolling": "opp_def_epa_rolling",
             "def_vs_11_epa_rolling": "opp_def_vs_11_epa_rolling",
             "def_vs_12_epa_rolling": "opp_def_vs_12_epa_rolling"}
)
opp_map = opp_map.merge(opp_def_rolling, on=["season", "week", "opponent"], how="left")
game_epa = game_epa.merge(opp_map, on=["season", "week", "team"], how="left")

# Adjusted raw EPA values (game-level, before rolling)
game_epa["off_epa_adj"] = game_epa["off_epa_per_play"] - game_epa["opp_def_epa_rolling"]
game_epa["off_11_epa_adj"] = game_epa["off_11_epa"] - game_epa["opp_def_vs_11_epa_rolling"]
game_epa["off_12_epa_adj"] = game_epa["off_12_epa"] - game_epa["opp_def_vs_12_epa_rolling"]

game_epa = game_epa.sort_values(["team", "season", "week"]).reset_index(drop=True)

# Roll the adjusted values
for _col in ["off_epa_adj", "off_11_epa_adj", "off_12_epa_adj"]:
    game_epa[f"{_col}_rolling"] = (
        game_epa.groupby(["team", "season"])[_col]
        .transform(lambda x: x.shift(1).rolling(WINDOW, min_periods=1).mean())
    )

# Iterative opponent-adjusted metrics (season-to-date, converging over N_ITER rounds).
# For each (off_col, def_col) pair, for each team going into week W:
#   adj_off[team] = mean(raw_off_game - adj_def[opponent]) over all prior games this season
# Iterating ensures the opponent's defensive quality is itself corrected for who they faced.
N_ITER = 5

def compute_iter_adj(df, off_col, def_col, out_off, out_def, n_iter=N_ITER):
    cols = ["season", "week", "team", "opponent", off_col, def_col]
    data = df[cols].copy()
    results = []
    for season in sorted(data["season"].dropna().unique()):
        sg = data[data["season"] == season].copy()
        weeks = sorted(sg["week"].dropna().unique())
        league_avg_off = sg[off_col].mean()
        league_avg_def = sg[def_col].mean()
        for week in weeks:
            prior = sg[sg["week"] < week].dropna(subset=[off_col, def_col, "opponent"])
            teams_this_week = sg[sg["week"] == week]["team"].unique()
            if len(prior) == 0:
                for team in teams_this_week:
                    results.append({"season": season, "week": week, "team": team,
                                    out_off: np.nan, out_def: np.nan})
                continue
            adj_off = prior.groupby("team")[off_col].mean().to_dict()
            adj_def = prior.groupby("team")[def_col].mean().to_dict()
            for _ in range(n_iter):
                new_adj_off, new_adj_def = {}, {}
                for team in prior["team"].unique():
                    tg = prior[prior["team"] == team]
                    off_vals, def_vals = [], []
                    for _, row in tg.iterrows():
                        opp = row["opponent"]
                        off_vals.append(row[off_col] - adj_def.get(opp, league_avg_def))
                        def_vals.append(row[def_col] - adj_off.get(opp, league_avg_off))
                    new_adj_off[team] = float(np.mean(off_vals))
                    new_adj_def[team] = float(np.mean(def_vals))
                adj_off, adj_def = new_adj_off, new_adj_def
            for team in teams_this_week:
                results.append({"season": season, "week": week, "team": team,
                                out_off: adj_off.get(team, np.nan),
                                out_def: adj_def.get(team, np.nan)})
    return pd.DataFrame(results)

# Pairs: (off_col, def_col, out_off_name, out_def_name)
ITER_ADJ_PAIRS = [
    ("off_epa_per_play", "def_epa_per_play",   "off_epa_iter_adj",        "def_epa_iter_adj"),
    ("rush_ypc",         "def_rush_ypc",        "rush_ypc_iter_adj",       "def_rush_ypc_iter_adj"),
    ("rush_epa",         "def_rush_epa",        "rush_epa_iter_adj",       "def_rush_epa_iter_adj"),
    ("off_11_epa",       "def_vs_11_epa",       "off_11_epa_iter_adj",     "def_vs_11_epa_iter_adj"),
    ("off_12_epa",       "def_vs_12_epa",       "off_12_epa_iter_adj",     "def_vs_12_epa_iter_adj"),
    ("cpoe_per_game",    "def_cpoe_allowed",    "cpoe_iter_adj",           "def_cpoe_iter_adj"),
    ("ypa",              "def_ypa_allowed",     "ypa_iter_adj",            "def_ypa_iter_adj"),
    ("qb_hit_rate",      "def_qb_hit_rate",     "qb_hit_rate_iter_adj",    "def_qb_hit_rate_iter_adj"),
    ("sack_rate",        "def_sack_rate",       "sack_rate_iter_adj",      "def_sack_rate_iter_adj"),
]

print("Computing iterative opponent-adjusted metrics...")
for off_col, def_col, out_off, out_def in ITER_ADJ_PAIRS:
    print(f"  {out_off} / {out_def}...")
    iter_df = compute_iter_adj(game_epa, off_col, def_col, out_off, out_def)
    game_epa = game_epa.merge(iter_df, on=["season", "week", "team"], how="left")
print("  Done.")

# 5-game rolling window versions
W5_STATS = {
    "off_epa_rolling_w5": "off_epa_per_play",
    "def_epa_rolling_w5": "def_epa_per_play",
    "cpoe_rolling_w5": "cpoe_per_game",
    "off_epa_no_to_rolling_w5": "off_epa_no_to",
    "def_epa_no_to_rolling_w5": "def_epa_no_to",
    "plays_per_game_rolling_w5": "plays",
    "off_epa_early_down_rolling_w5": "off_epa_early_down",
    "def_epa_early_down_rolling_w5": "def_epa_early_down",
    "off_epa_first_down_rolling_w5": "off_epa_first_down",
    "def_epa_first_down_rolling_w5": "def_epa_first_down",
    "top_rolling_w5": "top_seconds_per_game",
    "sack_rate_rolling_w5": "sack_rate",
    "qb_hit_rate_rolling_w5": "qb_hit_rate",
    "stuff_rate_rolling_w5": "stuff_rate",
}
for col_out, col_in in W5_STATS.items():
    game_epa[col_out] = (
        game_epa.groupby(["team", "season"])[col_in]
        .transform(lambda x: x.shift(1).rolling(WINDOW5, min_periods=1).mean())
    )

# Verification: spot-check that week 4+ rolling = mean of prior 3 raw EPAs
print("\nVerifying rolling values (checking week 4 of each team/season)...")
errors = 0
for (team, season), grp in game_epa.groupby(["team", "season"]):
    grp = grp.sort_values("week").reset_index(drop=True)
    for i in range(len(grp)):
        prior = grp["off_epa_per_play"].iloc[:i].tail(WINDOW)
        expected = prior.mean() if len(prior) > 0 else float("nan")
        actual = grp["off_epa_rolling"].iloc[i]
        if i == 0:
            if pd.notna(actual):
                print(f"  FAIL {team} {season} week {grp['week'].iloc[i]}: expected NaN, got {actual:.4f}")
                errors += 1
        else:
            if abs(actual - expected) > 1e-6:
                print(f"  FAIL {team} {season} week {grp['week'].iloc[i]}: expected {expected:.4f}, got {actual:.4f}")
                errors += 1
if errors == 0:
    print("  All rolling values verified correct.")
else:
    print(f"  {errors} errors found.")

# Rest days (within season only)
games = schedules[schedules["game_type"] == "REG"].copy()
games["gameday"] = pd.to_datetime(games["gameday"])
games["home_team"] = games["home_team"].replace(TEAM_MAP)
games["away_team"] = games["away_team"].replace(TEAM_MAP)

home = games[["season", "week", "home_team", "away_team", "gameday"]].rename(
    columns={"home_team": "team", "away_team": "opponent"}
)
away = games[["season", "week", "away_team", "home_team", "gameday"]].rename(
    columns={"away_team": "team", "home_team": "opponent"}
)
team_games = pd.concat([home, away], ignore_index=True)
team_games = team_games.sort_values(["team", "season", "week"]).reset_index(drop=True)

team_games["rest_days"] = (
    team_games.groupby(["team", "season"])["gameday"]
    .diff().dt.days.fillna(7)
)

# Merge opponent rest days and compute rest advantage
opp_rest = team_games[["season", "week", "team", "rest_days"]].copy()
team_games = team_games.merge(
    opp_rest.rename(columns={"team": "opponent", "rest_days": "opp_rest_days"}),
    on=["season", "week", "opponent"], how="left"
)
team_games["rest_advantage"] = team_games["rest_days"] - team_games["opp_rest_days"]

game_epa = game_epa.merge(
    team_games[["season", "week", "team", "rest_days", "rest_advantage"]],
    on=["season", "week", "team"], how="left"
)

# Point differential: score margin from each team's perspective
home_scores = games[["season", "week", "home_team", "away_team", "home_score", "away_score"]].copy()
home_scores = home_scores[home_scores["home_score"].notna() & home_scores["away_score"].notna()]
home_pd = home_scores.rename(columns={"home_team": "team"}).copy()
home_pd["point_diff"]    = home_pd["home_score"] - home_pd["away_score"]
home_pd["points_scored"] = home_pd["home_score"]
home_pd["points_allowed"]= home_pd["away_score"]
away_pd = home_scores.rename(columns={"away_team": "team"}).copy()
away_pd["point_diff"]    = away_pd["away_score"] - away_pd["home_score"]
away_pd["points_scored"] = away_pd["away_score"]
away_pd["points_allowed"]= away_pd["home_score"]
team_pd = pd.concat([
    home_pd[["season", "week", "team", "point_diff", "points_scored", "points_allowed"]],
    away_pd[["season", "week", "team", "point_diff", "points_scored", "points_allowed"]],
], ignore_index=True)
team_pd["team"] = team_pd["team"].replace(TEAM_MAP)

game_epa = game_epa.merge(team_pd, on=["season", "week", "team"], how="left")
game_epa = game_epa.sort_values(["team", "season", "week"]).reset_index(drop=True)
game_epa["point_diff_rolling"] = (
    game_epa.groupby(["team", "season"])["point_diff"]
    .transform(lambda x: x.shift(1).rolling(WINDOW, min_periods=1).mean())
)
game_epa["point_diff_rolling_w5"] = (
    game_epa.groupby(["team", "season"])["point_diff"]
    .transform(lambda x: x.shift(1).rolling(WINDOW5, min_periods=1).mean())
)
game_epa["points_scored_rolling"] = (
    game_epa.groupby(["team", "season"])["points_scored"]
    .transform(lambda x: x.shift(1).rolling(WINDOW, min_periods=1).mean())
)
game_epa["points_allowed_rolling"] = (
    game_epa.groupby(["team", "season"])["points_allowed"]
    .transform(lambda x: x.shift(1).rolling(WINDOW, min_periods=1).mean())
)
print("  points_scored_iter_adj / points_allowed_iter_adj...")
_pts_iter = compute_iter_adj(game_epa, "points_scored", "points_allowed",
                             "points_scored_iter_adj", "points_allowed_iter_adj")
game_epa = game_epa.merge(_pts_iter, on=["season", "week", "team"], how="left")
# Strength of schedule: rolling avg of opponent's point_diff_rolling going into each game.
# Build a team-game table (all game types, home + away perspective) to get opponent per game.
all_games = schedules[schedules["game_type"].isin(["REG", "WC", "DIV", "CON", "SB"])].copy()
all_games["home_team"] = all_games["home_team"].replace(TEAM_MAP)
all_games["away_team"] = all_games["away_team"].replace(TEAM_MAP)
all_team_games_home = all_games[["season", "week", "home_team", "away_team"]].rename(
    columns={"home_team": "team", "away_team": "opponent"}
)
all_team_games_away = all_games[["season", "week", "away_team", "home_team"]].rename(
    columns={"away_team": "team", "home_team": "opponent"}
)
all_team_games = pd.concat([all_team_games_home, all_team_games_away], ignore_index=True)

# Join opponent's point_diff_rolling (their form going into this game)
opp_strength = game_epa[["season", "week", "team", "point_diff_rolling"]].rename(
    columns={"team": "opponent", "point_diff_rolling": "opp_pd_rolling"}
)
all_team_games = all_team_games.merge(opp_strength, on=["season", "week", "opponent"], how="left")

game_epa = game_epa.merge(
    all_team_games[["season", "week", "team", "opp_pd_rolling"]],
    on=["season", "week", "team"], how="left"
)
game_epa = game_epa.sort_values(["team", "season", "week"]).reset_index(drop=True)
game_epa["sos_rolling"] = (
    game_epa.groupby(["team", "season"])["opp_pd_rolling"]
    .transform(lambda x: x.shift(1).rolling(WINDOW, min_periods=1).mean())
)

# Power Record: winner gets 1 + opponent's wins; loser gets 1 + opponent's losses.
# Processed week-by-week within each season so intra-week games don't affect each other.
# Uses regular wins/losses as the currency feeding into the power calculation.
from collections import defaultdict

completed_all = all_games[all_games["home_score"].notna() & all_games["away_score"].notna()].copy()

power_rows = []
for season in sorted(completed_all["season"].unique()):
    season_games = completed_all[completed_all["season"] == season]
    wins = defaultdict(int)
    losses = defaultdict(int)
    pw = defaultdict(int)
    pl = defaultdict(int)

    for week in sorted(season_games["week"].unique()):
        week_games = season_games[season_games["week"] == week]

        # Snapshot record going INTO this week for every team playing
        for _, g in week_games.iterrows():
            for team in [g["home_team"], g["away_team"]]:
                power_rows.append({
                    "season": season, "week": week, "team": team,
                    "wins": wins[team], "losses": losses[team],
                    "power_wins": pw[team], "power_losses": pl[team],
                })

        # Resolve all games in this week, then update tallies
        for _, g in week_games.iterrows():
            home, away = g["home_team"], g["away_team"]
            if g["home_score"] > g["away_score"]:
                winner, loser = home, away
            elif g["away_score"] > g["home_score"]:
                winner, loser = away, home
            else:
                continue  # tie — no update
            pw[winner] += 1 + wins[loser]
            pl[loser]  += 1 + losses[winner]
            wins[winner] += 1
            losses[loser] += 1

power_df = pd.DataFrame(power_rows).drop_duplicates(subset=["season", "week", "team"])
power_df["power_win_pct"] = power_df["power_wins"] / (power_df["power_wins"] + power_df["power_losses"]).replace(0, float("nan"))

game_epa = game_epa.merge(power_df, on=["season", "week", "team"], how="left")

game_epa["to_diff"] = game_epa["turnovers_forced"] - game_epa["turnovers_committed"]
game_epa["to_diff_rolling"] = (
    game_epa.groupby(["team", "season"])["to_diff"]
    .transform(lambda x: x.shift(1).rolling(WINDOW, min_periods=1).mean())
)
game_epa["turnovers_committed_rolling"] = (
    game_epa.groupby(["team", "season"])["turnovers_committed"]
    .transform(lambda x: x.shift(1).rolling(WINDOW, min_periods=1).mean())
)
game_epa["turnovers_committed_expanding"] = (
    game_epa.groupby(["team", "season"])["turnovers_committed"]
    .transform(lambda x: x.shift(1).expanding(min_periods=1).mean())
)

game_epa["off_11_epa_rolling_w5"] = (
    game_epa.groupby(["team", "season"])["off_11_epa"]
    .transform(lambda x: x.shift(1).rolling(WINDOW5, min_periods=1).mean())
)
game_epa["off_12_epa_rolling_w5"] = (
    game_epa.groupby(["team", "season"])["off_12_epa"]
    .transform(lambda x: x.shift(1).rolling(WINDOW5, min_periods=1).mean())
)

# Season-to-date expanding window (all prior games this season)
EXPANDING_STATS = {
    "off_epa_expanding": "off_epa_per_play",
    "def_epa_expanding": "def_epa_per_play",
    "cpoe_expanding": "cpoe_per_game",
    "off_epa_first_down_expanding": "off_epa_first_down",
    "def_epa_first_down_expanding": "def_epa_first_down",
    "point_diff_expanding": "point_diff",
    "plays_per_game_expanding": "plays",
    "qb_hit_rate_expanding": "qb_hit_rate",
}
for col_out, col_in in EXPANDING_STATS.items():
    game_epa[col_out] = (
        game_epa.groupby(["team", "season"])[col_in]
        .transform(lambda x: x.shift(1).expanding(min_periods=1).mean())
    )

PERSONNEL_EXPANDING = [
    "off_11_epa", "off_12_epa",
    "off_11_rate", "off_12_rate",
    "off_vs_nickel_epa", "off_vs_base_epa",
    "def_nickel_rate", "def_base_rate",
    "def_vs_11_epa", "def_vs_12_epa",
]
for col in PERSONNEL_EXPANDING:
    game_epa[f"{col}_expanding"] = (
        game_epa.groupby(["team", "season"])[col]
        .transform(lambda x: x.shift(1).expanding(min_periods=1).mean())
    )

# Rate-weighted formation EPA: off_11_epa * off_11_rate, off_12_epa * off_12_rate
game_epa["off_11_weighted_epa"] = game_epa["off_11_epa"] * game_epa["off_11_rate"]
game_epa["off_12_weighted_epa"] = game_epa["off_12_epa"] * game_epa["off_12_rate"]
personnel_mask = game_epa["off_11_rate"].isna()
game_epa.loc[personnel_mask, ["off_11_weighted_epa", "off_12_weighted_epa"]] = float("nan")

for col in ["off_11_weighted_epa", "off_12_weighted_epa"]:
    game_epa[f"{col}_rolling"] = (
        game_epa.groupby(["team", "season"])[col]
        .transform(lambda x: x.shift(1).rolling(WINDOW, min_periods=1).mean())
    )

# Formation composite: net formation edge weighted by usage rate
# (off_11_epa - def_vs_11_epa) * off_11_rate captures how much better the offense
# is in 11 personnel than the defense is at stopping it, scaled by usage
game_epa["formation_score"] = (
    (game_epa["off_11_epa"].fillna(0) - game_epa["def_vs_11_epa"].fillna(0)) * game_epa["off_11_rate"].fillna(0) +
    (game_epa["off_12_epa"].fillna(0) - game_epa["def_vs_12_epa"].fillna(0)) * game_epa["off_12_rate"].fillna(0)
)
personnel_mask = game_epa["off_11_rate"].isna()
game_epa.loc[personnel_mask, "formation_score"] = float("nan")

for col in ["formation_score"]:
    game_epa[f"{col}_rolling"] = (
        game_epa.groupby(["team", "season"])[col]
        .transform(lambda x: x.shift(1).rolling(WINDOW, min_periods=1).mean())
    )
    game_epa[f"{col}_expanding"] = (
        game_epa.groupby(["team", "season"])[col]
        .transform(lambda x: x.shift(1).expanding(min_periods=1).mean())
    )

# Coach ATS rolling: how many pts/game does a coach beat the spread on average
# Rolls across seasons (coach tendency is career-level, not season-level)
# Uses a 10-game window to capture stable signal
COACH_WINDOW = 10

games_with_spread = games[games["spread_line"].notna() & games["home_score"].notna()].copy()
games_with_spread["margin"] = games_with_spread["home_score"] - games_with_spread["away_score"]
games_with_spread["ats_margin"] = games_with_spread["margin"] - games_with_spread["spread_line"]

home_coach = games_with_spread[["season", "week", "home_team", "home_coach", "ats_margin"]].rename(
    columns={"home_team": "team", "home_coach": "coach"}
)
away_coach = games_with_spread[["season", "week", "away_team", "away_coach", "ats_margin"]].copy()
away_coach["ats_margin"] = -away_coach["ats_margin"]
away_coach = away_coach.rename(columns={"away_team": "team", "away_coach": "coach"})

coach_games = pd.concat([home_coach, away_coach], ignore_index=True)
coach_games = coach_games.sort_values(["coach", "season", "week"]).reset_index(drop=True)

# Rolling across seasons (no groupby season reset)
coach_games["coach_ats_rolling"] = (
    coach_games.groupby("coach")["ats_margin"]
    .transform(lambda x: x.shift(1).rolling(COACH_WINDOW, min_periods=1).mean())
)

game_epa = game_epa.merge(
    coach_games[["season", "week", "team", "coach", "coach_ats_rolling"]],
    on=["season", "week", "team"], how="left"
)

# Starting QB = QB with most pass attempts per team per game
pass_attempts = pbp[pbp["play_type"] == "pass"].copy()
starting_qb = (
    pass_attempts.groupby(["season", "week", "posteam", "passer_player_name"])
    .size()
    .reset_index(name="attempts")
    .sort_values("attempts", ascending=False)
    .groupby(["season", "week", "posteam"])
    .first()
    .reset_index()[["season", "week", "posteam", "passer_player_name"]]
    .rename(columns={"posteam": "team", "passer_player_name": "starting_qb"})
)
starting_qb["team"] = starting_qb["team"].replace(TEAM_MAP)

game_epa = game_epa.merge(starting_qb, on=["season", "week", "team"], how="left")

# Per-QB rolling stats (across team/season boundaries — no reset per season).
# Keyed on passer_player_name so they follow the QB regardless of team or year.
# This correctly handles returning starters (Murray back from ACL) and QB
# controversies (Fields/Wilson) by using the actual starter's own recent history.
qb_epa_game = (
    plays[plays["passer_player_name"].notna() & (plays["play_type"] == "pass")]
    .groupby(["season", "week", "passer_player_name"])
    .agg(qb_epa=("epa", "mean"))
    .reset_index()
)
qb_cpoe_game = (
    pass_plays[pass_plays["passer_player_name"].notna()]
    .groupby(["season", "week", "passer_player_name"])
    .agg(qb_cpoe=("cpoe", "mean"))
    .reset_index()
)
qb_off_11_game = (
    personnel_plays[
        (personnel_plays["off_pkg"] == "11") & personnel_plays["passer_player_name"].notna()
    ]
    .groupby(["season", "week", "passer_player_name"])
    .agg(qb_off_11_epa=("epa", "mean"))
    .reset_index()
)
qb_off_12_game = (
    personnel_plays[
        (personnel_plays["off_pkg"] == "12") & personnel_plays["passer_player_name"].notna()
    ]
    .groupby(["season", "week", "passer_player_name"])
    .agg(qb_off_12_epa=("epa", "mean"))
    .reset_index()
)

qb_stats = (
    qb_epa_game
    .merge(qb_cpoe_game, on=["season", "week", "passer_player_name"], how="outer")
    .merge(qb_off_11_game, on=["season", "week", "passer_player_name"], how="left")
    .merge(qb_off_12_game, on=["season", "week", "passer_player_name"], how="left")
    .sort_values(["passer_player_name", "season", "week"])
    .reset_index(drop=True)
)

for raw_col, roll_col in [
    ("qb_epa",        "qb_epa_rolling"),
    ("qb_cpoe",       "qb_cpoe_rolling"),
    ("qb_off_11_epa", "qb_off_11_epa_rolling"),
    ("qb_off_12_epa", "qb_off_12_epa_rolling"),
]:
    qb_stats[roll_col] = (
        qb_stats.groupby("passer_player_name")[raw_col]
        .transform(lambda x: x.shift(1).rolling(WINDOW, min_periods=1).mean())
    )

for raw_col, exp_col in [
    ("qb_off_11_epa", "qb_off_11_epa_expanding"),
    ("qb_off_12_epa", "qb_off_12_epa_expanding"),
]:
    qb_stats[exp_col] = (
        qb_stats.groupby("passer_player_name")[raw_col]
        .transform(lambda x: x.shift(1).expanding(min_periods=1).mean())
    )

game_epa = game_epa.merge(
    qb_stats[["season", "week", "passer_player_name",
              "qb_epa_rolling", "qb_cpoe_rolling",
              "qb_off_11_epa_rolling", "qb_off_12_epa_rolling",
              "qb_off_11_epa_expanding", "qb_off_12_epa_expanding"]]
    .rename(columns={"passer_player_name": "starting_qb"}),
    on=["season", "week", "starting_qb"],
    how="left",
)

# Rolling QBR: join QBR onto starting QB, then compute 3-game rolling average
qbr_reg = qbr[qbr["season_type"].isin(["Regular", "Playoffs"])].copy()
qbr_reg["team_abb"] = qbr_reg["team_abb"].replace(TEAM_MAP)
# Normalize to "F.Last" format — strip middle initials so "C.J. Stroud" → "C.Stroud"
# to match the passer_player_name format used in nflverse play-by-play
def _normalize_qbr_name(name: str) -> str:
    # Produce "F.Last" to match passer_player_name in PBP
    # Handles "M. Penix Jr." → "M.Penix", "C.J. Stroud" → "C.Stroud"
    parts = name.strip().split(".")
    first_initial = parts[0].strip()
    last = ""
    for part in parts[1:]:
        word = part.strip().split()[0] if part.strip() else ""
        if len(word) > 1:  # skip middle initials like "J"
            last = word
            break
    if first_initial and last:
        return f"{first_initial}.{last}"
    return name

qbr_reg["name_normalized"] = qbr_reg["name_short"].apply(_normalize_qbr_name)
qbr_reg = qbr_reg.drop(columns=["team"]).rename(columns={"week_num": "week", "team_abb": "team"})

# Join QBR to game_epa on season/week/starting_qb only (drop team from key because
# nflverse sometimes attributes a QB's QBR to his former team after a mid-season trade)
qbr_game = (
    qbr_reg[["season", "week", "name_normalized", "qbr_total"]]
    .sort_values("qbr_total", ascending=False)
    .drop_duplicates(subset=["season", "week", "name_normalized"])
)
game_epa = game_epa.merge(
    qbr_game.rename(columns={"name_normalized": "starting_qb", "qbr_total": "qbr"}),
    on=["season", "week", "starting_qb"], how="left"
)

game_epa = game_epa.sort_values(["team", "season", "week"]).reset_index(drop=True)
game_epa["qbr_rolling"] = (
    game_epa.groupby(["team", "season"])["qbr"]
    .transform(lambda x: x.shift(1).rolling(WINDOW, min_periods=1).mean())
)
game_epa["qbr_rolling_w5"] = (
    game_epa.groupby(["team", "season"])["qbr"]
    .transform(lambda x: x.shift(1).rolling(WINDOW5, min_periods=1).mean())
)
game_epa["qbr_expanding"] = (
    game_epa.groupby(["team", "season"])["qbr"]
    .transform(lambda x: x.shift(1).expanding(min_periods=1).mean())
)

result = game_epa[["season", "week", "team",
                    "off_epa_per_play", "off_epa_rolling",
                    "def_epa_per_play", "def_epa_rolling",
                    "cpoe_per_game", "cpoe_rolling",
                    "off_epa_no_to", "off_epa_no_to_rolling",
                    "def_epa_no_to", "def_epa_no_to_rolling",
                    "off_epa_early_down", "off_epa_early_down_rolling",
                    "def_epa_early_down", "def_epa_early_down_rolling",
                    "off_epa_first_down", "off_epa_first_down_rolling",
                    "def_epa_first_down", "def_epa_first_down_rolling",
                    "off_epa_second_down", "off_epa_second_down_rolling",
                    "def_epa_second_down", "def_epa_second_down_rolling",
                    "off_epa_second_long", "off_epa_second_long_rolling",
                    "def_epa_second_long", "def_epa_second_long_rolling",
                    "plays", "plays_per_game_rolling",
                    "first_down_rate", "first_down_rate_rolling",
                    "top_seconds_per_game", "top_rolling",
                    "rest_days", "rest_advantage",
                    "point_diff", "point_diff_rolling", "point_diff_rolling_w5",
                    "points_scored", "points_scored_rolling",
                    "points_allowed", "points_allowed_rolling",
                    "coach", "coach_ats_rolling",
                    "starting_qb", "qbr", "qbr_rolling",
                    "qb_epa_rolling", "qb_cpoe_rolling",
                    "qb_off_11_epa_rolling", "qb_off_12_epa_rolling",
                    "sack_rate", "sack_rate_rolling",
                    "qb_hit_rate", "qb_hit_rate_rolling",
                    "def_qb_hit_rate", "def_qb_hit_rate_rolling",
                    "stuff_rate", "stuff_rate_rolling",
                    "off_11_epa", "off_11_epa_rolling", "off_11_epa_rolling_w5",
                    "off_12_epa", "off_12_epa_rolling", "off_12_epa_rolling_w5",
                    "off_11_epa_early_down", "off_11_epa_early_down_rolling",
                    "off_12_epa_early_down", "off_12_epa_early_down_rolling",
                    "off_11_weighted_epa", "off_11_weighted_epa_rolling",
                    "off_12_weighted_epa", "off_12_weighted_epa_rolling",
                    "off_11_rate", "off_11_rate_rolling",
                    "off_12_rate", "off_12_rate_rolling",
                    "off_21_rate", "off_21_rate_rolling",
                    "off_vs_nickel_epa", "off_vs_nickel_epa_rolling",
                    "off_vs_base_epa", "off_vs_base_epa_rolling",
                    "off_vs_dime_epa", "off_vs_dime_epa_rolling",
                    "def_nickel_rate", "def_nickel_rate_rolling",
                    "def_base_rate", "def_base_rate_rolling",
                    "def_dime_rate", "def_dime_rate_rolling",
                    "def_vs_11_epa", "def_vs_11_epa_rolling",
                    "def_vs_12_epa", "def_vs_12_epa_rolling",
                    "def_vs_21_epa", "def_vs_21_epa_rolling",
                    "formation_score", "formation_score_rolling", "formation_score_expanding",
                    "off_epa_expanding", "def_epa_expanding",
                    "cpoe_expanding", "off_epa_first_down_expanding", "def_epa_first_down_expanding",
                    "point_diff_expanding", "plays_per_game_expanding", "qb_hit_rate_expanding",
                    "off_11_epa_expanding", "off_12_epa_expanding",
                    "off_11_rate_expanding", "off_12_rate_expanding",
                    "off_vs_nickel_epa_expanding", "off_vs_base_epa_expanding",
                    "def_nickel_rate_expanding", "def_base_rate_expanding",
                    "def_vs_11_epa_expanding", "def_vs_12_epa_expanding",
                    "off_epa_rolling_w5", "def_epa_rolling_w5",
                    "cpoe_rolling_w5", "off_epa_no_to_rolling_w5", "def_epa_no_to_rolling_w5",
                    "plays_per_game_rolling_w5",
                    "off_epa_early_down_rolling_w5", "def_epa_early_down_rolling_w5",
                    "off_epa_first_down_rolling_w5", "def_epa_first_down_rolling_w5",
                    "top_rolling_w5", "sack_rate_rolling_w5",
                    "qb_hit_rate_rolling_w5", "stuff_rate_rolling_w5",
                    "qbr_rolling_w5",
                    "turnovers_committed", "turnovers_forced", "to_diff", "to_diff_rolling",
                    "turnovers_committed_rolling", "turnovers_committed_expanding",
                    "qbr_expanding",
                    "qb_off_11_epa_expanding", "qb_off_12_epa_expanding",
                    "opp_pd_rolling", "sos_rolling",
                    "wins", "losses",
                    "power_wins", "power_losses", "power_win_pct",
                    "ypa", "ypa_rolling",
                    "adot", "adot_rolling",
                    "pass_attempts_pg", "pass_attempts_pg_rolling",
                    "completions_pg", "completions_pg_rolling",
                    "qb_rush_yards_pg", "qb_rush_yards_pg_rolling",
                    "explosive_plays_pg", "explosive_plays_pg_rolling",
                    "first_downs_pg", "first_downs_pg_rolling",
                    "third_down_rate", "third_down_rate_rolling",
                    "fourth_down_attempt_rate", "fourth_down_attempt_rate_rolling",
                    "rush_yards_pg", "rush_yards_pg_rolling",
                    "rush_ypc", "rush_ypc_rolling",
                    "rush_epa", "rush_epa_rolling",
                    "rush_first_down_rate", "rush_first_down_rate_rolling",
                    "rush_explosive_rate", "rush_explosive_rate_rolling",
                    "def_rush_yards_pg", "def_rush_yards_pg_rolling",
                    "def_rush_ypc", "def_rush_ypc_rolling",
                    "def_rush_epa", "def_rush_epa_rolling",
                    "def_rush_first_down_rate", "def_rush_first_down_rate_rolling",
                    "def_rush_explosive_rate", "def_rush_explosive_rate_rolling",
                    "def_tfl_rate", "def_tfl_rate_rolling",
                    "def_sack_rate", "def_sack_rate_rolling",
                    "off_epa_adj", "off_epa_adj_rolling",
                    "off_11_epa_adj", "off_11_epa_adj_rolling",
                    "off_12_epa_adj", "off_12_epa_adj_rolling",
                    "off_epa_iter_adj", "def_epa_iter_adj",
                    "rush_ypc_iter_adj", "def_rush_ypc_iter_adj",
                    "rush_epa_iter_adj", "def_rush_epa_iter_adj",
                    "off_11_epa_iter_adj", "def_vs_11_epa_iter_adj",
                    "off_12_epa_iter_adj", "def_vs_12_epa_iter_adj",
                    "points_scored_iter_adj", "points_allowed_iter_adj",
                    "cpoe_iter_adj", "def_cpoe_iter_adj",
                    "ypa_iter_adj", "def_ypa_iter_adj",
                    "qb_hit_rate_iter_adj", "def_qb_hit_rate_iter_adj",
                    "sack_rate_iter_adj", "def_sack_rate_iter_adj"]].copy()
result = result.sort_values(["season", "week", "team"]).reset_index(drop=True)

out_path = Path("exports/features.parquet")
out_path.parent.mkdir(exist_ok=True)
result.to_parquet(out_path, index=False)

print(f"\nExported {len(result):,} rows to {out_path}")
print("\nSample — DAL weeks 1-5 of 2022:")
sample = result[(result["team"] == "DAL") & (result["season"] == 2022)].head(5)
print(sample[["season", "week", "team", "off_epa_rolling", "def_epa_rolling", "rest_days", "rest_advantage"]].to_string(index=False))
