"""Fetch and cache NFL data from nflverse via nfl_data_py."""

import os
import time
from pathlib import Path

import nfl_data_py as nfl
import pandas as pd
import requests

DATA_DIR = Path(__file__).resolve().parent.parent / "data"


def _stringify_mixed(v):
    if pd.isna(v):
        return None
    if isinstance(v, float) and v.is_integer():
        return str(int(v))
    return str(v)


def _safe_to_parquet(df: pd.DataFrame, path: Path) -> None:
    """Write a DataFrame to parquet, working around two failure modes:

    1. Some nflverse columns (e.g. jersey_number) are typed inconsistently
       across seasons — str in some, float in others — which produces a
       mixed-type object column that fastparquet can't encode and fails on
       mid-write. Normalize any such column to a consistent string first.
    2. A failed write shouldn't leave a corrupt file sitting at `path`
       looking like a valid cache. Write to a temp file and rename into
       place only on success.
    """
    df = df.copy()
    for col in df.select_dtypes(include="object").columns:
        non_null_types = set(df[col].dropna().map(type))
        if len(non_null_types) > 1:
            df[col] = df[col].map(_stringify_mixed)

    tmp_path = path.with_suffix(".tmp.parquet")
    df.to_parquet(tmp_path, index=False)
    tmp_path.replace(path)


def get_pbp_data(seasons: list[int], refresh: bool = False) -> pd.DataFrame:
    cache_path = DATA_DIR / "pbp.parquet"
    if cache_path.exists() and not refresh:
        return pd.read_parquet(cache_path)

    print(f"Downloading play-by-play data for {seasons[0]}-{seasons[-1]}...")
    pbp = nfl.import_pbp_data(seasons)
    os.makedirs(DATA_DIR, exist_ok=True)
    _safe_to_parquet(pbp, cache_path)
    print(f"Cached {len(pbp):,} plays to {cache_path}")
    return pbp


def get_schedule_data(seasons: list[int], refresh: bool = False) -> pd.DataFrame:
    cache_path = DATA_DIR / "schedules.parquet"
    if cache_path.exists() and not refresh:
        return pd.read_parquet(cache_path)

    print(f"Downloading schedule data for {seasons[0]}-{seasons[-1]}...")
    schedules = nfl.import_schedules(seasons)
    os.makedirs(DATA_DIR, exist_ok=True)
    _safe_to_parquet(schedules, cache_path)
    print(f"Cached {len(schedules):,} games to {cache_path}")
    return schedules


def get_qbr_data(seasons: list[int], refresh: bool = False) -> pd.DataFrame:
    cache_path = DATA_DIR / "qbr.parquet"
    if cache_path.exists() and not refresh:
        return pd.read_parquet(cache_path)

    print(f"Downloading QBR data for {seasons[0]}-{seasons[-1]}...")
    qbr = nfl.import_qbr(seasons, level="nfl", frequency="weekly")
    os.makedirs(DATA_DIR, exist_ok=True)
    _safe_to_parquet(qbr, cache_path)
    print(f"Cached {len(qbr):,} QB-game rows to {cache_path}")
    return qbr


def get_roster_data(seasons: list[int], refresh: bool = False) -> pd.DataFrame:
    cache_path = DATA_DIR / "rosters.parquet"
    if cache_path.exists() and not refresh:
        return pd.read_parquet(cache_path)

    print(f"Downloading roster data for {seasons[0]}-{seasons[-1]}...")
    rosters = nfl.import_players()
    os.makedirs(DATA_DIR, exist_ok=True)
    _safe_to_parquet(rosters, cache_path)
    print(f"Cached {len(rosters):,} roster rows to {cache_path}")
    return rosters


def get_snap_counts(seasons: list[int], refresh: bool = False) -> pd.DataFrame:
    cache_path = DATA_DIR / "snap_counts.parquet"
    if cache_path.exists() and not refresh:
        return pd.read_parquet(cache_path)

    snap_seasons = [s for s in seasons if s >= 2012]  # not available before 2012
    print(f"Downloading snap count data for {snap_seasons[0]}-{snap_seasons[-1]}...")
    snaps = nfl.import_snap_counts(snap_seasons)
    os.makedirs(DATA_DIR, exist_ok=True)
    _safe_to_parquet(snaps, cache_path)
    print(f"Cached {len(snaps):,} snap count rows to {cache_path}")
    return snaps


def get_pfr_advstats(seasons: list[int], refresh: bool = False) -> pd.DataFrame:
    cache_path = DATA_DIR / "pfr_advstats.parquet"
    if cache_path.exists() and not refresh:
        return pd.read_parquet(cache_path)

    pfr_seasons = [s for s in seasons if s >= 2018]
    print(f"Downloading PFR weekly advanced stats for {pfr_seasons[0]}-{pfr_seasons[-1]}...")
    dfs = []
    for stat_type in ["pass", "rush", "rec"]:
        try:
            df = nfl.import_weekly_pfr(stat_type, pfr_seasons)
            df["stat_type"] = stat_type
            dfs.append(df)
            print(f"  {stat_type}: {len(df):,} rows")
        except Exception as e:
            print(f"  {stat_type}: skipped ({e})")

    if not dfs:
        return pd.DataFrame()

    combined = pd.concat(dfs, ignore_index=True)
    os.makedirs(DATA_DIR, exist_ok=True)
    _safe_to_parquet(combined, cache_path)
    print(f"Cached {len(combined):,} PFR rows to {cache_path}")
    return combined


def get_weekly_data(seasons: list[int], refresh: bool = False) -> pd.DataFrame:
    cache_path = DATA_DIR / "weekly.parquet"
    if cache_path.exists() and not refresh:
        return pd.read_parquet(cache_path)

    print(f"Downloading weekly player data for {seasons[0]}-{seasons[-1]}...")
    dfs = []
    for year in seasons:
        try:
            dfs.append(nfl.import_weekly_data([year]))
        except Exception as e:
            print(f"  {year}: skipped ({e})")

    weekly = pd.concat(dfs, ignore_index=True)
    os.makedirs(DATA_DIR, exist_ok=True)
    _safe_to_parquet(weekly, cache_path)
    print(f"Cached {len(weekly):,} weekly player rows to {cache_path}")
    return weekly


def get_seasonal_data(seasons: list[int], refresh: bool = False) -> pd.DataFrame:
    cache_path = DATA_DIR / "seasonal.parquet"
    if cache_path.exists() and not refresh:
        return pd.read_parquet(cache_path)

    print(f"Downloading seasonal player data for {seasons[0]}-{seasons[-1]}...")
    dfs = []
    for year in seasons:
        try:
            dfs.append(nfl.import_seasonal_data([year]))
        except Exception as e:
            print(f"  {year}: skipped ({e})")

    seasonal = pd.concat(dfs, ignore_index=True)
    os.makedirs(DATA_DIR, exist_ok=True)
    _safe_to_parquet(seasonal, cache_path)
    print(f"Cached {len(seasonal):,} seasonal player rows to {cache_path}")
    return seasonal


def get_seasonal_rosters(seasons: list[int], refresh: bool = False) -> pd.DataFrame:
    cache_path = DATA_DIR / "seasonal_rosters.parquet"
    if cache_path.exists() and not refresh:
        return pd.read_parquet(cache_path)

    print(f"Downloading seasonal rosters for {seasons[0]}-{seasons[-1]}...")
    rosters = nfl.import_seasonal_rosters(seasons)
    os.makedirs(DATA_DIR, exist_ok=True)
    _safe_to_parquet(rosters, cache_path)
    print(f"Cached {len(rosters):,} seasonal roster rows to {cache_path}")
    return rosters


def get_weekly_rosters(seasons: list[int], refresh: bool = False) -> pd.DataFrame:
    cache_path = DATA_DIR / "weekly_rosters.parquet"
    if cache_path.exists() and not refresh:
        return pd.read_parquet(cache_path)

    print(f"Downloading weekly rosters for {seasons[0]}-{seasons[-1]}...")
    rosters = nfl.import_weekly_rosters(seasons)
    os.makedirs(DATA_DIR, exist_ok=True)
    _safe_to_parquet(rosters, cache_path)
    print(f"Cached {len(rosters):,} weekly roster rows to {cache_path}")
    return rosters


def get_depth_charts(seasons: list[int], refresh: bool = False) -> pd.DataFrame:
    cache_path = DATA_DIR / "depth_charts.parquet"
    if cache_path.exists() and not refresh:
        return pd.read_parquet(cache_path)

    print(f"Downloading depth charts for {seasons[0]}-{seasons[-1]}...")
    charts = nfl.import_depth_charts(seasons)
    os.makedirs(DATA_DIR, exist_ok=True)
    _safe_to_parquet(charts, cache_path)
    print(f"Cached {len(charts):,} depth chart rows to {cache_path}")
    return charts


def get_injuries(seasons: list[int], refresh: bool = False) -> pd.DataFrame:
    cache_path = DATA_DIR / "injuries.parquet"
    if cache_path.exists() and not refresh:
        return pd.read_parquet(cache_path)

    injury_seasons = [s for s in seasons if s >= 2009]  # not available before 2009
    print(f"Downloading injury reports for {injury_seasons[0]}-{injury_seasons[-1]}...")
    injuries = nfl.import_injuries(injury_seasons)
    os.makedirs(DATA_DIR, exist_ok=True)
    _safe_to_parquet(injuries, cache_path)
    print(f"Cached {len(injuries):,} injury report rows to {cache_path}")
    return injuries


def get_ngs_data(seasons: list[int], refresh: bool = False) -> pd.DataFrame:
    cache_path = DATA_DIR / "ngs.parquet"
    if cache_path.exists() and not refresh:
        return pd.read_parquet(cache_path)

    print(f"Downloading Next Gen Stats for {seasons[0]}-{seasons[-1]}...")
    dfs = []
    for stat_type in ["passing", "rushing", "receiving"]:
        df = nfl.import_ngs_data(stat_type, seasons)
        df["stat_type"] = stat_type
        dfs.append(df)
        print(f"  {stat_type}: {len(df):,} rows")

    combined = pd.concat(dfs, ignore_index=True)
    os.makedirs(DATA_DIR, exist_ok=True)
    _safe_to_parquet(combined, cache_path)
    print(f"Cached {len(combined):,} NGS rows to {cache_path}")
    return combined


def get_ftn_data(seasons: list[int], refresh: bool = False) -> pd.DataFrame:
    cache_path = DATA_DIR / "ftn.parquet"
    if cache_path.exists() and not refresh:
        return pd.read_parquet(cache_path)

    ftn_seasons = [s for s in seasons if s >= 2022]
    print(f"Downloading FTN charting data for {ftn_seasons[0]}-{ftn_seasons[-1]}...")
    ftn = nfl.import_ftn_data(ftn_seasons)
    os.makedirs(DATA_DIR, exist_ok=True)
    _safe_to_parquet(ftn, cache_path)
    print(f"Cached {len(ftn):,} FTN rows to {cache_path}")
    return ftn


def get_seasonal_pfr(seasons: list[int], refresh: bool = False) -> pd.DataFrame:
    cache_path = DATA_DIR / "seasonal_pfr.parquet"
    if cache_path.exists() and not refresh:
        return pd.read_parquet(cache_path)

    pfr_seasons = [s for s in seasons if s >= 2018]
    print(f"Downloading PFR seasonal advanced stats for {pfr_seasons[0]}-{pfr_seasons[-1]}...")
    dfs = []
    for s_type in ["pass", "rush", "rec", "def"]:
        try:
            df = nfl.import_seasonal_pfr(s_type, pfr_seasons)
            df["stat_type"] = s_type
            dfs.append(df)
            print(f"  {s_type}: {len(df):,} rows")
        except Exception as e:
            print(f"  {s_type}: skipped ({e})")

    if not dfs:
        return pd.DataFrame()

    combined = pd.concat(dfs, ignore_index=True)
    os.makedirs(DATA_DIR, exist_ok=True)
    _safe_to_parquet(combined, cache_path)
    print(f"Cached {len(combined):,} PFR seasonal rows to {cache_path}")
    return combined


def get_combine_data(seasons: list[int], refresh: bool = False) -> pd.DataFrame:
    cache_path = DATA_DIR / "combine.parquet"
    if cache_path.exists() and not refresh:
        return pd.read_parquet(cache_path)

    print(f"Downloading combine data for {seasons[0]}-{seasons[-1]}...")
    combine = nfl.import_combine_data(seasons)
    os.makedirs(DATA_DIR, exist_ok=True)
    _safe_to_parquet(combine, cache_path)
    print(f"Cached {len(combine):,} combine rows to {cache_path}")
    return combine


def get_draft_picks(seasons: list[int], refresh: bool = False) -> pd.DataFrame:
    cache_path = DATA_DIR / "draft_picks.parquet"
    if cache_path.exists() and not refresh:
        return pd.read_parquet(cache_path)

    print(f"Downloading draft picks for {seasons[0]}-{seasons[-1]}...")
    picks = nfl.import_draft_picks(seasons)
    os.makedirs(DATA_DIR, exist_ok=True)
    _safe_to_parquet(picks, cache_path)
    print(f"Cached {len(picks):,} draft pick rows to {cache_path}")
    return picks


def get_draft_values(refresh: bool = False) -> pd.DataFrame:
    cache_path = DATA_DIR / "draft_values.parquet"
    if cache_path.exists() and not refresh:
        return pd.read_parquet(cache_path)

    print("Downloading draft pick values...")
    values = nfl.import_draft_values()
    os.makedirs(DATA_DIR, exist_ok=True)
    _safe_to_parquet(values, cache_path)
    print(f"Cached {len(values):,} draft value rows to {cache_path}")
    return values


def get_contracts(refresh: bool = False) -> pd.DataFrame:
    cache_path = DATA_DIR / "contracts.parquet"
    if cache_path.exists() and not refresh:
        return pd.read_parquet(cache_path)

    print("Downloading historical contract data...")
    contracts = nfl.import_contracts()
    os.makedirs(DATA_DIR, exist_ok=True)
    _safe_to_parquet(contracts, cache_path)
    print(f"Cached {len(contracts):,} contract rows to {cache_path}")
    return contracts


def get_officials(seasons: list[int], refresh: bool = False) -> pd.DataFrame:
    cache_path = DATA_DIR / "officials.parquet"
    if cache_path.exists() and not refresh:
        return pd.read_parquet(cache_path)

    print(f"Downloading game officials for {seasons[0]}-{seasons[-1]}...")
    officials = nfl.import_officials(seasons)
    os.makedirs(DATA_DIR, exist_ok=True)
    _safe_to_parquet(officials, cache_path)
    print(f"Cached {len(officials):,} official rows to {cache_path}")
    return officials


def get_team_desc(refresh: bool = False) -> pd.DataFrame:
    cache_path = DATA_DIR / "team_desc.parquet"
    if cache_path.exists() and not refresh:
        return pd.read_parquet(cache_path)

    print("Downloading team descriptive data...")
    teams = nfl.import_team_desc()
    os.makedirs(DATA_DIR, exist_ok=True)
    _safe_to_parquet(teams, cache_path)
    print(f"Cached {len(teams):,} team rows to {cache_path}")
    return teams


def get_ids(refresh: bool = False) -> pd.DataFrame:
    cache_path = DATA_DIR / "ids.parquet"
    if cache_path.exists() and not refresh:
        return pd.read_parquet(cache_path)

    print("Downloading cross-provider ID mapping table...")
    ids = nfl.import_ids()
    os.makedirs(DATA_DIR, exist_ok=True)
    _safe_to_parquet(ids, cache_path)
    print(f"Cached {len(ids):,} id mapping rows to {cache_path}")
    return ids


def get_sc_lines(seasons: list[int], refresh: bool = False) -> pd.DataFrame:
    cache_path = DATA_DIR / "sc_lines.parquet"
    if cache_path.exists() and not refresh:
        return pd.read_parquet(cache_path)

    print(f"Downloading weekly scoring lines for {seasons[0]}-{seasons[-1]}...")
    lines = nfl.import_sc_lines(seasons)
    os.makedirs(DATA_DIR, exist_ok=True)
    _safe_to_parquet(lines, cache_path)
    print(f"Cached {len(lines):,} scoring line rows to {cache_path}")
    return lines


def get_win_totals(seasons: list[int], refresh: bool = False) -> pd.DataFrame:
    """Preseason win-total market lines, scraped from sportsoddshistory.com.

    Delegates to pipeline/fetch_win_totals.py's scrape_season() rather than
    nfl_data_py's import_win_totals() — that source has been flaky/incomplete
    for recent seasons (it warns the source is "in flux" and has returned 0
    rows for some years even when others succeed).
    """
    cache_path = DATA_DIR / "win_totals.parquet"
    if cache_path.exists() and not refresh:
        return pd.read_parquet(cache_path)

    from pipeline.fetch_win_totals import scrape_season

    print(f"Scraping win totals for {seasons[0]}-{seasons[-1]}...")
    session = requests.Session()
    session.headers.update({"User-Agent": "Mozilla/5.0 (research scraper)"})

    all_rows = []
    for year in seasons:
        try:
            rows = scrape_season(year, session)
            print(f"  {year}: {len(rows)} teams")
            all_rows.extend(rows)
        except Exception as e:
            print(f"  {year}: ERROR — {e}")
        time.sleep(1.0)

    totals = pd.DataFrame(all_rows).sort_values(["season", "team"]).reset_index(drop=True)
    os.makedirs(DATA_DIR, exist_ok=True)
    _safe_to_parquet(totals, cache_path)
    print(f"Cached {len(totals):,} win total rows to {cache_path}")
    return totals


# (loader function, needs `seasons` arg) — every raw source fetch_all() pulls.
ALL_LOADERS = [
    (get_pbp_data, True),
    (get_schedule_data, True),
    (get_qbr_data, True),
    (get_roster_data, True),
    (get_snap_counts, True),
    (get_pfr_advstats, True),
    (get_weekly_data, True),
    (get_seasonal_data, True),
    (get_seasonal_rosters, True),
    (get_weekly_rosters, True),
    (get_depth_charts, True),
    (get_injuries, True),
    (get_ngs_data, True),
    (get_ftn_data, True),
    (get_seasonal_pfr, True),
    (get_combine_data, True),
    (get_draft_picks, True),
    (get_draft_values, False),
    (get_contracts, False),
    (get_officials, True),
    (get_team_desc, False),
    (get_ids, False),
    (get_sc_lines, True),
    (get_win_totals, True),
]


def fetch_all(seasons: list[int], refresh: bool = False) -> dict[str, pd.DataFrame]:
    """Call every loader in this module, caching each source to its own parquet file.

    Returns a dict of {function name: DataFrame} for whatever succeeded.
    A single source failing (e.g. an nflverse endpoint being down) doesn't
    stop the rest from being fetched.
    """
    results = {}
    for loader, needs_seasons in ALL_LOADERS:
        name = loader.__name__
        print(f"=== {name} ===")
        try:
            df = loader(seasons, refresh=refresh) if needs_seasons else loader(refresh=refresh)
            results[name] = df
            print(f"{df.shape[0]:,} rows, {df.shape[1]} columns\n")
        except Exception as e:
            print(f"  skipped ({e})\n")
    return results


if __name__ == "__main__":
    from config import SEASONS

    fetch_all(SEASONS)
