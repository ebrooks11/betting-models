"""Materialize every data/*.parquet file as a table in a DuckDB database
for ad hoc SQL exploration (e.g. via DBeaver).

Tables are dropped and rebuilt from the parquet files each run, so this is
safe to re-run any time data_loader.py refreshes the underlying data.

Usage:
    python3 pipeline/build_duckdb.py
"""

from pathlib import Path

import duckdb

DATA_DIR = Path(__file__).resolve().parent.parent / "data"
DB_PATH = DATA_DIR / "nfl.duckdb"


def build_duckdb() -> Path:
    """(Re)build data/nfl.duckdb with one table per data/*.parquet file.
    Requires an exclusive lock on the db file — fails if e.g. DBeaver is
    currently connected to it.
    """
    con = duckdb.connect(str(DB_PATH))

    for path in sorted(DATA_DIR.glob("*.parquet")):
        table = path.stem
        con.execute(f"CREATE OR REPLACE TABLE {table} AS SELECT * FROM read_parquet('{path}')")
        count = con.execute(f"SELECT count(*) FROM {table}").fetchone()[0]
        print(f"  {table}: {count:,} rows")

    con.close()
    print(f"\nBuilt {DB_PATH}")
    return DB_PATH


if __name__ == "__main__":
    build_duckdb()
