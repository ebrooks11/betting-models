"""
Scrapes historical NFL offensive/defensive coordinators by team-season from
Wikipedia's per-team-season articles (e.g. "2023 San Francisco 49ers season").
src.data_loader.get_coordinators() calls scrape_team_season() below and caches
the combined result to data/coordinators.parquet, one row per team/season/role.

Wikipedia's season articles embed staff info two different ways depending on
when the article was last edited:
- Newer articles use a templated {{NFL final staff}} table with the coaching
  groups as raw wikitext (e.g. "* Offensive coordinator – [[Norv Turner]]")
  inside the page's data-mw attribute — parsed directly from that JSON rather
  than scraping rendered text, since it's stable across rendering changes.
- Older articles (seen as far forward as ~2010-2011) render the same
  information as plain HTML instead: a "Coaching staff" heading followed by
  a table with "<b>Category</b>" headers (e.g. "Defensive coaches") each
  followed by a "<ul><li>Role – Name</li></ul>" list. There's a fallback
  parser for this shape.

Notes / known limitations:
- When a head coach calls his own plays, there may be no titled "offensive
  coordinator" at all (e.g. Kyle Shanahan, 49ers) — that's a real football
  fact, not a scraping bug. Those team-seasons will have no OC row.
- Mid-season coordinator changes can produce two rows for one team-season.
- Role titles vary ("Offensive coordinator", "Co-offensive coordinator",
  "Passing game coordinator/quarterbacks coach", etc.) — anything containing
  "coordinator" in the offensive/defensive coaching block is kept, with the
  raw title preserved in role_raw alongside the OC/DC category.

Run locally (blocked in remote containers, like fetch_win_totals.py):
    python3 pipeline/fetch_coordinators.py
"""

import json
import re
import time

import requests
from bs4 import BeautifulSoup

# nflverse team abbreviation -> Wikipedia article name, with relocations/
# renames handled as (start_season, end_season_inclusive_or_None, name) spans.
STATIC_TEAM_NAMES = {
    "ARI": "Arizona Cardinals",
    "ATL": "Atlanta Falcons",
    "BAL": "Baltimore Ravens",
    "BUF": "Buffalo Bills",
    "CAR": "Carolina Panthers",
    "CHI": "Chicago Bears",
    "CIN": "Cincinnati Bengals",
    "CLE": "Cleveland Browns",
    "DAL": "Dallas Cowboys",
    "DEN": "Denver Broncos",
    "DET": "Detroit Lions",
    "GB": "Green Bay Packers",
    "HOU": "Houston Texans",
    "IND": "Indianapolis Colts",
    "JAX": "Jacksonville Jaguars",
    "KC": "Kansas City Chiefs",
    "MIA": "Miami Dolphins",
    "MIN": "Minnesota Vikings",
    "NE": "New England Patriots",
    "NO": "New Orleans Saints",
    "NYG": "New York Giants",
    "NYJ": "New York Jets",
    "PHI": "Philadelphia Eagles",
    "PIT": "Pittsburgh Steelers",
    "SF": "San Francisco 49ers",
    "SEA": "Seattle Seahawks",
    "TB": "Tampa Bay Buccaneers",
    "TEN": "Tennessee Titans",
}

VARIABLE_TEAM_NAMES = {
    "LV": [(0, 2019, "Oakland Raiders"), (2020, None, "Las Vegas Raiders")],
    "LAC": [(0, 2016, "San Diego Chargers"), (2017, None, "Los Angeles Chargers")],
    "LA": [(0, 2015, "St. Louis Rams"), (2016, None, "Los Angeles Rams")],
    "WAS": [
        (0, 2019, "Washington Redskins"),
        (2020, 2021, "Washington Football Team"),
        (2022, None, "Washington Commanders"),
    ],
}

ALL_TEAMS = sorted(set(STATIC_TEAM_NAMES) | set(VARIABLE_TEAM_NAMES))

BASE_URL = "https://en.wikipedia.org/wiki/{title}"


def _wiki_team_name(abbr: str, season: int) -> str:
    if abbr in VARIABLE_TEAM_NAMES:
        for start, end, name in VARIABLE_TEAM_NAMES[abbr]:
            if season >= start and (end is None or season <= end):
                return name
        raise ValueError(f"No Wikipedia name mapped for {abbr} {season}")
    return STATIC_TEAM_NAMES[abbr]


def _extract_names(text: str) -> str:
    """Pull display names out of wikitext, e.g. '[[Norv Turner]]' -> 'Norv Turner'
    or '[[Billy Davis (American football coach)|Bill Davis]]' -> 'Bill Davis'."""
    links = re.findall(r"\[\[([^\]|]+)(?:\|([^\]]+))?\]\]", text)
    if links:
        names = [display if display else target for target, display in links]
        return " & ".join(n.strip() for n in names)
    return re.sub(r"'{2,}", "", text).strip()


_PRIMARY_COORDINATOR = re.compile(r"^(co-)?(offensive|defensive) coordinator\b", re.I)
_ANY_COORDINATOR = re.compile(r"coordinator", re.I)


def _split_role_name(line: str):
    parts = re.split(r"\s[-–—]\s", line, maxsplit=1)
    if len(parts) != 2:
        return None
    return parts[0].strip(), parts[1].strip()


def _select_coordinator_rows(pairs: list[tuple[str, str]]):
    """pairs: (role_raw, name) tuples already resolved to plain names.

    Prefers an exact "(Co-)Offensive/Defensive Coordinator" title when one is
    present; only falls back to any title containing the word "coordinator"
    (e.g. "Passing game coordinator", "Run game coordinator") when no primary
    title exists — some head coaches call their own plays and only
    split-duty coordinator titles are used that season.
    """
    primary = [p for p in pairs if _PRIMARY_COORDINATOR.search(p[0])]
    return primary or [p for p in pairs if _ANY_COORDINATOR.search(p[0])]


def _parse_wikitext_block(wikitext: str):
    """Parse a {{NFL final staff}} coaching-group wikitext block (bullet list
    of '* Role – [[Name]]' lines) into (role_raw, name) pairs."""
    pairs = []
    for line in wikitext.splitlines():
        line = line.strip().lstrip("*").strip()
        if not line:
            continue
        split = _split_role_name(line)
        if split:
            role, name_wikitext = split
            pairs.append((role, _extract_names(name_wikitext)))
    return pairs


def _find_staff_template_params(soup: BeautifulSoup):
    """Newer articles: staff data lives inside a templated table's data-mw
    JSON attribute. Returns the template's params dict, or None if this
    article doesn't use the template (older articles render plain HTML
    instead — see _find_staff_html_table)."""
    for table in soup.find_all("table", attrs={"data-mw": True}):
        try:
            dmw = json.loads(table["data-mw"])
            tmpl_name = dmw["parts"][0]["template"]["target"]["wt"].strip().lower()
        except (KeyError, IndexError, ValueError, json.JSONDecodeError):
            continue
        if "staff" in tmpl_name:
            return dmw["parts"][0]["template"]["params"]
    return None


def _find_staff_html_table(soup: BeautifulSoup):
    """Older articles: no template, just a 'Staff'/'Coaching staff' heading
    followed by a plain HTML table. Returns that table, or None.

    Some articles have multiple headings containing "staff" — a compound
    section heading like "Staff and roster" that doesn't directly precede
    the table (it wraps other subsections first), or a prose heading like
    "Coaching and staff personnel changes" with no table at all. Try every
    "staff" heading, shortest text first (favors an exact "Staff" match over
    a compound one), until one actually leads to a table.
    """
    candidates = [
        tag for tag in soup.find_all(["h2", "h3", "h4"])
        if "staff" in tag.get_text(strip=True).lower()
        and "change" not in tag.get_text(strip=True).lower()
    ]
    candidates.sort(key=lambda tag: len(tag.get_text(strip=True)))

    for heading in candidates:
        parent = heading.parent
        anchor = parent if "mw-heading" in (parent.get("class") or []) else heading
        for sib in anchor.find_next_siblings():
            classes = sib.get("class") or []
            if "mw-heading" in classes:
                break
            if sib.name == "table":
                return sib
    return None


def _parse_html_staff_table(table) -> dict:
    """Returns {category_label: [(role_raw, name), ...]} for each
    '<b>Category</b>' + '<ul><li>Role – Name</li></ul>' block in the table.

    The <ul> isn't reliably a direct sibling of its <b> — some categories on
    the same page wrap the <b> in a <p>, making the <ul> a sibling of the <p>
    instead. find_next() (whole-document order, not just siblings) with a
    ['ul', 'b'] stop set handles both shapes: whichever comes first after a
    given <b> is either its own list, or the next category's label (empty
    list, correctly skipped).
    """
    blocks = {}
    for b in table.find_all("b"):
        nxt = b.find_next(["ul", "b"])
        if nxt is None or nxt.name != "ul":
            continue
        pairs = []
        for li in nxt.find_all("li", recursive=False):
            split = _split_role_name(li.get_text(" ", strip=True))
            if split:
                pairs.append(split)
        blocks[b.get_text(strip=True)] = pairs
    return blocks


def scrape_team_season(abbr: str, season: int, session: requests.Session) -> list[dict]:
    """Fetch one team's Wikipedia season article and extract OC/DC rows.
    Returns an empty list (with a printed reason) if the page or the
    expected staff info isn't found — callers should treat that as a
    skip, not a fatal error.
    """
    name = _wiki_team_name(abbr, season)
    title = f"{season}_{name.replace(' ', '_')}_season"
    resp = session.get(BASE_URL.format(title=title), timeout=15)
    if resp.status_code == 404:
        print(f"  {season} {abbr}: page not found ({title})")
        return []
    resp.raise_for_status()

    soup = BeautifulSoup(resp.text, "html.parser")

    blocks = {}  # {category_label: [(role_raw, name), ...]}
    staff_params = _find_staff_template_params(soup)
    if staff_params is not None:
        for key, val in staff_params.items():
            blocks[key] = _parse_wikitext_block(val.get("wt", ""))
    else:
        html_table = _find_staff_html_table(soup)
        if html_table is not None:
            blocks = _parse_html_staff_table(html_table)

    if not blocks:
        print(f"  {season} {abbr}: no staff table found")
        return []

    rows = []
    for key, pairs in blocks.items():
        key_lower = key.lower()
        if key_lower.startswith("offensive"):
            category = "OC"
        elif key_lower.startswith("defensive"):
            category = "DC"
        else:
            continue
        for role_raw, coach_name in _select_coordinator_rows(pairs):
            rows.append({
                "season": season,
                "team": abbr,
                "role_category": category,
                "role_raw": role_raw,
                "name": coach_name,
            })

    return rows


def main():
    from src.data_loader import get_coordinators
    from config import SEASONS

    df = get_coordinators(list(SEASONS), ALL_TEAMS, refresh=True)
    print(df.head(20).to_string(index=False))


if __name__ == "__main__":
    main()
