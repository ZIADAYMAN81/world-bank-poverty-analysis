"""Download World Development Indicators from the public World Bank API, with a local cache.

No API key is needed. Each indicator is cached as a CSV under ``data/raw`` so analyses are
reproducible offline; pass ``refresh=True`` to download the latest release again.
"""
from __future__ import annotations

import time
from pathlib import Path
from typing import Dict, Iterable

import pandas as pd
import requests

API = "https://api.worldbank.org/v2"

INDICATORS: Dict[str, str] = {
    "SI.POV.DDAY": "poverty_300",        # % living on less than $3.00 a day (2021 PPP), extreme poverty
    "SI.POV.LMIC": "poverty_420",        # % below $4.20 a day (lower-middle-income line)
    "SI.POV.UMIC": "poverty_830",        # % below $8.30 a day (upper-middle-income line)
    "SI.POV.NAHC": "poverty_national",   # % below the national poverty line
    "SI.POV.GINI": "gini",               # Gini index (0 = equal, 100 = maximally unequal)
    "SI.DST.FRST.20": "share_bottom20",  # income share held by the poorest 20%
    "NY.GDP.PCAP.PP.KD": "gdp_pc_ppp",   # GDP per capita, PPP (constant 2021 international $)
    "SP.POP.TOTL": "population",
}
LABELS = {
    "poverty_300": "Extreme poverty, below $3.00/day (%)",
    "poverty_420": "Below $4.20/day (%)",
    "poverty_830": "Below $8.30/day (%)",
    "poverty_national": "Below national poverty line (%)",
    "gini": "Gini index",
    "share_bottom20": "Income share of poorest 20% (%)",
    "gdp_pc_ppp": "GDP per capita (2021 PPP $)",
    "population": "Population",
}


def _get(url: str, params: dict, retries: int = 4) -> list:
    for attempt in range(retries):
        try:
            r = requests.get(url, params={**params, "format": "json"}, timeout=60)
            r.raise_for_status()
            payload = r.json()
            if len(payload) < 2:
                raise ValueError(f"unexpected response: {payload}")
            return payload
        except (requests.RequestException, ValueError):
            if attempt == retries - 1:
                raise
            time.sleep(2 ** attempt)
    return []


def fetch_indicator(code: str, start: int = 1990, end: int = 2025, cache_dir: Path = Path("data/raw"),
                    refresh: bool = False) -> pd.DataFrame:
    """Long table: iso3, year, value for every country and aggregate."""
    cache = Path(cache_dir) / f"{code}.csv"
    if cache.exists() and not refresh:
        return pd.read_csv(cache)
    meta, rows = _get(f"{API}/country/all/indicator/{code}", {"date": f"{start}:{end}", "per_page": 20000})
    df = pd.DataFrame([{"iso3": r["countryiso3code"], "year": int(r["date"]), "value": r["value"]}
                       for r in rows if r["countryiso3code"]])
    df.attrs["last_updated"] = meta.get("lastupdated")
    cache.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(cache, index=False)
    return df


def fetch_countries(cache_dir: Path = Path("data/raw"), refresh: bool = False) -> pd.DataFrame:
    """Country metadata: name, region, income group, and whether the row is an aggregate (e.g. 'World')."""
    cache = Path(cache_dir) / "countries.csv"
    if cache.exists() and not refresh:
        df = pd.read_csv(cache, keep_default_na=False)
        for col in ("name", "region", "income_group"):
            df[col] = df[col].str.strip()
        return df
    _, rows = _get(f"{API}/country", {"per_page": 400})
    # The API returns some names with trailing spaces (e.g. "Sub-Saharan Africa "), so strip them.
    df = pd.DataFrame([{"iso3": r["id"], "name": r["name"].strip(), "region": r["region"]["value"].strip(),
                        "income_group": r["incomeLevel"]["value"].strip(),
                        "is_aggregate": r["region"]["value"] == "Aggregates"} for r in rows])
    cache.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(cache, index=False)
    return df


def build_panel(codes: Iterable[str] = INDICATORS, start: int = 1990, end: int = 2025,
                cache_dir: Path = Path("data/raw"), refresh: bool = False) -> pd.DataFrame:
    """Wide country-year panel with one column per indicator, joined to country metadata."""
    frames = []
    for code in codes:
        d = fetch_indicator(code, start, end, cache_dir, refresh).dropna(subset=["value"])
        frames.append(d.assign(indicator=INDICATORS[code]))
    long = pd.concat(frames, ignore_index=True)
    wide = long.pivot_table(index=["iso3", "year"], columns="indicator", values="value").reset_index()
    wide.columns.name = None
    return wide.merge(fetch_countries(cache_dir, refresh), on="iso3", how="left")
