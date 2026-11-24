"""Analyses on the country-year panel built by ``wb.data.build_panel``."""
from __future__ import annotations

from typing import Dict, List

import numpy as np
import pandas as pd
from scipy.stats import spearmanr
from sklearn.cluster import KMeans
from sklearn.linear_model import LinearRegression
from sklearn.metrics import silhouette_score
from sklearn.preprocessing import StandardScaler

REGIONS = ["Sub-Saharan Africa", "South Asia", "East Asia & Pacific",
           "Middle East, North Africa, Afghanistan & Pakistan", "Latin America & Caribbean",
           "Europe & Central Asia", "World"]


def countries(panel: pd.DataFrame) -> pd.DataFrame:
    return panel[panel["is_aggregate"] == False]  # noqa: E712  (column may hold numpy bools)


def latest(panel: pd.DataFrame, column: str, since: int = 2014) -> pd.DataFrame:
    """Most recent non-missing value per country since ``since`` (survey years differ by country)."""
    d = countries(panel).dropna(subset=[column])
    d = d[d["year"] >= since].sort_values("year")
    return d.groupby("iso3").tail(1)[["iso3", "name", "region", "income_group", "year", column]]


def regional_trends(panel: pd.DataFrame, column: str = "poverty_300", regions: List[str] = REGIONS) -> pd.DataFrame:
    """World Bank's own regional aggregates (population-weighted and gap-filled by the Bank)."""
    agg = panel[panel["name"].isin(regions)]
    return agg.pivot_table(index="year", columns="name", values=column).reindex(columns=[r for r in regions if r in agg["name"].unique()])


def latest_profile(panel: pd.DataFrame, columns: List[str], since: int = 2014) -> pd.DataFrame:
    """One row per country with the latest value of each column (each may come from a different year)."""
    base = None
    for c in columns:
        part = latest(panel, c, since)[["iso3", "name", "region", "income_group", c]]
        base = part if base is None else base.merge(part[["iso3", c]], on="iso3", how="inner")
    return base.reset_index(drop=True)


def poverty_vs_income(panel: pd.DataFrame, since: int = 2014) -> Dict:
    """How well does average income explain the $8.30/day poverty rate, and who deviates most?

    Poverty is modelled on log GDP per capita (log-odds of the poverty rate, so predictions stay
    between 0 and 100%). Large positive residuals mean more poverty than income alone predicts,
    which points to inequality or how growth is shared.
    """
    prof = latest_profile(panel, ["poverty_830", "gdp_pc_ppp", "gini"], since)
    p = prof["poverty_830"].clip(0.5, 99.5) / 100
    y = np.log(p / (1 - p))
    X = np.log(prof[["gdp_pc_ppp"]])
    model = LinearRegression().fit(X, y)
    pred = 100 / (1 + np.exp(-model.predict(X)))
    prof = prof.assign(predicted_poverty_830=pred, residual=prof["poverty_830"] - pred)
    rho_gini, p_gini = spearmanr(prof["residual"], prof["gini"])
    return {
        "n_countries": int(len(prof)),
        "r2_log_odds": float(model.score(X, y)),
        "spearman_gdp_vs_poverty": float(spearmanr(prof["gdp_pc_ppp"], prof["poverty_830"])[0]),
        "spearman_residual_vs_gini": float(rho_gini), "p_value_residual_vs_gini": float(p_gini),
        "table": prof.sort_values("residual"),
    }


def cluster_countries(panel: pd.DataFrame, k_range=range(3, 7), since: int = 2014, seed: int = 42) -> Dict:
    """Group countries by poverty and inequality profile with k-means; k chosen by silhouette score.

    k starts at 3 because two clusters only separate rich from poor countries, which adds nothing
    beyond GDP per capita.
    """
    feats = ["poverty_300", "poverty_830", "gini", "share_bottom20", "gdp_pc_ppp"]
    prof = latest_profile(panel, feats, since)
    X = prof[feats].copy()
    X["gdp_pc_ppp"] = np.log(X["gdp_pc_ppp"])
    Xs = StandardScaler().fit_transform(X)
    scores = {k: float(silhouette_score(Xs, KMeans(k, n_init=20, random_state=seed).fit_predict(Xs))) for k in k_range}
    best_k = max(scores, key=scores.get)
    labels = KMeans(best_k, n_init=20, random_state=seed).fit_predict(Xs)
    prof = prof.assign(cluster=labels)
    centres = prof.groupby("cluster")[feats].median().round(1)
    order = centres["poverty_830"].sort_values().index          # least to most poor
    names = {c: f"Group {i + 1}" for i, c in enumerate(order)}
    if best_k == 3:  # descriptive names: poorest group, then the two others split by inequality
        poorest = order[-1]
        rest = centres.drop(index=poorest)["gini"].sort_values().index
        names = {poorest: "High poverty", rest[0]: "Lower inequality", rest[1]: "High inequality"}
    prof["cluster"] = prof["cluster"].map(names)
    centres.index = centres.index.map(names)
    return {"k": best_k, "silhouette": scores, "profile": prof, "centres": centres.sort_index(),
            "sizes": prof["cluster"].value_counts().sort_index().to_dict()}


def country_story(panel: pd.DataFrame, iso3: str = "EGY") -> pd.DataFrame:
    cols = ["year", "poverty_300", "poverty_420", "poverty_830", "poverty_national", "gini", "share_bottom20"]
    d = panel[panel["iso3"] == iso3][cols]
    return d.dropna(subset=["poverty_830"]).set_index("year")
