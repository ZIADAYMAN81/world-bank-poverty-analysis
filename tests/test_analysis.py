from pathlib import Path

import numpy as np
import pandas as pd

from wb import analysis as A

DATA = Path(__file__).resolve().parents[1] / "data" / "raw"


def toy_panel():
    rng = np.random.default_rng(0)
    rows = []
    for i in range(60):
        gdp = float(np.exp(rng.uniform(7, 11)))
        p830 = float(100 / (1 + np.exp(1.5 * (np.log(gdp) - 9.3))))
        for year in (2015, 2019):
            rows.append({"iso3": f"C{i:02d}", "name": f"Country {i}", "region": "R", "income_group": "G",
                         "is_aggregate": False, "year": year, "gdp_pc_ppp": gdp, "poverty_830": p830,
                         "poverty_300": p830 / 3, "gini": float(rng.uniform(25, 55)),
                         "share_bottom20": float(rng.uniform(4, 9))})
    rows.append({"iso3": "WLD", "name": "World", "region": "Aggregates", "income_group": "Aggregates",
                 "is_aggregate": True, "year": 2019, "poverty_300": 10.0})
    return pd.DataFrame(rows)


def test_latest_picks_most_recent_year_and_skips_aggregates():
    out = A.latest(toy_panel(), "poverty_830")
    assert set(out["year"]) == {2019}
    assert "WLD" not in set(out["iso3"])


def test_income_model_fits_a_clean_relationship():
    res = A.poverty_vs_income(toy_panel())
    assert res["n_countries"] == 60
    assert res["r2_log_odds"] > 0.95
    assert res["spearman_gdp_vs_poverty"] < -0.9


def test_clusters_cover_every_country():
    res = A.cluster_countries(toy_panel())
    assert sum(res["sizes"].values()) == 60
    assert res["k"] in range(3, 7)


def test_regional_trends_uses_aggregate_rows():
    t = A.regional_trends(toy_panel(), regions=["World"])
    assert t.loc[2019, "World"] == 10.0


def test_dashboard_runs_without_errors():
    if not (DATA / "countries.csv").exists():
        import pytest
        pytest.skip("run `python -m wb.report` once to download the data")
    from streamlit.testing.v1 import AppTest

    at = AppTest.from_file(str(Path(__file__).resolve().parents[1] / "dashboard" / "app.py"), default_timeout=120)
    at.run()
    assert not at.exception
    assert at.title[0].value == "World Poverty and Inequality Dashboard"
