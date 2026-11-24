"""Build every chart and the findings summary: ``python -m wb.report`` (add ``--refresh`` for the latest data)."""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402

from . import analysis as A  # noqa: E402
from .data import LABELS, build_panel  # noqa: E402

NAVY = "#1F3864"
plt.rcParams.update({"figure.dpi": 140, "axes.spines.top": False, "axes.spines.right": False})


def fig_regions(panel, out):
    t = A.regional_trends(panel, "poverty_300").interpolate(limit_area="inside")
    fig, ax = plt.subplots(figsize=(9, 4.8))
    for col in t.columns:
        ax.plot(t.index, t[col], lw=2.6 if col == "World" else 1.6, color="black" if col == "World" else None,
                label=col.replace(", Afghanistan & Pakistan", "*"))
    ax.set(title="Extreme poverty by region, 1990-2024", ylabel="% below $3.00 a day (2021 PPP)")
    ax.legend(fontsize=8, frameon=False)
    ax.text(0, -0.16, "* including Afghanistan and Pakistan.  Source: World Bank, World Development Indicators.",
            transform=ax.transAxes, fontsize=7, color="grey")
    fig.tight_layout(); fig.savefig(out / "regional_extreme_poverty.png"); plt.close(fig)
    return t


def fig_country(panel, out, iso3="EGY", name="Egypt"):
    d = A.country_story(panel, iso3)
    fig, ax = plt.subplots(figsize=(9, 4.4))
    for col, colour in [("poverty_830", NAVY), ("poverty_420", "#4472C4"), ("poverty_300", "#9DC3E6")]:
        ax.plot(d.index, d[col], marker="o", color=colour, label=LABELS[col])
    nat = d["poverty_national"].dropna()
    if len(nat):
        ax.scatter(nat.index, nat, marker="D", color="#C00000", zorder=5, label=LABELS["poverty_national"])
    ax.set(title=f"{name}: share of people below each poverty line", ylabel="% of population", ylim=(0, 100))
    ax.legend(fontsize=8, frameon=False, loc="upper right")
    fig.tight_layout(); fig.savefig(out / f"{iso3.lower()}_poverty_lines.png"); plt.close(fig)
    return d


def fig_income(res, out, highlight=("EGY",)):
    t = res["table"]
    fig, ax = plt.subplots(figsize=(9, 5.2))
    ax.scatter(t["gdp_pc_ppp"], t["poverty_830"], s=18, color="#9DC3E6", edgecolor="none")
    order = np.argsort(t["gdp_pc_ppp"].to_numpy())
    ax.plot(t["gdp_pc_ppp"].to_numpy()[order], t["predicted_poverty_830"].to_numpy()[order], color=NAVY, lw=2,
            label=f"Fitted on log income (R² {res['r2_log_odds']:.2f}, log-odds scale)")
    labelled = set(t.nlargest(4, "residual")["iso3"]) | set(t.nsmallest(3, "residual")["iso3"]) | set(highlight)
    for _, r in t[t["iso3"].isin(labelled)].iterrows():
        bold = r["iso3"] in highlight
        ax.annotate(r["name"].split(",")[0], (r["gdp_pc_ppp"], r["poverty_830"]), fontsize=8 if not bold else 9,
                    fontweight="bold" if bold else None, color="#C00000" if bold else "black",
                    xytext=(4, 3), textcoords="offset points")
    ax.set_xscale("log")
    ax.set(title="Poverty versus average income (latest year per country)",
           xlabel="GDP per capita, PPP (constant 2021 $, log scale)", ylabel="% below $8.30 a day")
    ax.legend(fontsize=8, frameon=False)
    fig.tight_layout(); fig.savefig(out / "poverty_vs_income.png"); plt.close(fig)


def fig_clusters(res, out):
    prof = res["profile"]
    fig, ax = plt.subplots(figsize=(9, 5.2))
    for g, d in prof.groupby("cluster"):
        ax.scatter(d["gdp_pc_ppp"], d["poverty_830"], s=22, label=f"{g} ({len(d)} countries)")
    egy = prof[prof["iso3"] == "EGY"]
    if len(egy):
        ax.annotate("Egypt", (egy["gdp_pc_ppp"].iloc[0], egy["poverty_830"].iloc[0]), color="#C00000",
                    fontweight="bold", xytext=(5, 4), textcoords="offset points")
    ax.set_xscale("log")
    ax.set(title=f"Country groups from k-means on poverty and inequality (k = {res['k']})",
           xlabel="GDP per capita, PPP (log scale)", ylabel="% below $8.30 a day")
    ax.legend(fontsize=8, frameon=False)
    fig.tight_layout(); fig.savefig(out / "country_clusters.png"); plt.close(fig)


def main(argv=None) -> dict:
    ap = argparse.ArgumentParser(description="Generate poverty analysis charts and findings.")
    ap.add_argument("--refresh", action="store_true", help="download the latest data from the World Bank")
    ap.add_argument("--out", default="reports")
    args = ap.parse_args(argv)
    out = Path(args.out); out.mkdir(parents=True, exist_ok=True)

    panel = build_panel(refresh=args.refresh)
    regions = fig_regions(panel, out)
    egypt = fig_country(panel, out)
    income = A.poverty_vs_income(panel)
    fig_income(income, out)
    clusters = A.cluster_countries(panel)
    fig_clusters(clusters, out)

    t = income["table"].reset_index(drop=True)
    egy = t[t["iso3"] == "EGY"].iloc[0]
    summary = {
        "countries_in_panel": int(A.countries(panel)["iso3"].nunique()),
        "years": [int(panel["year"].min()), int(panel["year"].max())],
        "world_extreme_poverty": {str(y): round(float(regions.loc[y, "World"]), 1)
                                  for y in (1990, 2000, 2010, 2024) if y in regions.index and not np.isnan(regions.loc[y, "World"])},
        "income_model": {k: round(v, 3) for k, v in income.items() if k != "table" and isinstance(v, float)} | {"n_countries": income["n_countries"]},
        "egypt": {"poverty_830_latest": float(egypt["poverty_830"].iloc[-1]), "latest_year": int(egypt.index[-1]),
                  "predicted_from_income": round(float(egy["predicted_poverty_830"]), 1),
                  "excess_poverty_rank": int((t["residual"] > egy["residual"]).sum()) + 1,
                  "gini_latest": float(egypt["gini"].iloc[-1])},
        "largest_excess_poverty": t.nlargest(5, "residual")["name"].tolist(),
        "clusters": {"k": clusters["k"], "silhouette": {str(k): round(v, 3) for k, v in clusters["silhouette"].items()},
                     "sizes": clusters["sizes"], "medians": clusters["centres"].to_dict(orient="index")},
    }
    (out / "summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    clusters["profile"].sort_values(["cluster", "name"]).to_csv(out / "country_clusters.csv", index=False)
    t[["iso3", "name", "region", "gdp_pc_ppp", "poverty_830", "predicted_poverty_830", "residual", "gini"]].round(2) \
        .sort_values("residual", ascending=False).to_csv(out / "poverty_vs_income.csv", index=False)
    print(json.dumps(summary, indent=2))
    return summary


if __name__ == "__main__":
    main()
