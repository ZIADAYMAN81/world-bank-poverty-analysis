"""Interactive poverty dashboard: ``streamlit run dashboard/app.py``."""
from __future__ import annotations

import sys
from pathlib import Path

import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from wb import analysis as A  # noqa: E402
from wb.data import LABELS, build_panel  # noqa: E402

st.set_page_config(page_title="World Poverty Dashboard", layout="wide")


@st.cache_data(show_spinner="Loading World Bank data...")
def load() -> pd.DataFrame:
    return build_panel(cache_dir=Path(__file__).resolve().parents[1] / "data" / "raw")


panel = load()
countries = A.countries(panel)
indicators = [c for c in ("poverty_300", "poverty_420", "poverty_830", "poverty_national", "gini", "share_bottom20")
              if c in panel.columns]

st.title("World Poverty and Inequality Dashboard")
st.caption("Source: World Bank, World Development Indicators (poverty lines in 2021 PPP dollars).")

indicator = st.sidebar.selectbox("Indicator", indicators, format_func=LABELS.get)
since = st.sidebar.slider("Use surveys from this year onwards", 2000, 2022, 2014)

latest = A.latest(panel, indicator, since)
col1, col2, col3 = st.columns(3)
col1.metric("Countries with data", f"{latest['iso3'].nunique()}")
col2.metric("Median country", f"{latest[indicator].median():.1f}")
world = panel[(panel["iso3"] == "WLD")].dropna(subset=[indicator])
if len(world):
    col3.metric(f"World ({int(world['year'].iloc[-1])})", f"{world[indicator].iloc[-1]:.1f}")

st.subheader("Latest value by country")
fig = px.choropleth(latest, locations="iso3", color=indicator, hover_name="name",
                    hover_data={"year": True, "iso3": False}, color_continuous_scale="Blues",
                    labels={indicator: LABELS[indicator]})
fig.update_layout(margin=dict(l=0, r=0, t=0, b=0), height=430)
st.plotly_chart(fig, use_container_width=True)

st.subheader("Compare countries over time")
names = sorted(countries["name"].dropna().unique())
default = [n for n in ("Egypt, Arab Rep.", "India", "Brazil", "Nigeria") if n in names]
chosen = st.multiselect("Countries", names, default=default)
trend = countries[countries["name"].isin(chosen)].dropna(subset=[indicator])
if len(trend):
    line = go.Figure([go.Scatter(x=d["year"], y=d[indicator], mode="lines+markers", name=str(n))
                      for n, d in trend.groupby("name")])
    line.update_layout(yaxis_title=LABELS[indicator], margin=dict(l=0, r=0, t=10, b=0), height=380)
    st.plotly_chart(line, use_container_width=True)
else:
    st.info("No data for the selected countries and indicator.")

st.subheader("Country groups (k-means on poverty and inequality)")
clusters = A.cluster_countries(panel, since=since)
st.dataframe(clusters["centres"].rename(columns=LABELS), use_container_width=True)
st.dataframe(clusters["profile"][["name", "region", "cluster"]].sort_values(["cluster", "name"]),
             use_container_width=True, height=300, hide_index=True)
