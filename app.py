"""Streamlit market forecasting dashboard."""
from __future__ import annotations

import re
from datetime import datetime, timezone

import networkx as nx
import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st

from forecasting import analyze
from market_data import INDEXES, get_prices, intraday_benchmark

st.set_page_config(page_title="Market Relationship Project", page_icon="M", layout="wide")
st.markdown("""
<style>
.block-container {padding-top: 1.5rem; padding-bottom: 3rem; max-width: 1440px;}
[data-testid="stMetric"] {border:1px solid rgba(128,128,128,.22); border-radius:12px; padding:14px;}
.small-header {font-size:.8rem;letter-spacing:.14em;text-transform:uppercase;color:#78d9cd;font-weight:700;}
</style>
""", unsafe_allow_html=True)

with st.sidebar:
    st.title("Market Relationship Project")
    index_name = st.selectbox("Tracked index", list(INDEXES), index=0)
    benchmark = INDEXES[index_name]["benchmark"]
    default_stocks = INDEXES[index_name]["constituents"]
    st.caption("Basket tickers are a research sample, not official up-to-date index constituent weights.")
    custom = st.text_area("Sample basket tickers (comma separated)", value=", ".join(default_stocks), height=104)
    members = [s.strip().upper() for s in custom.replace("\n", ",").split(",") if s.strip()]
    members = list(dict.fromkeys(s for s in members if s != benchmark))
    is_valid = all(re.fullmatch(r"[A-Z0-9^.-]+", s) for s in members) and 5 <= len(members) <= 18
    source = st.selectbox("Data source", ["Yahoo Finance", "Simulated demo"], index=0)
    method = st.selectbox("Direct index predictor", ["Ridge", "Random Forest"])
    correlation_window = st.slider("Rolling correlation window (days)", 25, 70, 40, 5)
    threshold = st.slider("Graph edge |correlation| threshold", 0.3, 0.9, 0.6, 0.05)
    blend = st.slider("Weight of reconciled basket in index scenario", 0.0, 0.5, 0.25, 0.05)
    horizon = st.selectbox("Scenario horizon (trading days)", [1, 5, 10], index=0)
    auto_refresh = st.toggle("Automatically refresh dashboard", value=True)
    if st.button("Refresh data now", use_container_width=True):
        st.cache_data.clear()
        st.rerun()
    st.caption("Automatic view refresh: every 120 s. Yahoo daily bars may update only after market close; this is not tick-level streaming.")
    st.divider()
    st.caption("Academic prototype • research and paper trading only. No broker orders. No guarantee of prediction accuracy.")

st.markdown('<div class="small-header">DS 340W • Three-paper research prototype</div>', unsafe_allow_html=True)
st.title("Market Relationship Project")
st.write("**Index forecasts, changing stock relationships, and interpretable risk scenarios.**")

if not is_valid:
    st.error("Enter between 5 and 18 unique valid ticker symbols, separated by commas.")
    st.stop()


@st.cache_data(ttl=300, show_spinner=False)
def get_data_cached(symbol: str, tickers: tuple[str, ...], mode: str):
    return get_prices(symbol, list(tickers), mode=mode)


@st.cache_data(ttl=900, show_spinner=False)
def get_analysis_cached(prices: pd.DataFrame, symbol: str, window: int, edge_threshold: float, predictor: str):
    return analyze(prices, symbol, correlation_window=window,
                   edge_threshold=edge_threshold, model_name=predictor)


@st.cache_data(ttl=120, show_spinner=False)
def get_intraday_cached(symbol: str) -> pd.DataFrame:
    return intraday_benchmark(symbol)


def pct(x: float) -> str:
    return f"{x * 100:+.2f}%"


def relation_graph(result):
    graph = nx.Graph()
    for stock in result.clusters.index:
        graph.add_node(stock)
    for row in result.graph_edges.itertuples(index=False):
        graph.add_edge(row.Source, row.Target, weight=abs(row.Correlation), corr=row.Correlation)
    pos = nx.spring_layout(graph, seed=19, k=1.05 / np.sqrt(max(len(graph), 1)))
    fig = go.Figure()
    for category, color in [("Positive", "#3fc5ad"), ("Negative", "#f09a73")]:
        group = result.graph_edges[result.graph_edges["Type"] == category]
        xs, ys = [], []
        for row in group.itertuples(index=False):
            xs.extend([pos[row.Source][0], pos[row.Target][0], None])
            ys.extend([pos[row.Source][1], pos[row.Target][1], None])
        if xs:
            fig.add_trace(go.Scatter(x=xs, y=ys, mode="lines", name=f"{category} relationship",
                                     line={"color": color, "width": 2.3}, hoverinfo="skip"))
    stocks = list(graph.nodes())
    cs = [int(result.clusters[s]) for s in stocks]
    fig.add_trace(go.Scatter(
        x=[pos[s][0] for s in stocks], y=[pos[s][1] for s in stocks],
        mode="markers+text", name="Stock", text=stocks, textposition="top center",
        textfont={"size": 11},
        marker={"size": 18, "color": cs, "colorscale": "Viridis", "showscale": True,
                "colorbar": {"title": "Cluster"}, "line": {"width": 1.7, "color": "#fff"}},
        customdata=np.array(cs).reshape(-1, 1),
        hovertemplate="%{text}<br>Cluster %{customdata[0]}<extra></extra>",
    ))
    fig.update_layout(height=510, margin=dict(l=10, r=10, t=30, b=10),
                      plot_bgcolor="rgba(0,0,0,0)",
                      showlegend=True, legend=dict(orientation="h", y=-0.1),
                      xaxis={"visible": False}, yaxis={"visible": False})
    return fig


def forecast_chart(res, daily_scenario: float, days: int):
    recent = res.prices[res.index_symbol].tail(140)
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=recent.index, y=recent.values, mode="lines", name="Observed index",
                             line={"color": "#47baa8", "width": 2.5}))
    future_dates = pd.bdate_range(recent.index[-1] + pd.offsets.BDay(1), periods=days)
    predictions = [float(recent.iloc[-1]) * (1 + daily_scenario)**k for k in range(1, days+1)]
    fig.add_trace(go.Scatter(x=[recent.index[-1]] + list(future_dates),
                             y=[float(recent.iloc[-1])] + predictions,
                             name="Model scenario (not actual)", mode="lines+markers",
                             line={"color": "#ef9c58", "dash": "dash", "width": 2}))
    fig.update_layout(height=350, margin=dict(l=8, r=8, t=20, b=5),
                      yaxis_title="Index level", legend=dict(orientation="h", y=-0.21),
                      hovermode="x unified")
    return fig


@st.fragment(run_every="120s" if auto_refresh else None)
def live_dashboard():
    with st.spinner("Loading prices, estimating relationships and validating forecasts..."):
        try:
            md = get_data_cached(benchmark, tuple(members), source)
            result = get_analysis_cached(md.prices, benchmark, correlation_window, threshold, method)
        except Exception as exc:
            st.error(f"Could not build the dashboard: {exc}")
            st.info("Try the **Simulated demo** data source to test the app offline, or use fewer tickers.")
            return

    if source == "Simulated demo":
        st.warning("SIMULATED DEMO DATA — All prices, clusters, forecasts and backtest scores in this mode are synthetic. Do not interpret as market facts.")
    else:
        st.info("Yahoo Finance research data may be delayed, incomplete or revised. Models use daily closing-price information, not a guaranteed real-time market feed.")
    if md.dropped:
        st.warning("Tickers unavailable and excluded: " + ", ".join(md.dropped))
    st.caption(f"Price history through **{result.asof_date}** | Requested source: {md.source} | Last fetched: {md.pulled_at_utc} | Sample stocks: {len(result.clusters)}")
    last_dt = pd.Timestamp(result.prices.index[-1]).date()
    stale_days = (datetime.now(timezone.utc).date() - last_dt).days
    if stale_days > 4 and source != "Simulated demo":
        st.error(f"Historical data is {stale_days} calendar days old — treat all signals as stale.")

    # The sample basket is not the published index.
    scenario_daily = (1-blend)*result.direct_forecast + blend*result.proxy_reconciled_return
    scenario_cumulative = (1+scenario_daily)**horizon - 1
    sigma = result.forecast_std * np.sqrt(horizon)
    upper = scenario_cumulative + 1.96*sigma
    lower = scenario_cumulative - 1.96*sigma
    lead = "Upward scenario" if scenario_cumulative > 0.002 else (
        "Downward scenario" if scenario_cumulative < -0.002 else "No clear direction"
    )
    weak = lower <= 0 <= upper

    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Official index last close", f"{result.index_last:,.2f}")
    c2.metric(f"{horizon}-day index scenario", pct(scenario_cumulative))
    c3.metric("Daily direct-index estimate", pct(result.direct_forecast))
    c4.metric("Relationship clusters", str(result.cluster_count))
    if weak:
        st.caption(f"**Signal: {lead} — low conviction.** A rough 95% scenario band spans {pct(lower)} to {pct(upper)}. "
                   "Intervals are historical-residual heuristics, not calibrated probabilities or executable trading instructions.")
    else:
        st.caption(f"**Signal: {lead}.** Approximate scenario band: {pct(lower)} to {pct(upper)}. "
                   "Use for research only; check risk, costs, and independent evidence before decisions.")

    overview, relation_tab, eval_tab, research_tab = st.tabs([
        "Index scenarios", "Dynamic graph & clusters", "Walk-forward validation", "Research design"
    ])
    with overview:
        st.subheader("Market history + future scenario")
        st.plotly_chart(forecast_chart(result, scenario_daily, horizon), use_container_width=True)
        st.caption("Dashed segment is a *hypothetical extrapolation* from a one-day estimate. "
                   "Longer horizons are not independently trained or validated.")
        x, y, z = st.columns(3)
        x.metric("Index-only prediction (1 day)", pct(result.direct_forecast))
        y.metric("Direct sample-basket estimate", pct(result.proxy_direct_return))
        z.metric("MinT-reconciled basket estimate", pct(result.proxy_reconciled_return))
        st.info("**Index vs. basket:** The official index uses market/price weights and potentially more stocks. "
                "Our basket is a selected equal-weighted proxy. MinT makes *that basket hierarchy* coherent; "
                "it does NOT guarantee mathematical coherence with the official published index. "
                "The displayed index scenario blends the index-only prediction with the basket return signal.")
        st.subheader("Constituent scenarios and clusters")
        st.dataframe(result.stock_forecasts, hide_index=True, use_container_width=True)
        if source == "Yahoo Finance":
            st.subheader("Intraday provider snapshot (optional)")
            try:
                day = get_intraday_cached(benchmark)
                if day.empty:
                    st.caption("No intraday observations are currently available from the provider.")
                else:
                    st.line_chart(day["Close"], height=220)
                    st.caption("Intraday chart is a provider snapshot and is not fed into the daily-close forecast model.")
            except Exception:
                st.caption("Intraday data unavailable. Daily forecasting and graph analysis remain usable.")

    with relation_tab:
        st.subheader("Positive and negative stock correlations")
        st.plotly_chart(relation_graph(result), use_container_width=True)
        st.caption(f"Edges show {correlation_window}-day correlations with |correlation| ≥ {threshold:.2f}. "
                   "Colors indicate statistical associations, NOT causal influence or a learned THGNN attention weight.")
        st.subheader("Current feature-based K-means clusters")
        st.dataframe(result.cluster_table, hide_index=True, use_container_width=True)
        left, right = st.columns([1.3, 1])
        with left:
            st.subheader("Correlation heatmap")
            heat = go.Figure(go.Heatmap(z=result.correlation.to_numpy(),
                                       x=list(result.correlation.columns), y=list(result.correlation.index),
                                       zmin=-1, zmax=1, colorscale="RdBu", reversescale=True,
                                       colorbar={"title": "Pearson r"}))
            heat.update_layout(height=430, margin=dict(l=10, r=10, t=15, b=10))
            st.plotly_chart(heat, use_container_width=True)
        with right:
            st.subheader("Strong connections")
            strongest = result.graph_edges.copy()
            if strongest.empty:
                st.caption("No edges met the threshold. Lower it in the sidebar.")
            else:
                strongest["Strength"] = strongest["Correlation"].abs()
                st.dataframe(strongest.sort_values("Strength", ascending=False)
                             .head(12).drop(columns="Strength"), hide_index=True, use_container_width=True)
        st.caption("Clusters are built from momentum, volatility, mean return, and basket beta. "
                   "Daily rolling-correlation signals also enter the graph-aware constituent forecast model.")

    with eval_tab:
        st.subheader("Strictly prior-information walk-forward backtest")
        st.caption("Each next-day prediction uses a model trained only on labels already observable at that date. "
                   "Training parameters refit every five test sessions. The table covers the most recent ~65 completed sessions.")
        st.dataframe(result.metric_table, hide_index=True, use_container_width=True)
        b = result.backtest.copy() * 100
        fig = go.Figure()
        for k, c in [("actual", "#a4abb4"), ("Ridge", "#3fc5ad"), ("Random Forest", "#f6aa6a")]:
            fig.add_trace(go.Scatter(x=b.index, y=b[k], name=k, mode="lines",
                                     line={"color": c, "width": 1.9}))
        fig.update_layout(height=320, yaxis_title="Next-day return (%)",
                          legend=dict(orientation="h", y=-0.2), margin=dict(l=8,r=8,t=15,b=5))
        st.plotly_chart(fig, use_container_width=True)
        st.caption("The reported backtest is for *direct-index Ridge/Random Forest models only*. "
                   "The dynamic graph + MinT blended index scenario is exploratory and is NOT covered by these backtest scores.")
        st.subheader("What drives the current index-only prediction?")
        st.dataframe(result.contributions, hide_index=True, use_container_width=True)
        st.caption("Ridge: standardized feature contributions to its one-day return prediction (not SHAP). "
                   "Random Forest: global impurity-based feature importance (not a local explanation or causal proof).")

    with research_tab:
        st.subheader("How your three parent papers contribute")
        st.markdown("""
**Paper 1 — Stawarz (2025):** K-means-driven market segmentation and an explanation-oriented research question. Here we implement *stock-level* K-means clusters and show feature contributions; SHAP is **not yet implemented**.

**Paper 2 — Mattera et al. (2024):** We use **MinT-style forecast reconciliation** with a summing matrix, shrinkage covariance and cluster hierarchy. It reconciles the selected **equal-weighted stock basket**, not the official S&P 500 / DJIA / Nasdaq-100.

**Paper 3 — Xiang et al. (2022):** We recalculate **signed correlation networks** over rolling historical windows and feed positive/negative neighbor signals into a pooled stock-return predictor. This is **THGNN-inspired**, not a trained Transformer or graph-attention neural network.
""")
        st.subheader("Research hypotheses to test next")
        st.markdown("""
1. Does graph-aware modeling reduce *out-of-sample* MAE/RMSE versus a direct-index and random-walk baseline?
2. Does **dynamic** correlation or clustering outperform a fixed stock structure when used for reconciliation?
3. Can full THGNN-style temporal and heterogeneous attention add value beyond this lightweight rolling-correlation proxy?
4. After realistic fees, slippage, and turnover, do forecasts improve **risk-adjusted** paper-trading performance?
""")
        st.code(f"Sample-basket reconciliation max coherence error: {result.reconciliation_check:.3e}", language="text")
        st.caption("This is an algebraic coherence check on the tracked equal-weighted basket, not evidence of forecasting skill.")


live_dashboard()
