# Market Relationship Project

An **educational, paper-trading research dashboard** based on the themes of your three DS 340W parent papers. It forecasts a published market index **directly**, explores a dynamic network of sample constituent stocks, and generates a **MinT-reconciled sample-basket forecast**. It refreshes while the app is open. It **does not** send orders or claim a live, validated alpha strategy.

## Quick start on Mac / VS Code

1. Extract `DS-340W-Project.zip`.
2. Open Terminal (or VS Code → Terminal → New Terminal) in the extracted `DS-340W-Project` folder.
3. Enter:

```bash
python3 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
streamlit run app.py
```

4. Visit the local address shown by Streamlit (usually `http://localhost:8501`).
5. **First**, choose **Simulated demo** in the sidebar to ensure the full model runs without an internet connection.
6. Then switch to **Yahoo Finance** for actual market-history retrieval. Yahoo's quote and chart data can be delayed, missing, or revised. If the provider is unavailable, the app shows an error; it never silently substitutes simulated prices.

Alternative, fast Python-only check:

```bash
python sample_run.py
python -m pytest -q
```

> If `python` does not point to the virtual environment, use `python3` for commands outside the environment.

## Where the code lives

| File | Purpose |
| --- | --- |
| `app.py` | Streamlit dashboard: index chart, forecasts, relationship network, cluster tables, walk-forward backtest, research caveats. |
| `market_data.py` | Yahoo daily OHLC **close** retrieval, optional intraday preview, explicitly simulated example data and index presets. |
| `forecasting.py` | Rolling graph features, K-means segmentation, direct benchmark Ridge/Random Forest forecasting, stock forecasting, MinT-style basket coherence, true chronological direct-index backtest. |
| `sample_run.py` | Offline CLI demo showing all core computations without the UI. |
| `tests/test_smoke.py` | Deterministic no-network tests for main outputs and MinT consistency. |

## Reading the dashboard

**Published benchmark** (e.g. S&P 500 `^GSPC`) is forecast *directly*, using its historical returns plus cross-sectional signals calculated from a **sample stock basket**.

**Selected sample basket** is **not the actual underlying composition or official weighting of the index**. The basket's members are user-editable, equal-weighted and normalized to 100 at the sample start. We estimate its one-day forecast both as a standalone series and through stock-level forecasts reconciled with cluster aggregates.

**Index scenario** is an optional user-controlled blend of (1) the direct benchmark forecast and (2) the MinT-reconciled sample-basket return forecast. Blending is an **experimental heuristic**, not a formally reconciled exact official index forecast. The default uses 75% direct index model and 25% basket proxy. A 5- or 10-day scenario compounds the **same one-day estimate**; it is not a separately trained multi-horizon prediction.

**Trading insight** is a descriptive *scenario* (upward/downward/no clear direction) with a rough uncertainty band derived from recent forecast errors. It is **not a probability-calibrated buy/sell recommendation**. When zero lies inside the band, the dashboard marks the scenario low conviction.

## The three paper connections — clearly separated from full replication

1. **Stawarz (2025), global stock market development:** K-means grouping and interpretation of market structures inspire our rolling stock-level clusters. We show cluster descriptors and index-model feature contributions. **We do not reproduce the original paper's exchange data, Random Forest cluster classification or SHAP calculations in v0.1.**
2. **Mattera et al. (2024), clustering + forecast reconciliation:** A hierarchy with a top equal-weighted basket, K-means-based cluster contributions, and constituent levels is generated. Separate base forecasts are reconciled by a **MinT-style projection** `S(S'W⁻¹S)⁻¹S'W⁻¹ŷ`; `W` uses covariance shrinkage of historical held-out forecast errors. This is a **new Python implementation adapted to the sample basket**, not a reproduction of their original R experiments or exact full-index weights.
3. **Xiang et al. (2022), THGNN:** On each historical day, rolling signed correlations form positive/negative edges, and neighbor momentum becomes an input to a pooled stock-return Ridge model. The graph visualizes these relationships. This is a **lightweight dynamic relational proxy, not the paper's trained Transformer + heterogeneous GNN**. It does not contain graph-attention weights. A full THGNN integration is future work.

Paper / official replication links:

- [Stawarz 2025 paper](https://journals.plos.org/plosone/article?id=10.1371/journal.pone.0326809)
- [Mattera et al. 2024 source and R replication](https://github.com/raffmattera/stockprices_reco)
- [Xiang et al. 2022 THGNN official implementation](https://github.com/TongjiFinLab/THGNN)

## Validation boundaries

- **Direct-index backtest:** Last ~65 complete trading sessions. At each date `t`, model fitting uses feature rows up to `t-1` and therefore only return labels known by close of `t`. Each test prediction targets the **next daily** return. The model refits every five sessions. Compare walk-forward MAE/RMSE to a random-walk (zero-return) benchmark. Future-leaking `train_test_split(shuffle=True)` is **not** used.
- **Graph signals:** Computed using return windows ending at observation date, never future returns. The latest K-means segmentation is used to define the **current** sample-basket hierarchy.
- **MinT sample-basket calibration:** The covariance estimator uses a 45-observation held-out historical prediction period to calibrate the current hierarchy. This is an *algebraic, current-structure forecast*, **not a validated trading strategy**.
- **Important:** Reported walk-forward test scores cover **direct-index Ridge and Random Forest only**, **not** the blended or MinT stock-graph strategy. Before claiming scientific superiority, you must implement a comparable full walk-forward graph + reconciliation backtest, including cluster recomputation at every historical cutoff.
- Approximate 95% bands assume residual scale stability; these bands are **not empirically calibrated** for market tails.
- No trading fees, bid/ask spreads, borrow costs, short restrictions, or liquidity constraints are modeled. No real-money execution.

## How it refreshes

- While a Streamlit session is active, the dashboard fragment reruns every **120 seconds** (`st.fragment(run_every="120s")`).
- Daily historical data are cached for **300 seconds** and model outputs for **900 seconds**. They are recomputed after cache expiry or a sidebar setting change.
- An optional provider intraday **display only** uses a 5-minute series cached for 120 seconds. It is **not used as a training signal** for the daily forecasting model.
- **Continuous 24/7 service** needs separately hosted market ingestion/model jobs with persisted data, monitoring, licensed live feeds and appropriate market-hours awareness. Browser auto-refresh alone is not background service.

## Next genuine research milestones

1. Reproduce each parent's relevant published experiment (or document why you cannot), including Mattera's R replication and THGNN's Python reference workflow.
2. Build a **fully walk-forward graph + MinT backtest** with market-consistent constituents/weights and compare against equal-compute baselines.
3. Implement the full THGNN temporal encoder/graph attention (or another explicitly specified architecture) and forecast **numeric returns**, not only direction probabilities.
4. Add model-level interpretability (such as SHAP for Tree models), component ablations, uncertainty calibration and risk-adjusted transaction-cost-aware paper trading.
5. Only then consider authenticated market streams, persistent services or user portfolios.

## Practical notes

- Data is downloaded only when the app is running. It is not a streaming quote subscription. `yfinance` is a free experimental data source and may impose rate limits or change its API.
- Sample tickers **are not certified as current constituents** of the selected index and **do not reproduce capitalization/price weighting**. The official index itself comes from its own benchmark history.
- In demo mode **EVERY chart, result and backtest number is simulated** (random but deterministic). The demo should be used to test layout and code, never as market evidence.
- No part of the tool is investment advice or a reliable prediction of future prices.
