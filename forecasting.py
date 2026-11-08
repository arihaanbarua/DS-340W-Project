"""Index and constituent forecasting models."""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd
from sklearn.cluster import KMeans
from sklearn.covariance import LedoitWolf
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import Ridge
from sklearn.metrics import silhouette_score
from sklearn.preprocessing import StandardScaler


STOCK_FACTORS = [
    "return_1d", "momentum_5d", "momentum_20d", "volatility_20d",
    "index_return", "peer_positive", "peer_negative", "signed_neighbor",
    "degree_fraction", "market_dispersion",
]
INDEX_FACTORS = [
    "index_return_1d", "index_momentum_5d", "index_momentum_20d",
    "index_volatility_20d", "stock_basket_momentum_5d",
    "stock_dispersion", "market_avg_signed_neighbor", "mean_abs_correlation",
]


@dataclass
class AnalysisResult:
    index_symbol: str
    asof_date: str
    prices: pd.DataFrame
    clusters: pd.Series
    cluster_table: pd.DataFrame
    graph_edges: pd.DataFrame
    correlation: pd.DataFrame
    stock_forecasts: pd.DataFrame
    proxy_direct_return: float
    proxy_reconciled_return: float
    direct_forecast: float
    direct_alternative: float
    forecast_std: float
    index_last: float
    contributions: pd.DataFrame
    backtest: pd.DataFrame
    metric_table: pd.DataFrame
    reconciliation_check: float
    model_name: str
    cluster_count: int


def infer_clusters(stock_prices: pd.DataFrame) -> tuple[pd.Series, pd.DataFrame]:
    returns = stock_prices.pct_change(fill_method=None).dropna()
    active = returns.tail(min(len(returns), 90))
    market = active.mean(axis=1)
    var = float(market.var())
    characteristics = pd.DataFrame(index=stock_prices.columns)
    characteristics["mean_return"] = active.mean(axis=0)
    characteristics["volatility"] = active.std(axis=0)
    characteristics["last_5d_momentum"] = active.tail(5).sum(axis=0)
    characteristics["beta_to_basket"] = [
        active[col].cov(market) / var if var > 1e-12 else 0.0
        for col in active.columns
    ]
    characteristics = characteristics.replace([np.inf, -np.inf], np.nan).fillna(0)
    scaled = StandardScaler().fit_transform(characteristics)
    n = len(stock_prices.columns)
    k_max = min(5, n - 1)
    best_score = -np.inf
    best_labels = None
    for k in range(2, k_max + 1):
        km = KMeans(n_clusters=k, n_init=10, random_state=42)
        labels = km.fit_predict(scaled)
        if len(set(labels)) <= 1:
            continue
        score = silhouette_score(scaled, labels)
        if score > best_score:
            best_score, best_labels = score, labels
    if best_labels is None:
        best_labels = np.arange(n) % 2
    assignments = pd.Series(best_labels + 1, index=stock_prices.columns, name="Cluster")
    characteristics["Cluster"] = assignments
    return assignments, characteristics


def mint_reconcile(base: np.ndarray, summing: np.ndarray, residuals: np.ndarray) -> np.ndarray:
    """Reconcile base forecasts using a shrinkage covariance."""
    y = np.asarray(base, dtype=float).reshape(-1)
    S = np.asarray(summing, dtype=float)
    E = np.asarray(residuals, dtype=float)
    if E.ndim != 2 or E.shape[0] < 10 or E.shape[1] != len(y):
        W = np.eye(len(y))
    else:
        W = LedoitWolf().fit(E).covariance_
    W += np.eye(len(y)) * max(1e-10, float(np.trace(W)) / len(y) * 1e-6)
    WinvS = np.linalg.solve(W, S)
    Winvy = np.linalg.solve(W, y)
    bottom = np.linalg.solve(S.T @ WinvS, S.T @ Winvy)
    return S @ bottom


def _ridge(X: np.ndarray, y: np.ndarray):
    scaler = StandardScaler().fit(X)
    model = Ridge(alpha=22.0).fit(scaler.transform(X), y)
    return model, scaler


def _rf(X: np.ndarray, y: np.ndarray):
    model = RandomForestRegressor(
        n_estimators=65, max_depth=6, min_samples_leaf=8, max_features=0.75,
        random_state=42, n_jobs=1,
    ).fit(X, y)
    return model, None


def _predict_pair(pair, X):
    model, scaler = pair
    return np.asarray(model.predict(scaler.transform(X) if scaler is not None else X), dtype=float)


def _feature_arrays(prices: pd.DataFrame, benchmark: str, window: int, threshold: float):
    members = [c for c in prices.columns if c != benchmark]
    all_ret = prices.pct_change(fill_method=None).fillna(0)
    stock_ret = all_ret[members].to_numpy()
    idx_ret = all_ret[benchmark].to_numpy()
    n, T = len(members), len(prices)
    start = max(window + 5, 45)
    stock_x = np.full((T, n, len(STOCK_FACTORS)), np.nan)
    idx_x = np.full((T, len(INDEX_FACTORS)), np.nan)
    basket_level = (prices[members] / prices[members].iloc[0] * 100).mean(axis=1).to_numpy()
    basket_ret = np.concatenate([[0.0], np.diff(basket_level) / basket_level[:-1]])
    proxy_x = np.full_like(idx_x, np.nan)
    graph_at_last = np.eye(n)
    for t in range(start, T):
        history = stock_ret[t - window + 1: t + 1]
        corr = np.corrcoef(history, rowvar=False)
        corr = np.nan_to_num(corr, nan=0.0, posinf=0.0, neginf=0.0)
        np.fill_diagonal(corr, 0.0)
        adj = (np.abs(corr) >= threshold)
        signed_corr = corr * adj
        momentum5 = np.sum(stock_ret[t - 4:t + 1], axis=0)
        momentum20 = np.sum(stock_ret[t - 19:t + 1], axis=0)
        vol20 = np.std(stock_ret[t - 19:t + 1], axis=0)
        denom = np.maximum(np.sum(adj, axis=1), 1)
        pos = np.where(corr >= threshold, corr, 0)
        neg = np.where(corr <= -threshold, -corr, 0)
        pos_feat = pos @ momentum5 / np.maximum(pos.sum(axis=1), 1e-8)
        neg_feat = neg @ momentum5 / np.maximum(neg.sum(axis=1), 1e-8)
        signed_feat = signed_corr @ momentum5 / np.maximum(np.sum(np.abs(signed_corr), axis=1), 1e-8)
        dispersion = float(np.std(stock_ret[t]))
        stock_x[t] = np.column_stack([
            stock_ret[t], momentum5, momentum20, vol20,
            np.full(n, idx_ret[t]), pos_feat, neg_feat, signed_feat,
            np.sum(adj, axis=1) / max(n - 1, 1), np.full(n, dispersion),
        ])
        factor_common = [
            float(np.mean(momentum5)), dispersion, float(np.mean(signed_feat)),
            float(np.mean(np.abs(corr))),
        ]
        idx_x[t] = [
            idx_ret[t], float(np.sum(idx_ret[t-4:t+1])), float(np.sum(idx_ret[t-19:t+1])),
            float(np.std(idx_ret[t-19:t+1])), *factor_common,
        ]
        proxy_x[t] = [
            basket_ret[t], float(np.sum(basket_ret[t-4:t+1])),
            float(np.sum(basket_ret[t-19:t+1])), float(np.std(basket_ret[t-19:t+1])),
            *factor_common,
        ]
        if t == T - 1:
            graph_at_last = corr
    return members, stock_x, idx_x, proxy_x, stock_ret, idx_ret, basket_level, basket_ret, graph_at_last, start


def _training_rows(X: np.ndarray, series_next: np.ndarray, start: int, stop: int):
    """Return features with observed next-day labels."""
    rows = np.arange(start, stop + 1)
    return X[rows], series_next[rows + 1]


def _direct_backtest(idx_x, idx_ret, start: int, last: int, size: int = 65):
    """Walk-forward predictions using only observed labels."""
    start_test = max(start + 160, last - size)
    records = []
    ridge = forest = None
    for t in range(start_test, last):
        if (t - start_test) % 5 == 0 or ridge is None:
            X, y = _training_rows(idx_x, idx_ret, start, t - 1)
            ridge = _ridge(X, y)
            forest = _rf(X, y)
        x = idx_x[t:t+1]
        records.append((t + 1, float(idx_ret[t+1]), float(_predict_pair(ridge, x)[0]),
                        float(_predict_pair(forest, x)[0])))
    return pd.DataFrame(records, columns=["position", "actual", "Ridge", "Random Forest"])


def _reconcile_proxy(stock_x, proxy_x, stock_ret, basket_level, prices, members,
                     clusters, start: int):
    """Reconcile the sample basket, not the official index."""
    T = len(prices)
    n = len(members)
    end_fit_calib = T - 47
    if end_fit_calib - start < 120:
        raise ValueError("Not enough observations for the covariance calibration window.")
    flat_train = stock_x[start:end_fit_calib-1].reshape(-1, stock_x.shape[-1])
    flat_labels = stock_ret[start+1:end_fit_calib].reshape(-1)
    pre_model = _ridge(flat_train, flat_labels)
    current_model = _ridge(stock_x[start:T-1].reshape(-1, stock_x.shape[-1]),
                           stock_ret[start+1:T].reshape(-1))
    ret_pred = _predict_pair(current_model, stock_x[T-1]).clip(-0.12, 0.12)
    z = (prices[members] / prices[members].iloc[0] * 100).to_numpy()
    zn = z[-1]
    stock_base = zn * (1 + ret_pred)

    # Use the latest cluster structure.
    unique_clusters = sorted(set(clusters.astype(int)))
    weights = np.ones(n) / n
    cluster_weights = np.stack([(clusters.loc[members].to_numpy() == c) * weights
                                for c in unique_clusters])
    S = np.vstack([weights, cluster_weights, np.eye(n)])
    basket_x_train, basket_y_train = _training_rows(
        proxy_x, np.r_[0, np.diff(basket_level) / basket_level[:-1]], start, T - 2
    )
    basket_pair = _ridge(basket_x_train, basket_y_train)
    basket_pred = float(basket_level[-1] * (1 + _predict_pair(basket_pair, proxy_x[T-1:T])[0]))

    cluster_levels = z @ cluster_weights.T
    cluster_pairs = []
    cluster_base = []
    for c in range(len(unique_clusters)):
        levels = cluster_levels[:, c]
        ret = np.r_[0.0, np.diff(levels) / levels[:-1]]
        x = proxy_x.copy()
        x[:, 0] = ret
        x[:, 1] = pd.Series(ret).rolling(5).sum().to_numpy()
        x[:, 2] = pd.Series(ret).rolling(20).sum().to_numpy()
        x[:, 3] = pd.Series(ret).rolling(20).std(ddof=0).to_numpy()
        pair = _ridge(x[start:T-1], ret[start+1:T])
        cluster_pairs.append((pair, x, ret))
        cluster_base.append(levels[-1] * (1 + _predict_pair(pair, x[T-1:T])[0]))
    base = np.concatenate([[basket_pred], cluster_base, stock_base])

    # Calibrate covariance on earlier observations.
    pre_basket = _ridge(proxy_x[start:end_fit_calib-1],
                        np.r_[0, np.diff(basket_level) / basket_level[:-1]][start+1:end_fit_calib])
    pre_cluster = []
    for pair, x, ret in cluster_pairs:
        pre_cluster.append(_ridge(x[start:end_fit_calib-1], ret[start+1:end_fit_calib]))
    residual_rows = []
    for t in range(end_fit_calib, T-1):
        stock_pred = z[t] * (1 + _predict_pair(pre_model, stock_x[t]).clip(-0.12, 0.12))
        direct_b = basket_level[t] * (1 + _predict_pair(pre_basket, proxy_x[t:t+1])[0])
        direct_c = [cluster_levels[t, c] * (1 + _predict_pair(pre_cluster[c], x[t:t+1])[0])
                    for c, (_, x, _) in enumerate(cluster_pairs)]
        forecast_vector = np.r_[direct_b, direct_c, stock_pred]
        realized = S @ z[t+1]
        residual_rows.append(realized - forecast_vector)
    residuals = np.stack(residual_rows)
    reconciled = mint_reconcile(base, S, residuals)
    return {
        "base": base, "reconciled": reconciled,
        "direct_proxy_return": base[0] / basket_level[-1] - 1,
        "reconciled_proxy_return": reconciled[0] / basket_level[-1] - 1,
        "stock_returns": reconciled[1+len(unique_clusters):] / zn - 1,
        "coherence_max_error": float(np.max(np.abs(reconciled - S @ reconciled[-n:]))),
    }


def analyze(prices: pd.DataFrame, benchmark: str, *, correlation_window: int = 40,
            edge_threshold: float = 0.60, model_name: str = "Ridge") -> AnalysisResult:
    if len(prices) < 260:
        raise ValueError("At least 260 daily data points are required.")
    if benchmark not in prices.columns:
        raise ValueError("The benchmark column is missing.")
    if not 0.3 <= edge_threshold <= 0.95:
        raise ValueError("edge_threshold must be between 0.3 and 0.95")
    names, stock_x, index_x, proxy_x, stock_ret, idx_ret, basket_level, basket_ret, corr_np, start = _feature_arrays(
        prices, benchmark, correlation_window, edge_threshold,
    )
    clusters, chars = infer_clusters(prices[names])
    counts = clusters.value_counts().sort_index()
    cluster_table = pd.DataFrame({
        "Cluster": counts.index.astype(int), "Stocks": counts.values,
        "Members": [", ".join(clusters.index[clusters == c].tolist()) for c in counts.index],
        "Avg 20-day momentum (%)": [round(float(chars.loc[clusters == c, "last_5d_momentum"].mean() * 100), 2) for c in counts.index],
        "Avg daily vol (%)": [round(float(chars.loc[clusters == c, "volatility"].mean() * 100), 2) for c in counts.index],
    })
    corr = pd.DataFrame(corr_np, index=names, columns=names)
    edges = []
    for i in range(len(names)):
        for j in range(i+1, len(names)):
            value = corr_np[i, j]
            if abs(value) >= edge_threshold:
                edges.append({"Source": names[i], "Target": names[j], "Correlation": float(value),
                              "Type": "Positive" if value > 0 else "Negative"})
    edges_df = pd.DataFrame(edges, columns=["Source", "Target", "Correlation", "Type"])
    last = len(prices) - 1
    x_train, y_train = _training_rows(index_x, idx_ret, start, last - 1)
    ridge = _ridge(x_train, y_train)
    rf = _rf(x_train, y_train)
    latest_features = index_x[last:last+1]
    ridge_pred = float(_predict_pair(ridge, latest_features)[0])
    rf_pred = float(_predict_pair(rf, latest_features)[0])
    chosen_pred = ridge_pred if model_name == "Ridge" else rf_pred
    other_pred = rf_pred if model_name == "Ridge" else ridge_pred
    back = _direct_backtest(index_x, idx_ret, start, last)
    back.index = prices.index[back["position"].astype(int).to_numpy()]
    back = back.drop(columns=["position"])
    chosen_resid = back["actual"] - back[model_name]
    forecast_std = max(float(chosen_resid.std(ddof=1)), 0.001)
    comparison = []
    for label, value in [("Random walk", np.zeros(len(back))),
                         ("Ridge", back["Ridge"].to_numpy()),
                         ("Random Forest", back["Random Forest"].to_numpy())]:
        errors = back["actual"].to_numpy() - value
        comparison.append({
            "Model": label,
            "MAE (bps)": round(float(np.mean(np.abs(errors))) * 10000, 2),
            "RMSE (bps)": round(float(np.sqrt(np.mean(errors**2))) * 10000, 2),
            "Up/down hit rate (%)": round(float(np.mean((value > 0) == (back["actual"].to_numpy() > 0))) * 100, 1)
                if label != "Random walk" else np.nan,
        })
    comps = pd.DataFrame(comparison)
    rec = _reconcile_proxy(stock_x, proxy_x, stock_ret, basket_level, prices,
                           names, clusters, start)
    stock_forecasts = pd.DataFrame({
        "Ticker": names,
        "Cluster": clusters.loc[names].to_numpy(),
        "1-day scenario (%)": np.round(rec["stock_returns"] * 100, 3),
        "Last 5-day return (%)": np.round(prices[names].pct_change(5).iloc[-1].to_numpy() * 100, 2),
    }).sort_values("1-day scenario (%)", ascending=False)
    if model_name == "Ridge":
        model, scaler = ridge
        zval = scaler.transform(latest_features)[0]
        contributions = np.asarray(model.coef_) * zval
        explain = pd.DataFrame({"Factor": INDEX_FACTORS, "Contribution (bps)": contributions * 10000})
        explain["Importance"] = explain["Contribution (bps)"].abs()
        explain = explain.sort_values("Importance", ascending=False).drop(columns=["Importance"])
    else:
        model = rf[0]
        explain = pd.DataFrame({"Factor": INDEX_FACTORS,
                                "Global importance (%)": model.feature_importances_ * 100}).sort_values(
                                    "Global importance (%)", ascending=False)
    return AnalysisResult(
        index_symbol=benchmark, asof_date=str(prices.index[-1].date()), prices=prices,
        clusters=clusters, cluster_table=cluster_table,
        graph_edges=edges_df, correlation=corr, stock_forecasts=stock_forecasts,
        proxy_direct_return=float(rec["direct_proxy_return"]),
        proxy_reconciled_return=float(rec["reconciled_proxy_return"]),
        direct_forecast=chosen_pred, direct_alternative=other_pred,
        forecast_std=forecast_std, index_last=float(prices[benchmark].iloc[-1]),
        contributions=explain, backtest=back, metric_table=comps,
        reconciliation_check=float(rec["coherence_max_error"]),
        model_name=model_name, cluster_count=len(cluster_table),
    )
