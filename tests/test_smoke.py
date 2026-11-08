"""No internet needed. Run: python -m pytest -q"""
import numpy as np
import pandas as pd

from forecasting import analyze, infer_clusters, mint_reconcile
from market_data import demo_prices

MEMBERS = ["AAA", "BBB", "CCC", "DDD", "EEE", "FFF", "GGG", "HHH"]


def test_demo_predictors_and_coherence():
    prices = demo_prices("^TEST", MEMBERS, n_days=540)
    result = analyze(prices, "^TEST", correlation_window=40,
                     edge_threshold=0.55, model_name="Ridge")
    assert result.cluster_count >= 2
    assert np.isfinite(result.direct_forecast)
    assert np.isfinite(result.proxy_reconciled_return)
    assert result.reconciliation_check < 1e-7
    assert len(result.backtest) >= 40
    assert len(result.stock_forecasts) == len(MEMBERS)
    assert (result.clusters.index == pd.Index(MEMBERS)).all()
    assert result.asof_date


def test_random_forest_runs():
    prices = demo_prices("^TEST", MEMBERS, n_days=450)
    result = analyze(prices, "^TEST", model_name="Random Forest")
    assert np.isfinite(result.direct_forecast)
    assert "Global importance (%)" in result.contributions.columns


def test_mint_projects_to_hierarchy():
    S = np.array([[1., 1.], [1., 0.], [0., 1.]])
    base = np.array([19., 10., 12.])
    rng = np.random.default_rng(9)
    residuals = rng.normal(size=(80, 3))
    result = mint_reconcile(base, S, residuals)
    assert np.isclose(result[0], result[1] + result[2], atol=1e-9)
