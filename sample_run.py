"""Offline model check using simulated prices."""
from market_data import demo_prices
from forecasting import analyze

stocks = ["AAA", "BBB", "CCC", "DDD", "EEE", "FFF", "GGG", "HHH"]
prices = demo_prices("^SIM", stocks, n_days=540)
result = analyze(prices, "^SIM", model_name="Ridge")
print("SIMULATED DATA ONLY — not actual market performance")
print("As of:", result.asof_date)
print("Direct index 1-day predicted return: {:.3%}".format(result.direct_forecast))
print("MinT-reconciled sample basket 1-day return: {:.3%}".format(result.proxy_reconciled_return))
print("Max basket coherence error:", result.reconciliation_check)
print("\nClusters:\n", result.cluster_table.to_string(index=False))
print("\nStrict walk-forward direct-index score:\n", result.metric_table.to_string(index=False))
