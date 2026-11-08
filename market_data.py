"""Market data retrieval and simulated examples."""
from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, timezone

import numpy as np
import pandas as pd

INDEXES = {
    "S&P 500": {
        "benchmark": "^GSPC",
        "constituents": ["AAPL", "MSFT", "NVDA", "AMZN", "GOOGL", "META", "BRK-B", "JPM", "XOM", "JNJ", "PG", "UNH"],
    },
    "Dow Jones": {
        "benchmark": "^DJI",
        "constituents": ["AAPL", "MSFT", "AMZN", "JPM", "GS", "CAT", "HD", "DIS", "MCD", "KO", "WMT", "V"],
    },
    "Nasdaq-100": {
        "benchmark": "^NDX",
        "constituents": ["AAPL", "MSFT", "NVDA", "AMZN", "META", "GOOGL", "AVGO", "COST", "NFLX", "AMD", "ADBE", "CSCO"],
    },
}


@dataclass
class MarketData:
    prices: pd.DataFrame
    benchmark: str
    source: str
    pulled_at_utc: str
    dropped: list[str]


def demo_prices(benchmark: str, members: list[str], n_days: int = 760) -> pd.DataFrame:
    """Generate deterministic test prices."""
    rng = np.random.default_rng(20261020)
    n = len(members)
    date_index = pd.bdate_range(end=pd.Timestamp.today().normalize(), periods=n_days)
    market = rng.normal(0.00024, 0.010, size=n_days)
    sector = rng.normal(0, 0.006, size=(n_days, 3))
    sector_id = np.arange(n) % 3
    stock_returns = np.zeros((n_days, n))
    for i in range(n):
        stock_returns[:, i] = (
            market * (0.75 + 0.45 * rng.random())
            + sector[:, sector_id[i]]
            + rng.normal(0, 0.0075, size=n_days)
            + (0.00015 if i % 3 == 0 else 0)
        )
    if n_days > 350:
        stock_returns[-180:, 0] += sector[-180:, 1] * 0.7
    stock_prices = 100 * np.exp(np.cumsum(stock_returns, axis=0))
    benchmark_prices = 4000 * np.exp(np.cumsum(market + rng.normal(0, 0.001, n_days)))
    data = pd.DataFrame(stock_prices, index=date_index, columns=members)
    data[benchmark] = benchmark_prices
    return data


def get_prices(benchmark: str, members: list[str], mode: str = "Yahoo Finance") -> MarketData:
    symbols = list(dict.fromkeys([benchmark] + members))
    if mode == "Simulated demo":
        prices = demo_prices(benchmark, members)
        dropped = []
    else:
        try:
            import yfinance as yf
        except ImportError as exc:
            raise RuntimeError("Install packages from requirements.txt to use Yahoo Finance.") from exc
        try:
            raw = yf.download(
                tickers=symbols,
                period="3y",
                interval="1d",
                group_by="column",
                auto_adjust=True,
                progress=False,
                threads=False,
                timeout=20,
            )
            if raw is None or raw.empty:
                raise ValueError("The provider returned no price history.")
            if isinstance(raw.columns, pd.MultiIndex):
                if "Close" not in raw.columns.get_level_values(0):
                    raise ValueError("No Close prices were returned.")
                prices = raw["Close"].copy()
            else:
                if len(symbols) != 1 or "Close" not in raw.columns:
                    raise ValueError("Could not parse the provider response.")
                prices = raw[["Close"]].rename(columns={"Close": symbols[0]})
            prices.columns = [str(c).upper() for c in prices.columns]
            prices.index = pd.to_datetime(prices.index).tz_localize(None)
        except Exception as exc:
            raise RuntimeError(
                f"Yahoo Finance could not supply data ({exc}). Try again later, reduce tickers, "
                "or explicitly choose Simulated demo in the sidebar."
            ) from exc
        dropped = [s for s in symbols if s not in prices.columns or prices[s].dropna().empty]
        if benchmark in dropped:
            raise RuntimeError(f"No benchmark series for {benchmark}; try another index or demo mode.")
        prices = prices.drop(columns=[s for s in dropped if s in prices.columns])
        # Do not fill prices backward.
        prices = prices.ffill(limit=3).dropna()
    if benchmark not in prices.columns or len(prices) < 250:
        raise RuntimeError("At least 250 aligned daily observations are required.")
    active = [s for s in members if s in prices.columns and s != benchmark]
    if len(active) < 5:
        raise RuntimeError("At least five stocks must have price histories for the relationship graph.")
    prices = prices[[benchmark] + active].astype(float)
    if (prices <= 0).any().any():
        raise RuntimeError("The downloaded dataset contains invalid nonpositive prices.")
    return MarketData(
        prices=prices,
        benchmark=benchmark,
        source=mode,
        pulled_at_utc=datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M UTC"),
        dropped=dropped,
    )


def intraday_benchmark(benchmark: str) -> pd.DataFrame:
    """Get an optional intraday display series."""
    import yfinance as yf

    chart = yf.Ticker(benchmark).history(period="5d", interval="5m", auto_adjust=True)
    if chart is None or chart.empty:
        return pd.DataFrame()
    out = chart[["Close"]].dropna().copy()
    out.index = pd.to_datetime(out.index)
    return out.tail(400)
