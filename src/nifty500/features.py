"""Behavioural feature engineering for the Nifty 500 OHLCV master.

Turns a long OHLCV frame into one row per symbol describing *how the stock
trades* — risk, trend, downside and liquidity — rather than what the company
sells. The output feeds the clustering in notebooks/02_clustering.ipynb.
"""

import numpy as np
import pandas as pd

TRADING_DAYS = 252

# A symbol needs near-complete history inside the window for its features to be
# comparable with the rest of the universe; recent IPOs are dropped rather than
# padded, for the same reason the pipeline never forward-fills prices.
DEFAULT_WINDOW_YEARS = 3
DEFAULT_MIN_COVERAGE = 0.95

# A single session move beyond this is not equity behaviour — for an index
# constituent it is a split, bonus issue or demerger that auto_adjust missed
# (e.g. VEDL -65% on its 2026 demerger). Left in, these few points dominate
# the kurtosis and skew features: masking them drops peak kurtosis from
# 331 to 44 while discarding 0.002% of observations.
CORPORATE_ACTION_THRESHOLD = 0.40

FEATURE_COLUMNS = [
    "ann_volatility",
    "downside_volatility",
    "beta",
    "mom_3m",
    "mom_6m",
    "mom_12m",
    "max_drawdown",
    "return_skew",
    "return_kurtosis",
    "log_turnover",
]


def select_universe(prices, window_years=DEFAULT_WINDOW_YEARS,
                    min_coverage=DEFAULT_MIN_COVERAGE):
    """Restrict to a trailing window and to symbols that traded through it.

    Returns (windowed_frame, kept_symbols, dropped_symbols).
    """
    end = prices["date"].max()
    start = end - pd.DateOffset(years=window_years)
    window = prices[prices["date"] >= start].copy()

    sessions = window["date"].nunique()
    bars = window.groupby("symbol")["date"].count()
    kept = bars[bars >= sessions * min_coverage].index
    dropped = bars.index.difference(kept)

    return window[window["symbol"].isin(kept)].copy(), list(kept), list(dropped)


def daily_returns(window, mask_corporate_actions=True):
    """Wide (date x symbol) frame of daily simple returns.

    fill_method=None matters: the pandas default pads missing prices forward,
    which invents a zero return for every non-traded session instead of
    leaving a gap.
    """
    closes = window.pivot(index="date", columns="symbol", values="close").sort_index()
    returns = closes.pct_change(fill_method=None)
    if mask_corporate_actions:
        returns = returns.mask(returns.abs() > CORPORATE_ACTION_THRESHOLD)
    return returns


def _max_drawdown(series):
    """Most negative peak-to-trough move on a price path, as a fraction."""
    curve = (1 + series.fillna(0)).cumprod()
    return float((curve / curve.cummax() - 1).min())


def _trailing_return(closes, lookback):
    """Simple return over the last `lookback` sessions."""
    if len(closes) <= lookback:
        return np.nan
    first, last = closes.iloc[-lookback - 1], closes.iloc[-1]
    if not np.isfinite(first) or first <= 0:
        return np.nan
    return float(last / first - 1)


def build_features(prices, window_years=DEFAULT_WINDOW_YEARS,
                   min_coverage=DEFAULT_MIN_COVERAGE):
    """One row of behavioural features per surviving symbol.

    `prices` is the long OHLCV master: symbol, date, open, high, low, close, volume.
    """
    window, kept, dropped = select_universe(prices, window_years, min_coverage)

    raw_returns = daily_returns(window, mask_corporate_actions=False)
    returns = daily_returns(window)
    masked_count = int((raw_returns.notna() & returns.isna()).sum().sum())
    closes = window.pivot(index="date", columns="symbol", values="close").sort_index()

    # Equal-weight index built from the universe itself — no external benchmark
    # file to drift out of sync with the price master.
    market = returns.mean(axis=1)
    market_var = market.var()

    turnover = (window["close"] * window["volume"]).groupby(window["symbol"]).median()

    rows = []
    for symbol in returns.columns:
        r = returns[symbol].dropna()
        if r.empty:
            continue

        downside = r[r < 0]
        aligned = pd.concat([r, market], axis=1, join="inner").dropna()

        rows.append({
            "symbol": symbol,
            "ann_volatility": float(r.std() * np.sqrt(TRADING_DAYS)),
            "downside_volatility": float(downside.std() * np.sqrt(TRADING_DAYS)),
            "beta": float(aligned.cov().iloc[0, 1] / market_var) if market_var else np.nan,
            "mom_3m": _trailing_return(closes[symbol].dropna(), 63),
            "mom_6m": _trailing_return(closes[symbol].dropna(), 126),
            "mom_12m": _trailing_return(closes[symbol].dropna(), TRADING_DAYS),
            "max_drawdown": _max_drawdown(r),
            "return_skew": float(r.skew()),
            "return_kurtosis": float(r.kurtosis()),
            # Turnover spans several orders of magnitude across the index;
            # logging it stops the largest names dominating the distance metric.
            "log_turnover": float(np.log10(turnover[symbol])) if turnover.get(symbol, 0) > 0 else np.nan,
        })

    features = pd.DataFrame(rows).set_index("symbol").sort_index()
    features.attrs["dropped_symbols"] = dropped
    features.attrs["masked_observations"] = masked_count
    features.attrs["window_start"] = str(window["date"].min().date())
    features.attrs["window_end"] = str(window["date"].max().date())
    return features


def attach_sectors(features, constituents):
    """Join the official Industry label on for comparison against the clusters."""
    labels = (constituents.assign(symbol=constituents["Symbol"].str.strip())
                          .set_index("symbol")["Industry"])
    return features.join(labels.rename("industry"), how="left")
