"""Behavioural feature engineering for the Nifty 500 OHLCV master.

Turns a long OHLCV frame into one row per symbol describing *how the stock
trades* — risk, trend, downside and liquidity — rather than what the company
sells. The output feeds the clustering in notebooks/02_clustering.ipynb.
"""

from pathlib import Path

import numpy as np
import pandas as pd

from nifty500.bars import clean_bars

TRADING_DAYS = 252

# A symbol needs near-complete history inside the window for its features to be
# comparable with the rest of the universe; recent IPOs are dropped rather than
# padded, for the same reason the pipeline never forward-fills prices.
DEFAULT_WINDOW_YEARS = 3
DEFAULT_MIN_COVERAGE = 0.95

# ...and it must still be trading when the window ends. A stock delisted or merged
# away part-way through has coverage to spare but no *current* behaviour to
# measure. A few sessions of slack keep one missing bar from ejecting a healthy name.
LAST_BAR_TOLERANCE_SESSIONS = 5

# A single session move beyond this is not equity behaviour — for an index
# constituent it is a split, bonus issue or demerger that auto_adjust missed
# (e.g. VEDL -65% on its 2026 demerger). Left in, these few points dominate
# the kurtosis and skew features: masking them drops peak kurtosis from
# 331 to 44 while discarding 0.002% of observations.
CORPORATE_ACTION_THRESHOLD = 0.40

# The threshold is blunt: a 1:2 bonus moves a price by only a third and walks
# straight under it. Sessions found by review -- a large move with no volume behind
# it -- are listed here and masked exactly like the threshold's catches.
REVIEWED_ACTIONS_PATH = Path("data/reference/corporate_actions.csv")

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


def load_reviewed_actions(path=REVIEWED_ACTIONS_PATH):
    """The register of sessions masked in addition to the threshold's catches."""
    return pd.read_csv(path, parse_dates=["date"])


def session_evidence(prices, symbol, date, base_sessions=30, after_sessions=20):
    """Numbers that tell a genuine crash from a re-basing of the price.

    A real shock trades heavily on the day and the volume fades. A split, bonus or
    demerger re-bases the price with no trading behind it, and -- for a bonus or
    split -- leaves the *share* volume permanently higher. Returns the session's
    price ratio, its volume as a multiple of the median of the `base_sessions`
    before it, and the median volume of the `after_sessions` that follow, as a
    multiple of the same base.
    """
    bars = prices[prices["symbol"] == symbol].sort_values("date").set_index("date")
    i = bars.index.get_loc(pd.Timestamp(date))
    base = bars["volume"].iloc[max(0, i - base_sessions):i].median()
    after = bars["volume"].iloc[i + 1:i + 1 + after_sessions].median()
    return {
        "price_ratio": float(bars["close"].iloc[i] / bars["close"].iloc[i - 1]),
        "day_volume_x": float(bars["volume"].iloc[i] / base),
        "next20_volume_x": float(after / base),
    }


def select_universe(prices, window_years=DEFAULT_WINDOW_YEARS,
                    min_coverage=DEFAULT_MIN_COVERAGE, as_of=None):
    """Restrict to a trailing window and to symbols that traded through it.

    The window ends at `as_of` if given, else at the newest bar. Pinning it is what
    makes a result reproducible: the master gains a session every weekday, and
    without a pin the same code gives a different answer each day.

    A symbol is kept if it has bars on at least `min_coverage` of the window's
    sessions and one of them is among the window's last few sessions.

    Returns (windowed_frame, kept_symbols, dropped_symbols).
    """
    if as_of is not None:
        prices = prices[prices["date"] <= pd.Timestamp(as_of)]
    end = prices["date"].max()
    start = end - pd.DateOffset(years=window_years)
    window = prices[prices["date"] >= start].copy()

    sessions = window["date"].nunique()
    bars = window.groupby("symbol")["date"].count()

    last_sessions = np.sort(window["date"].unique())[-LAST_BAR_TOLERANCE_SESSIONS:]
    still_trading = set(window.loc[window["date"] >= last_sessions[0], "symbol"])

    covered = bars >= sessions * min_coverage
    kept = bars[covered & bars.index.isin(still_trading)].index
    dropped = bars.index.difference(kept)

    return window[window["symbol"].isin(kept)].copy(), list(kept), list(dropped)


def _closes(window):
    return window.pivot(index="date", columns="symbol", values="close").sort_index()


def corporate_action_mask(raw_returns, reviewed_actions=None):
    """Boolean frame, True where a session is masked as a corporate action.

    The threshold's catches, plus any (symbol, date) in `reviewed_actions`.
    """
    mask = raw_returns.abs() > CORPORATE_ACTION_THRESHOLD
    if reviewed_actions is not None:
        for symbol, day in zip(reviewed_actions["symbol"],
                               pd.to_datetime(reviewed_actions["date"]), strict=True):
            if symbol in mask.columns and day in mask.index:
                mask.loc[day, symbol] = True
    return mask


def daily_returns(window, mask_corporate_actions=True, reviewed_actions=None):
    """Wide (date x symbol) frame of daily simple returns.

    fill_method=None matters: the pandas default pads missing prices forward,
    which invents a zero return for every non-traded session instead of
    leaving a gap.
    """
    returns = _closes(window).pct_change(fill_method=None)
    if mask_corporate_actions:
        returns = returns.mask(corporate_action_mask(returns, reviewed_actions))
    return returns


def adjusted_closes(window, reviewed_actions=None):
    """Closes with the level shift of every masked session divided out.

    Volatility, skew and kurtosis are built from returns, where masking a session
    simply removes it. Momentum and drawdown are built from the price *path*, where
    the same session would survive as a phantom crash -- a stock that lost 65% to a
    demerger would show as -65% momentum. Dividing the shift out keeps one
    definition of "what a corporate action does" across all ten features, and
    leaves every symbol without a masked session exactly as it was.
    """
    closes = _closes(window)
    raw = closes.pct_change(fill_method=None)
    shift = (1 + raw).where(corporate_action_mask(raw, reviewed_actions), 1.0)
    return closes / shift.cumprod()


def _max_drawdown(prices):
    """Most negative peak-to-trough move on a price path, as a fraction.

    The first observation is itself a candidate peak, so a stock that falls from
    the very first bar is measured from there.
    """
    return float((prices / prices.cummax() - 1).min())


def _trailing_return(closes, lookback):
    """Simple return over the last `lookback` sessions."""
    if len(closes) <= lookback:
        return np.nan
    first, last = closes.iloc[-lookback - 1], closes.iloc[-1]
    if not np.isfinite(first) or first <= 0:
        return np.nan
    return float(last / first - 1)


def _beta(returns, market):
    """OLS slope of a stock's returns on the market's, on the sessions both have.

    Covariance and variance must come from the *same* sessions. Dividing a
    covariance over the stock's sessions by the market's variance over all of them
    -- the obvious shortcut -- biases beta for any stock with a masked or missing day.
    """
    aligned = pd.concat([returns, market], axis=1, join="inner").dropna()
    variance = aligned.iloc[:, 1].var()
    if len(aligned) < 2 or not variance:
        return np.nan
    return float(aligned.iloc[:, 0].cov(aligned.iloc[:, 1]) / variance)


def build_features(prices, window_years=DEFAULT_WINDOW_YEARS,
                   min_coverage=DEFAULT_MIN_COVERAGE, as_of=None, reviewed_actions=None):
    """One row of behavioural features per surviving symbol.

    `prices` is the long OHLCV master: symbol, date, open, high, low, close, volume.
    Rows that are not real bars (see `nifty500.bars`) are dropped first, so the
    result does not depend on whether the caller cleaned the master.
    """
    window, kept, dropped = select_universe(clean_bars(prices), window_years, min_coverage,
                                            as_of)

    raw_returns = daily_returns(window, mask_corporate_actions=False)
    returns = daily_returns(window, reviewed_actions=reviewed_actions)
    masked_count = int((raw_returns.notna() & returns.isna()).sum().sum())
    closes = adjusted_closes(window, reviewed_actions)

    # Equal-weight index built from the universe itself — no external benchmark
    # file to drift out of sync with the price master.
    market = returns.mean(axis=1)

    turnover = (window["close"] * window["volume"]).groupby(window["symbol"]).median()

    rows = []
    for symbol in returns.columns:
        r = returns[symbol].dropna()
        if r.empty:
            continue

        path = closes[symbol].dropna()
        downside = r[r < 0]

        rows.append({
            "symbol": symbol,
            "ann_volatility": float(r.std() * np.sqrt(TRADING_DAYS)),
            "downside_volatility": float(downside.std() * np.sqrt(TRADING_DAYS)),
            "beta": _beta(r, market),
            "mom_3m": _trailing_return(path, 63),
            "mom_6m": _trailing_return(path, 126),
            "mom_12m": _trailing_return(path, TRADING_DAYS),
            "max_drawdown": _max_drawdown(path),
            "return_skew": float(r.skew()),
            "return_kurtosis": float(r.kurtosis()),
            # Turnover spans several orders of magnitude across the index;
            # logging it stops the largest names dominating the distance metric.
            "log_turnover": (float(np.log10(turnover[symbol]))
                             if turnover.get(symbol, 0) > 0 else np.nan),
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
