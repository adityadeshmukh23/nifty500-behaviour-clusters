"""Hygiene for daily OHLCV bars.

Yahoo Finance returns rows that look like bars but are not, and yfinance only
drops a row when *every* column is NaN or zero. Two shapes get through, and both
have reached the master:

* **Unpriced bars.** A session Yahoo has not finalised comes back with a volume
  but null open/high/low/close. On 2026-09-21 that was all 503 symbols at once.
* **Placeholder bars.** On a day a stock did not trade -- most visibly a market
  holiday -- Yahoo carries the previous close forward with zero volume. Five such
  holidays, and a few hundred single-stock days, sat in the master as sessions.

Neither is a price the market printed, so neither belongs in a dataset that
promises no synthetic prices. The pipeline, the freshness check and the analysis
all filter through this one module so they agree on what a bar is.
"""

import pandas as pd

BAR_COLUMNS = ["symbol", "date", "open", "high", "low", "close", "volume"]
PRICE_COLUMNS = ["open", "high", "low", "close"]

# On a normal session a handful of illiquid names show up as placeholders. When
# at least this share of a date's priced rows are placeholders, the exchange was
# shut and the whole date is dropped -- including the few stragglers that slipped
# through row by row.
CLOSED_SESSION_SHARE = 0.5


def is_placeholder(bars):
    """Zero-volume bars whose four prices are identical.

    A stock that traded has a volume. One that did not has no printed price, only
    a quote carried across the gap -- which is what this catches. Zero volume with
    *moving* prices is deliberately left alone: that is a real bar whose volume
    Yahoo failed to report, and its prices are still good.
    """
    flat = ((bars["open"] == bars["high"]) & (bars["high"] == bars["low"])
            & (bars["low"] == bars["close"]))
    return (bars["volume"] == 0) & flat


def flag_invalid_bars(bars, closed_session_share=CLOSED_SESSION_SHARE):
    """Boolean frame, one column per reason a row is not a real bar.

    Reasons are mutually exclusive, so the column sums are a straight tally of
    what was wrong and how often.
    """
    unpriced = bars[PRICE_COLUMNS].isna().any(axis=1) | ~(bars["close"] > 0)
    priced = ~unpriced
    placeholder = is_placeholder(bars) & priced

    n_priced = priced.groupby(bars["date"]).transform("sum")
    n_placeholder = placeholder.groupby(bars["date"]).transform("sum")
    share = n_placeholder / n_priced.where(n_priced > 0)
    closed_session = (share >= closed_session_share) & priced & ~placeholder

    return pd.DataFrame({
        "unpriced": unpriced,
        "placeholder": placeholder,
        "closed_session": closed_session,
    })


def clean_bars(bars, closed_session_share=CLOSED_SESSION_SHARE):
    """`bars` without unpriced rows, placeholder rows or closed-market dates."""
    flags = flag_invalid_bars(bars, closed_session_share)
    return bars[~flags.any(axis=1)]
