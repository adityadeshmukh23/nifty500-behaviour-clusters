"""Per-symbol coverage report for the OHLCV master.

`data/raw/coverage_report.csv` is published next to the data so a reader can see,
without loading half a million rows, which symbols carry the full history and
which are recent listings left deliberately short. It is derived from the master,
so the pipeline regenerates it on every run rather than letting it go stale.
"""

from nifty500.bars import clean_bars

# A symbol "carries the full history" when it has at least this share of the
# bars of the longest-lived symbol. It absorbs the odd missing session without
# admitting a listing that arrived a year late.
FULL_HISTORY_SHARE = 0.90


def build_coverage_report(bars, symbols=None):
    """First date, last date and bar count per symbol, over valid bars only.

    Pass `symbols` (the constituents) to also list any that have no bars at all,
    with a count of zero, instead of leaving them out of the report silently.
    """
    valid = clean_bars(bars)
    report = (valid.groupby("symbol")["date"]
                   .agg(first_date="min", last_date="max", bars="count"))
    if symbols is not None:
        report = report.reindex(sorted(set(symbols) | set(report.index)))
        report["bars"] = report["bars"].fillna(0).astype("int64")

    report["full_history"] = report["bars"] >= FULL_HISTORY_SHARE * report["bars"].max()
    for column in ("first_date", "last_date"):
        report[column] = report[column].dt.strftime("%Y-%m-%d")
    return report.rename_axis("symbol").reset_index()
