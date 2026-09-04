"""Fail if the OHLCV master has gone stale.

This is the backstop for the failure mode the pipeline's own alerting cannot
see. Failure alerting only fires when a run *fails*; it says nothing when runs
stop happening at all -- a disabled schedule, a deleted secret, a workflow that
succeeds while quietly writing nothing. This check looks at the data itself and
asks a single question: is the newest bar recent enough to be plausible?
"""

import argparse
import sys
from datetime import date, timedelta
from pathlib import Path

import pandas as pd

DEFAULT_PARQUET = Path("data/raw/nifty500_ohlcv_raw.parquet")

# NSE closes for a scattering of holidays, some of them adjacent to weekends.
# Three business days of slack tolerates the longest normal cluster without
# tolerating a pipeline that has actually stopped.
DEFAULT_MAX_LAG_BUSINESS_DAYS = 3


def business_days_between(start, end):
    """Weekdays falling strictly after `start` and on or before `end`."""
    if end <= start:
        return 0
    days = 0
    cursor = start + timedelta(days=1)
    while cursor <= end:
        if cursor.weekday() < 5:          # Mon-Fri
            days += 1
        cursor += timedelta(days=1)
    return days


def latest_bar_date(parquet_path):
    """Newest date present in the master, as a date."""
    master = pd.read_parquet(parquet_path, columns=["date"])
    if master.empty:
        raise SystemExit(f"FAIL: {parquet_path} contains no rows.")
    return pd.to_datetime(master["date"]).max().date()


def check(parquet_path=DEFAULT_PARQUET, max_lag=DEFAULT_MAX_LAG_BUSINESS_DAYS,
          today=None):
    """Return (is_fresh, lag_in_business_days, latest_date)."""
    today = today or date.today()
    latest = latest_bar_date(parquet_path)
    lag = business_days_between(latest, today)
    return lag <= max_lag, lag, latest


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--path", type=Path, default=DEFAULT_PARQUET)
    parser.add_argument("--max-lag", type=int, default=DEFAULT_MAX_LAG_BUSINESS_DAYS,
                        help="business days of staleness to tolerate")
    args = parser.parse_args()

    fresh, lag, latest = check(args.path, args.max_lag)

    print(f"latest bar in master : {latest}")
    print(f"today                : {date.today()}")
    print(f"lag                  : {lag} business day(s)")
    print(f"tolerated            : {args.max_lag}")

    if fresh:
        print("\nOK: dataset is current.")
        return 0

    print(f"\nFAIL: dataset is {lag} business days stale (limit {args.max_lag}).")
    print("The daily pull is not landing new data. Likely causes:")
    print("  1. The scheduled workflow was disabled after repository inactivity.")
    print("  2. It is running but writing nothing -- check the most recent run's log.")
    print("  3. KAGGLE_USERNAME / KAGGLE_KEY expired, failing the job before the commit.")
    return 1


if __name__ == "__main__":
    sys.exit(main())
