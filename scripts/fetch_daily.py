"""Refresh the OHLCV master from Yahoo Finance, then rebuild the coverage report.

Run from the repository root. The logic lives in `nifty500.ingest`; this script
only supplies the network call and the file paths.

It exits non-zero whenever the result cannot be trusted -- an empty response, no
usable bars, too many symbols missing -- so a scheduled run that fetched nothing
is red rather than a green run that quietly wrote nothing.
"""

import argparse
import sys
from datetime import UTC, datetime
from pathlib import Path

import pandas as pd
import yfinance as yf

from nifty500.coverage import build_coverage_report
from nifty500.ingest import PipelineError, load_symbols, update_master, write_master

MASTER_PATH = Path("data/raw/nifty500_ohlcv_raw.parquet")
CONSTITUENTS_PATH = Path("data/raw/nifty500_constituents.csv")
COVERAGE_PATH = Path("data/raw/coverage_report.csv")


def yahoo_download(symbols_ns, start, end):
    """The one place that touches the network. `end` is exclusive."""
    print(f"Fetching {start} to {end} (end exclusive) for {len(symbols_ns)} symbols...")
    return yf.download(
        symbols_ns,
        start=start.strftime("%Y-%m-%d"),
        end=end.strftime("%Y-%m-%d"),
        auto_adjust=True,
        group_by="ticker",
        threads=True,
        progress=False,
    )


def main(argv=None, download=yahoo_download, today=None):
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--full-refresh", action="store_true",
                        help="re-download every symbol's whole history, not only the recent "
                             "sessions. A replacement shorter than the stored history is "
                             "rejected, so this cannot shorten the master.")
    args = parser.parse_args(argv)

    symbols = load_symbols(CONSTITUENTS_PATH)
    master = pd.read_parquet(MASTER_PATH)
    today = today or datetime.now(UTC).date()

    try:
        updated, report = update_master(
            master, symbols, download, today,
            refresh_symbols=symbols if args.full_refresh else (),
        )
    except PipelineError as error:
        print(f"FAIL: {error}")
        return 1

    for line in report.lines():
        print(line)

    if not report.changed:
        print("Master is unchanged: no new sessions (a market holiday, or Yahoo has not "
              "finalised today's bars yet).")
        return 0

    write_master(updated, MASTER_PATH)
    build_coverage_report(updated, symbols).to_csv(COVERAGE_PATH, index=False)
    print(f"Master updated: {len(updated):,} rows, latest date: {report.last_bar_after}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
