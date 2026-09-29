"""Tests for the per-symbol coverage report."""

import numpy as np
import pandas as pd

from nifty500.bars import BAR_COLUMNS
from nifty500.coverage import FULL_HISTORY_SHARE, build_coverage_report


def history(symbol, start, sessions, volume=1_000.0):
    dates = pd.bdate_range(start, periods=sessions)
    price = np.linspace(100, 110, sessions)
    return pd.DataFrame({"symbol": symbol, "date": dates, "open": price, "high": price + 1,
                         "low": price - 1, "close": price, "volume": volume},
                        columns=BAR_COLUMNS)


class TestBuildCoverageReport:
    def test_reports_first_last_and_count_per_symbol(self):
        report = build_coverage_report(history("AAA", "2026-01-05", 10))
        row = report.iloc[0]
        assert row["symbol"] == "AAA"
        assert row["first_date"] == "2026-01-05"
        assert row["last_date"] == "2026-01-16"
        assert row["bars"] == 10

    def test_matches_the_published_layout(self):
        report = build_coverage_report(history("AAA", "2026-01-05", 5))
        assert list(report.columns) == ["symbol", "first_date", "last_date", "bars",
                                        "full_history"]
        assert report["full_history"].dtype == bool
        assert report["bars"].dtype == "int64"

    def test_full_history_is_relative_to_the_longest_symbol(self):
        bars = pd.concat([
            history("LONG", "2022-01-03", 100),
            history("GAPPY", "2022-01-03", 95),          # 95% of the longest: still full
            history("RECENT", "2022-01-03", 40),         # a late listing: not full
        ])
        report = build_coverage_report(bars).set_index("symbol")
        assert 95 >= FULL_HISTORY_SHARE * 100
        assert report.loc["LONG", "full_history"]
        assert report.loc["GAPPY", "full_history"]
        assert not report.loc["RECENT", "full_history"]

    def test_bars_that_are_not_real_do_not_extend_the_range(self):
        bars = history("AAA", "2026-01-05", 10)
        junk = bars.iloc[[-1]].copy()
        junk["date"] = pd.Timestamp("2026-01-30")
        junk[["open", "high", "low", "close"]] = np.nan      # volume but no price
        report = build_coverage_report(pd.concat([bars, junk]))
        assert report.iloc[0]["last_date"] == "2026-01-16"
        assert report.iloc[0]["bars"] == 10

    def test_constituents_with_no_bars_are_listed_rather_than_omitted(self):
        report = build_coverage_report(history("AAA", "2026-01-05", 10),
                                       symbols=["AAA", "GHOST"]).set_index("symbol")
        assert report.loc["GHOST", "bars"] == 0
        assert not report.loc["GHOST", "full_history"]
        assert pd.isna(report.loc["GHOST", "first_date"])

    def test_output_is_ordered_by_symbol_so_reruns_do_not_churn_the_diff(self):
        bars = pd.concat([history("ZZZ", "2026-01-05", 5), history("AAA", "2026-01-05", 5)])
        assert build_coverage_report(bars)["symbol"].tolist() == ["AAA", "ZZZ"]
