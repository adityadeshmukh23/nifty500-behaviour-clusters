"""Tests for the staleness backstop.

The interesting cases are all calendar edges: a Monday check against Friday's
close must read as fresh, while a genuinely stalled pipeline must read as stale
no matter which day the check happens to run on.
"""

from datetime import date

import pandas as pd
import pytest

from scripts.check_freshness import (DEFAULT_MAX_LAG_BUSINESS_DAYS,
                                     business_days_between, check,
                                     latest_bar_date)


@pytest.fixture
def master(tmp_path):
    """Write a minimal master parquet whose newest bar is `last`."""
    def _write(last):
        path = tmp_path / "master.parquet"
        dates = pd.bdate_range(end=pd.Timestamp(last), periods=10)
        pd.DataFrame({
            "symbol": "AAA",
            "date": dates,
            "open": 100.0, "high": 100.0, "low": 100.0,
            "close": 100.0, "volume": 1_000.0,
        }).to_parquet(path, index=False)
        return path
    return _write


class TestBusinessDaysBetween:
    def test_consecutive_weekdays_is_one(self):
        # Thu 2026-09-03 -> Fri 2026-09-04
        assert business_days_between(date(2026, 9, 3), date(2026, 9, 4)) == 1

    def test_weekend_is_not_counted(self):
        # Fri 2026-09-04 -> Mon 2026-09-07 crosses a weekend: one business day.
        assert business_days_between(date(2026, 9, 4), date(2026, 9, 7)) == 1

    def test_friday_to_saturday_is_zero(self):
        assert business_days_between(date(2026, 9, 4), date(2026, 9, 5)) == 0

    def test_same_day_is_zero(self):
        assert business_days_between(date(2026, 9, 4), date(2026, 9, 4)) == 0

    def test_past_end_date_is_zero_not_negative(self):
        assert business_days_between(date(2026, 9, 10), date(2026, 9, 4)) == 0

    def test_full_week(self):
        # Fri 2026-09-04 -> Fri 2026-09-11 is five business days.
        assert business_days_between(date(2026, 9, 4), date(2026, 9, 11)) == 5


class TestLatestBarDate:
    def test_reads_the_newest_date(self, master):
        assert latest_bar_date(master("2026-09-04")) == date(2026, 9, 4)


class TestCheck:
    def test_same_day_is_fresh(self, master):
        fresh, lag, _ = check(master("2026-09-04"), today=date(2026, 9, 4))
        assert fresh and lag == 0

    def test_monday_against_friday_close_is_fresh(self, master):
        # The weekend must not be read as the pipeline having stalled.
        fresh, lag, _ = check(master("2026-09-04"), today=date(2026, 9, 7))
        assert fresh and lag == 1

    def test_long_holiday_cluster_is_tolerated(self, master):
        # Three business days of slack is the documented tolerance.
        fresh, lag, _ = check(master("2026-09-04"), today=date(2026, 9, 9))
        assert lag == DEFAULT_MAX_LAG_BUSINESS_DAYS
        assert fresh

    def test_one_day_past_the_limit_is_stale(self, master):
        fresh, lag, _ = check(master("2026-09-04"), today=date(2026, 9, 10))
        assert lag == 4
        assert not fresh

    def test_a_stalled_pipeline_is_stale(self, master):
        # The three-month outage this check exists to catch.
        fresh, lag, _ = check(master("2026-06-14"), today=date(2026, 9, 4))
        assert not fresh
        assert lag > 50

    def test_threshold_is_configurable(self, master):
        fresh, _, _ = check(master("2026-09-04"), max_lag=10, today=date(2026, 9, 18))
        assert fresh
