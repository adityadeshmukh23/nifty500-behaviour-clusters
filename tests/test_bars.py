"""Tests for bar hygiene.

Each case is a shape of row that really was in the master: unpriced bars (all 503
symbols on 2026-09-21), and zero-volume placeholders (five market holidays plus a
scatter of single-stock days).
"""

import numpy as np
import pandas as pd
import pytest

from nifty500.bars import BAR_COLUMNS, clean_bars, flag_invalid_bars, is_placeholder


def bar(symbol="AAA", date="2026-09-01", open_=100.0, high=101.0, low=99.0, close=100.5,
        volume=1_000.0):
    return {"symbol": symbol, "date": pd.Timestamp(date), "open": open_, "high": high,
            "low": low, "close": close, "volume": volume}


def frame(*rows):
    return pd.DataFrame(list(rows), columns=BAR_COLUMNS)


def flat(symbol, date, volume=0.0, price=100.0):
    return bar(symbol, date, price, price, price, price, volume)


class TestUnpricedBars:
    def test_a_real_bar_is_kept(self):
        assert len(clean_bars(frame(bar()))) == 1

    def test_volume_without_prices_is_dropped(self):
        # The 2026-09-21 shape: Yahoo had a volume but no price for the session.
        unpriced = bar(open_=np.nan, high=np.nan, low=np.nan, close=np.nan, volume=573_949.0)
        assert clean_bars(frame(unpriced)).empty

    def test_a_partly_priced_bar_is_dropped(self):
        # Every stored bar carries all four prices, so consumers can rely on that.
        assert clean_bars(frame(bar(open_=np.nan))).empty

    @pytest.mark.parametrize("close", [0.0, -5.0])
    def test_a_non_positive_close_is_dropped(self, close):
        assert clean_bars(frame(bar(close=close))).empty


class TestPlaceholderBars:
    def test_zero_volume_and_identical_prices_is_a_placeholder(self):
        assert is_placeholder(frame(flat("AAA", "2026-09-01"))).all()

    def test_zero_volume_with_moving_prices_is_a_real_bar(self):
        # Yahoo failed to report the volume, but the prices were genuinely printed.
        assert len(clean_bars(frame(bar(volume=0.0)))) == 1

    def test_flat_prices_with_volume_is_a_real_bar(self):
        # A thin stock that only ever traded at one price that day.
        assert len(clean_bars(frame(flat("AAA", "2026-09-01", volume=250.0)))) == 1

    def test_isolated_placeholders_are_dropped_and_the_rest_kept(self):
        rows = [bar(f"S{i}", "2026-09-01") for i in range(9)]
        rows.append(flat("ILLIQUID", "2026-09-01"))
        cleaned = clean_bars(frame(*rows))
        assert len(cleaned) == 9
        assert "ILLIQUID" not in set(cleaned["symbol"])


class TestClosedSessions:
    def test_a_market_holiday_is_dropped_whole(self):
        # 9 of 10 symbols carry a padded bar; the tenth has a real-looking bar.
        # The exchange was shut, so that one is not trustworthy either.
        rows = [flat(f"S{i}", "2026-06-26") for i in range(9)]
        rows.append(bar("STRAGGLER", "2026-06-26"))
        rows += [bar(f"S{i}", "2026-06-25") for i in range(10)]
        cleaned = clean_bars(frame(*rows))
        assert set(cleaned["date"]) == {pd.Timestamp("2026-06-25")}
        assert len(cleaned) == 10

    def test_a_normal_day_with_a_few_placeholders_is_not_a_holiday(self):
        rows = [bar(f"S{i}", "2026-09-01") for i in range(8)]
        rows += [flat("X", "2026-09-01"), flat("Y", "2026-09-01")]
        assert len(clean_bars(frame(*rows))) == 8

    def test_a_date_with_no_priced_rows_does_not_break_the_share_maths(self):
        unpriced = [bar(f"S{i}", "2026-09-21", open_=np.nan, high=np.nan, low=np.nan,
                        close=np.nan, volume=10.0) for i in range(5)]
        assert clean_bars(frame(*unpriced)).empty


class TestFlagInvalidBars:
    def test_reasons_are_mutually_exclusive(self):
        rows = [
            bar("OK", "2026-09-01"),
            bar("UNPRICED", "2026-09-01", open_=np.nan, high=np.nan, low=np.nan, close=np.nan),
            flat("PLACEHOLDER", "2026-09-01"),
        ]
        flags = flag_invalid_bars(frame(*rows))
        assert flags.sum(axis=1).max() == 1
        assert flags["unpriced"].tolist() == [False, True, False]
        assert flags["placeholder"].tolist() == [False, False, True]

    def test_the_tally_names_what_was_wrong(self):
        rows = [flat(f"S{i}", "2026-06-26") for i in range(9)] + [bar("S9", "2026-06-26")]
        flags = flag_invalid_bars(frame(*rows))
        assert int(flags["placeholder"].sum()) == 9
        assert int(flags["closed_session"].sum()) == 1

    def test_the_input_is_not_modified(self):
        original = frame(bar(), flat("B", "2026-09-01"))
        snapshot = original.copy()
        clean_bars(original)
        pd.testing.assert_frame_equal(original, snapshot)
