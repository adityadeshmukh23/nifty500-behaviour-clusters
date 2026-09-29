"""Tests for the incremental refresh of the OHLCV master.

The scenarios are the ones that actually hurt this pipeline, replayed against a
Yahoo they can steer (see `fake_yahoo.py`): a run that found "nothing to fetch"
every other evening, a session Yahoo had not finalised, holidays padded with
stale prices, and history re-based by a corporate action after it was stored.
"""

from datetime import date, timedelta

import numpy as np
import pandas as pd
import pytest

from fake_yahoo import FakeYahoo, initial_master
from nifty500.ingest import (
    OVERLAP_DAYS,
    PipelineError,
    UpdateReport,
    find_readjusted_symbols,
    load_symbols,
    merge_bars,
    plan_window,
    reshape_to_long,
    update_master,
    write_master,
)

SYMBOLS = ["AAA", "BBB", "CCC", "DDD", "EEE", "FFF"]
MANY = [f"S{i:02d}" for i in range(20)]          # 20 symbols: one missing is 5%, under the limit
SESSIONS = pd.bdate_range("2026-08-03", "2026-09-30")

MON, TUE, WED, THU, FRI = (date(2026, 9, d) for d in (14, 15, 16, 17, 18))


def run(master, yahoo, today, symbols=SYMBOLS, **kwargs):
    """One scheduled run on the evening of `today`, with Yahoo holding data through it."""
    yahoo.available_through = pd.Timestamp(today)
    return update_master(master, symbols, yahoo, today, **kwargs)


@pytest.fixture
def yahoo():
    return FakeYahoo(SESSIONS, SYMBOLS)


@pytest.fixture
def master(yahoo):
    """A master that holds everything through Monday 2026-09-14."""
    return initial_master(yahoo, SYMBOLS, MON)


def closes(bars, symbol):
    return bars[bars["symbol"] == symbol].set_index("date")["close"]


class TestReshapeToLong:
    def test_produces_the_master_schema(self, yahoo):
        raw = yahoo([s + ".NS" for s in SYMBOLS], date(2026, 9, 8), date(2026, 9, 12))
        bars, missing = reshape_to_long(raw, [s + ".NS" for s in SYMBOLS])
        assert list(bars.columns) == ["symbol", "date", "open", "high", "low", "close", "volume"]
        assert set(bars["symbol"]) == set(SYMBOLS)         # ".NS" is stripped
        assert str(bars["date"].dtype) == "datetime64[ns]"
        assert missing == []

    def test_a_symbol_absent_from_the_response_is_missing(self, yahoo):
        raw = yahoo(["AAA.NS", "BBB.NS"], date(2026, 9, 8), date(2026, 9, 12))
        bars, missing = reshape_to_long(raw, ["AAA.NS", "BBB.NS", "ZZZ.NS"])
        assert missing == ["ZZZ"]
        assert set(bars["symbol"]) == {"AAA", "BBB"}

    def test_a_symbol_yahoo_has_nothing_for_is_missing(self, yahoo):
        yahoo.absent = {"BBB"}
        raw = yahoo(["AAA.NS", "BBB.NS"], date(2026, 9, 8), date(2026, 9, 12))
        bars, missing = reshape_to_long(raw, ["AAA.NS", "BBB.NS"])
        assert missing == ["BBB"]
        assert set(bars["symbol"]) == {"AAA"}

    def test_unpriced_rows_are_dropped_and_leave_the_symbol_missing(self, yahoo):
        # A symbol whose only rows are volume-without-prices has no usable bar.
        yahoo.unfinalised = set(SESSIONS)
        raw = yahoo(["AAA.NS"], date(2026, 9, 8), date(2026, 9, 12))
        bars, missing = reshape_to_long(raw, ["AAA.NS"])
        assert bars.empty
        assert missing == ["AAA"]

    def test_a_single_symbol_response_with_flat_columns_is_accepted(self, yahoo):
        # Older yfinance versions returned flat columns for one ticker.
        raw = yahoo(["AAA.NS"], date(2026, 9, 8), date(2026, 9, 12))["AAA.NS"]
        bars, missing = reshape_to_long(raw, ["AAA.NS"])
        assert set(bars["symbol"]) == {"AAA"}
        assert missing == []

    def test_timezone_aware_dates_are_made_naive(self, yahoo):
        raw = yahoo(["AAA.NS"], date(2026, 9, 8), date(2026, 9, 12))
        raw.index = raw.index.tz_localize("Asia/Kolkata")
        bars, _ = reshape_to_long(raw, ["AAA.NS"])
        assert bars["date"].dt.tz is None
        assert bars["date"].iloc[0] == pd.Timestamp("2026-09-08")

    def test_a_response_missing_a_column_fails_loudly(self, yahoo):
        raw = yahoo(["AAA.NS"], date(2026, 9, 8), date(2026, 9, 12))
        raw = raw.drop(columns="Volume", level="Price")
        with pytest.raises(PipelineError, match="schema"):
            reshape_to_long(raw, ["AAA.NS"])


class TestPlanWindow:
    def test_end_is_the_day_after_today_so_a_session_that_just_closed_is_included(self):
        # `end` is exclusive: asking for [.., Tue) would leave out Tuesday's own bar.
        _, end = plan_window(MON, TUE)
        assert end == WED

    def test_start_re_covers_sessions_already_held(self):
        start, _ = plan_window(MON, TUE)
        assert start == MON - timedelta(days=OVERLAP_DAYS)

    def test_a_clock_behind_the_master_still_gives_a_valid_window(self):
        start, end = plan_window(MON, date(2026, 9, 10))
        assert start < end


class TestMergeBars:
    def test_a_newer_bar_supersedes_the_stored_one(self, master):
        stored = master.iloc[[0]].copy()
        newer = stored.copy()
        newer["close"] = 999.0
        merged = merge_bars(stored, newer)
        assert len(merged) == 1
        assert merged["close"].iloc[0] == 999.0

    def test_the_result_is_sorted_and_unique(self, master):
        shuffled = master.sample(frac=1, random_state=1)
        merged = merge_bars(shuffled, master)
        assert not merged.duplicated(["symbol", "date"]).any()
        assert merged.equals(merged.sort_values(["symbol", "date"]).reset_index(drop=True))


class TestFindReadjustedSymbols:
    def test_identical_overlap_is_not_drift(self, master):
        assert find_readjusted_symbols(master, master.copy()) == []

    def test_a_rescaled_symbol_is_found_and_only_that_one(self, master):
        fetched = master.copy()
        fetched.loc[fetched["symbol"] == "CCC", "close"] *= 0.98
        assert find_readjusted_symbols(master, fetched) == ["CCC"]

    def test_a_young_bar_that_firms_up_is_not_a_re_adjustment(self, master):
        # The newest stored bar may still be provisional. If its close settles 0.3%
        # differently, that is the fresh fetch replacing it -- not a corporate action,
        # and certainly not a reason to re-download every symbol's history.
        fetched = master.copy()
        newest = fetched["date"] == fetched["date"].max()
        fetched.loc[newest, "close"] *= 1.003
        assert find_readjusted_symbols(master, fetched) == []

    def test_an_older_bar_that_differs_is_a_re_adjustment(self, master):
        fetched = master.copy()
        older = sorted(fetched["date"].unique())[-5]
        row = (fetched["symbol"] == "AAA") & (fetched["date"] == older)
        fetched.loc[row, "close"] *= 1.01
        assert find_readjusted_symbols(master, fetched) == ["AAA"]

    def test_noise_below_the_tolerance_is_ignored(self, master):
        fetched = master.copy()
        fetched["close"] *= 1 + 1e-9
        assert find_readjusted_symbols(master, fetched) == []


class TestUpdateMaster:
    def test_an_evening_run_lands_that_evenings_session(self, yahoo, master):
        # Regression. The master already holds Monday; the run on Tuesday evening
        # used to find `last_date + 1 == today` and exit "already up to date",
        # so data only landed every other day.
        updated, report = run(master, yahoo, TUE)
        assert report.last_bar_after == TUE
        assert report.new_rows == len(SYMBOLS)
        assert updated["date"].max() == pd.Timestamp(TUE)
        assert yahoo.calls[-1][2] > TUE                     # the request reached past today

    def test_every_consecutive_evening_lands_its_session(self, yahoo, master):
        for day in (TUE, WED, THU, FRI):
            master, report = run(master, yahoo, day)
            assert report.last_bar_after == day
            assert report.new_rows == len(SYMBOLS)

    def test_an_evening_with_no_new_session_is_neither_an_error_nor_a_change(self, yahoo, master):
        # A market holiday: Yahoo has nothing past Monday, but the overlap is still
        # answered, so this is distinguishable from an upstream that is down.
        yahoo.available_through = pd.Timestamp(MON)
        updated, report = update_master(master, SYMBOLS, yahoo, TUE)
        assert report.new_rows == 0
        assert not report.changed
        assert updated.equals(master)

    def test_running_twice_changes_nothing_the_second_time(self, yahoo, master):
        once, first = run(master, yahoo, TUE)
        twice, second = run(once, yahoo, TUE)
        assert first.changed and not second.changed
        assert twice.equals(once)

    def test_an_empty_response_fails_loudly_even_on_a_holiday(self, master):
        with pytest.raises(PipelineError, match="no data"):
            update_master(master, SYMBOLS, lambda *a: pd.DataFrame(), TUE)

    def test_a_none_response_fails_loudly(self, master):
        with pytest.raises(PipelineError, match="no data"):
            update_master(master, SYMBOLS, lambda *a: None, TUE)

    def test_a_response_with_no_valid_bar_at_all_fails(self, yahoo, master):
        yahoo.unfinalised = set(SESSIONS)
        with pytest.raises(PipelineError, match="valid bar"):
            run(master, yahoo, TUE)

    def test_a_master_with_nothing_valid_in_it_is_an_error(self, yahoo, master):
        broken = master.copy()
        broken[["open", "high", "low", "close"]] = np.nan
        with pytest.raises(PipelineError, match="no valid bars"):
            run(broken, yahoo, TUE)

    def test_no_symbols_is_an_error(self, yahoo, master):
        with pytest.raises(PipelineError, match="no symbols"):
            run(master, yahoo, TUE, symbols=[])

    def test_a_few_symbols_without_data_are_tolerated(self):
        # A delisted constituent is expected; it must not fail every run.
        many = FakeYahoo(SESSIONS, MANY)
        held = initial_master(many, MANY, MON)
        many.absent = {"S07"}
        _, report = run(held, many, TUE, symbols=MANY)
        assert report.missing == ["S07"]

    def test_too_many_symbols_without_data_fails(self):
        many = FakeYahoo(SESSIONS, MANY)
        held = initial_master(many, MANY, MON)
        many.absent = {"S01", "S02", "S03"}
        with pytest.raises(PipelineError, match="partial update"):
            run(held, many, TUE, symbols=MANY)


class TestSelfHealing:
    def test_an_unfinalised_session_is_left_out_and_picked_up_the_next_day(self, yahoo, master):
        yahoo.unfinalised = {pd.Timestamp(TUE)}
        held, report = run(master, yahoo, TUE)
        assert report.new_rows == 0
        assert held["date"].max() == pd.Timestamp(MON)

        yahoo.unfinalised = set()                            # Yahoo has finalised Tuesday
        healed, report = run(held, yahoo, WED)
        assert report.new_rows == 2 * len(SYMBOLS)           # Tuesday and Wednesday
        assert not healed[["open", "high", "low", "close"]].isna().any().any()

    def test_a_revised_bar_supersedes_the_provisional_one(self, yahoo, master):
        # Yahoo's volume for the newest session firms up after the close.
        held, _ = run(master, yahoo, TUE)
        yahoo.raw["AAA"].loc[pd.Timestamp(TUE), "volume"] *= 1.5
        revised, report = run(held, yahoo, WED)
        assert report.revised_rows == 1
        want = yahoo.raw["AAA"].loc[pd.Timestamp(TUE), "volume"]
        got = revised[(revised["symbol"] == "AAA")
                      & (revised["date"] == pd.Timestamp(TUE))]["volume"].iloc[0]
        assert got == want

    def test_unpriced_rows_already_in_the_master_are_removed(self, yahoo, master):
        # The live master carried a whole session of these: 503 rows, 2026-09-21.
        poisoned = master.copy()
        extra = poisoned[poisoned["date"] == pd.Timestamp(MON)].copy()
        extra["date"] = pd.Timestamp(TUE)
        extra[["open", "high", "low", "close"]] = np.nan
        poisoned = pd.concat([poisoned, extra], ignore_index=True)

        healed, report = run(poisoned, yahoo, WED)
        assert report.removed_from_master["unpriced"] == len(SYMBOLS)
        assert not healed[["open", "high", "low", "close"]].isna().any().any()
        assert report.new_rows == 2 * len(SYMBOLS)           # Tuesday re-fetched, plus Wednesday

    def test_placeholder_sessions_already_in_the_master_are_removed(self, yahoo, master):
        holiday = pd.Timestamp("2026-09-11")
        padded = master[master["date"] == pd.Timestamp("2026-09-10")].copy()
        padded["date"] = holiday
        for column in ("open", "high", "low"):
            padded[column] = padded["close"]
        padded["volume"] = 0.0
        polluted = pd.concat([master, padded], ignore_index=True)

        healed, report = run(polluted, yahoo, TUE)
        # 2026-09-11 is a real session in FakeYahoo, so it comes back real -- but the
        # padded copy must not survive next to it.
        assert report.removed_from_master["placeholder"] == len(SYMBOLS)
        assert not ((healed["volume"] == 0) & (healed["open"] == healed["close"])).any()

    def test_a_padded_holiday_from_upstream_never_enters_the_master(self, yahoo, master):
        holiday = date(2026, 9, 16)
        yahoo.padded_holidays = [holiday]
        yahoo.sessions = yahoo.sessions[yahoo.sessions != pd.Timestamp(holiday)]
        updated, _ = run(master, yahoo, THU)
        assert pd.Timestamp(holiday) not in set(updated["date"])
        assert updated["date"].max() == pd.Timestamp(THU)


class TestAdjustmentDrift:
    def test_a_dividend_after_the_last_update_refreshes_that_symbols_history(self, yahoo, master):
        yahoo.add_action("BBB", TUE, 0.98)                   # 2% dividend, ex-date Tuesday
        updated, report = run(master, yahoo, TUE)
        assert report.refreshed == ["BBB"]
        # The stored history now equals what a full re-download returns.
        want = yahoo.adjusted("BBB").set_index("date")["close"]
        pd.testing.assert_series_equal(closes(updated, "BBB"), want, check_names=False)

    def test_a_bonus_issue_does_not_leave_a_seam_in_the_series(self, yahoo, master):
        # Regression for the append-only design: a 1:2 bonus multiplies earlier
        # prices by 2/3. Appending Tuesday on the old basis would show a -33% day.
        yahoo.add_action("CCC", TUE, 2 / 3)
        updated, report = run(master, yahoo, TUE)
        assert report.refreshed == ["CCC"]
        assert closes(updated, "CCC").pct_change().abs().max() < 0.05

    def test_symbols_without_an_action_are_left_alone(self, yahoo, master):
        yahoo.add_action("CCC", TUE, 0.98)
        updated, _ = run(master, yahoo, TUE)
        untouched = master[master["symbol"] == "AAA"].reset_index(drop=True)
        kept = updated[(updated["symbol"] == "AAA") & (updated["date"] <= pd.Timestamp(MON))]
        pd.testing.assert_frame_equal(kept.reset_index(drop=True), untouched)

    def test_no_action_means_no_extra_download(self, yahoo, master):
        before = len(yahoo.calls)
        _, report = run(master, yahoo, TUE)
        assert report.refreshed == []
        assert len(yahoo.calls) - before == 1

    def test_a_failed_refresh_keeps_the_stored_rows_and_adds_no_new_ones(self):
        many = FakeYahoo(SESSIONS, MANY)
        held = initial_master(many, MANY, MON)
        many.add_action("S05", TUE, 0.97)

        def flaky(symbols_ns, start, end):
            if start < date(2026, 9, 1):                     # the whole-history request
                return pd.DataFrame()
            return many(symbols_ns, start, end)

        many.available_through = pd.Timestamp(TUE)
        updated, report = update_master(held, MANY, flaky, TUE)
        assert report.refresh_failed == ["S05"]
        stored = held[held["symbol"] == "S05"].reset_index(drop=True)
        got = updated[updated["symbol"] == "S05"].reset_index(drop=True)
        pd.testing.assert_frame_equal(got, stored)
        # The mismatch is still there next run, so it is retried rather than lost.
        assert find_readjusted_symbols(updated, many.adjusted("S05")) == ["S05"]

    def test_a_truncated_refresh_cannot_shorten_the_history(self):
        many = FakeYahoo(SESSIONS, MANY)
        held = initial_master(many, MANY, MON)
        many.add_action("S05", TUE, 0.97)

        def truncating(symbols_ns, start, end):
            frame = many(symbols_ns, start, end)
            if start < date(2026, 9, 1):
                return frame.iloc[-5:]                       # only the last five sessions
            return frame

        many.available_through = pd.Timestamp(TUE)
        updated, report = update_master(held, MANY, truncating, TUE)
        assert report.refresh_failed == ["S05"]
        assert len(updated[updated["symbol"] == "S05"]) == len(held[held["symbol"] == "S05"])

    def test_a_symbol_new_to_the_constituents_is_backfilled(self, yahoo):
        wider = FakeYahoo(SESSIONS, [*SYMBOLS, "NEWCO"])
        held = initial_master(wider, SYMBOLS, MON)           # NEWCO was never stored
        updated, report = run(held, wider, TUE, symbols=[*SYMBOLS, "NEWCO"])
        assert "NEWCO" in report.refreshed
        assert len(updated[updated["symbol"] == "NEWCO"]) == len(wider.adjusted("NEWCO"))

    def test_a_forced_full_refresh_redownloads_every_symbol(self, yahoo, master):
        updated, report = run(master, yahoo, TUE, refresh_symbols=SYMBOLS)
        assert sorted(report.refreshed) == sorted(SYMBOLS)
        for symbol in SYMBOLS:
            want = yahoo.adjusted(symbol).set_index("date")["close"]
            pd.testing.assert_series_equal(closes(updated, symbol), want, check_names=False)


class TestReportAndIo:
    def test_the_report_reads_as_a_log(self, yahoo, master):
        _, report = run(master, yahoo, TUE)
        text = "\n".join(report.lines())
        assert isinstance(report, UpdateReport)
        assert "last bar" in text and "->" in text

    def test_write_master_round_trips_and_leaves_no_scratch_file(self, master, tmp_path):
        path = tmp_path / "master.parquet"
        write_master(master, path)
        pd.testing.assert_frame_equal(pd.read_parquet(path), master)
        assert list(tmp_path.iterdir()) == [path]

    def test_load_symbols_trims_whitespace(self, tmp_path):
        path = tmp_path / "constituents.csv"
        path.write_text("Company Name,Industry,Symbol,Series,ISIN Code\n"
                        "A Ltd.,Auto,AAA ,EQ,X1\nB Ltd.,Auto, BBB,EQ,X2\n")
        assert load_symbols(path) == ["AAA", "BBB"]
