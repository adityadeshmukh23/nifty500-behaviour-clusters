"""Tests for the behavioural feature math.

Each case uses a synthetic price path with a known answer, so a regression in
the feature definitions fails here rather than silently shifting every cluster.
"""

import numpy as np
import pandas as pd
import pytest

from nifty500.features import (
    CORPORATE_ACTION_THRESHOLD,
    LAST_BAR_TOLERANCE_SESSIONS,
    TRADING_DAYS,
    _beta,
    _max_drawdown,
    _trailing_return,
    adjusted_closes,
    build_features,
    daily_returns,
    select_universe,
    session_evidence,
)


def make_prices(symbol, closes, start="2022-01-03", volume=1_000.0):
    dates = pd.bdate_range(start, periods=len(closes))
    return pd.DataFrame({
        "symbol": symbol,
        "date": dates,
        "open": closes,
        "high": closes,
        "low": closes,
        "close": closes,
        "volume": volume,
    })


def steady_growth(sessions=800, daily=0.0005):
    return 100 * np.exp(daily * np.arange(sessions))


def reviewed(symbol, closes, position):
    """A one-row register naming the session at `position` on a `make_prices` path."""
    day = pd.bdate_range("2022-01-03", periods=len(closes))[position]
    return pd.DataFrame({"symbol": [symbol], "date": [day]})


class TestMaxDrawdown:
    def test_monotonic_rise_has_no_drawdown(self):
        assert _max_drawdown(pd.Series(np.linspace(100, 200, 10))) == pytest.approx(0.0)

    def test_halving_is_minus_fifty_percent(self):
        assert _max_drawdown(pd.Series([100.0, 100.0, 50.0])) == pytest.approx(-0.5)

    def test_measures_peak_to_trough_not_start_to_end(self):
        # Up 100%, then down 50% -> back to the start, but the drawdown is -50%.
        assert _max_drawdown(pd.Series([100.0, 200.0, 100.0])) == pytest.approx(-0.5)

    def test_a_fall_from_the_very_first_bar_counts(self):
        # The first observation is a candidate peak. Measuring from the second bar,
        # as an earlier version did, would report no drawdown here at all.
        assert _max_drawdown(pd.Series([100.0, 50.0])) == pytest.approx(-0.5)


class TestTrailingReturn:
    def test_simple_return_over_lookback(self):
        closes = pd.Series([100.0] * 10 + [110.0])
        assert _trailing_return(closes, 10) == pytest.approx(0.10)

    def test_returns_nan_when_history_is_shorter_than_lookback(self):
        assert np.isnan(_trailing_return(pd.Series([100.0, 101.0]), 63))


class TestBeta:
    def test_a_scaled_market_has_exactly_that_beta_even_across_gaps(self):
        # Covariance and variance must come from the same sessions. Dividing the
        # covariance over the stock's sessions by the market's variance over *all*
        # sessions understates this beta once a few days are masked.
        rng = np.random.default_rng(3)
        market = pd.Series(rng.normal(0, 0.01, 300))
        stock = 2 * market
        stock.iloc[[10, 11, 100, 250]] = np.nan
        assert _beta(stock.dropna(), market) == pytest.approx(2.0, abs=1e-12)

    def test_too_few_common_sessions_gives_nan(self):
        assert np.isnan(_beta(pd.Series([0.01]), pd.Series([0.02])))


class TestDailyReturns:
    def test_gaps_are_not_padded_into_zero_returns(self):
        # A missing session must stay NaN, not become a fabricated 0% day.
        prices = make_prices("AAA", [100.0, 101.0, 102.0])
        prices.loc[1, "close"] = np.nan
        assert daily_returns(prices)["AAA"].isna().sum() >= 1

    def test_corporate_action_jump_is_masked(self):
        # A -80% single session is a split/demerger artifact, not behaviour.
        prices = make_prices("AAA", [100.0, 20.0, 21.0])
        returns = daily_returns(prices, mask_corporate_actions=True)["AAA"]
        assert returns.abs().max() < CORPORATE_ACTION_THRESHOLD

    def test_masking_can_be_disabled(self):
        prices = make_prices("AAA", [100.0, 20.0, 21.0])
        returns = daily_returns(prices, mask_corporate_actions=False)["AAA"]
        assert returns.min() == pytest.approx(-0.8)

    def test_a_reviewed_session_is_masked_even_below_the_threshold(self):
        # A 1:2 bonus moves the price by a third: under the 40% threshold, so only
        # the reviewed register can catch it.
        closes = [100.0, 100.0, 66.7, 66.7]
        prices = make_prices("AAA", closes)
        assert daily_returns(prices)["AAA"].min() == pytest.approx(-0.333, abs=1e-3)

        masked = daily_returns(prices, reviewed_actions=reviewed("AAA", closes, 2))["AAA"]
        assert masked.isna().sum() == 2            # the undefined first return, and the session
        assert masked.min() == pytest.approx(0.0)

    def test_a_register_naming_an_unknown_symbol_or_date_is_ignored(self):
        closes = [100.0, 101.0, 102.0]
        ghost = pd.DataFrame({"symbol": ["ZZZ", "AAA"],
                              "date": [pd.Timestamp("2022-01-04"), pd.Timestamp("2030-01-01")]})
        returns = daily_returns(make_prices("AAA", closes), reviewed_actions=ghost)["AAA"]
        assert returns.notna().sum() == 2


class TestAdjustedCloses:
    def test_a_masked_level_shift_is_divided_out_of_the_path(self):
        closes = [100.0, 101.0, 35.35, 35.7]                  # -65% demerger on day 3
        adjusted = adjusted_closes(make_prices("AAA", closes))["AAA"]
        assert adjusted.iloc[-1] / adjusted.iloc[0] == pytest.approx(35.7 / 100 / 0.35)
        assert adjusted.pct_change().abs().max() < CORPORATE_ACTION_THRESHOLD

    def test_a_path_without_masked_sessions_is_unchanged(self):
        closes = list(steady_growth(50))
        adjusted = adjusted_closes(make_prices("AAA", closes))["AAA"]
        np.testing.assert_allclose(adjusted.to_numpy(), closes)


class TestSelectUniverse:
    def test_symbol_with_short_history_is_dropped(self):
        full = make_prices("FULL", list(np.linspace(100, 120, 800)))
        # Same calendar, but only the last 50 sessions are present.
        short = make_prices("SHORT", list(np.linspace(100, 120, 800)))[-50:]
        prices = pd.concat([full, short], ignore_index=True)

        _, kept, dropped = select_universe(prices, window_years=3, min_coverage=0.95)
        assert "FULL" in kept
        assert "SHORT" in dropped

    def test_a_stock_that_stopped_trading_is_dropped_despite_ample_coverage(self):
        # 20 missing sessions is 97.5% coverage, but a stock with no recent bar has
        # no current behaviour to measure.
        full = make_prices("LIVE", list(np.linspace(100, 120, 800)))
        gone = make_prices("GONE", list(np.linspace(100, 120, 800)))[:-20]
        prices = pd.concat([full, gone], ignore_index=True)

        _, kept, dropped = select_universe(prices, window_years=3, min_coverage=0.95)
        assert "LIVE" in kept
        assert "GONE" in dropped

    def test_one_missing_recent_bar_does_not_eject_a_healthy_stock(self):
        full = make_prices("LIVE", list(np.linspace(100, 120, 800)))
        blip = make_prices("BLIP", list(np.linspace(100, 120, 800))).drop(index=799)
        assert LAST_BAR_TOLERANCE_SESSIONS > 1

        _, kept, _ = select_universe(pd.concat([full, blip], ignore_index=True))
        assert "BLIP" in kept

    def test_as_of_anchors_the_window_and_ignores_later_bars(self):
        prices = make_prices("AAA", list(np.linspace(100, 120, 800)))
        cutoff = prices["date"].iloc[699]
        window, _, _ = select_universe(prices, as_of=cutoff)
        assert window["date"].max() == cutoff


class TestBuildFeatures:
    def test_volatility_is_annualised(self):
        rng = np.random.default_rng(0)
        daily_sigma = 0.02
        steps = rng.normal(0, daily_sigma, 800)
        path = 100 * np.exp(np.cumsum(steps))

        features = build_features(make_prices("AAA", list(path)))
        expected = daily_sigma * np.sqrt(TRADING_DAYS)
        assert features.loc["AAA", "ann_volatility"] == pytest.approx(expected, rel=0.15)

    def test_flat_price_has_zero_volatility_and_no_drawdown(self):
        features = build_features(make_prices("AAA", [100.0] * 800))
        assert features.loc["AAA", "ann_volatility"] == pytest.approx(0.0)
        assert features.loc["AAA", "max_drawdown"] == pytest.approx(0.0)

    def test_masked_observations_are_counted(self):
        path = [100.0] * 400 + [20.0] + [20.0] * 399
        features = build_features(make_prices("AAA", path))
        assert features.attrs["masked_observations"] == 1

    def test_momentum_and_drawdown_do_not_see_a_masked_corporate_action(self):
        # Regression. Returns were masked but momentum read the raw close path, so a
        # stock that lost 65% to a demerger looked like -65% momentum and a -65%
        # drawdown -- one of the ten features contradicting the other eight.
        path = steady_growth()
        path[700:] *= 0.35
        features = build_features(make_prices("AAA", list(path)))
        row = features.loc["AAA"]
        # The masked session counts as no move, exactly as it does in the returns:
        # inside the 12-month lookback that leaves 251 days of growth, not 252.
        expected = np.exp(0.0005 * (TRADING_DAYS - 1)) - 1
        assert row["mom_12m"] == pytest.approx(expected, abs=1e-6)
        assert row["mom_3m"] == pytest.approx(np.exp(0.0005 * 63) - 1, abs=1e-6)   # outside it
        assert row["max_drawdown"] == pytest.approx(0.0, abs=1e-9)

    def test_reviewed_session_is_kept_out_of_every_feature(self):
        path = steady_growth()
        path[700:] *= 2 / 3                                     # a 1:2 bonus: -33%, under the bar
        closes = list(path)

        naive = build_features(make_prices("AAA", closes)).loc["AAA"]
        assert naive["max_drawdown"] < -0.3                     # the artefact reads as a crash

        register = reviewed("AAA", closes, 700)
        result = build_features(make_prices("AAA", closes), reviewed_actions=register)
        assert result.loc["AAA", "max_drawdown"] == pytest.approx(0.0, abs=1e-9)
        assert result.loc["AAA", "return_kurtosis"] < naive["return_kurtosis"]
        assert result.attrs["masked_observations"] == 1

    def test_as_of_makes_the_result_independent_of_later_data(self):
        rng = np.random.default_rng(1)
        prices = make_prices("AAA", list(100 * np.exp(np.cumsum(rng.normal(0, 0.01, 800)))))
        cutoff = prices["date"].iloc[699]

        pinned = build_features(prices, as_of=cutoff)
        truncated = build_features(prices[prices["date"] <= cutoff])
        pd.testing.assert_frame_equal(pinned, truncated)

    def test_bars_that_are_not_real_do_not_change_the_features(self):
        # Placeholder sessions are zero-return days invented by the data source; the
        # features must come out the same whether or not the caller cleaned them.
        rng = np.random.default_rng(2)
        clean = make_prices("AAA", list(100 * np.exp(np.cumsum(rng.normal(0, 0.01, 800)))))
        padded = clean.iloc[[300, 500]].copy()
        padded["date"] = padded["date"] + pd.offsets.Week(weekday=5)   # Saturday: no session
        for column in ("open", "high", "low"):
            padded[column] = padded["close"]
        padded["volume"] = 0.0
        dirty = pd.concat([clean, padded], ignore_index=True)

        pd.testing.assert_frame_equal(build_features(dirty), build_features(clean))


class TestSessionEvidence:
    def test_separates_a_re_basing_from_a_crash(self):
        volume = np.full(60, 100.0)
        closes = np.full(60, 100.0)
        # A crash: price -30% on 10x volume, volume fading afterwards.
        closes[40:] = 70.0
        volume[40] = 1_000.0
        volume[41:] = 130.0
        prices = make_prices("AAA", list(closes))
        prices["volume"] = volume

        evidence = session_evidence(prices, "AAA", prices["date"].iloc[40])
        assert evidence["price_ratio"] == pytest.approx(0.7)
        assert evidence["day_volume_x"] == pytest.approx(10.0)
        assert evidence["next20_volume_x"] == pytest.approx(1.3)
