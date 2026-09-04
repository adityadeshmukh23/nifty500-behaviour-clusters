"""Tests for the behavioural feature math.

Each case uses a synthetic price path with a known answer, so a regression in
the feature definitions fails here rather than silently shifting every cluster.
"""

import numpy as np
import pandas as pd
import pytest

from src.features import (CORPORATE_ACTION_THRESHOLD, TRADING_DAYS,
                          _max_drawdown, _trailing_return, build_features,
                          daily_returns, select_universe)


def make_prices(symbol, closes, start="2022-01-03"):
    dates = pd.bdate_range(start, periods=len(closes))
    return pd.DataFrame({
        "symbol": symbol,
        "date": dates,
        "open": closes,
        "high": closes,
        "low": closes,
        "close": closes,
        "volume": 1_000.0,
    })


class TestMaxDrawdown:
    def test_monotonic_rise_has_no_drawdown(self):
        assert _max_drawdown(pd.Series([0.01] * 10)) == pytest.approx(0.0)

    def test_halving_is_minus_fifty_percent(self):
        # +0% then -50% in one step.
        assert _max_drawdown(pd.Series([0.0, -0.5])) == pytest.approx(-0.5)

    def test_measures_peak_to_trough_not_start_to_end(self):
        # Up 100%, then down 50% -> back to the start, but the drawdown is -50%.
        assert _max_drawdown(pd.Series([1.0, -0.5])) == pytest.approx(-0.5)


class TestTrailingReturn:
    def test_simple_return_over_lookback(self):
        closes = pd.Series([100.0] * 10 + [110.0])
        assert _trailing_return(closes, 10) == pytest.approx(0.10)

    def test_returns_nan_when_history_is_shorter_than_lookback(self):
        assert np.isnan(_trailing_return(pd.Series([100.0, 101.0]), 63))


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


class TestSelectUniverse:
    def test_symbol_with_short_history_is_dropped(self):
        full = make_prices("FULL", list(np.linspace(100, 120, 800)))
        # Same calendar, but only the last 50 sessions are present.
        short = make_prices("SHORT", list(np.linspace(100, 120, 800)))[-50:]
        prices = pd.concat([full, short], ignore_index=True)

        _, kept, dropped = select_universe(prices, window_years=3, min_coverage=0.95)
        assert "FULL" in kept
        assert "SHORT" in dropped


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
