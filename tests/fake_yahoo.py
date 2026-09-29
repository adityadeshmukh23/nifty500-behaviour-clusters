"""A stand-in for `yf.download(..., auto_adjust=True, group_by="ticker")`.

The pipeline's hard cases are properties of what Yahoo *returns*, not of our code,
so the tests need a Yahoo they can steer. `FakeYahoo` holds raw (unadjusted) bars
per symbol and answers the way the real thing does:

* the frame has MultiIndex columns (Ticker, Price) and an index named "Date",
  reindexed to the union of every ticker's dates -- so a symbol with no data
  comes back as all-NaN rows, exactly as `yfinance.multi.reindex_dfs` produces;
* every bar before a corporate action's ex-date is rescaled by that action's
  factor as of the moment of the call (`auto_adjust=True`), which is what makes
  bars stored on an earlier day go stale;
* a session can come back *unfinalised* (prices null, volume present) or a market
  holiday can come back *padded* (previous close, zero volume) -- the two shapes
  that got into the real master.
"""

from datetime import timedelta

import numpy as np
import pandas as pd

PRICE = ["open", "high", "low", "close"]


class FakeYahoo:
    def __init__(self, sessions, symbols, seed=0):
        self.sessions = pd.DatetimeIndex(sessions)
        self.available_through = self.sessions.max()
        self.actions = []            # (symbol, ex_date, factor)
        self.unfinalised = set()     # dates whose prices come back null
        self.padded_holidays = []    # extra dates returned as carried-forward bars
        self.absent = set()          # symbols Yahoo has nothing for
        self.calls = []              # (symbols_ns, start, end) of every request
        self.raw = {s: self._raw_bars(seed + i) for i, s in enumerate(symbols)}

    def _raw_bars(self, seed):
        rng = np.random.default_rng(seed)
        close = 100 * np.exp(np.cumsum(rng.normal(0, 0.01, len(self.sessions))))
        return pd.DataFrame({
            "open": close * (1 + rng.normal(0, 0.002, len(close))),
            "high": close * 1.01,
            "low": close * 0.99,
            "close": close,
            "volume": rng.integers(100_000, 1_000_000, len(close)).astype(float),
        }, index=self.sessions)

    def add_action(self, symbol, ex_date, factor):
        """A split, bonus or dividend that takes effect on `ex_date`.

        Two things happen, as in the real market. The raw price steps down by
        `factor` on the ex-date; and from then on Yahoo rescales every earlier bar
        by the same factor, which is what makes the *adjusted* series continuous
        and any bar stored before the event stale.
        """
        ex_date = pd.Timestamp(ex_date)
        self.actions.append((symbol, ex_date, factor))
        raw = self.raw[symbol]
        raw.loc[raw.index >= ex_date, PRICE] *= factor

    def _factor(self, symbol, index):
        factor = pd.Series(1.0, index=index)
        for action_symbol, ex_date, action_factor in self.actions:
            if action_symbol == symbol:
                factor[index < ex_date] *= action_factor
        return factor

    def adjusted(self, symbol):
        """The series a full re-download would return right now, as a long frame."""
        window = self.sessions[self.sessions <= self.available_through]
        return self._long(symbol, window)

    def _long(self, symbol, index):
        frame = self._symbol_frame(symbol, index)
        frame = frame.rename_axis("date").reset_index()
        frame.columns = [c.lower() for c in frame.columns]
        frame.insert(0, "symbol", symbol)
        return frame

    def _symbol_frame(self, symbol, index):
        raw = self.raw[symbol].loc[index]
        factor = self._factor(symbol, index)
        frame = pd.DataFrame({
            "Open": raw["open"] * factor,
            "High": raw["high"] * factor,
            "Low": raw["low"] * factor,
            "Close": raw["close"] * factor,
            "Volume": raw["volume"],
        }, index=pd.DatetimeIndex(index, name="Date"))
        for day in self.unfinalised & set(index):
            frame.loc[day, ["Open", "High", "Low", "Close"]] = np.nan
        return frame

    def _padded(self, symbol, day):
        """A holiday bar: the previous close carried forward, zero volume."""
        earlier = self.sessions[self.sessions < day]
        close = self._symbol_frame(symbol, earlier)["Close"].iloc[-1]
        return pd.DataFrame(
            {"Open": close, "High": close, "Low": close, "Close": close, "Volume": 0.0},
            index=pd.DatetimeIndex([day], name="Date"))

    def __call__(self, symbols_ns, start, end):
        """`start` inclusive, `end` exclusive, like the real thing."""
        self.calls.append((list(symbols_ns), start, end))
        wanted = self.sessions[(self.sessions >= pd.Timestamp(start))
                               & (self.sessions < pd.Timestamp(end))
                               & (self.sessions <= self.available_through)]
        holidays = [d for d in map(pd.Timestamp, self.padded_holidays)
                    if pd.Timestamp(start) <= d < pd.Timestamp(end)
                    and d <= self.available_through]

        frames = {}
        for symbol_ns in symbols_ns:
            symbol = symbol_ns.removesuffix(".NS")
            if symbol in self.absent or symbol not in self.raw:
                frames[symbol_ns] = pd.DataFrame(
                    columns=["Open", "High", "Low", "Close", "Volume"], dtype=float,
                    index=pd.DatetimeIndex([], name="Date"))
                continue
            parts = [self._symbol_frame(symbol, wanted)]
            parts += [self._padded(symbol, day) for day in holidays]
            frames[symbol_ns] = pd.concat(parts).sort_index()

        union = pd.DatetimeIndex(sorted({d for f in frames.values() for d in f.index}),
                                 name="Date")
        frames = {k: f.reindex(union) for k, f in frames.items()}
        return pd.concat(frames.values(), axis=1, sort=True, keys=frames.keys(),
                         names=["Ticker", "Price"])


def initial_master(yahoo, symbols, through):
    """What a first full download leaves on disk, as of `through`."""
    from nifty500.ingest import reshape_to_long

    yahoo.available_through = pd.Timestamp(through)
    raw = yahoo([s + ".NS" for s in symbols], yahoo.sessions.min().date(),
                (pd.Timestamp(through) + timedelta(days=1)).date())
    bars, _ = reshape_to_long(raw, [s + ".NS" for s in symbols])
    return bars
