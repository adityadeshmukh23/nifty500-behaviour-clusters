"""Incremental, self-healing refresh of the OHLCV master.

Everything here is a pure function of DataFrames plus an injected `download`
callable. The failure modes that hurt this pipeline -- an upstream that quietly
returns nothing, a bar that is not final yet, prices re-based by a corporate
action -- are all properties of the *data*, so they can be reproduced in a test
without touching the network. `scripts/fetch_daily.py` only wires in yfinance.

Design, in one paragraph. Every run re-fetches a short overlap of sessions it
already holds instead of starting at `last_date + 1`. That overlap does three
jobs at once: it proves the upstream is alive (an overlap can never legitimately
be empty), it lets a fresh bar supersede a provisional one, and it exposes prices
that Yahoo has since re-adjusted for a split or dividend -- for those symbols the
whole history is re-downloaded, so the stored series stays on a single basis.
"""

from collections.abc import Callable
from dataclasses import dataclass, field
from datetime import date, timedelta
from pathlib import Path

import pandas as pd

from nifty500.bars import BAR_COLUMNS, PRICE_COLUMNS, clean_bars, flag_invalid_bars

YAHOO_SUFFIX = ".NS"

# Fail loudly rather than exiting 0 on a partial or empty fetch: a silent
# success here is what let 50 scheduled runs look like market holidays.
MAX_MISSING_RATIO = 0.10

# Calendar days of already-stored sessions that every run fetches again. A week
# is about five sessions: enough to span a long weekend and still be a small
# request.
OVERLAP_DAYS = 7

# Stored and freshly fetched closes for the same session agree to floating-point
# noise unless Yahoo has re-adjusted the series since. This is the relative gap
# treated as a re-adjustment; even a small dividend clears it comfortably.
DRIFT_TOLERANCE = 1e-4

# The newest stored sessions are left out of the re-adjustment comparison. A bar
# that young may still be provisional, and its firming up is not a re-adjustment --
# the fresh fetch simply replaces it. A real re-adjustment rescales *every* earlier
# bar, so the older sessions of the overlap still expose it.
SETTLED_AFTER_SESSIONS = 2

# A re-downloaded history replaces the stored one only if it is about as long.
# A truncated response must never be allowed to shorten the master.
MIN_REFRESH_COVERAGE = 0.95

# start (inclusive), end (exclusive) -> yfinance-shaped frame, or None.
Downloader = Callable[[list[str], date, date], pd.DataFrame | None]


class PipelineError(RuntimeError):
    """A condition that must fail the scheduled job rather than exit 0."""


@dataclass
class UpdateReport:
    """What one refresh did, in numbers a log reader can act on."""

    last_bar_before: date
    last_bar_after: date
    rows_before: int
    rows_after: int
    removed_from_master: dict[str, int]
    new_rows: int
    revised_rows: int
    refreshed: list[str] = field(default_factory=list)
    refresh_failed: list[str] = field(default_factory=list)
    missing: list[str] = field(default_factory=list)
    changed: bool = False

    def lines(self):
        removed = ", ".join(f"{count:,} {reason}" for reason, count
                            in self.removed_from_master.items() if count) or "nothing"
        rows = [
            ("last bar", f"{self.last_bar_before} -> {self.last_bar_after}"),
            ("rows", f"{self.rows_before:,} -> {self.rows_after:,} "
                     f"({self.new_rows:,} new, {self.revised_rows:,} revised)"),
            ("removed as non-bars", removed),
        ]
        if self.refreshed:
            rows.append(("history refreshed",
                         f"{_preview(self.refreshed)} -- prices were re-adjusted upstream"))
        if self.refresh_failed:
            rows.append(("refresh FAILED", _preview(self.refresh_failed)))
        if self.missing:
            rows.append(("no usable data", _preview(self.missing)))
        return [f"{label:<20}: {value}" for label, value in rows]


def _preview(symbols, limit=10):
    shown = ", ".join(symbols[:limit])
    return f"{len(symbols)} symbol(s): {shown}{' ...' if len(symbols) > limit else ''}"


def _plain(symbol_ns):
    return symbol_ns.removesuffix(YAHOO_SUFFIX)


def _canonical(bars):
    """Master schema and dtypes, sorted, with a fresh RangeIndex."""
    out = bars[BAR_COLUMNS].copy()
    dates = pd.to_datetime(out["date"])
    if dates.dt.tz is not None:
        dates = dates.dt.tz_localize(None)
    out["date"] = dates.dt.normalize()
    out["symbol"] = out["symbol"].astype(str)
    for column in [*PRICE_COLUMNS, "volume"]:
        out[column] = out[column].astype("float64")
    return out.sort_values(["symbol", "date"]).reset_index(drop=True)


def _empty_bars():
    return _canonical(pd.DataFrame({column: [] for column in BAR_COLUMNS}))


def reshape_to_long(data, symbols_ns):
    """Wide yfinance frame -> long bars in the master's schema.

    Returns `(bars, missing)`. `missing` lists the plain symbols that produced no
    usable bar in the window: absent from the response, all-NaN, or nothing but
    unpriced / placeholder rows (see `nifty500.bars`).
    """
    if not isinstance(data.columns, pd.MultiIndex) and len(symbols_ns) == 1:
        data = pd.concat({symbols_ns[0]: data}, axis=1)

    returned = set(data.columns.get_level_values(0))
    frames = []
    for symbol_ns in symbols_ns:
        if symbol_ns not in returned:
            continue
        frame = data[symbol_ns].dropna(how="all")
        if frame.empty:
            continue
        frame = frame.rename_axis("date").reset_index()
        frame.columns = [str(column).lower() for column in frame.columns]
        frame["symbol"] = _plain(symbol_ns)
        frames.append(frame)

    if not frames:
        return _empty_bars(), [_plain(s) for s in symbols_ns]

    combined = pd.concat(frames, ignore_index=True)
    absent = [column for column in BAR_COLUMNS if column not in combined.columns]
    if absent:
        raise PipelineError(f"the upstream response has no {absent} column(s); its schema "
                            f"has changed, which is what pinning yfinance is meant to prevent")

    bars = clean_bars(_canonical(combined)).reset_index(drop=True)
    have = set(bars["symbol"])
    return bars, [_plain(s) for s in symbols_ns if _plain(s) not in have]


def plan_window(last_bar, today, overlap_days=OVERLAP_DAYS):
    """`(start, end)` to request: `start` is inclusive, `end` exclusive.

    `end` is the day after today, so a session that has just closed is included.
    Starting at `last_bar + 1` instead -- as an earlier version did -- meant a run
    on the evening of session N found "nothing to fetch" whenever the master
    already held N-1, and so the pipeline only landed data every other day.
    """
    end = max(today, last_bar) + timedelta(days=1)
    return last_bar - timedelta(days=overlap_days), end


def find_readjusted_symbols(stored, fetched, tolerance=DRIFT_TOLERANCE,
                            settled_after=SETTLED_AFTER_SESSIONS):
    """Symbols whose stored closes disagree with the same sessions fetched now.

    With `auto_adjust=True` Yahoo rescales *all* earlier prices whenever a split,
    bonus or dividend occurs. Bars appended earlier are on the old basis, so the
    stored series would carry a seam at the event. Comparing the overlap finds
    those symbols -- over sessions old enough to be settled (`settled_after`).
    """
    keys = ["symbol", "date"]
    sessions = pd.Series(stored["date"].unique()).sort_values()
    settled = sessions.iloc[:-settled_after] if settled_after else sessions
    old_enough = stored["date"].isin(settled) & (stored["date"] >= fetched["date"].min())
    both = (stored.loc[old_enough, [*keys, "close"]]
            .merge(fetched[[*keys, "close"]], on=keys, suffixes=("_stored", "_fetched")))
    gap = (both["close_fetched"] / both["close_stored"] - 1).abs()
    return sorted(both.loc[gap > tolerance, "symbol"].unique())


def merge_bars(master, *fresh):
    """Master plus new bars, unique on (symbol, date), newest value winning.

    `keep="last"` is the point: a bar fetched today replaces the provisional or
    stale one already stored. Keeping the first, as an earlier version did, made
    any bad row permanent.
    """
    frames = [frame for frame in (master, *fresh) if not frame.empty]
    merged = pd.concat(frames, ignore_index=True)
    merged = merged.drop_duplicates(subset=["symbol", "date"], keep="last")
    return merged.sort_values(["symbol", "date"]).reset_index(drop=True)


def _covers(new, old):
    """Is `new` a plausible full replacement for the stored history `old`?"""
    if new.empty:
        return False
    if old.empty:
        return True
    long_enough = len(new) >= MIN_REFRESH_COVERAGE * len(old)
    starts_early = new["date"].min() <= old["date"].min() + timedelta(days=OVERLAP_DAYS)
    return long_enough and starts_early


def refresh_histories(symbols, stored, download, end):
    """Re-download the whole history of `symbols`.

    Returns `(bars, refreshed, failed)`. A symbol only counts as refreshed if the
    new history covers the stored one; otherwise the stored rows are left alone
    and the symbol is reported as failed, to be retried on the next run.
    """
    symbols = list(symbols)
    if not symbols:
        return _empty_bars(), [], []

    start = stored["date"].min().date()
    raw = download([s + YAHOO_SUFFIX for s in symbols], start, end)
    if raw is None or raw.empty:
        return _empty_bars(), [], symbols

    full, _ = reshape_to_long(raw, [s + YAHOO_SUFFIX for s in symbols])
    kept, failed = [], []
    for symbol in symbols:
        replacement = full[full["symbol"] == symbol]
        if _covers(replacement, stored[stored["symbol"] == symbol]):
            kept.append(replacement)
        else:
            failed.append(symbol)
    bars = pd.concat(kept, ignore_index=True) if kept else _empty_bars()
    return bars, [s for s in symbols if s not in failed], failed


def _count_changes(before, after):
    """(new rows, revised rows) going from `before` to `after`."""
    keys = ["symbol", "date"]
    old = before.set_index(keys)
    new = after.set_index(keys)
    added = len(new.index.difference(old.index))
    common = new.index.intersection(old.index)
    columns = [*PRICE_COLUMNS, "volume"]
    revised = int((new.loc[common, columns] != old.loc[common, columns]).any(axis=1).sum())
    return added, revised


def update_master(master, symbols, download, today, *, overlap_days=OVERLAP_DAYS,
                  refresh_symbols=()):
    """Bring `master` up to date. Returns `(updated_bars, UpdateReport)`.

    Raises `PipelineError` whenever the result cannot be trusted: an empty
    response, a response with no usable bars, too many symbols missing, or a
    master with nothing valid in it.

    `refresh_symbols` forces a full history re-download for those symbols, on top
    of the ones found re-adjusted -- and symbols that are not in the master yet
    are always backfilled, so adding a constituent needs no manual step.
    """
    original = _canonical(master)
    invalid = flag_invalid_bars(original).sum().to_dict()
    stored = clean_bars(original).reset_index(drop=True)      # heal what an earlier run let in
    if stored.empty:
        raise PipelineError("the master holds no valid bars to update from")

    symbols = list(dict.fromkeys(symbols))
    if not symbols:
        raise PipelineError("no symbols to fetch: is the constituents file empty?")

    last_bar = stored["date"].max().date()
    start, end = plan_window(last_bar, today, overlap_days)
    symbols_ns = [s + YAHOO_SUFFIX for s in symbols]

    raw = download(symbols_ns, start, end)
    if raw is None or raw.empty:
        # The window always re-covers sessions we already hold, so "no data" is
        # never a holiday: it means the upstream, or the yfinance version, broke.
        raise PipelineError(f"no data came back for {start} to {end} across "
                            f"{len(symbols)} symbols, although that window overlaps sessions "
                            f"already held. The upstream API or the yfinance version is broken.")

    fresh, missing = reshape_to_long(raw, symbols_ns)
    if fresh.empty:
        raise PipelineError("the response held rows but not one valid bar "
                            "(all unpriced or placeholders).")

    unseen = sorted(set(symbols) - set(stored["symbol"]))
    readjusted = find_readjusted_symbols(stored, fresh)
    drifted = sorted({*readjusted, *refresh_symbols, *unseen})
    refreshed_bars, refreshed, refresh_failed = refresh_histories(drifted, stored, download, end)

    unusable = sorted(set(missing) | set(refresh_failed))
    ratio = len(unusable) / len(symbols)
    if ratio > MAX_MISSING_RATIO:
        raise PipelineError(f"{ratio:.0%} of symbols returned no usable data "
                            f"(threshold {MAX_MISSING_RATIO:.0%}): {_preview(unusable)}. "
                            f"Refusing to commit a partial update.")

    # Refreshed symbols come back whole; failed ones keep their stored rows and get
    # no new ones, so the mismatch is seen again -- and retried -- next run.
    kept = stored[~stored["symbol"].isin(refreshed)]
    new_rows = fresh[~fresh["symbol"].isin(drifted)]
    updated = merge_bars(kept, refreshed_bars, new_rows)

    added, revised = _count_changes(stored, updated)
    report = UpdateReport(
        last_bar_before=last_bar,
        last_bar_after=updated["date"].max().date(),
        rows_before=len(original),
        rows_after=len(updated),
        removed_from_master={reason: int(count) for reason, count in invalid.items()},
        new_rows=added,
        revised_rows=revised,
        refreshed=refreshed,
        refresh_failed=refresh_failed,
        missing=missing,
        changed=not updated.equals(original),
    )
    return updated, report


def write_master(bars, path):
    """Write the master atomically and verify it, so a bad file is never committed."""
    path = Path(path)
    scratch = path.with_name(path.name + ".tmp")
    bars.to_parquet(scratch, index=False)
    written = len(pd.read_parquet(scratch, columns=["symbol"]))
    if written != len(bars):
        scratch.unlink()
        raise PipelineError(f"wrote {written:,} rows but expected {len(bars):,}")
    scratch.replace(path)


def load_symbols(constituents_path):
    """Plain NSE symbols from the constituents CSV, whitespace-trimmed."""
    symbols = pd.read_csv(constituents_path)["Symbol"]
    return [s.strip() for s in symbols.tolist()]
