import yfinance as yf
import pandas as pd
from pathlib import Path
from datetime import datetime, timedelta
import sys

PARQUET_PATH = Path("data/raw/nifty500_ohlcv_raw.parquet")
CONSTITUENTS_PATH = Path("data/raw/nifty500_constituents.csv")

# Fail loudly rather than exiting 0 on a partial or empty fetch: a silent
# success here is what let 50 scheduled runs look like market holidays.
MAX_MISSING_RATIO = 0.10
MAX_HOLIDAY_GAP_DAYS = 5

def is_weekend():
    if datetime.today().weekday() >= 5:
        print("Today is a weekend. No market data expected.")
        return True
    return False

def get_last_date_in_master():
    master = pd.read_parquet(PARQUET_PATH)
    last_date = pd.to_datetime(master["date"]).max()
    print(f"Last date in master: {last_date.date()}")
    return last_date

def fetch_new_data(symbols_ns, start_date, end_date):
    print(f"Fetching {start_date.date()} to {end_date.date()} for {len(symbols_ns)} symbols...")
    data = yf.download(
        symbols_ns,
        start=start_date.strftime("%Y-%m-%d"),
        end=end_date.strftime("%Y-%m-%d"),
        auto_adjust=True,
        group_by="ticker",
        threads=True,
        progress=False
    )
    return data

def reshape_to_long(data, symbols_ns):
    frames = []
    missing = []
    for symbol_ns in symbols_ns:
        try:
            df = data[symbol_ns].copy()
            df = df.dropna(how="all")
            if len(df) == 0:
                missing.append(symbol_ns)
                continue
            df["symbol"] = symbol_ns.replace(".NS", "")
            df = df.reset_index()
            df.columns = [c.lower() for c in df.columns]
            frames.append(df)
        except KeyError:
            missing.append(symbol_ns)
    if missing:
        print(f"No data for {len(missing)}/{len(symbols_ns)} symbols: "
              f"{', '.join(missing[:10])}{' ...' if len(missing) > 10 else ''}")
    if not frames:
        return pd.DataFrame(), missing
    combined = pd.concat(frames, ignore_index=True)
    combined = combined[["symbol", "date", "open", "high", "low", "close", "volume"]]
    combined["date"] = pd.to_datetime(combined["date"])
    return combined, missing

def main():
    if is_weekend():
        print("Skipping — weekend.")
        sys.exit(0)

    constituents = pd.read_csv(CONSTITUENTS_PATH)
    symbols_ns = [s.strip() + ".NS" for s in constituents["Symbol"].tolist()]

    last_date = get_last_date_in_master()
    fetch_start = last_date + timedelta(days=1)
    fetch_end = datetime.today() + timedelta(days=1)

    if fetch_start.date() >= datetime.today().date():
        print("Master is already up to date. Nothing to fetch.")
        sys.exit(0)

    raw = fetch_new_data(symbols_ns, fetch_start, fetch_end)

    requested_days = (fetch_end - fetch_start).days

    if raw is None or raw.empty:
        # A one- or two-day window can legitimately be a market holiday.
        # A wide window returning nothing means the upstream API broke.
        if requested_days > MAX_HOLIDAY_GAP_DAYS:
            print(f"FAIL: no data for a {requested_days}-day window across "
                  f"{len(symbols_ns)} symbols. This is not a holiday — "
                  f"the upstream API or the yfinance version is broken.")
            sys.exit(1)
        print("No new data returned — likely a market holiday.")
        sys.exit(0)

    new_df, missing = reshape_to_long(raw, symbols_ns)

    missing_ratio = len(missing) / len(symbols_ns)
    if missing_ratio > MAX_MISSING_RATIO:
        print(f"FAIL: {missing_ratio:.0%} of symbols returned no data "
              f"(threshold {MAX_MISSING_RATIO:.0%}). Refusing to commit a "
              f"partial update.")
        sys.exit(1)

    if new_df.empty:
        print("FAIL: no rows after reshape despite a non-empty response.")
        sys.exit(1)

    print(f"New rows fetched: {len(new_df)}")

    master = pd.read_parquet(PARQUET_PATH)
    master = pd.concat([master, new_df], ignore_index=True)
    master = master.drop_duplicates(subset=["symbol", "date"])
    master = master.sort_values(["symbol", "date"]).reset_index(drop=True)
    master.to_parquet(PARQUET_PATH, index=False)

    print(f"Master updated: {master.shape[0]} rows, latest date: {master['date'].max().date()}")

if __name__ == "__main__":
    main()
