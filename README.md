# Nifty 500 Behaviour Clusters

Daily OHLCV data for all Nifty 500 stocks from January 2022, auto-updated every weekday.

## Dataset
- 500 NSE-listed stocks
- Daily Open, High, Low, Close, Volume
- Updated automatically via GitHub Actions every weekday at 6:30 PM IST

## Source
Yahoo Finance (yfinance) — NSE India data

## Project Goal
Cluster Nifty 500 stocks by behavioural patterns (volatility, momentum, drawdown)
to discover groupings beyond traditional sector labels.

## Files
- `data/raw/nifty500_ohlcv_raw.parquet` — master OHLCV dataset
- `data/raw/nifty500_constituents.csv` — Nifty 500 stock list with sectors
- `scripts/fetch_daily.py` — daily data fetch script
