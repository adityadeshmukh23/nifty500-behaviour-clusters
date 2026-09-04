# Nifty 500 Behaviour Clusters

**An automated NSE market-data pipeline that maintains a 527k-row daily OHLCV dataset for the Nifty 500 — built as the foundation for clustering stocks by how they *behave* rather than by which sector they're labelled with.**

[![Daily NSE Pull](https://github.com/adityadeshmukh23/nifty500-behaviour-clusters/actions/workflows/daily_nse_pull.yml/badge.svg)](https://github.com/adityadeshmukh23/nifty500-behaviour-clusters/actions/workflows/daily_nse_pull.yml)
[![Python 3.11](https://img.shields.io/badge/python-3.11-blue.svg)](https://www.python.org/downloads/)
[![pandas](https://img.shields.io/badge/pandas-2.2-150458.svg?logo=pandas&logoColor=white)](https://pandas.pydata.org/)
[![Kaggle Dataset](https://img.shields.io/badge/Kaggle-dataset-20BEFF.svg?logo=kaggle&logoColor=white)](https://www.kaggle.com/datasets/adityadeshmukh05/nifty500-daily-ohlcv)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)

**Dataset on Kaggle →** https://www.kaggle.com/datasets/adityadeshmukh05/nifty500-daily-ohlcv

> **Status —** The data pipeline is built, automated, and publishing to Kaggle. The clustering analysis it was built for is **in progress** and not yet in this repo; the planned approach is documented [below](#planned-analysis-not-yet-implemented).

---

## The problem

Indian equities are conventionally grouped by GICS-style sector labels — "Financial Services", "Capital Goods", "IT". Those labels describe *what a company sells*, not *how its stock trades*. A small-cap NBFC and a large private bank sit in the same sector bucket while behaving nothing alike in volatility, drawdown depth, or momentum persistence.

The goal is to group the Nifty 500 by **measured trading behaviour** — realised volatility, momentum, max drawdown, liquidity — and see how far those clusters diverge from the official sector map.

Two things had to exist before any of that could happen:

1. **A clean, deep, reproducible price history.** Free NSE data is inconsistent — delisted tickers, recent IPOs with no history, symbol renames. There is no single trustworthy CSV to download.
2. **An automated way to keep it current**, so the dataset doesn't decay the moment it's committed.

This repo is that infrastructure.

---

## Dataset at a glance

| | |
|---|---|
| Rows | **526,815** |
| Symbols | **504** — every listed constituent |
| Coverage | **2022-01-03 → 2026-09-04** (4.7 years of daily bars) |
| Fields | `symbol, date, open, high, low, close, volume` |
| Prices | Split- and dividend-adjusted (`auto_adjust=True`) |
| Storage | Parquet, **19.7 MB** — 62% smaller than the equivalent CSV (51.8 MB) |
| Source | Yahoo Finance via [`yfinance`](https://github.com/ranaroussi/yfinance) |
| Refresh | GitHub Actions, weekdays 18:30 IST (13:00 UTC) |

**Coverage is uneven by construction, and that's recorded rather than hidden.** 420 of 504 symbols carry the full ~1,159-bar history; the remaining 84 are recent IPOs and demergers (Meesho, Lenskart, Groww, HDB Financial, Tata Capital, ITC Hotels) that simply did not trade for the whole window. Per-symbol first date, last date, and bar count are published in [`data/raw/coverage_report.csv`](data/raw/coverage_report.csv). Short histories are left short — never forward-filled — so no synthetic prices enter the dataset.

---

## Key features

- **Incremental fetch, not full re-download.** The job reads the last date already in the Parquet master and requests only the delta — a normal weekday pull is one trading day for 504 tickers, not 4.7 years.
- **Idempotent by construction.** New rows are appended, then de-duplicated on `(symbol, date)` and re-sorted. Re-running the job on the same day is a no-op, so a retry after a failure can never double-count a bar.
- **Fails closed on non-trading days.** Weekends exit before any network call; market holidays return an empty frame and exit cleanly instead of committing a no-change file.
- **Fails loudly, not silently.** An empty response over a multi-day window, or more than 10% of symbols returning nothing, exits non-zero instead of being written off as a market holiday. A partial update is never committed.
- **Explicit gap accounting.** Per-symbol coverage is published alongside the data, so anyone using the dataset can see exactly which histories are short.
- **Columnar storage.** Parquet with typed columns, chosen over CSV for the 62% size reduction and for selective column reads during the analysis step.
- **Zero-touch publishing.** The same workflow that updates the repo pushes a new version of the public Kaggle dataset.

---

## Architecture

```
                        ┌──────────────────────────────┐
                        │  GitHub Actions (cron)       │
                        │  weekdays 13:00 UTC          │
                        └──────────────┬───────────────┘
                                       │
                                       ▼
   ┌───────────────────────┐   ┌───────────────────────────┐
   │ nifty500_             │──▶│  scripts/fetch_daily.py   │
   │ constituents.csv      │   │                           │
   │ (504 symbols +        │   │  1. weekend? → exit 0     │
   │  sector labels)       │   │  2. read master → last dt │
   └───────────────────────┘   │  3. yfinance delta pull   │
                               │  4. wide → long reshape   │
                               │  5. append + dedup + sort │
                               └─────────────┬─────────────┘
                                             │
                          ┌──────────────────┴──────────────────┐
                          ▼                                     ▼
              ┌───────────────────────┐            ┌────────────────────────┐
              │ nifty500_ohlcv_raw    │            │  Kaggle dataset        │
              │ .parquet  (master)    │            │  (new version pushed)  │
              └───────────┬───────────┘            └────────────────────────┘
                          │
                          ▼
              ┌────────────────────────────────────────────┐
              │ analysis layer  —  PLANNED, see below      │
              └────────────────────────────────────────────┘
```

**Design decisions worth defending in review:**

- *Long format over wide.* A 504-column wide frame breaks whenever the index is rebalanced. Long format (`symbol, date, …`) absorbs constituent changes without a schema migration.
- *Parquet as the master, not a database.* The dataset is append-only, single-writer, and read in full by the analysis step — a columnar file beats the operational cost of hosting Postgres for this access pattern.
- *De-duplication on the natural key rather than trusting the API.* `yfinance` will happily return an overlapping window; `(symbol, date)` uniqueness is enforced on our side.

---

## Setup

**Requirements:** Python 3.11+, [Git LFS](https://git-lfs.com) (the Parquet master is LFS-tracked).

```bash
git lfs install
git clone https://github.com/adityadeshmukh23/nifty500-behaviour-clusters.git
cd nifty500-behaviour-clusters

python -m venv .venv && source .venv/bin/activate   # Windows: .venv\Scripts\activate
pip install -r requirements.txt
```

Run the incremental fetch locally:

```bash
python scripts/fetch_daily.py
```

Load the dataset:

```python
import pandas as pd

df = pd.read_parquet("data/raw/nifty500_ohlcv_raw.parquet")
print(df.shape)                                  # (526815, 7)
print(df["symbol"].nunique(), "symbols")         # 504 symbols
print(df["date"].min().date(), "→", df["date"].max().date())
```

Prefer no clone? Pull the same data from Kaggle:

```bash
kaggle datasets download -d adityadeshmukh05/nifty500-daily-ohlcv
```

### Automation setup (optional)

The scheduled workflow needs two repository secrets under **Settings → Secrets and variables → Actions**:

| Secret | Purpose |
|---|---|
| `KAGGLE_USERNAME` | Kaggle account name |
| `KAGGLE_KEY` | Kaggle API token (`kaggle.json` → `key`) |

No credentials are stored in the repository; `kaggle.json`, `.env`, and `secrets/` are git-ignored.

---

## Repository structure

```
nifty500-behaviour-clusters/
├── .github/workflows/
│   └── daily_nse_pull.yml          # cron: fetch → commit → publish to Kaggle
├── data/
│   └── raw/
│       ├── nifty500_ohlcv_raw.parquet   # master dataset (LFS, 19.7 MB)
│       ├── nifty500_constituents.csv    # 504 symbols + sector/industry/ISIN
│       ├── coverage_report.csv          # per-symbol first/last date + bar count
│       └── dataset-metadata.json        # Kaggle publishing manifest
├── scripts/
│   └── fetch_daily.py                   # incremental, idempotent fetch
├── requirements.txt
├── LICENSE
└── README.md
```

---

## Planned analysis (not yet implemented)

Per symbol, over a trailing window on adjusted closes:

| Feature | Captures |
|---|---|
| Annualised realised volatility (σ of daily log returns × √252) | risk regime |
| 3M / 6M / 12M momentum | trend persistence |
| Maximum drawdown | downside behaviour |
| Return skew & kurtosis | tail asymmetry |
| Median turnover (close × volume) | liquidity tier |

The intent is to z-score these (they differ by orders of magnitude), reduce with PCA to strip the correlation between volatility and drawdown, then cluster with KMeans — selecting `k` by silhouette score rather than assuming it. The resulting clusters get cross-tabulated against the `Industry` column in `nifty500_constituents.csv` to quantify where behaviour and sector labels disagree.

---

## What I learned

**Git LFS pointers are not files, and CI doesn't know that.** The scheduled job reads the master Parquet on every run. `actions/checkout` fetches LFS *pointers* by default — 133-byte text stubs — so `pd.read_parquet` was handed a stub, not a dataset, and every scheduled run failed. The fix is one line (`lfs: true` on checkout), but finding it meant learning that a green local run and a green CI run test genuinely different filesystems.

**A cron job with no alerting is a cron job you don't have.** The workflow was scheduled and the runs were failing. Nothing told me. Scheduled automation needs a failure signal — a notification, or a freshness assertion that fails loudly — or it quietly rots while still *looking* live.

**Committing a growing binary is a design decision with a bill attached.** Appending to a 14 MB Parquet and committing it daily writes a *new full copy* into history every weekday — several GB of LFS storage a year for a few kilobytes of new prices. Understanding that changed how I think about where mutable state belongs relative to version control.

**A green build is not a working build.** Fixing the LFS checkout turned the badge green — and the job still fetched nothing, because the pinned `yfinance` was too old to talk to the current Yahoo API and the script reported the empty result as "likely a market holiday". Exit code 0 was lying. The real fix was making the failure *loud*: an empty multi-day window or a >10% symbol miss now exits non-zero. A pipeline that cannot fail visibly cannot be trusted when it succeeds.

**Vendor data is dirty in specific, enumerable ways.** 84 of 504 constituents have partial history — recent IPOs and demerged entities that did not trade for the full window. The instinct is to forward-fill and get a clean rectangle. That fabricates prices. Publishing `coverage_report.csv` keeps the gap auditable by anyone using the dataset.

**Wide is convenient until the index rebalances.** Storing 504 tickers as columns is fast to write and brittle to maintain; every constituent change becomes a schema change. Long format costs a reshape and buys stability.

---

## Roadmap

- [ ] Feature engineering + clustering notebooks, with cluster-vs-sector comparison
- [ ] Cluster map and sector-disagreement plots in the README
- [ ] Alerting on scheduled-run failure
- [ ] Freshness assertion in CI (fail if the master's max date lags the last trading day)
- [ ] Extend history to 2015 for a full market-cycle view
- [ ] Corporate-action audit against NSE bhavcopy

---

## Licence & disclaimer

Code released under the [MIT Licence](LICENSE). The dataset is published on Kaggle under CC0-1.0.

Price data is sourced from Yahoo Finance and is provided as-is for research and education. **This is not investment advice.**
