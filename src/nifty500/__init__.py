"""Nifty 500 data pipeline and behavioural clustering.

bars        hygiene for daily OHLCV bars (shared by every layer below)
ingest      incremental, self-healing refresh of the OHLCV master
coverage    per-symbol coverage report for the master
freshness   staleness check on the master
features    OHLCV -> behavioural features, one row per stock
clustering  PCA + KMeans on those features, plus the diagnostics used to choose k
"""
