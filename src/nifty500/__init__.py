"""Nifty 500 data pipeline and behavioural clustering.

features    OHLCV -> behavioural features, one row per stock
clustering  PCA + KMeans on those features, plus the diagnostics used to choose k
"""
