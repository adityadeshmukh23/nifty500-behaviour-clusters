"""Fail if the OHLCV master has gone stale. The logic lives in `nifty500.freshness`."""

import sys

from nifty500.freshness import main

if __name__ == "__main__":
    sys.exit(main())
