"""End-to-end tests for `scripts/fetch_daily.py`: files in, files out, exit code.

`main()` is run against temporary copies of the three paths it touches and a
steerable Yahoo, so what is checked is the whole script -- including that a failed
run leaves the master exactly as it found it.
"""

import importlib.util
from datetime import date
from pathlib import Path

import pandas as pd
import pytest

from fake_yahoo import FakeYahoo, initial_master

SCRIPT = Path(__file__).resolve().parents[1] / "scripts" / "fetch_daily.py"
SYMBOLS = ["AAA", "BBB", "CCC", "DDD", "EEE", "FFF"]
SESSIONS = pd.bdate_range("2026-08-03", "2026-09-30")
MON, TUE = date(2026, 9, 14), date(2026, 9, 15)


@pytest.fixture
def cli(tmp_path, monkeypatch):
    spec = importlib.util.spec_from_file_location("fetch_daily", SCRIPT)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)

    monkeypatch.setattr(module, "MASTER_PATH", tmp_path / "master.parquet")
    monkeypatch.setattr(module, "CONSTITUENTS_PATH", tmp_path / "constituents.csv")
    monkeypatch.setattr(module, "COVERAGE_PATH", tmp_path / "coverage.csv")

    pd.DataFrame({"Company Name": SYMBOLS, "Industry": "Auto", "Symbol": SYMBOLS,
                  "Series": "EQ", "ISIN Code": SYMBOLS}).to_csv(module.CONSTITUENTS_PATH,
                                                                  index=False)
    yahoo = FakeYahoo(SESSIONS, SYMBOLS)
    initial_master(yahoo, SYMBOLS, MON).to_parquet(module.MASTER_PATH, index=False)
    module.yahoo = yahoo
    return module


def go(cli, today, argv=(), download=None):
    cli.yahoo.available_through = pd.Timestamp(today)
    return cli.main(list(argv), download=download or cli.yahoo, today=today)


class TestMain:
    def test_a_normal_run_updates_the_master_and_rebuilds_the_coverage_report(self, cli, capsys):
        assert go(cli, TUE) == 0
        master = pd.read_parquet(cli.MASTER_PATH)
        assert master["date"].max() == pd.Timestamp(TUE)

        coverage = pd.read_csv(cli.COVERAGE_PATH)
        assert sorted(coverage["symbol"]) == sorted(SYMBOLS)
        assert set(coverage["last_date"]) == {"2026-09-15"}
        assert "Master updated" in capsys.readouterr().out

    def test_an_empty_upstream_fails_and_leaves_every_file_untouched(self, cli, capsys):
        before = cli.MASTER_PATH.read_bytes()
        assert go(cli, TUE, download=lambda *a: pd.DataFrame()) == 1
        assert cli.MASTER_PATH.read_bytes() == before
        assert not cli.COVERAGE_PATH.exists()
        assert "FAIL" in capsys.readouterr().out

    def test_a_day_with_nothing_new_writes_nothing(self, cli, capsys):
        before = cli.MASTER_PATH.read_bytes()
        assert go(cli, MON) == 0                              # Yahoo holds nothing past Monday
        assert cli.MASTER_PATH.read_bytes() == before
        assert not cli.COVERAGE_PATH.exists()
        assert "unchanged" in capsys.readouterr().out

    def test_a_polluted_master_is_repaired_on_the_next_run(self, cli):
        polluted = pd.read_parquet(cli.MASTER_PATH)
        junk = polluted[polluted["date"] == pd.Timestamp(MON)].copy()
        junk["date"] = pd.Timestamp(TUE)
        junk[["open", "high", "low", "close"]] = float("nan")   # the 2026-09-21 shape
        pd.concat([polluted, junk], ignore_index=True).to_parquet(cli.MASTER_PATH, index=False)

        assert go(cli, TUE) == 0
        healed = pd.read_parquet(cli.MASTER_PATH)
        assert not healed[["open", "high", "low", "close"]].isna().any().any()
        assert healed["date"].max() == pd.Timestamp(TUE)

    def test_full_refresh_re_downloads_the_whole_history(self, cli):
        assert go(cli, TUE, argv=["--full-refresh"]) == 0
        whole_history_requests = [c for c in cli.yahoo.calls if c[1] <= date(2026, 8, 3)]
        assert whole_history_requests, "expected a request reaching back to the first session"
