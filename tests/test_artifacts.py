"""The committed outputs must agree with each other, with the register and with the README.

These are the files a reader actually looks at, and the notebooks regenerate them,
so nothing else stops them drifting apart -- which is how this README once came to
quote numbers from an earlier run. None of these tests needs the LFS master.
"""

import re
import tomllib
from pathlib import Path

import pandas as pd
import pytest
from packaging.requirements import Requirement
from sklearn.metrics import adjusted_rand_score, normalized_mutual_info_score

from nifty500.features import CORPORATE_ACTION_THRESHOLD, FEATURE_COLUMNS

ROOT = Path(__file__).resolve().parents[1]
PROCESSED = ROOT / "data" / "processed"


@pytest.fixture(scope="module")
def features():
    return pd.read_parquet(PROCESSED / "features.parquet")


@pytest.fixture(scope="module")
def clusters():
    return pd.read_csv(PROCESSED / "clusters.csv", index_col="symbol")


@pytest.fixture(scope="module")
def profiles():
    return pd.read_csv(PROCESSED / "cluster_profiles.csv", index_col="cluster")


@pytest.fixture(scope="module")
def register():
    return pd.read_csv(ROOT / "data" / "reference" / "corporate_actions.csv",
                       parse_dates=["date"])


@pytest.fixture(scope="module")
def readme():
    return (ROOT / "README.md").read_text(encoding="utf-8")


class TestFeatures:
    def test_has_exactly_the_ten_documented_columns_and_no_gaps(self, features):
        assert list(features.columns) == FEATURE_COLUMNS
        assert not features.isna().any().any()

    def test_one_row_per_symbol(self, features):
        assert features.index.is_unique
        assert features.index.name == "symbol"


class TestClusters:
    def test_covers_exactly_the_feature_universe(self, features, clusters):
        assert set(clusters.index) == set(features.index)

    def test_carries_the_same_feature_values(self, features, clusters):
        joined = clusters[FEATURE_COLUMNS].sort_index()
        pd.testing.assert_frame_equal(joined, features.sort_index()[FEATURE_COLUMNS],
                                      check_freq=False, atol=1e-12)

    def test_every_cluster_id_has_exactly_one_distinct_name(self, clusters):
        names = clusters.groupby("cluster")["cluster_name"].nunique()
        assert (names == 1).all()
        assert clusters["cluster_name"].nunique() == clusters["cluster"].nunique()

    def test_the_profile_sizes_and_means_are_those_of_the_assignments(self, clusters, profiles):
        assert profiles["n"].to_dict() == clusters.groupby("cluster").size().to_dict()
        means = clusters.groupby("cluster")[FEATURE_COLUMNS].mean()
        pd.testing.assert_frame_equal(profiles[FEATURE_COLUMNS], means, atol=1e-9,
                                      check_names=False)

    def test_profile_names_match_the_assignments(self, clusters, profiles):
        named = clusters.groupby("cluster")["cluster_name"].first()
        assert profiles["name"].to_dict() == named.to_dict()


class TestReviewedRegister:
    def test_is_well_formed(self, register):
        assert list(register.columns) == ["symbol", "date", "detected_by", "price_ratio",
                                          "day_volume_x", "next20_volume_x", "note"]
        assert not register.duplicated(["symbol", "date"]).any()
        assert set(register["detected_by"]) <= {"threshold", "volume audit"}

    def test_every_symbol_is_a_constituent(self, register):
        constituents = pd.read_csv(ROOT / "data" / "raw" / "nifty500_constituents.csv")
        assert set(register["symbol"]) <= set(constituents["Symbol"].str.strip())

    def test_threshold_rows_really_are_beyond_the_threshold(self, register):
        rows = register[register["detected_by"] == "threshold"]
        assert ((rows["price_ratio"] - 1).abs() > CORPORATE_ACTION_THRESHOLD).all()

    def test_audit_rows_carry_the_signature_that_justifies_them(self, register):
        # A big drop on below-normal volume: the register's own evidence must show it.
        rows = register[register["detected_by"] == "volume audit"]
        assert not rows.empty
        assert ((1 - rows["price_ratio"]) >= 0.15).all()
        assert (rows["day_volume_x"] < 1.0).all()

    def test_audit_rows_ask_for_verification(self, register):
        rows = register[register["detected_by"] == "volume audit"]
        assert rows["note"].str.contains("Verify", case=False).all()


class TestReadmeAgreesWithTheOutputs:
    def test_the_result_table_quotes_the_committed_cluster_sizes(self, readme, profiles):
        quoted = {name: int(n) for name, n in
                  re.findall(r"^\| \*\*(.+?)\*\* \| (\d+) \|", readme, flags=re.MULTILINE)}
        assert quoted == dict(zip(profiles["name"], profiles["n"], strict=True))

    def test_the_universe_size_is_the_committed_one(self, readme, features):
        assert f"from {len(features)} stocks" in readme

    def test_the_headline_agreement_with_sector_is_the_computed_one(self, readme, clusters):
        codes = clusters["industry"].astype("category").cat.codes
        ari = adjusted_rand_score(clusters["cluster"], codes)
        nmi = normalized_mutual_info_score(clusters["cluster"], codes)
        match = re.search(r"Adjusted Rand Index (\d\.\d+), Normalized Mutual Information (\d\.\d+)",
                          readme)
        assert match, "the README no longer states the ARI and NMI"
        assert float(match.group(1)) == pytest.approx(ari, abs=5e-4)
        assert float(match.group(2)) == pytest.approx(nmi, abs=5e-4)

    def test_the_analysis_date_is_the_one_the_notebook_pins(self, readme):
        notebook = (ROOT / "notebooks" / "01_feature_engineering.ipynb").read_text()
        pinned = re.search(r'AS_OF = \\"(\d{4}-\d{2}-\d{2})\\"', notebook)
        assert pinned, "notebook 01 no longer pins AS_OF"
        assert f"through **{pinned.group(1)}**" in readme


class TestSupportedPythonAndPins:
    def test_the_readme_states_the_range_pyproject_declares(self, readme):
        project = tomllib.loads((ROOT / "pyproject.toml").read_text())["project"]
        assert project["requires-python"] == ">=3.12,<3.14"
        assert "Python 3.12 or 3.13" in readme
        assert "3.11" not in readme

    def test_workflows_only_use_supported_python_versions(self):
        for workflow in (ROOT / ".github" / "workflows").glob("*.yml"):
            for line in workflow.read_text().splitlines():
                if "python-version" in line:
                    for version in re.findall(r'"(3\.\d+)"', line):
                        assert version in {"3.12", "3.13"}, f"{workflow.name}: {version}"

    def test_every_declared_dependency_is_pinned_exactly_in_a_requirements_file(self):
        project = tomllib.loads((ROOT / "pyproject.toml").read_text())["project"]
        declared = [*project["dependencies"],
                    *(dep for deps in project["optional-dependencies"].values() for dep in deps)]
        pinned = {}
        for name in ("requirements.txt", "requirements-analysis.txt"):
            for line in (ROOT / name).read_text().splitlines():
                line = line.split("#")[0].strip()
                if "==" in line:
                    requirement = Requirement(line)
                    pinned[requirement.name.lower()] = requirement
        for dependency in declared:
            requirement = Requirement(dependency)
            name = requirement.name.lower()
            assert name in pinned, f"{name} is declared but not pinned"
            version = next(iter(pinned[name].specifier)).version
            assert requirement.specifier.contains(version), \
                f"pinned {name}=={version} violates the declared {requirement.specifier}"
