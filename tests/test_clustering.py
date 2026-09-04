"""Tests for the clustering pipeline and its diagnostics."""

import numpy as np
import pandas as pd
import pytest

from src.clustering import (fit_clusters, name_clusters, profile_clusters,
                            reduce_features, sweep_k)
from src.features import FEATURE_COLUMNS


def make_features(n_per_group=40, seed=0):
    """Three deliberately separated behaviour groups in feature space."""
    rng = np.random.default_rng(seed)
    blocks = []
    # (volatility, beta, 12m momentum, kurtosis) centres for each synthetic group.
    for vol, beta, mom, kurt in [(0.20, 0.7, 0.02, 3.0),
                                 (0.45, 1.4, 0.05, 5.0),
                                 (0.40, 1.1, 0.60, 4.0)]:
        block = pd.DataFrame({
            "ann_volatility": rng.normal(vol, 0.01, n_per_group),
            "downside_volatility": rng.normal(vol * 0.6, 0.01, n_per_group),
            "beta": rng.normal(beta, 0.05, n_per_group),
            "mom_3m": rng.normal(mom / 4, 0.01, n_per_group),
            "mom_6m": rng.normal(mom / 2, 0.01, n_per_group),
            "mom_12m": rng.normal(mom, 0.02, n_per_group),
            "max_drawdown": rng.normal(-vol, 0.01, n_per_group),
            "return_skew": rng.normal(0.3, 0.05, n_per_group),
            "return_kurtosis": rng.normal(kurt, 0.2, n_per_group),
            "log_turnover": rng.normal(9.0, 0.1, n_per_group),
        })
        blocks.append(block)

    features = pd.concat(blocks, ignore_index=True)[FEATURE_COLUMNS]
    features.index = [f"SYM{i:03d}" for i in range(len(features))]
    features.index.name = "symbol"
    return features


class TestReduceFeatures:
    def test_retains_requested_variance(self):
        features = make_features()
        components, _, pca = reduce_features(features, variance_target=0.90)
        assert pca.explained_variance_ratio_.sum() >= 0.90
        assert components.shape[0] == len(features)

    def test_a_lower_target_keeps_fewer_components(self):
        features = make_features()
        _, _, loose = reduce_features(features, variance_target=0.70)
        _, _, tight = reduce_features(features, variance_target=0.95)
        assert loose.n_components_ <= tight.n_components_


class TestFitClusters:
    def test_recovers_planted_groups(self):
        features = make_features()
        components, _, _ = reduce_features(features)
        labels = fit_clusters(components, 3)
        # Each planted block of 40 should land in a single cluster.
        for start in (0, 40, 80):
            assert len(set(labels[start:start + 40])) == 1

    def test_is_deterministic_for_a_fixed_seed(self):
        features = make_features()
        components, _, _ = reduce_features(features)
        assert np.array_equal(fit_clusters(components, 3), fit_clusters(components, 3))


class TestSweepK:
    def test_reports_every_requested_k(self):
        features = make_features()
        components, _, _ = reduce_features(features)
        diagnostics = sweep_k(components, k_values=range(2, 5), n_trials=5)
        assert list(diagnostics.index) == [2, 3, 4]
        assert {"silhouette", "davies_bouldin", "stability"} <= set(diagnostics.columns)

    def test_planted_structure_is_stable_at_the_true_k(self):
        features = make_features()
        components, _, _ = reduce_features(features)
        diagnostics = sweep_k(components, k_values=[3], n_trials=10)
        assert diagnostics.loc[3, "stability"] > 0.9

    def test_sector_columns_appear_only_when_labels_are_given(self):
        features = make_features()
        components, _, _ = reduce_features(features)
        without = sweep_k(components, k_values=[3], n_trials=3)
        assert "ari_vs_sector" not in without.columns

        codes = np.repeat([0, 1, 2], 40)
        with_sectors = sweep_k(components, k_values=[3], n_trials=3, sector_codes=codes)
        assert with_sectors.loc[3, "ari_vs_sector"] == pytest.approx(1.0)


class TestClusterNaming:
    def test_every_cluster_gets_a_distinct_name(self):
        features = make_features()
        components, _, _ = reduce_features(features)
        profile = profile_clusters(features, fit_clusters(components, 3))
        assert profile["name"].notna().all()
        assert profile["name"].nunique() == len(profile)

    def test_names_follow_the_profile_not_the_cluster_id(self):
        # The lowest-volatility cluster must be the defensive one whichever id it got.
        features = make_features()
        components, _, _ = reduce_features(features)
        profile = profile_clusters(features, fit_clusters(components, 3))
        quietest = profile["ann_volatility"].idxmin()
        assert profile.loc[quietest, "name"] == "Large-cap defensives"

    def test_profile_sizes_sum_to_the_universe(self):
        features = make_features()
        components, _, _ = reduce_features(features)
        profile = profile_clusters(features, fit_clusters(components, 3))
        assert profile["n"].sum() == len(features)


class TestNameClusters:
    def test_momentum_group_is_named_from_its_returns(self):
        profile = pd.DataFrame({
            "ann_volatility": [0.20, 0.45, 0.40],
            "beta": [0.7, 1.4, 1.1],
            "mom_12m": [0.02, 0.05, 0.60],
            "return_skew": [0.3, 0.3, 0.3],
            "return_kurtosis": [3.0, 5.0, 4.0],
        })
        names = name_clusters(profile)
        assert names[2] == "Momentum leaders"
        assert names[1] == "High-beta cyclicals"
        assert names[0] == "Large-cap defensives"

    def test_no_tail_risk_group_when_no_cluster_has_a_left_tail(self):
        # All three skews are positive, so "Tail-risk" must not be handed out.
        profile = pd.DataFrame({
            "ann_volatility": [0.20, 0.45, 0.40],
            "beta": [0.7, 1.4, 1.1],
            "mom_12m": [0.02, 0.05, 0.60],
            "return_skew": [0.3, 0.3, 0.3],
            "return_kurtosis": [3.0, 5.0, 4.0],
        })
        assert "Tail-risk" not in name_clusters(profile).values()

    def test_tail_risk_group_is_found_when_one_exists(self):
        profile = pd.DataFrame({
            "ann_volatility": [0.20, 0.45, 0.40],
            "beta": [0.7, 1.4, 1.1],
            "mom_12m": [0.02, 0.05, 0.60],
            "return_skew": [0.3, -1.8, 0.3],
            "return_kurtosis": [3.0, 22.0, 4.0],
        })
        assert name_clusters(profile)[1] == "Tail-risk"
