"""Clustering of behavioural features, plus the diagnostics used to choose k.

Silhouette alone is not enough here: it rises monotonically toward small k on
this data, and its absolute values (~0.2) say the behaviour space is a
continuum rather than a set of well-separated blobs. Cluster count is therefore
chosen against a bootstrap stability curve as well, and the trade-off is
recorded rather than hidden behind a single number.
"""

import numpy as np
import pandas as pd
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from sklearn.metrics import (adjusted_rand_score, davies_bouldin_score,
                             normalized_mutual_info_score, silhouette_score)
from sklearn.preprocessing import StandardScaler

RANDOM_STATE = 42
PCA_VARIANCE_TARGET = 0.90

# Qualifying bars for the two conditional cluster names. Skew is compared
# absolutely (a left tail is meaningful in itself); momentum is compared to the
# other clusters, since a whole-market drawdown would otherwise leave no
# "momentum" group and a raging bull market would make every cluster one.
TAIL_SKEW_THRESHOLD = -0.5
MOMENTUM_EXCESS_THRESHOLD = 0.15


def reduce_features(features, variance_target=PCA_VARIANCE_TARGET):
    """Scale to comparable units, then PCA away the volatility/drawdown overlap.

    Returns (components, fitted_scaler, fitted_pca).
    """
    scaler = StandardScaler()
    scaled = scaler.fit_transform(features.values)

    probe = PCA(random_state=RANDOM_STATE).fit(scaled)
    n_components = int(np.searchsorted(np.cumsum(probe.explained_variance_ratio_),
                                       variance_target) + 1)

    pca = PCA(n_components=n_components, random_state=RANDOM_STATE)
    return pca.fit_transform(scaled), scaler, pca


def bootstrap_stability(components, k, n_trials=30, sample_frac=0.8,
                        random_state=RANDOM_STATE):
    """How reproducible a k-cluster solution is under resampling.

    Re-clusters random 80% subsamples and scores each against the full-sample
    labels on the shared symbols. A partition that only exists for one
    particular sample is not a finding.
    """
    rng = np.random.default_rng(random_state)
    reference = KMeans(n_clusters=k, random_state=random_state,
                       n_init=20).fit_predict(components)

    scores = []
    for trial in range(n_trials):
        idx = rng.choice(len(components), int(sample_frac * len(components)),
                         replace=False)
        resampled = KMeans(n_clusters=k, random_state=trial,
                           n_init=10).fit_predict(components[idx])
        scores.append(adjusted_rand_score(reference[idx], resampled))
    return float(np.mean(scores))


def sweep_k(components, k_values=range(2, 13), sector_codes=None, n_trials=30):
    """Diagnostics for every candidate k, as a tidy frame."""
    rows = []
    for k in k_values:
        labels = KMeans(n_clusters=k, random_state=RANDOM_STATE,
                        n_init=20).fit_predict(components)
        row = {
            "k": k,
            "silhouette": silhouette_score(components, labels),
            "davies_bouldin": davies_bouldin_score(components, labels),
            "stability": bootstrap_stability(components, k, n_trials=n_trials),
        }
        if sector_codes is not None:
            row["ari_vs_sector"] = adjusted_rand_score(labels, sector_codes)
            row["nmi_vs_sector"] = normalized_mutual_info_score(labels, sector_codes)
        rows.append(row)
    return pd.DataFrame(rows).set_index("k")


def fit_clusters(components, k, random_state=RANDOM_STATE):
    return KMeans(n_clusters=k, random_state=random_state,
                  n_init=50).fit_predict(components)


def name_clusters(profile):
    """Give each cluster a descriptive name derived from its own profile.

    Rule-based rather than hand-written, so the names survive a re-run that
    permutes the integer cluster ids.

    The two distinctive names are *conditional*: a partition only contains a
    tail-risk or momentum group if some cluster actually looks like one. Naming
    them unconditionally would relabel the calmest cluster in a three-way split
    as "Tail-risk" purely for being the least calm.
    """
    remaining = list(profile.index)
    names = {}

    def claim(pick, name, qualifies=None):
        if not remaining:
            return
        candidate = pick(profile.loc[remaining])
        if qualifies is not None and not qualifies(profile.loc[candidate]):
            return
        names[candidate] = name
        remaining.remove(candidate)

    median_momentum = profile["mom_12m"].median()

    # Ordered most-distinctive first: a fat left tail is unmistakable, a
    # momentum regime next, then the two ends of the risk range.
    claim(lambda p: p["return_skew"].idxmin(), "Tail-risk",
          lambda row: row["return_skew"] < TAIL_SKEW_THRESHOLD)
    claim(lambda p: p["mom_12m"].idxmax(), "Momentum leaders",
          lambda row: row["mom_12m"] - median_momentum > MOMENTUM_EXCESS_THRESHOLD)
    claim(lambda p: p["beta"].idxmax(), "High-beta cyclicals")
    claim(lambda p: p["ann_volatility"].idxmin(), "Large-cap defensives")

    for leftover in list(remaining):
        names[leftover] = "Quiet mid-caps"
        remaining.remove(leftover)

    return names


def profile_clusters(features, labels):
    """Mean feature vector per cluster, with sizes and descriptive names."""
    frame = features.copy()
    frame["cluster"] = labels
    profile = frame.groupby("cluster")[list(features.columns)].mean()
    profile.insert(0, "n", frame.groupby("cluster").size())
    profile.insert(1, "name", pd.Series(name_clusters(profile)))
    return profile
