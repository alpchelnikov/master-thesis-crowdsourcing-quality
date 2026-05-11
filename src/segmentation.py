# src/segmentation.py
"""
Worker segmentation for the crowdsourcing quality thesis.

Implements the clustering, robustness, and segment-naming pipeline from
Section 2.12 / 3.6 of the thesis.

Three components are exposed:

  * :func:`run_clustering` — fit one of three clustering algorithms
    (k-means, GMM, agglomerative) on a scaled feature matrix.

  * :func:`compare_clusterings` — sweep ``K ∈ {3..6}`` across all three
    algorithms and report both silhouette and Calinski–Harabasz scores.

  * :func:`segmentation_robustness` — two robustness checks documented
    in Section 2.12:

      1. *Stability under alternative K* — Adjusted Rand Index between
         the chosen K and neighbouring K values.
      2. *Behaviour-only re-clustering* — re-fit k-means without
         ``quality_score`` in the feature set; if the behaviour-only
         partition still stratifies by quality, the segmentation is
         grounded in genuine behavioural differences rather than in the
         composite-score feature itself.

A deterministic segment-naming function :func:`name_segments` maps
cluster centroids in (score, activity) space to the four operational
labels used in the thesis: Reliable veteran / Promising newcomer /
Average worker / Low quality.

These utilities mirror the implementation in
``notebooks/03_Advanced_Models.ipynb``.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Optional

import numpy as np
import pandas as pd
from sklearn.cluster import KMeans, AgglomerativeClustering
from sklearn.metrics import (
    adjusted_rand_score,
    calinski_harabasz_score,
    silhouette_score,
)
from sklearn.mixture import GaussianMixture
from sklearn.preprocessing import StandardScaler


# ── Defaults documented in Section 2.12 ──────────────────────────────────────

DEFAULT_CLUSTER_FEATURES = [
    "quality_score",
    "log_n_answers",
    "answer_entropy",
    "per_task_sec_median",
    "pct_label_1",
]

DEFAULT_BEHAVIOUR_FEATURES = [
    "log_n_answers",
    "answer_entropy",
    "per_task_sec_median",
    "pct_label_1",
]

SEGMENT_NAMES = [
    "Reliable veteran",
    "Promising newcomer",
    "Average worker",
    "Low quality",
]


# ── Feature-matrix preparation ───────────────────────────────────────────────

def prepare_cluster_features(
    worker_features: pd.DataFrame,
    feature_cols: Optional[list] = None,
    *,
    min_answers: int = 10,
    require: Optional[list] = ("quality_score",),
    add_log_n_answers: bool = True,
) -> tuple[np.ndarray, pd.Index, pd.DataFrame]:
    """
    Filter, impute, log-transform and standardise the worker feature
    table for clustering.

    Parameters
    ----------
    worker_features : pd.DataFrame
        Feature table indexed by ``ozon_id``.  Must contain at least the
        columns named in ``feature_cols``.
    feature_cols : list of str, optional
        Columns to standardise and pass to the clustering algorithm.
        Defaults to :data:`DEFAULT_CLUSTER_FEATURES`.
    min_answers : int
        Workers with fewer than this many answers are excluded.
    require : iterable of str, optional
        Columns whose presence (non-NaN) is required before scaling.
        Workers missing any of these are dropped to avoid imputing the
        composite score itself.
    add_log_n_answers : bool
        Whether to derive ``log_n_answers = log(1 + n_answers)`` if
        ``n_answers`` is present and ``log_n_answers`` is not.

    Returns
    -------
    X_scaled : np.ndarray
        Standardised feature matrix.
    index : pd.Index
        ``ozon_id`` values aligned with the rows of ``X_scaled``.
    wc : pd.DataFrame
        Pre-scaling table of the same rows, including ``log_n_answers``
        and any imputation that was applied — useful for plotting and
        for segment naming.
    """
    if feature_cols is None:
        feature_cols = list(DEFAULT_CLUSTER_FEATURES)

    wc = worker_features.copy()
    if add_log_n_answers and "n_answers" in wc.columns and "log_n_answers" not in wc.columns:
        wc["log_n_answers"] = np.log1p(wc["n_answers"])

    # Require non-NaN for fields we don't want to impute (typically the
    # composite quality score, which encodes the model output).
    if require:
        require = [r for r in require if r in wc.columns]
        if require:
            wc = wc[wc[require].notna().all(axis=1)]

    if "n_answers" in wc.columns:
        wc = wc[wc["n_answers"] >= min_answers]

    wc = wc.copy()
    for feat in feature_cols:
        if feat not in wc.columns:
            raise KeyError(
                f"Feature {feat!r} not in worker_features columns: "
                f"{list(wc.columns)}"
            )
        wc[feat] = wc[feat].fillna(wc[feat].median())

    X = wc[feature_cols].values
    X_scaled = StandardScaler().fit_transform(X)
    return X_scaled, wc.index, wc


# ── Single-shot clustering ───────────────────────────────────────────────────

def run_clustering(
    X: np.ndarray,
    index: pd.Index,
    *,
    n_clusters: int = 3,
    method: str = "kmeans",
    random_state: int = 42,
) -> pd.Series:
    """
    Fit a clustering algorithm and return integer labels indexed by
    ``ozon_id``.

    Parameters
    ----------
    X : np.ndarray
        Scaled feature matrix.
    index : pd.Index
        ``ozon_id`` values aligned with ``X``.
    n_clusters : int
        Number of clusters.  The thesis fixes this operationally
        (see Section 2.12).
    method : {'kmeans', 'gmm', 'agglomerative'}
        Algorithm used.  Three are compared because they make different
        assumptions (spherical hard, ellipsoidal soft, hierarchical
        without centroid).
    random_state : int

    Returns
    -------
    pd.Series
        Cluster id per worker.
    """
    method = method.lower()
    if method == "kmeans":
        model = KMeans(n_clusters=n_clusters, random_state=random_state, n_init=10)
    elif method == "gmm":
        model = GaussianMixture(
            n_components=n_clusters, random_state=random_state, n_init=3
        )
    elif method == "agglomerative":
        model = AgglomerativeClustering(n_clusters=n_clusters)
    else:
        raise ValueError(
            f"Unknown method {method!r}. Choose 'kmeans', 'gmm' or 'agglomerative'."
        )
    labels = model.fit_predict(X)
    return pd.Series(labels, index=index, name="cluster")


# ── Method/K sweep ───────────────────────────────────────────────────────────

def compare_clusterings(
    X: np.ndarray,
    *,
    k_grid: tuple = (3, 4, 5, 6),
    methods: tuple = ("kmeans", "gmm", "agglomerative"),
    random_state: int = 42,
) -> pd.DataFrame:
    """
    Sweep the (method, K) grid and return silhouette and Calinski–Harabasz
    scores for each combination.

    Higher is better for both metrics; the two often disagree on the
    optimal K, which is exactly why the thesis reports both rather than
    picking either one in isolation.

    Parameters
    ----------
    X : np.ndarray
        Scaled feature matrix.
    k_grid : iterable of int
        Cluster counts to evaluate.
    methods : iterable of str
        Algorithms to evaluate.
    random_state : int

    Returns
    -------
    pd.DataFrame
        Columns: ``Method``, ``K``, ``Silhouette``, ``Calinski-Harabasz``.
    """
    rows = []
    for K in k_grid:
        for m in methods:
            labels = run_clustering(
                X, pd.RangeIndex(len(X)),
                n_clusters=K, method=m, random_state=random_state,
            ).values
            rows.append({
                "Method": {"kmeans": "KMeans",
                           "gmm": "GMM",
                           "agglomerative": "Agglomerative"}[m],
                "K": K,
                "Silhouette": silhouette_score(X, labels),
                "Calinski-Harabasz": calinski_harabasz_score(X, labels),
            })
    return pd.DataFrame(rows)


# ── Robustness checks (Section 2.12) ─────────────────────────────────────────

@dataclass
class RobustnessReport:
    """Output of :func:`segmentation_robustness`."""

    ari_under_alt_k: pd.DataFrame
    behaviour_only_summary: pd.DataFrame
    ari_behaviour_vs_full: float
    base_K: int
    base_labels: pd.Series
    behaviour_labels: pd.Series


def segmentation_robustness(
    wc: pd.DataFrame,
    X_full: np.ndarray,
    *,
    base_K: int = 3,
    alternative_Ks: tuple = (3, 5),
    behaviour_features: Optional[list] = None,
    quality_col: str = "quality_score",
    gold_col: str = "gold_acc",
    random_state: int = 42,
) -> RobustnessReport:
    """
    Two-part robustness analysis for the worker segmentation.

    Part 1 – *Stability under alternative K.* For each ``K_alt`` in
    ``alternative_Ks``, fit k-means with that K on the same scaled
    feature matrix and compute the Adjusted Rand Index against the base
    K labels.  Also reports the mean ``quality_score`` of the top- and
    bottom-quality cluster under the alternative K; a large quality
    spread indicates the alternative K preserves the quality ordering
    of the workers even when the cluster count changes.

    Part 2 – *Behaviour-only re-clustering.* Re-fit k-means on the
    behaviour features alone (no ``quality_score`` among the inputs).
    If the resulting behaviour-only partition still stratifies workers
    by quality — i.e. cluster means of ``quality_score`` differ
    materially across clusters — the four-segment structure is grounded
    in genuine behavioural differences and not in the composite-score
    feature that the base clustering also receives.

    Parameters
    ----------
    wc : pd.DataFrame
        Pre-scaling table aligned with ``X_full`` (output of
        :func:`prepare_cluster_features`).  Must include
        ``quality_score`` and the behaviour features.
    X_full : np.ndarray
        Scaled feature matrix used to fit the base clustering.
    base_K : int
        Number of clusters in the segmentation under audit.
    alternative_Ks : iterable of int
        Alternative K values for the ARI stability test.
    behaviour_features : list of str, optional
        Columns used for the behaviour-only re-clustering.  Defaults to
        :data:`DEFAULT_BEHAVIOUR_FEATURES`.
    quality_col, gold_col : str
        Column names used in the behaviour-only quality summary.
    random_state : int

    Returns
    -------
    RobustnessReport
    """
    if behaviour_features is None:
        behaviour_features = list(DEFAULT_BEHAVIOUR_FEATURES)

    # Base labels at the chosen K
    base_labels = run_clustering(
        X_full, wc.index,
        n_clusters=base_K, method="kmeans", random_state=random_state,
    )

    # Part 1 — ARI vs alternative K
    rows = []
    for K_alt in alternative_Ks:
        alt = run_clustering(
            X_full, wc.index,
            n_clusters=K_alt, method="kmeans", random_state=random_state,
        )
        ari = adjusted_rand_score(base_labels.values, alt.values)
        cluster_q = (
            pd.DataFrame({"cluster": alt.values,
                          quality_col: wc[quality_col].values})
            .groupby("cluster")[quality_col].mean()
        )
        rows.append({
            "K": K_alt,
            f"ARI vs K={base_K}": round(ari, 3),
            "top cluster mean score": round(float(cluster_q.max()), 1),
            "bottom cluster mean score": round(float(cluster_q.min()), 1),
            "spread": round(float(cluster_q.max() - cluster_q.min()), 1),
        })
    ari_table = pd.DataFrame(rows)

    # Part 2 — behaviour-only k-means
    X_behav = StandardScaler().fit_transform(wc[behaviour_features].values)
    behav_labels = run_clustering(
        X_behav, wc.index,
        n_clusters=base_K, method="kmeans", random_state=random_state,
    )

    behav_profile = pd.DataFrame({
        "cluster": behav_labels.values,
        "quality_score": wc[quality_col].values,
        "gold_acc": wc[gold_col].values if gold_col in wc.columns else np.nan,
    })
    behav_summary = (
        behav_profile
        .groupby("cluster")
        .agg(
            size=("quality_score", "size"),
            mean_score=("quality_score", "mean"),
            mean_gold_acc=("gold_acc", "mean"),
        )
        .round(2)
        .sort_values("mean_score", ascending=False)
    )
    ari_behaviour = float(adjusted_rand_score(base_labels.values, behav_labels.values))

    return RobustnessReport(
        ari_under_alt_k=ari_table,
        behaviour_only_summary=behav_summary,
        ari_behaviour_vs_full=round(ari_behaviour, 3),
        base_K=base_K,
        base_labels=base_labels,
        behaviour_labels=behav_labels,
    )


# ── Deterministic segment naming ─────────────────────────────────────────────

def name_segments(
    profile: pd.DataFrame,
    *,
    high_quality_threshold: float = 75.0,
    low_quality_threshold: float = 65.0,
    veteran_n_answers: int = 100,
    quality_col: str = "quality_score",
    activity_col: str = "n_answers",
) -> dict:
    """
    Map cluster ids to operational segment names using a deterministic
    rule on cluster centroid position in ``(quality_score, n_answers)``
    space.

    The rule, documented in Section 2.12 of the thesis:

      * ``quality_score ≥ 75`` and ``n_answers ≥ 100`` → *Reliable veteran*
      * ``quality_score ≥ 75`` and ``n_answers < 100``  → *Promising newcomer*
      * ``quality_score < 65``                          → *Low quality*
      * otherwise                                       → *Average worker*

    The rule does not involve analyst judgement: given a cluster profile
    table, the mapping from cluster id to name is fully determined.
    Reproducibility is the point — re-running the clustering must yield
    the same named segments up to a permutation of cluster ids.

    Parameters
    ----------
    profile : pd.DataFrame
        Per-cluster mean profile, indexed by cluster id.  Must contain
        ``quality_col`` and ``activity_col``.
    high_quality_threshold, low_quality_threshold : float
        Thresholds for the high- and low-quality decisions, expressed on
        the 0–100 composite scale.
    veteran_n_answers : int
        Activity threshold separating Reliable veterans from Promising
        newcomers among high-quality clusters.
    quality_col, activity_col : str
        Column names within ``profile``.

    Returns
    -------
    dict
        Mapping ``{cluster_id: segment_name}``.
    """
    if quality_col not in profile.columns or activity_col not in profile.columns:
        raise KeyError(
            f"profile must contain {quality_col!r} and {activity_col!r} columns. "
            f"Got: {list(profile.columns)}"
        )

    out = {}
    for cluster_id, row in profile.iterrows():
        q = float(row[quality_col])
        n = float(row[activity_col])
        if q >= high_quality_threshold and n >= veteran_n_answers:
            name = "Reliable veteran"
        elif q >= high_quality_threshold and n < veteran_n_answers:
            name = "Promising newcomer"
        elif q < low_quality_threshold:
            name = "Low quality"
        else:
            name = "Average worker"
        out[cluster_id] = name
    return out


def describe_clusters(
    worker_features: pd.DataFrame,
    labels: pd.Series,
    feature_cols: list,
) -> pd.DataFrame:
    """
    Per-cluster mean for selected features and the cluster size.

    Used to build the profile that :func:`name_segments` consumes.

    Parameters
    ----------
    worker_features : pd.DataFrame
        Feature table indexed by ``ozon_id``.
    labels : pd.Series
        Cluster assignments indexed by ``ozon_id``.
    feature_cols : list of str
        Features to summarise.

    Returns
    -------
    pd.DataFrame
        Indexed by cluster id, with ``size`` and one column per feature.
    """
    merged = worker_features[feature_cols].join(labels, how="inner")
    out = merged.groupby("cluster")[feature_cols].mean()
    out["size"] = merged.groupby("cluster").size()
    return out
