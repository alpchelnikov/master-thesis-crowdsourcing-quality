# src/segmentation.py
"""
Worker segmentation (clustering) for the crowdsourcing quality thesis.

Wraps KMeans, GMM, and Agglomerative clustering with a consistent interface
and adds cluster labelling helpers.  Used in notebook 03.
"""

import numpy as np
import pandas as pd
from sklearn.cluster import KMeans, AgglomerativeClustering
from sklearn.mixture import GaussianMixture
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import silhouette_score


def prepare_cluster_features(
    worker_features: pd.DataFrame,
    feature_cols: list,
) -> tuple[np.ndarray, pd.Index]:
    """
    Scale worker features for clustering.

    Parameters
    ----------
    worker_features : pd.DataFrame
        Feature table indexed by ozon_id.
    feature_cols : list of str
        Columns to include.  Rows with NaN in any of these columns are dropped.

    Returns
    -------
    X_scaled : np.ndarray
        Standardised feature matrix.
    index : pd.Index
        ozon_id values corresponding to rows of X_scaled.
    """
    sub = worker_features[feature_cols].dropna()
    X = StandardScaler().fit_transform(sub.values)
    return X, sub.index


def run_clustering(
    X: np.ndarray,
    index: pd.Index,
    n_clusters: int = 4,
    method: str = "kmeans",
    random_state: int = 42,
) -> pd.Series:
    """
    Cluster workers and return cluster labels.

    Parameters
    ----------
    X : np.ndarray
        Scaled feature matrix (output of prepare_cluster_features).
    index : pd.Index
        ozon_id values aligned with X.
    n_clusters : int
        Number of clusters (K=4 used in the thesis).
    method : str
        One of 'kmeans', 'gmm', 'agglomerative'.
    random_state : int
        Random seed for reproducibility.

    Returns
    -------
    pd.Series
        Cluster label (0-indexed integer) per worker, indexed by ozon_id.
    """
    if method == "kmeans":
        model = KMeans(n_clusters=n_clusters, random_state=random_state, n_init=10)
        labels = model.fit_predict(X)

    elif method == "gmm":
        model = GaussianMixture(
            n_components=n_clusters, random_state=random_state, n_init=3
        )
        labels = model.fit_predict(X)

    elif method == "agglomerative":
        model = AgglomerativeClustering(n_clusters=n_clusters)
        labels = model.fit_predict(X)

    else:
        raise ValueError(f"Unknown method: {method!r}.  Choose kmeans, gmm, or agglomerative.")

    return pd.Series(labels, index=index, name="cluster")


def silhouette(X: np.ndarray, labels: np.ndarray) -> float:
    """Convenience wrapper for silhouette score."""
    return silhouette_score(X, labels)


def describe_clusters(
    worker_features: pd.DataFrame,
    labels: pd.Series,
    feature_cols: list,
) -> pd.DataFrame:
    """
    Compute per-cluster mean and std for selected features.

    Parameters
    ----------
    worker_features : pd.DataFrame
        Feature table indexed by ozon_id.
    labels : pd.Series
        Cluster assignments indexed by ozon_id.
    feature_cols : list of str
        Features to include in the summary.

    Returns
    -------
    pd.DataFrame
        MultiIndex columns (feature, stat) for each cluster.
    """
    merged = worker_features[feature_cols].join(labels, how="inner")
    return merged.groupby("cluster")[feature_cols].agg(["mean", "std"]).round(3)
