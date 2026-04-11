# src/scoring.py
"""
Worker quality scoring utilities for the crowdsourcing quality thesis.

Combines raw gold accuracy, aggregation model outputs, and behavioural features
into composite quality scores used in notebooks 02 and 03.
"""

import numpy as np
import pandas as pd
from sklearn.preprocessing import MinMaxScaler


def compute_gold_accuracy(df: pd.DataFrame, min_gold: int = 5) -> pd.DataFrame:
    """
    Compute gold-task accuracy per worker from the annotation DataFrame.

    Parameters
    ----------
    df : pd.DataFrame
        Output of data_prep.load_data().
    min_gold : int
        Minimum number of gold tasks required to include a worker.
        Workers below this threshold get NaN.

    Returns
    -------
    pd.DataFrame
        Columns: ozon_id, gold_acc, n_gold.
    """
    gold = df[(df["task_type"] == 1) & df["correct"].notna()]
    result = (
        gold.groupby("ozon_id")["correct"]
        .agg(gold_acc="mean", n_gold="count")
        .reset_index()
    )
    result.loc[result["n_gold"] < min_gold, "gold_acc"] = np.nan
    return result


def compute_composite_score(
    worker_features: pd.DataFrame,
    ds_quality: pd.Series,
    mace_competence: pd.Series,
    weights: dict = None,
) -> pd.Series:
    """
    Combine gold accuracy, Dawid-Skene quality, and MACE competence into
    a single normalised composite worker score.

    All components are Min-Max scaled to [0, 1] before combining.

    Parameters
    ----------
    worker_features : pd.DataFrame
        Output of features.build_worker_features(), indexed by ozon_id.
        Must contain 'gold_acc'.
    ds_quality : pd.Series
        Dawid-Skene diagonal quality, indexed by ozon_id.
    mace_competence : pd.Series
        MACE competence (1 - spam_prob), indexed by ozon_id.
    weights : dict, optional
        Keys: 'gold_acc', 'ds_quality', 'mace_competence'.
        Default: equal weights (1/3 each).

    Returns
    -------
    pd.Series
        Composite score in [0, 1], indexed by ozon_id.
    """
    if weights is None:
        weights = {"gold_acc": 1 / 3, "ds_quality": 1 / 3, "mace_competence": 1 / 3}

    scaler = MinMaxScaler()

    df_score = pd.DataFrame({
        "gold_acc": worker_features["gold_acc"],
        "ds_quality": ds_quality,
        "mace_competence": mace_competence,
    }).dropna()

    scaled = pd.DataFrame(
        scaler.fit_transform(df_score),
        index=df_score.index,
        columns=df_score.columns,
    )

    composite = (
        scaled["gold_acc"] * weights["gold_acc"]
        + scaled["ds_quality"] * weights["ds_quality"]
        + scaled["mace_competence"] * weights["mace_competence"]
    )
    composite.name = "composite_score"
    return composite


def assign_confidence_tiers(
    scores: pd.Series,
    thresholds: tuple = (0.33, 0.66),
) -> pd.Series:
    """
    Segment workers into three confidence tiers based on their composite score.

    Parameters
    ----------
    scores : pd.Series
        Composite quality scores (output of compute_composite_score).
    thresholds : tuple of two floats
        Lower and upper cut-points for Low / Medium / High tiers.

    Returns
    -------
    pd.Series
        Tier labels ('Low', 'Medium', 'High'), indexed like scores.
    """
    lo, hi = thresholds
    tiers = pd.cut(
        scores,
        bins=[-np.inf, lo, hi, np.inf],
        labels=["Low", "Medium", "High"],
    )
    tiers.name = "confidence_tier"
    return tiers
