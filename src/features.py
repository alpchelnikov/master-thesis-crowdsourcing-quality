# src/features.py
"""
Worker-level feature engineering for the crowdsourcing quality thesis.

All features are constructed at the worker (ozon_id) level.
The temporal split logic lives here so notebooks 02/03 stay clean.
"""

import numpy as np
import pandas as pd


# ── Feature groups (mirrors the six groups described in notebook 02) ──────────
#
#   1. Activity       — n_answers, n_projects, n_pools, lifetime_h
#   2. Accuracy       — gold_acc, n_gold, reg_acc, n_regular, agreement_rate
#   3. Timing         — med_duration, std_duration, p10_duration, p90_duration
#   4. Diversity      — answer_entropy, skip_rate
#   5. Platform flags — has_regular, has_rehab, has_training
#   6. Pricing        — mean_price, max_price
#
# The list below is used in notebooks to select the feature matrix.
FEATURE_COLS = [
    # activity
    "n_answers", "n_projects", "n_pools", "lifetime_h",
    # accuracy
    "gold_acc", "reg_acc", "agreement_rate",
    # timing
    "med_duration", "std_duration", "p10_duration", "p90_duration",
    # diversity
    "answer_entropy", "skip_rate",
    # platform flags
    "has_regular", "has_rehab", "has_training",
    # pricing
    "mean_price", "max_price",
]


def _answer_entropy(series: pd.Series) -> float:
    """Shannon entropy of answer distribution for one worker."""
    p = series.value_counts(normalize=True)
    return -(p * np.log2(p.clip(lower=1e-10))).sum()


def build_worker_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Construct the full 21-column worker feature table from a DataFrame.

    Parameters
    ----------
    df : pd.DataFrame
        Output of data_prep.load_data().  Must contain columns:
        ozon_id, task_id, project_id, pool_id, pool_type, task_type,
        user_ans, task_ans, correct, skipped, per_task_sec,
        created_at, price.

    Returns
    -------
    pd.DataFrame
        One row per worker (ozon_id), shape ~(n_workers, 21).

    Notes
    -----
    Call this function on the *feature window* subset of the data, not the
    full DataFrame, to avoid leakage into the prediction target.
    See build_temporal_split() for the correct usage pattern.
    """
    # ── 1. Activity ───────────────────────────────────────────────────────────
    w = df.groupby("ozon_id").agg(
        n_answers=("task_id", "size"),
        n_projects=("project_id", "nunique"),
        n_pools=("pool_id", "nunique"),
        skip_rate=("skipped", "mean"),
    ).reset_index()

    # ── 2. Timing (filter to sensible range) ──────────────────────────────────
    dur = (
        df[df["per_task_sec"].between(1, 600)]
        .groupby("ozon_id")["per_task_sec"]
        .agg(
            med_duration="median",
            std_duration="std",
            p10_duration=lambda x: x.quantile(0.1),
            p90_duration=lambda x: x.quantile(0.9),
        )
    )
    w = w.merge(dur, on="ozon_id", how="left")

    # ── 3. Gold accuracy ──────────────────────────────────────────────────────
    gold = df[(df["task_type"] == 1) & df["correct"].notna()]
    ga = (
        gold.groupby("ozon_id")["correct"]
        .agg(gold_acc="mean", n_gold="count")
    )
    w = w.merge(ga, on="ozon_id", how="left")

    # ── 4. Regular-task accuracy ──────────────────────────────────────────────
    ra = (
        df[(df["pool_type"] == 0) & df["correct"].notna()]
        .groupby("ozon_id")["correct"]
        .agg(reg_acc="mean", n_regular="count")
    )
    w = w.merge(ra, on="ozon_id", how="left")

    # ── 5. Agreement rate (regular tasks only) ────────────────────────────────
    ag = (
        df[(df["pool_type"] == 0) & (df["task_type"] == 0) & df["correct"].notna()]
        .groupby("ozon_id")["correct"]
        .mean()
        .rename("agreement_rate")
    )
    w = w.merge(ag, on="ozon_id", how="left")

    # ── 6. Answer entropy ─────────────────────────────────────────────────────
    ent = (
        df[df["user_ans"].notna()]
        .groupby("ozon_id")["user_ans"]
        .apply(_answer_entropy)
        .rename("answer_entropy")
    )
    w = w.merge(ent, on="ozon_id", how="left")

    # ── 7. Pool-type flags ────────────────────────────────────────────────────
    for pt, lbl in [(0, "has_regular"), (1, "has_rehab"), (3, "has_training")]:
        w[lbl] = w["ozon_id"].isin(
            df[df["pool_type"] == pt]["ozon_id"]
        ).astype(int)

    # ── 8. Lifetime ───────────────────────────────────────────────────────────
    lt = df.groupby("ozon_id")["created_at"].agg(["min", "max"])
    lt["lifetime_h"] = (lt["max"] - lt["min"]).dt.total_seconds() / 3600
    w = w.merge(lt[["lifetime_h"]], on="ozon_id", how="left")

    # ── 9. Pricing ────────────────────────────────────────────────────────────
    pr = (
        df[df["pool_type"] == 0]
        .groupby("ozon_id")["price"]
        .agg(mean_price="mean", max_price="max")
    )
    w = w.merge(pr, on="ozon_id", how="left")

    return w


def build_temporal_split(
    df: pd.DataFrame,
    feature_days: int = 14,
    target_days: int = 7,
    gap_days: int = 0,
) -> tuple[pd.DataFrame, pd.Series]:
    """
    Split the data into a feature window and a target window to prevent leakage.

    The feature window covers the first `feature_days` of the observation period.
    The target window covers the last `target_days` (with an optional gap).
    The target variable is future gold accuracy in the target window.

    Parameters
    ----------
    df : pd.DataFrame
        Full preprocessed DataFrame.
    feature_days : int
        Length of the feature construction window in days.
    target_days : int
        Length of the target measurement window in days.
    gap_days : int
        Optional gap between feature and target windows.

    Returns
    -------
    X : pd.DataFrame
        Worker feature matrix (one row per worker who appears in both windows).
    y : pd.Series
        Target: gold accuracy in the target window, indexed by ozon_id.

    Notes
    -----
    Workers who do not appear in both windows are dropped.
    Adjust window sizes based on the temporal span of your dataset.
    """
    t_min = df["created_at"].min()

    feature_end = t_min + pd.Timedelta(days=feature_days)
    target_start = feature_end + pd.Timedelta(days=gap_days)
    target_end = target_start + pd.Timedelta(days=target_days)

    df_feat = df[df["created_at"] < feature_end]
    df_tgt = df[
        (df["created_at"] >= target_start)
        & (df["created_at"] < target_end)
        & (df["task_type"] == 1)
        & df["correct"].notna()
    ]

    X = build_worker_features(df_feat).set_index("ozon_id")

    y = (
        df_tgt.groupby("ozon_id")["correct"]
        .mean()
        .rename("future_gold_acc")
    )

    # Keep only workers present in both windows
    common = X.index.intersection(y.index)
    return X.loc[common], y.loc[common]
