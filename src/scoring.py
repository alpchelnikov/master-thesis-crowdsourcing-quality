# src/scoring.py
"""
Composite worker quality score and task-level confidence tiers.

This module implements two related pieces of the pipeline described in
Chapter 2 of the thesis:

  * :func:`compute_composite_score` — fits a ridge regression that predicts
    held-out gold accuracy from three signals (agreement rate, Dawid–Skene
    diagonal, MACE competence) and reports the learned linear weights
    together with a 0–100 winsorised score for every worker.

  * :func:`assign_confidence_tiers` — partitions *tasks* (not workers) into
    four tiers based on worker agreement and Dawid–Skene posterior
    confidence: Confident / Likely correct / Borderline / Contested.

Both are vectorised; both match the implementations in
``notebooks/03_Advanced_Models.ipynb``.

A small helper :func:`compute_gold_accuracy` is also provided since
several notebooks need it.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

import numpy as np
import pandas as pd
from sklearn.linear_model import Ridge
from sklearn.model_selection import cross_val_score, train_test_split
from sklearn.preprocessing import MinMaxScaler


# ── Gold-task accuracy helper ────────────────────────────────────────────────

def compute_gold_accuracy(df: pd.DataFrame, min_gold: int = 5) -> pd.DataFrame:
    """
    Per-worker gold-task accuracy and the count it is based on.

    Parameters
    ----------
    df : pd.DataFrame
        Pre-processed annotation table (output of ``data_prep.load_data``).
        Must contain ``ozon_id``, ``task_type`` and ``correct``.
    min_gold : int
        Workers with fewer than this many gold observations are returned
        with ``gold_acc = NaN`` rather than a noisy point estimate.

    Returns
    -------
    pd.DataFrame
        Columns: ``ozon_id``, ``gold_acc``, ``n_gold``.
    """
    gold = df[(df["task_type"] == 1) & df["correct"].notna()]
    out = (
        gold.groupby("ozon_id")["correct"]
        .agg(gold_acc="mean", n_gold="count")
        .reset_index()
    )
    out.loc[out["n_gold"] < min_gold, "gold_acc"] = np.nan
    return out


# ── Composite quality score ──────────────────────────────────────────────────

@dataclass
class CompositeScoreResult:
    """
    Output of :func:`compute_composite_score`.

    Attributes
    ----------
    scores : pd.Series
        Final composite score in [0, 100], indexed by ``ozon_id``.
    raw_scores : pd.Series
        Ridge predictions before winsorisation and 0–100 scaling.
    ridge : sklearn.linear_model.Ridge
        Fitted ridge model (coefficients aligned with ``feature_names``).
    feature_names : list of str
        The three component signals, in the order used to fit the ridge.
    cv_r2_mean, cv_r2_std : float
        5-fold cross-validated R² on the training cohort.
    holdout_r2, holdout_mae : float
        R² and MAE on the worker-disjoint held-out cohort. Holdout workers
        are never seen by the ridge fit; this is the honest validation.
    train_workers, test_workers : np.ndarray
        Worker ids used to fit the ridge and to hold out, respectively.
    """

    scores: pd.Series
    raw_scores: pd.Series
    ridge: Ridge
    feature_names: list
    cv_r2_mean: float
    cv_r2_std: float
    holdout_r2: float
    holdout_mae: float
    train_workers: np.ndarray
    test_workers: np.ndarray


def compute_composite_score(
    worker_features: pd.DataFrame,
    ds_quality: pd.Series,
    mace_competence: pd.Series,
    *,
    min_gold: int = 40,
    alpha: float = 1.0,
    test_size: float = 0.25,
    winsor: tuple = (0.01, 0.99),
    score_range: tuple = (0, 100),
    random_state: int = 42,
) -> CompositeScoreResult:
    """
    Fit a ridge regression that predicts held-out gold accuracy from three
    quality signals and project every worker onto a 0–100 composite score.

    Three signals are used as predictors:
      1. ``agreement_rate`` – worker's agreement with the platform majority
         on regular tasks (cheap but biased proxy);
      2. ``ds_score``       – mean diagonal of the Dawid–Skene confusion
         matrix (from :class:`aggregation.DawidSkene`);
      3. ``mace_score``     – MACE competence ``1 − σ_w`` (from
         :class:`aggregation.MACE`).

    The ridge is fit on workers with at least ``min_gold`` gold
    observations — the *rated* set in the thesis — and validated on a
    worker-disjoint held-out cohort.  The raw ridge predictions are then
    winsorised at the ``winsor`` percentiles and Min-Max scaled to
    ``score_range`` to produce the published composite score for every
    worker who has at least one of the three input signals.

    The procedure mirrors the implementation in
    ``notebooks/03_Advanced_Models.ipynb``.

    Parameters
    ----------
    worker_features : pd.DataFrame
        Worker feature table, indexed by ``ozon_id``.  Must contain at
        least ``gold_acc``, ``n_gold`` and ``agreement_rate``.
    ds_quality : pd.Series
        Dawid–Skene per-worker score, indexed by ``ozon_id``.
    mace_competence : pd.Series
        MACE competence, indexed by ``ozon_id``.
    min_gold : int
        Minimum gold observations a worker must have to enter the ridge
        training set.  Workers below this threshold still receive a
        composite score (computed from the fitted weights and median-
        imputed predictors) but do not influence the fit.
    alpha : float
        L2 regularisation strength for the ridge.
    test_size : float
        Fraction of rated workers held out for honest validation.
    winsor : (float, float)
        Lower and upper quantiles used to winsorise the raw ridge
        predictions before scaling.
    score_range : (float, float)
        Range of the final composite score.
    random_state : int

    Returns
    -------
    CompositeScoreResult
    """
    feature_names = ["agreement_rate", "ds_score", "mace_score"]

    # Build a single signal table indexed by worker
    wf = worker_features.copy()
    if "agreement_rate" not in wf.columns:
        raise KeyError(
            "worker_features must contain an 'agreement_rate' column. "
            "If your feature table uses a different name (e.g. "
            "'reg_agreement_proxy'), rename it before calling this function."
        )
    signals = pd.DataFrame({
        "agreement_rate": wf["agreement_rate"],
        "ds_score":       ds_quality.reindex(wf.index),
        "mace_score":     mace_competence.reindex(wf.index),
    })
    signals.index.name = wf.index.name

    # Rated workers form the training pool
    rated_mask = (wf["n_gold"] >= min_gold) & wf["gold_acc"].notna()
    train_pool = (
        signals.loc[rated_mask]
        .dropna(subset=feature_names)
        .join(wf.loc[rated_mask, "gold_acc"], how="inner")
    )
    if len(train_pool) < 20:
        raise ValueError(
            f"Only {len(train_pool)} rated workers met the min_gold={min_gold} "
            "threshold. Lower the threshold or check the data."
        )

    train_workers, test_workers = train_test_split(
        train_pool.index.values,
        test_size=test_size, random_state=random_state,
    )

    X_train = train_pool.loc[train_workers, feature_names].values
    y_train = train_pool.loc[train_workers, "gold_acc"].values
    X_test = train_pool.loc[test_workers, feature_names].values
    y_test = train_pool.loc[test_workers, "gold_acc"].values

    ridge = Ridge(alpha=alpha)
    cv_scores = cross_val_score(ridge, X_train, y_train, cv=5, scoring="r2")
    ridge.fit(X_train, y_train)

    holdout_pred = ridge.predict(X_test)
    holdout_r2 = float(ridge.score(X_test, y_test))
    holdout_mae = float(np.mean(np.abs(holdout_pred - y_test)))

    # Score every worker. Missing signals are imputed with the training
    # median, which gives a cautious anchor for under-observed workers.
    scoring_df = signals.copy()
    medians = train_pool[feature_names].median()
    for col in feature_names:
        scoring_df[col] = scoring_df[col].fillna(medians[col])

    raw = pd.Series(
        ridge.predict(scoring_df[feature_names].values),
        index=scoring_df.index,
        name="raw_score",
    )

    lo, hi = raw.quantile(winsor[0]), raw.quantile(winsor[1])
    clipped = raw.clip(lower=lo, upper=hi).values.reshape(-1, 1)
    scaler = MinMaxScaler(feature_range=score_range)
    scaled = scaler.fit_transform(clipped).ravel()
    scores = pd.Series(scaled, index=scoring_df.index, name="quality_score")

    return CompositeScoreResult(
        scores=scores,
        raw_scores=raw,
        ridge=ridge,
        feature_names=feature_names,
        cv_r2_mean=float(cv_scores.mean()),
        cv_r2_std=float(cv_scores.std()),
        holdout_r2=holdout_r2,
        holdout_mae=holdout_mae,
        train_workers=np.asarray(train_workers),
        test_workers=np.asarray(test_workers),
    )


# ── Task-level confidence tiers ──────────────────────────────────────────────

# Default thresholds match notebook 03 / Section 2.11 of the thesis.
DEFAULT_TIER_THRESHOLDS = {
    "confident_agreement": 1.00,   # unanimity required
    "confident_ds":        0.90,
    "likely_agreement":    0.67,
    "likely_ds":           0.70,
    "borderline_agreement": 0.50,
}

# Canonical ordering for plots and tables
TIER_ORDER = ["Confident", "Likely correct", "Borderline", "Contested"]


def assign_confidence_tiers(
    annotations: pd.DataFrame,
    ds_task_confidence: Optional[pd.Series] = None,
    *,
    worker_col: str = "ozon_id",
    task_col: str = "task_id",
    answer_col: str = "user_ans",
    platform_col: Optional[str] = "task_ans",
    thresholds: Optional[dict] = None,
) -> pd.DataFrame:
    """
    Partition tasks into four reliability tiers based on worker agreement
    and Dawid–Skene posterior confidence.

    This is **task-level** triage (not worker-level), as described in
    Section 2.11 of the thesis.  The four tiers, in decreasing reliability:

      * **Confident**     – all workers agreed AND ``ds_conf ≥ 0.90``;
      * **Likely correct** – ``agreement ≥ 0.67`` AND ``ds_conf ≥ 0.70``;
      * **Borderline**    – ``agreement ≥ 0.50`` (otherwise);
      * **Contested**     – none of the above.

    The operational value of the partition is tested by the *error
    concentration* in the lower tiers: routing the small Contested subset
    to higher overlap or expert review should address a large share of
    label errors at a small share of cost.

    Parameters
    ----------
    annotations : pd.DataFrame
        Long-format annotation table.  Must contain ``worker_col``,
        ``task_col``, ``answer_col``.  If ``platform_col`` is given, it is
        used to compute agreement with the platform's own majority label.
    ds_task_confidence : pd.Series, optional
        Maximum posterior probability per task, e.g. from
        ``DawidSkene.task_confidence()``.  If omitted, tasks default to a
        neutral confidence of 0.5 and the DS-based criteria do not bind.
    worker_col, task_col, answer_col : str
        Column names.
    platform_col : str, optional
        Column carrying the platform's task-level label.  When present,
        the output includes an ``agrees_with_platform`` boolean.
    thresholds : dict, optional
        Override any of the keys in ``DEFAULT_TIER_THRESHOLDS``.

    Returns
    -------
    pd.DataFrame
        Indexed by task id with columns:
        ``n_answers``, ``majority_ans``, ``agreement``, ``ds_conf``,
        ``tier``, and (if ``platform_col`` is given) ``agrees_with_platform``.
    """
    thr = dict(DEFAULT_TIER_THRESHOLDS)
    if thresholds:
        thr.update(thresholds)

    agg_dict = dict(
        n_answers=(answer_col, "count"),
        n_agree=(answer_col, lambda x: x.value_counts().iloc[0]),
        majority_ans=(answer_col, lambda x: x.mode().iloc[0]),
    )
    if platform_col and platform_col in annotations.columns:
        agg_dict["platform_ans"] = (platform_col, "first")

    out = annotations.groupby(task_col).agg(**agg_dict)
    out["agreement"] = out["n_agree"] / out["n_answers"]

    if ds_task_confidence is not None:
        out["ds_conf"] = out.index.map(ds_task_confidence)
        out["ds_conf"] = out["ds_conf"].fillna(0.5)
    else:
        out["ds_conf"] = 0.5

    # Classify
    tiers = np.full(len(out), "Contested", dtype=object)
    agr = out["agreement"].values
    cf = out["ds_conf"].values

    is_borderline = agr >= thr["borderline_agreement"]
    is_likely = (agr >= thr["likely_agreement"]) & (cf >= thr["likely_ds"])
    is_confident = (agr >= thr["confident_agreement"]) & (cf >= thr["confident_ds"])

    tiers[is_borderline] = "Borderline"
    tiers[is_likely] = "Likely correct"
    tiers[is_confident] = "Confident"
    out["tier"] = pd.Categorical(tiers, categories=TIER_ORDER, ordered=True)

    if "platform_ans" in out.columns:
        out["agrees_with_platform"] = out["majority_ans"] == out["platform_ans"]

    out = out.drop(columns=["n_agree"])
    return out


def tier_summary(tiers: pd.DataFrame) -> pd.DataFrame:
    """
    Per-tier summary: count, share, mean agreement and DS confidence, and
    (if available) the within-tier agreement with the platform label and
    the share of *all* disagreements falling into each tier.

    This reproduces the diagnostic table in notebook 03 §8 (confidence
    tiers).
    """
    cols = dict(
        count=("agreement", "count"),
        mean_agreement=("agreement", "mean"),
        mean_ds_conf=("ds_conf", "mean"),
    )
    if "agrees_with_platform" in tiers.columns:
        cols["agreement_with_platform"] = ("agrees_with_platform", "mean")

    out = tiers.groupby("tier", observed=True).agg(**cols)
    out["share_pct"] = (out["count"] / out["count"].sum() * 100).round(2)

    if "agrees_with_platform" in tiers.columns:
        total_err = (~tiers["agrees_with_platform"]).sum()
        if total_err > 0:
            err_per_tier = (
                tiers.loc[~tiers["agrees_with_platform"], "tier"]
                .value_counts()
                .reindex(out.index)
                .fillna(0)
            )
            out["share_of_errors_pct"] = (err_per_tier / total_err * 100).round(2)

    return out.reindex([t for t in TIER_ORDER if t in out.index])
