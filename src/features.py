# src/features.py
"""
Worker-level feature engineering for the crowdsourcing quality thesis.

The module defines a single function :func:`build_worker_features` that
takes the pre-processed annotation table and returns one row per worker
with the full inventory of behavioural and accuracy features.  The
inventory is split into five groups (activity, gold history,
regular-task proxy, speed, behaviour) and matches the
``worker_features.csv`` artefact produced by
``notebooks/02_Feature_Engineering_and_Baseline.ipynb``.

Two named subsets are also exposed for downstream use:

  * :data:`FULL_FEATURES`       – every column produced by
    ``build_worker_features``, suitable for serialisation.
  * :data:`MODELLING_FEATURES`  – the compact 16-feature subset selected
    for the supervised baselines in Chapter 3 (Section 3.2).  This is the
    *modelling* representation referenced by Table 2.2 of the thesis.

Two temporal-split helpers — :func:`add_worker_split` and
:func:`build_temporal_split` — implement the per-worker 80/20
chronological split used to construct features and targets without
leakage (Section 2.7).

Implementation is shared with notebook 02 so the two stay in sync.
"""

from __future__ import annotations

from typing import Optional, Tuple

import numpy as np
import pandas as pd


# ── Feature groups (mirrors Section 2.6 of the thesis) ───────────────────────
#
#   1. Activity              – scale of platform engagement
#   2. Gold history          – correctness on control tasks (the honest signal)
#   3. Regular-task proxy    – cheap noisy quality signal from regular tasks
#   4. Speed                 – per-task timing distribution
#   5. Behaviour             – answer-distribution shape and project breadth
#
# Each group below lists the columns ``build_worker_features`` populates.

FEATURE_GROUPS = {
    "activity": [
        "n_rows", "n_answers", "n_tasks", "n_pages",
        "n_projects", "n_pools",
        "active_days", "span_days",
        "skip_rate", "tasks_per_page_mean",
        "overlap_mean", "overlap_std",
        "price_mean",
        "share_gold_tasks", "share_rehab_pools",
    ],
    "gold_history": [
        "n_gold", "gold_acc", "gold_acc_std",
        "gold_longest_success_streak", "gold_longest_error_streak",
        "gold_recent5_acc", "gold_recent10_acc", "gold_first5_acc",
        "gold_class_gap", "gold_learning_delta",
    ],
    "regular_proxy": [
        "n_regular",
        "reg_agreement_proxy", "reg_agreement_std",
        "regular_answer_entropy", "regular_answer_mode_share",
    ],
    "speed": [
        "per_task_sec_median", "per_task_sec_mean", "per_task_sec_std",
        "fast_task_share", "slow_task_share",
        "mean_hour", "weekend_share",
    ],
    "behaviour": [
        "answer_entropy", "answer_mode_share", "main_project_share",
        "pct_label_1",
    ],
    "derived": [
        "answers_per_active_day", "answers_per_span_day",
        "gold_share_among_answers",
    ],
}

# Full inventory – every column the function produces (≈ 45 columns).
FULL_FEATURES = [c for cols in FEATURE_GROUPS.values() for c in cols]

# Compact modelling subset used in Chapter 3 (Section 3.2).
# This is the 16-feature representation Table 2.2 of the thesis describes;
# selecting from the full inventory keeps the two consistent without
# requiring the supervised stage to carry ~45 mostly-collinear columns.
MODELLING_FEATURES = [
    # activity
    "n_answers", "n_projects", "active_days", "skip_rate",
    # gold history
    "n_gold", "gold_acc",
    "gold_recent5_acc", "gold_recent10_acc", "gold_learning_delta",
    # regular-task proxy
    "reg_agreement_proxy",
    # speed
    "per_task_sec_median", "per_task_sec_std",
    # behaviour
    "answer_entropy", "answer_mode_share", "main_project_share",
    # activity (continued)
    "overlap_mean",
]


# ── Small statistical helpers (vectorised over a single worker's answers) ───

def _shannon_entropy(values) -> float:
    """Shannon entropy of a value distribution; NaN if no observations."""
    s = pd.Series(values).dropna()
    if s.empty:
        return np.nan
    p = s.value_counts(normalize=True)
    return float(-(p * np.log2(p)).sum())


def _top_share(values) -> float:
    """Share of the most frequent value; NaN if no observations."""
    s = pd.Series(values).dropna()
    if s.empty:
        return np.nan
    return float(s.value_counts(normalize=True).iloc[0])


def _longest_run(series, value) -> int:
    """Longest contiguous run of ``value`` in ``series``."""
    best = 0
    cur = 0
    for x in pd.Series(series).fillna(-1):
        if x == value:
            cur += 1
            best = max(best, cur)
        else:
            cur = 0
    return best


# ── Base-frame preparation ───────────────────────────────────────────────────

def prepare_base(df: pd.DataFrame) -> pd.DataFrame:
    """
    Add the derived columns that the feature builder relies on.

    This is the exact ``prepare_base`` from notebook 02: it parses the
    timestamps, drops the ``user_ans == 3`` service-flag rows, adds
    ``per_task_sec``, the task-type and pool-type indicator columns,
    the page-level ``agreement_proxy`` and ``gold_correct`` flags, and a
    few useful hour-of-day / day-of-week derivatives.

    Parameters
    ----------
    df : pd.DataFrame
        Raw annotation table loaded from ``data.csv`` /
        ``sample_data.csv``.

    Returns
    -------
    pd.DataFrame
        Annotation table with derived columns ready for
        ``build_worker_features``.
    """
    out = df.copy()

    out["created_at"] = pd.to_datetime(out["created_at"], errors="coerce")
    out["finished_at"] = pd.to_datetime(out["finished_at"], errors="coerce")

    # Service-flag exclusion (Section 2.4)
    out["is_error_report"] = out["user_ans"].eq(3)
    out = out[~out["is_error_report"]].copy()

    out["answered"] = out["user_ans"].notna()
    out["duration_sec"] = (out["finished_at"] - out["created_at"]).dt.total_seconds()
    out["per_task_sec"] = out["duration_sec"] / out["tasks_per_page"].replace(0, np.nan)

    out["is_gold_task"] = out["task_type"].eq(1)
    out["is_training_task"] = out["task_type"].eq(2)
    out["is_regular_task"] = out["task_type"].eq(0)

    out["is_regular_pool"] = out["pool_type"].eq(0)
    out["is_rehab_pool"] = out["pool_type"].eq(1)
    out["is_training_pool"] = out["pool_type"].eq(3)

    out["valid_target_label"] = out["task_ans"].notna() & out["task_ans"].ne(3)

    out["agreement_proxy"] = np.where(
        out["answered"] & out["valid_target_label"],
        (out["user_ans"] == out["task_ans"]).astype(float),
        np.nan,
    )
    out["gold_correct"] = np.where(
        out["is_gold_task"] & out["answered"] & out["valid_target_label"],
        (out["user_ans"] == out["task_ans"]).astype(float),
        np.nan,
    )

    out["hour"] = out["created_at"].dt.hour
    out["dayofweek"] = out["created_at"].dt.dayofweek
    out["is_weekend"] = out["dayofweek"].isin([5, 6]).astype(int)

    return out


# ── Temporal split (per-worker 80/20) ────────────────────────────────────────

def add_worker_split(df: pd.DataFrame, history_share: float = 0.80) -> pd.DataFrame:
    """
    Add ``is_history`` indicating whether each row falls in the first
    ``history_share`` of a worker's chronologically ordered timeline.

    Per-worker rather than global splitting is used because the
    observation window is short (~1 month) and a global calendar split
    would systematically under-sample features for workers who joined
    late.  See Section 2.7 for the full justification.

    The split is on the row-index *within* each worker, not on calendar
    time: each worker's first 80% of answers form their feature window.

    Parameters
    ----------
    df : pd.DataFrame
        Output of :func:`prepare_base`.
    history_share : float
        Fraction of each worker's timeline used as the feature window.

    Returns
    -------
    pd.DataFrame
        Sorted copy of ``df`` with three added columns:
        ``worker_row`` (0-indexed position in the worker's timeline),
        ``worker_n``   (total rows for this worker),
        ``is_history`` (True for feature-window rows).
    """
    out = df.sort_values(["ozon_id", "created_at", "task_id"]).copy()
    out["worker_row"] = out.groupby("ozon_id").cumcount()
    out["worker_n"] = out.groupby("ozon_id")["task_id"].transform("size")
    split_point = np.maximum(
        1, np.floor(out["worker_n"] * history_share)
    ).astype(int)
    out["is_history"] = out["worker_row"] < split_point
    return out


# ── Main feature builder ─────────────────────────────────────────────────────

def build_worker_features(frame: pd.DataFrame) -> pd.DataFrame:
    """
    Aggregate the annotation table to one row per worker.

    The function consumes a DataFrame already passed through
    :func:`prepare_base` and produces the full ~45-column inventory
    listed in :data:`FULL_FEATURES`.  Workers absent from a particular
    sub-population (no gold tasks, no regular tasks, etc.) receive NaN
    on the dependent columns; downstream code is responsible for either
    imputing or excluding those rows.

    Always call this on the *feature window* subset of the data — never
    on the full table — to avoid leakage into the prediction target.
    See :func:`build_temporal_split` for the standard usage pattern.

    Parameters
    ----------
    frame : pd.DataFrame
        Output of :func:`prepare_base`.

    Returns
    -------
    pd.DataFrame
        One row per worker (``ozon_id``).  Shape ≈ (n_workers, 45).
    """
    frame = frame.sort_values(["ozon_id", "created_at", "task_id"]).copy()

    # ── 1. Activity ──────────────────────────────────────────────────────
    activity = frame.groupby("ozon_id").agg(
        n_rows=("task_id", "size"),
        n_answers=("answered", "sum"),
        n_tasks=("task_id", "nunique"),
        n_pages=("page_id", "nunique"),
        n_projects=("project_id", "nunique"),
        n_pools=("pool_id", "nunique"),
        active_days=("created_at", lambda s: s.dt.date.nunique()),
        span_days=("created_at", lambda s: max((s.max() - s.min()).days + 1, 1)),
        skip_rate=("skipped", "mean"),
        tasks_per_page_mean=("tasks_per_page", "mean"),
        overlap_mean=("overlap", "mean"),
        overlap_std=("overlap", "std"),
        price_mean=("price", "mean"),
        share_gold_tasks=("is_gold_task", "mean"),
        share_rehab_pools=("is_rehab_pool", "mean"),
    )

    # ── 2. Gold history ─────────────────────────────────────────────────
    gold = frame[
        frame["is_gold_task"] & frame["answered"] & frame["valid_target_label"]
    ].copy()

    if len(gold):
        gold["gold_order"] = gold.groupby("ozon_id").cumcount()
        gold["gold_n"] = gold.groupby("ozon_id")["task_id"].transform("size")
        gold["from_end"] = gold["gold_n"] - gold["gold_order"]

        gold_summary = gold.groupby("ozon_id").agg(
            n_gold=("task_id", "size"),
            gold_acc=("gold_correct", "mean"),
            gold_acc_std=("gold_correct", "std"),
            gold_longest_success_streak=(
                "gold_correct", lambda s: _longest_run(s, 1.0)
            ),
            gold_longest_error_streak=(
                "gold_correct", lambda s: _longest_run(s, 0.0)
            ),
        )

        gold_recent5 = (
            gold[gold["from_end"] <= 5]
            .groupby("ozon_id")["gold_correct"].mean()
            .rename("gold_recent5_acc")
        )
        gold_recent10 = (
            gold[gold["from_end"] <= 10]
            .groupby("ozon_id")["gold_correct"].mean()
            .rename("gold_recent10_acc")
        )
        gold_first5 = (
            gold[gold["gold_order"] < 5]
            .groupby("ozon_id")["gold_correct"].mean()
            .rename("gold_first5_acc")
        )

        # Per-class accuracy gap (max class acc − min class acc), which
        # flags workers who systematically miss one of the labels even
        # while looking fine on the diagonal average.
        class_gap = gold[gold["user_ans"].isin([1, 2, 4])].pivot_table(
            index="ozon_id", columns="user_ans",
            values="gold_correct", aggfunc="mean",
        )
        if class_gap.shape[1] >= 2:
            class_gap["gold_class_gap"] = (
                class_gap.max(axis=1) - class_gap.min(axis=1)
            )
            class_gap = class_gap[["gold_class_gap"]]
        else:
            class_gap = pd.DataFrame(index=gold_summary.index)

        gold_features = pd.concat(
            [gold_summary, gold_recent5, gold_recent10, gold_first5, class_gap],
            axis=1,
        )
        gold_features["gold_learning_delta"] = (
            gold_features["gold_recent10_acc"] - gold_features["gold_first5_acc"]
        )
    else:
        gold_features = pd.DataFrame(index=activity.index)

    # ── 3. Regular-task proxy ───────────────────────────────────────────
    regular = frame[
        frame["is_regular_pool"]
        & frame["is_regular_task"]
        & frame["answered"]
        & frame["valid_target_label"]
    ].copy()

    if len(regular):
        regular_features = regular.groupby("ozon_id").agg(
            n_regular=("task_id", "size"),
            reg_agreement_proxy=("agreement_proxy", "mean"),
            reg_agreement_std=("agreement_proxy", "std"),
            regular_answer_entropy=("user_ans", _shannon_entropy),
            regular_answer_mode_share=("user_ans", _top_share),
        )
    else:
        regular_features = pd.DataFrame(index=activity.index)

    # ── 4. Speed ────────────────────────────────────────────────────────
    timed = frame[frame["per_task_sec"].between(1, 600)].copy()
    if len(timed):
        speed = timed.groupby("ozon_id").agg(
            per_task_sec_median=("per_task_sec", "median"),
            per_task_sec_mean=("per_task_sec", "mean"),
            per_task_sec_std=("per_task_sec", "std"),
            fast_task_share=("per_task_sec", lambda s: (s < 5).mean()),
            slow_task_share=("per_task_sec", lambda s: (s > 30).mean()),
            mean_hour=("hour", "mean"),
            weekend_share=("is_weekend", "mean"),
        )
    else:
        speed = pd.DataFrame(index=activity.index)

    # ── 5. Behaviour ────────────────────────────────────────────────────
    answered = frame[frame["answered"]].copy()
    if len(answered):
        behaviour = answered.groupby("ozon_id").agg(
            answer_entropy=("user_ans", _shannon_entropy),
            answer_mode_share=("user_ans", _top_share),
            main_project_share=("project_id", _top_share),
            pct_label_1=("user_ans", lambda s: (s == 1).mean()),
        )
    else:
        behaviour = pd.DataFrame(index=activity.index)

    feat = (
        activity
        .join([gold_features, regular_features, speed, behaviour], how="left")
        .reset_index()
    )

    # ── 6. Derived rates ─────────────────────────────────────────────────
    feat["answers_per_active_day"] = (
        feat["n_answers"] / feat["active_days"].replace(0, np.nan)
    )
    feat["answers_per_span_day"] = (
        feat["n_answers"] / feat["span_days"].replace(0, np.nan)
    )
    feat["gold_share_among_answers"] = (
        feat["n_gold"] / feat["n_answers"].replace(0, np.nan)
    )

    return feat


# ── Convenience wrapper: build features + targets in one call ────────────────

def build_temporal_split(
    df: pd.DataFrame,
    *,
    history_share: float = 0.80,
    min_future_gold: int = 5,
    high_quality_threshold: float = 0.85,
) -> Tuple[pd.DataFrame, pd.Series, pd.Series]:
    """
    End-to-end feature/target construction with no leakage.

    The function:

      1. runs :func:`prepare_base` and :func:`add_worker_split` on the
         raw annotation table;
      2. builds worker features on the per-worker history window only
         (the first ``history_share`` of each worker's answers);
      3. computes the supervised target ``future_gold_acc`` on the
         holdout window, restricted to gold tasks with a valid label;
      4. additionally exposes the binary target ``future_high_quality``
         (Section 2.7) that combines a high-accuracy threshold with a
         minimum number of future gold observations.

    Parameters
    ----------
    df : pd.DataFrame
        Raw annotation table (output of ``pd.read_csv`` on the
        crowdsourcing CSV — *not* yet processed).
    history_share : float
        Fraction of each worker's timeline used as the feature window.
    min_future_gold : int
        Minimum future-window gold observations a worker must have for
        their target to be considered defined.
    high_quality_threshold : float
        Future gold accuracy threshold for the binary target.

    Returns
    -------
    X : pd.DataFrame
        Worker feature matrix indexed by ``ozon_id``.  Workers absent
        from either window are excluded.
    y_reg : pd.Series
        Continuous target ``future_gold_acc`` indexed by ``ozon_id``.
    y_cls : pd.Series
        Binary target ``future_high_quality`` indexed by ``ozon_id``.
    """
    base = prepare_base(df)
    base = add_worker_split(base, history_share=history_share)

    history_df = base[base["is_history"]]
    holdout_df = base[~base["is_history"]]

    X = build_worker_features(history_df).set_index("ozon_id")

    holdout_gold = holdout_df[
        holdout_df["is_gold_task"]
        & holdout_df["answered"]
        & holdout_df["valid_target_label"]
    ]
    targets = holdout_gold.groupby("ozon_id").agg(
        future_n_gold=("task_id", "size"),
        future_gold_acc=("gold_correct", "mean"),
    )
    targets["future_high_quality"] = (
        (targets["future_gold_acc"] >= high_quality_threshold)
        & (targets["future_n_gold"] >= min_future_gold)
    ).astype(int)

    # Workers must appear in both windows
    common = X.index.intersection(targets.index)
    X = X.loc[common]
    y_reg = targets.loc[common, "future_gold_acc"].rename("future_gold_acc")
    y_cls = targets.loc[common, "future_high_quality"].rename("future_high_quality")

    return X, y_reg, y_cls
