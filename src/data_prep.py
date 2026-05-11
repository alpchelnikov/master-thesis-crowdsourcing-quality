# src/data_prep.py
"""
Data loading and preprocessing for the crowdsourcing quality thesis.

Centralises the loading and uniform pre-processing steps applied at the
top of every notebook, so that downstream code starts from an identical
clean DataFrame.  The exclusions applied here are the ones documented in
Section 2.4 of the thesis (service-flag removal, page/per-task timing
normalisation, correctness flag construction).

For richer per-worker feature engineering, see :mod:`src.features`,
which consumes the output of :func:`load_data`.
"""

from __future__ import annotations

import pandas as pd
import numpy as np


# ── Domain constants ─────────────────────────────────────────────────────────

POOL_MAP = {0: "Regular", 1: "Rehabilitation", 3: "Training"}
TASK_MAP = {0: "Regular", 1: "Gold (control)", 2: "Training"}

# Platform service flag — see Section 2.4 of the thesis. user_ans == 3 is
# emitted when a task page is malformed and is not a semantic annotation.
SERVICE_FLAG = 3

# Valid annotation classes after service-flag removal.
VALID_ANSWERS = (1, 2, 4)


def load_data(path: str, *, binary_only: bool = False) -> pd.DataFrame:
    """
    Load the crowdsourcing CSV and apply standard pre-processing.

    Steps applied (in order):

      1. parse the ``created_at`` and ``finished_at`` timestamps;
      2. compute page-level duration and the page-normalised per-task
         duration ``per_task_sec = page_duration_sec / tasks_per_page``;
      3. add the binary correctness flag ``correct`` where both
         ``user_ans`` and ``task_ans`` are non-missing — note that on
         regular tasks ``task_ans`` is the platform majority vote, see
         Section 2.3.5 of the thesis;
      4. map ``pool_type`` and ``task_type`` to readable label columns;
      5. drop rows with ``user_ans == 3`` (the service flag).

    Parameters
    ----------
    path : str
        Path to the CSV file (``data.csv`` or ``sample_data.csv``).
    binary_only : bool
        If True, restrict to ``user_ans ∈ {1, 2}`` and matching
        ``task_ans`` so the table is safe to pass directly to
        :class:`aggregation.DawidSkene` or :class:`aggregation.MACE` at
        ``n_classes=2``.

    Returns
    -------
    pd.DataFrame
        Pre-processed annotation table.  Each row is still one worker's
        answer on one task.
    """
    df = pd.read_csv(path, index_col=0)

    # Timestamps
    df["created_at"] = pd.to_datetime(df["created_at"], errors="coerce")
    df["finished_at"] = pd.to_datetime(df["finished_at"], errors="coerce")

    # Page-level duration and per-task approximation
    df["page_duration_sec"] = (
        (df["finished_at"] - df["created_at"]).dt.total_seconds()
    )
    df["per_task_sec"] = df["page_duration_sec"] / df["tasks_per_page"].replace(0, np.nan)

    # Correctness flag (NaN where labels are unknown). On gold tasks
    # this is the honest accuracy signal; on regular tasks it is
    # agreement with the platform majority and is used as a proxy only.
    df["correct"] = (df["user_ans"] == df["task_ans"]).where(
        df["user_ans"].notna() & df["task_ans"].notna()
    )

    # Readable labels
    df["pool_label"] = df["pool_type"].map(POOL_MAP)
    df["task_label"] = df["task_type"].map(TASK_MAP)

    # Exclude service-flag rows
    df = df[df["user_ans"] != SERVICE_FLAG].copy()

    if binary_only:
        df = df[df["user_ans"].isin([1, 2])]
        df = df[df["task_ans"].isna() | df["task_ans"].isin([1, 2])]

    return df


def get_gold_rows(df: pd.DataFrame) -> pd.DataFrame:
    """
    Return rows that are gold (control) tasks with a valid worker answer.

    Gold tasks have ``task_type == 1`` and carry an expert-verified
    label; they are the primary quality signal throughout the thesis.
    """
    return df[
        (df["task_type"] == 1)
        & df["user_ans"].notna()
        & df["task_ans"].notna()
    ].copy()


def get_regular_scored(df: pd.DataFrame) -> pd.DataFrame:
    """
    Return regular-pool, regular-task rows where correctness can be computed.

    On regular tasks ``task_ans`` is the platform's majority-vote
    aggregation, not an independent label.  Using ``correct`` on
    these rows as a quality criterion creates the circular-evaluation
    pathology described in Section 2.3.5 of the thesis; treat it as a
    feature, never as a target.
    """
    return df[
        (df["pool_type"] == 0)
        & (df["task_type"] == 0)
        & df["correct"].notna()
    ].copy()
