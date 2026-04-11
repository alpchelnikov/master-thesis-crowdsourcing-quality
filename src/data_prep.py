# src/data_prep.py
"""
Data loading and preprocessing for the crowdsourcing quality thesis.

Centralises all column derivations and exclusion rules so every notebook
starts from an identical, clean DataFrame.
"""

import pandas as pd
import numpy as np


# ── Domain constants ──────────────────────────────────────────────────────────

POOL_MAP = {0: "Regular", 1: "Rehabilitation", 3: "Training"}
TASK_MAP = {0: "Regular", 1: "Gold (control)", 2: "Training"}

# user_ans == 3 is a platform service flag for malformed pages — not a valid class.
SERVICE_FLAG = 3


def load_data(path: str) -> pd.DataFrame:
    """
    Load the raw crowdsourcing CSV and apply all standard preprocessing steps:
      - parse timestamps
      - compute page-level duration and per-task duration
      - add correctness flag (only where task_ans is known)
      - add human-readable label columns
      - exclude service-flag answers (user_ans == 3)

    Parameters
    ----------
    path : str
        Path to the CSV file (e.g. "data.csv" or "sample_data.csv").

    Returns
    -------
    pd.DataFrame
        Preprocessed DataFrame ready for analysis.

    Notes
    -----
    Each row is one worker's answer to one task.
    Timing is recorded at page level; per-task time = page_duration / tasks_per_page.
    """
    df = pd.read_csv(path, index_col=0)

    # Timestamps
    df["created_at"] = pd.to_datetime(df["created_at"])
    df["finished_at"] = pd.to_datetime(df["finished_at"])

    # Page-level duration and per-task approximation
    df["page_duration_sec"] = (
        (df["finished_at"] - df["created_at"]).dt.total_seconds()
    )
    df["per_task_sec"] = df["page_duration_sec"] / df["tasks_per_page"]

    # Correctness flag (NaN where task_ans is unknown)
    df["correct"] = (df["user_ans"] == df["task_ans"]).where(
        df["user_ans"].notna() & df["task_ans"].notna()
    )

    # Readable labels
    df["pool_label"] = df["pool_type"].map(POOL_MAP)
    df["task_label"] = df["task_type"].map(TASK_MAP)

    # Exclude service-flag rows
    df = df[df["user_ans"] != SERVICE_FLAG].copy()

    return df


def get_gold_rows(df: pd.DataFrame) -> pd.DataFrame:
    """
    Return rows that are gold (control) tasks with a valid worker answer.

    Gold tasks have task_type == 1 and provide independent ground truth.
    These are used as the primary quality signal throughout the thesis.
    """
    return df[
        (df["task_type"] == 1)
        & df["user_ans"].notna()
        & df["task_ans"].notna()
    ].copy()


def get_regular_scored(df: pd.DataFrame) -> pd.DataFrame:
    """
    Return regular-pool, regular-task rows where correctness can be computed.

    Note: task_ans on regular tasks reflects the platform's majority-vote
    aggregation, not an independent ground truth. Use with caution when
    evaluating aggregation models (circular comparison risk).
    """
    return df[
        (df["pool_type"] == 0)
        & (df["task_type"] == 0)
        & df["correct"].notna()
    ].copy()
