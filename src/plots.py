# src/plots.py
"""
Shared plotting utilities for the crowdsourcing quality thesis.

Contains the colour palette, matplotlib style settings, and reusable
helper functions that appear across all three notebooks.
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker


# ── Colour palette (matches all three notebooks) ─────────────────────────────

C = dict(
    blue="#4A90D9", teal="#2BA89E", amber="#E5A832", coral="#E06350",
    purple="#7C6BC4", green="#5AAE61", pink="#D96BA0", gray="#8C8C8C",
    red="#D94F4F", slate="#5C6B7A", light="#F5F5F5",
)

PROJECT_C = {575: "#7C6BC4", 576: "#4A90D9", 577: "#2BA89E", 578: "#E5A832", 581: "#E06350"}
POOL_C = {0: C["blue"], 1: C["coral"], 3: C["teal"]}


def set_style():
    """
    Apply the thesis matplotlib style.
    Call once at the top of each notebook (after importing this module).
    """
    plt.rcParams.update({
        "figure.facecolor": "white",
        "axes.facecolor": "white",
        "axes.edgecolor": "#CCCCCC",
        "axes.spines.top": False,
        "axes.spines.right": False,
        "axes.grid": True,
        "grid.color": "#EEEEEE",
        "grid.linewidth": 0.6,
        "font.size": 11,
        "axes.titlesize": 12,
        "axes.titleweight": "600",
        "figure.dpi": 110,
    })


# ── Low-level helpers ─────────────────────────────────────────────────────────

def despine(ax):
    """Remove top and right spines from an axes object."""
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)


def bar_labels(ax, bars, fmt="{:,.0f}", horizontal=False, fontsize=9):
    """
    Add value labels to the end of each bar in a bar chart.

    Parameters
    ----------
    ax : matplotlib.axes.Axes
    bars : BarContainer
    fmt : str
        Format string for the label value.
    horizontal : bool
        True for horizontal bar charts (barh).
    fontsize : int
    """
    for bar in bars:
        if horizontal:
            w = bar.get_width()
            ax.text(
                w * 1.01, bar.get_y() + bar.get_height() / 2,
                fmt.format(w), va="center", fontsize=fontsize,
            )
        else:
            h = bar.get_height()
            ax.text(
                bar.get_x() + bar.get_width() / 2, h * 1.01,
                fmt.format(h), ha="center", fontsize=fontsize,
            )


# ── Reusable chart functions ──────────────────────────────────────────────────

def plot_accuracy_by_pool(df: pd.DataFrame, pool_map: dict, pool_c: dict):
    """
    Bar chart of mean worker accuracy by pool type.

    Parameters
    ----------
    df : pd.DataFrame
        Preprocessed DataFrame with 'pool_type', 'user_ans', 'task_ans', 'correct'.
    pool_map : dict
        Mapping of pool_type int to string label.
    pool_c : dict
        Mapping of pool_type int to colour hex.

    Returns
    -------
    fig, ax
    """
    pool_acc = {}
    for pt, nm in pool_map.items():
        s = df[(df["pool_type"] == pt) & df["user_ans"].notna() & df["task_ans"].notna()]
        if len(s) > 0:
            pool_acc[nm] = s["correct"].mean()

    fig, ax = plt.subplots(figsize=(6, 3.5))
    bars = ax.bar(pool_acc.keys(), pool_acc.values(), width=0.5,
                  color=[pool_c[k] for k in pool_map if pool_map[k] in pool_acc])
    for b, v in zip(bars, pool_acc.values()):
        ax.text(b.get_x() + b.get_width() / 2, v + 0.008,
                f"{v:.1%}", ha="center", fontsize=10, fontweight=600)
    ax.set_title("Accuracy by pool type")
    ax.set_ylabel("Accuracy")
    ax.set_ylim(0, 1.08)
    despine(ax)
    plt.tight_layout()
    return fig, ax


def plot_gold_accuracy_distribution(gold_acc: pd.Series, min_gold: int = 5):
    """
    Histogram of per-worker gold accuracy for workers with sufficient data.

    Parameters
    ----------
    gold_acc : pd.Series
        Gold accuracy per worker (output of scoring.compute_gold_accuracy).
    min_gold : int
        Workers with fewer gold tasks are excluded.

    Returns
    -------
    fig, ax
    """
    reliable = gold_acc.dropna()

    fig, ax = plt.subplots(figsize=(7, 3.5))
    ax.hist(reliable, bins=20, color=C["amber"], alpha=0.85,
            edgecolor="white", linewidth=0.3)
    ax.axvline(reliable.mean(), color="#333", ls="--", lw=0.8,
               label=f"mean = {reliable.mean():.2f}")
    ax.set_title(f"Gold accuracy distribution (workers with ≥ {min_gold} gold tasks)")
    ax.set_xlabel("Accuracy")
    ax.set_ylabel("Workers")
    ax.legend()
    despine(ax)
    plt.tight_layout()
    return fig, ax


def plot_cluster_scatter(
    X_2d: np.ndarray,
    labels: pd.Series,
    title: str = "Worker clusters",
):
    """
    2-D scatter plot of worker clusters (after PCA or UMAP reduction).

    Parameters
    ----------
    X_2d : np.ndarray
        Shape (n_workers, 2) — 2-D projection of the feature space.
    labels : pd.Series
        Cluster assignments (integer labels).
    title : str

    Returns
    -------
    fig, ax
    """
    unique_labels = sorted(labels.unique())
    palette = [C["blue"], C["coral"], C["teal"], C["amber"],
               C["purple"], C["green"]]

    fig, ax = plt.subplots(figsize=(7, 5))
    for i, cluster_id in enumerate(unique_labels):
        mask = labels.values == cluster_id
        ax.scatter(
            X_2d[mask, 0], X_2d[mask, 1],
            c=palette[i % len(palette)], s=15, alpha=0.6,
            edgecolors="none", label=f"Cluster {cluster_id}",
        )
    ax.legend(markerscale=2, fontsize=9)
    ax.set_title(title)
    ax.set_xlabel("Component 1")
    ax.set_ylabel("Component 2")
    despine(ax)
    plt.tight_layout()
    return fig, ax


def plot_speed_vs_accuracy(df: pd.DataFrame):
    """
    Line plot of accuracy by response-speed decile (regular pool).

    Reproduces the speed-vs-quality chart from EDA section 11.

    Returns
    -------
    fig, ax
    """
    scored = df[
        (df["pool_type"] == 0)
        & df["correct"].notna()
        & df["per_task_sec"].between(1, 300)
    ].copy()

    scored["speed_q"] = pd.qcut(scored["per_task_sec"], 10, duplicates="drop")
    sa = scored.groupby("speed_q", observed=True).agg(
        accuracy=("correct", "mean"),
        count=("correct", "count"),
        median_sec=("per_task_sec", "median"),
    )

    fig, ax = plt.subplots(figsize=(9, 4))
    ax.plot(sa["median_sec"], sa["accuracy"], color=C["purple"],
            lw=2.5, marker="o", ms=6, zorder=5)
    ax.fill_between(sa["median_sec"], sa["accuracy"],
                    sa["accuracy"].min() - 0.005,
                    alpha=0.07, color=C["purple"])
    ax.axhline(scored["correct"].mean(), color=C["gray"], ls="--", lw=0.8,
               label=f"overall mean = {scored['correct'].mean():.1%}")
    ax.set_title("Accuracy by response speed decile (regular pool)")
    ax.set_xlabel("Median seconds per task")
    ax.set_ylabel("Accuracy")
    ax.legend(fontsize=9)
    despine(ax)
    plt.tight_layout()
    return fig, ax
