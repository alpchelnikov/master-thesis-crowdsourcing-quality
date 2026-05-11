# src/__init__.py
"""
Shared utilities for the crowdsourcing quality thesis.

Re-exports the most commonly used names so notebooks can write
``from src import build_worker_features, DawidSkene`` instead of
remembering the sub-module layout.

Modules:
    data_prep    – CSV loading and uniform pre-processing
    features     – per-worker feature engineering + temporal split
    aggregation  – majority vote, Dawid–Skene, MACE
    scoring      – ridge-based composite score, task-level confidence tiers
    segmentation – k-means / GMM / agglomerative clustering with robustness
    plots        – shared matplotlib helpers and the thesis colour palette
"""

from .data_prep import (
    POOL_MAP, TASK_MAP, SERVICE_FLAG, VALID_ANSWERS,
    load_data, get_gold_rows, get_regular_scored,
)
from .features import (
    FEATURE_GROUPS, FULL_FEATURES, MODELLING_FEATURES,
    prepare_base, add_worker_split,
    build_worker_features, build_temporal_split,
)
from .aggregation import (
    majority_vote, DawidSkene, MACE,
)
from .scoring import (
    compute_gold_accuracy, compute_composite_score,
    CompositeScoreResult,
    assign_confidence_tiers, tier_summary,
    DEFAULT_TIER_THRESHOLDS, TIER_ORDER,
)
from .segmentation import (
    prepare_cluster_features,
    run_clustering, compare_clusterings,
    segmentation_robustness, RobustnessReport,
    name_segments, describe_clusters,
    DEFAULT_CLUSTER_FEATURES, DEFAULT_BEHAVIOUR_FEATURES, SEGMENT_NAMES,
)

__all__ = [
    # data_prep
    "POOL_MAP", "TASK_MAP", "SERVICE_FLAG", "VALID_ANSWERS",
    "load_data", "get_gold_rows", "get_regular_scored",
    # features
    "FEATURE_GROUPS", "FULL_FEATURES", "MODELLING_FEATURES",
    "prepare_base", "add_worker_split",
    "build_worker_features", "build_temporal_split",
    # aggregation
    "majority_vote", "DawidSkene", "MACE",
    # scoring
    "compute_gold_accuracy", "compute_composite_score",
    "CompositeScoreResult",
    "assign_confidence_tiers", "tier_summary",
    "DEFAULT_TIER_THRESHOLDS", "TIER_ORDER",
    # segmentation
    "prepare_cluster_features",
    "run_clustering", "compare_clusterings",
    "segmentation_robustness", "RobustnessReport",
    "name_segments", "describe_clusters",
    "DEFAULT_CLUSTER_FEATURES", "DEFAULT_BEHAVIOUR_FEATURES", "SEGMENT_NAMES",
]
