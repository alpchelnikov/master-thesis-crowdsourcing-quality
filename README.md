# figures/

This folder contains **exported visualisations** from all three notebooks.
PNG files are excluded from git (see `.gitignore`) and should be regenerated
by running the notebooks on the full dataset.

See [`docs/reproducibility.md`](../docs/reproducibility.md) — each notebook step
lists which figures to export and the exact `fig.savefig()` call to use.

---

## Expected files

### From `01_EDA.ipynb`

| File | Section | Description |
|---|---|---|
| `eda_accuracy_by_pool.png` | §8 | Bar chart: mean accuracy per pool type |
| `eda_accuracy_by_project.png` | §8 | Bar chart: accuracy per project (regular pool) |
| `eda_gold_accuracy_distribution.png` | §8.3 | Histogram: gold accuracy per worker |
| `eda_worker_activity_distribution.png` | §6 | Log-scale histogram + Pareto curve |
| `eda_response_time_by_pool.png` | §7 | Three-panel per-task timing histogram |
| `eda_speed_vs_quality.png` | §11 | Line chart: accuracy by speed decile |
| `eda_feature_correlation_heatmap.png` | §15.1 | Correlation matrix (13 × 13) |
| `eda_rehab_before_after.png` | §17 | Paired histogram + bucket chart |
| `eda_confusion_matrix.png` | §8.2 | Confusion matrix (absolute + row-normalised) |
| `eda_class_balance_by_project.png` | §8.1 | Class distribution per project |

### From `02_Feature_Engineering_Baseline.ipynb`

| File | Description |
|---|---|
| `baseline_roc_curves.png` | ROC curves for all three baseline models |
| `feature_importance_rf.png` | Random Forest feature importances |
| `baseline_metrics_comparison.png` | Bar chart comparing AUC / F1 across baselines |

### From `03_Advanced_Models.ipynb`

| File | Description |
|---|---|
| `aggregation_comparison.png` | Grouped bar: Dawid-Skene vs MACE vs Majority Vote |
| `worker_score_distribution.png` | Histogram of composite quality scores |
| `cluster_scatter.png` | 2-D scatter (PCA/UMAP) coloured by cluster |
| `cluster_profiles.png` | Radar or bar chart of mean features per cluster |
| `adaptive_overlap_simulation.png` | Cost vs accuracy trade-off curve |

---

## Export snippet

```python
# At the end of any plot cell:
fig.savefig("figures/eda_accuracy_by_pool.png", dpi=150, bbox_inches="tight")
```

Recommended settings: `dpi=150`, `bbox_inches="tight"`, white background (set in `src/plots.set_style()`).
