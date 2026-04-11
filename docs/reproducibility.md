# Reproducibility Guide

This document describes how to fully reproduce the results of this thesis from scratch.

---

## Requirements

| Requirement | Version |
|---|---|
| Python | 3.10+ |
| pandas | ≥ 2.0 |
| numpy | ≥ 1.24 |
| scikit-learn | ≥ 1.3 |
| matplotlib | ≥ 3.7 |
| scipy | ≥ 1.11 |
| jupyter | ≥ 1.0 |

Install all dependencies:

```bash
pip install -r requirements.txt
```

---

## Data access

The raw data is **not included** in this repository (proprietary platform data).

To reproduce results you need:

- A CSV file named `data.csv` (full dataset, ~4.12 M rows) **or**
- `sample_data.csv` (100 000-row sample, included) for development / review

Place the data file in the **project root** (same directory as the notebooks folder),
or update the `path` argument in `src/data_prep.load_data()`.

See [`data/README.md`](../data/README.md) for the full schema.

---

## Reproduction pipeline

Run the notebooks **in order**. Each notebook reads from the previous stage's output.

### Step 1 — Exploratory Data Analysis

```bash
jupyter notebook notebooks/01_EDA.ipynb
```

**Inputs:** `data.csv` (or `sample_data.csv`)  
**Outputs (to export manually):**
- `figures/eda_accuracy_by_pool.png`
- `figures/eda_gold_accuracy_distribution.png`
- `figures/eda_speed_vs_quality.png`
- `figures/eda_rehab_before_after.png`
- `figures/eda_feature_correlation_heatmap.png`
- `results/eda_key_findings.md` ← copy the Section 18 summary table

### Step 2 — Feature Engineering & Baseline Models

```bash
jupyter notebook notebooks/02_Feature_Engineering_Baseline.ipynb
```

**Inputs:** `data.csv`  
**Outputs (to export manually):**
- `artifacts/worker_features.csv` — full 21-column feature table
- `artifacts/train_test_split_info.txt` — temporal window parameters
- `results/baseline_metrics.csv` — model comparison table
- `figures/baseline_roc_curves.png`
- `figures/feature_importance_rf.png`

### Step 3 — Advanced Models & Segmentation

```bash
jupyter notebook notebooks/03_Advanced_Models.ipynb
```

**Inputs:** `data.csv`, optionally `artifacts/worker_features.csv`  
**Outputs (to export manually):**
- `artifacts/worker_scores.csv` — composite quality scores per worker
- `artifacts/worker_clusters.csv` — cluster assignments (K=4)
- `artifacts/ds_worker_quality.csv` — Dawid-Skene per-worker quality scores
- `artifacts/mace_worker_competence.csv` — MACE competence scores
- `results/aggregation_comparison.csv` — Dawid-Skene vs MACE vs Majority Vote
- `results/cluster_profiles.csv` — mean features per cluster
- `figures/cluster_scatter.png`
- `figures/worker_score_distribution.png`
- `figures/adaptive_overlap_simulation.png`

---

## Naming convention for artifacts

All exported files follow this pattern:

```
{stage}_{description}.{ext}
```

| Prefix | Notebook |
|---|---|
| `eda_` | 01_EDA |
| `baseline_` | 02_Feature_Engineering_Baseline |
| (none / descriptive) | 03_Advanced_Models |

Examples:
- `eda_gold_accuracy_distribution.png`
- `baseline_roc_curves.png`
- `worker_clusters.csv`

---

## How to export figures from notebooks

At the end of any plot cell, add:

```python
fig.savefig("figures/eda_accuracy_by_pool.png", dpi=150, bbox_inches="tight")
```

Or use the helper in `src/plots.py`:

```python
from src.plots import set_style
set_style()
# ... your plot code ...
fig.savefig("figures/my_plot.png", dpi=150, bbox_inches="tight")
```

---

## How to export tables from notebooks

For CSV artifacts:

```python
worker_features.to_csv("artifacts/worker_features.csv", index=True)
```

For markdown result tables:

```python
print(results_df.to_markdown(index=False))
# copy output into results/aggregation_comparison.md
```

---

## Random seeds

All stochastic steps use `random_state=42` by default (set in `src/segmentation.py`
and model calls in notebooks). Change this parameter if you want to test stability
across seeds.

---

## Known platform differences

Running on the 100 K **sample** vs the full 4.12 M dataset produces slightly
different numbers in some analyses (most notably the rehabilitation section,
where the sample showed a small *negative* effect while the full dataset shows
+1.24 pp). Where the two differ materially, the notebook text notes this explicitly.

All figures and results in `figures/` and `results/` should be regenerated
from the **full dataset** for the final thesis submission.
