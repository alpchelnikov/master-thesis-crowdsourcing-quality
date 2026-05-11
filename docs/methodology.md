# Methodology

**Thesis:** Evaluation of Performer Quality and Segmentation of Results in Crowdsourcing Systems
**Author:** Aleksandr Pchelnikov · HSE FCS MNAD, 2024–2026

---

## Overview

The study analyses a large-scale industrial crowdsourcing dataset with three goals:
(1) **characterising worker quality** against gold-task ground truth,
(2) **predicting future worker quality** under a leakage-free temporal split,
(3) **segmenting workers** into operationally interpretable groups and pricing the cost–quality trade-off of an adaptive overlap policy.

The pipeline executes in three notebooks; the shared algorithmic core lives in `src/` and is imported by each notebook.

```
01_EDA  →  02_Feature_Engineering_and_Baseline  →  03_Advanced_Models
```

---

## Stage 1 — Exploratory Data Analysis (`01_EDA.ipynb`)

### Data

Each row in the dataset is one worker's answer to one task. The hierarchy is:
**answer → task → page → pool → project**.

| Pool type | Role | Tasks inside |
|---|---|---|
| `pool_type = 0` | Regular (paid) production | Regular + gold tasks |
| `pool_type = 1` | Rehabilitation — triggered after ≥ 2 errors in last 10 gold | Gold tasks only |
| `pool_type = 3` | Training — onboarding before regular pools | Training tasks only |

### Ground truth and correctness

Gold tasks (`task_type = 1`) carry an expert-verified label (`task_ans`) that is independent of worker votes. They are the primary quality signal throughout this study.

**`task_ans` caveat (Section 2.3.5 of the thesis).** On regular tasks `task_ans` is the platform's own majority-vote aggregation. Comparing a new aggregation model against `task_ans` on regular tasks creates a **circular evaluation** — majority vote agrees with it by construction. All model comparisons use gold-task ground truth or held-out workers as the external criterion; the regular-task comparison is reported only with the circularity caveat made explicit.

### Exclusions

* `user_ans = 3` (service flag for malformed pages) is dropped before any correctness computation.
* `user_ans = 4` is dropped when fitting binary-class aggregation models (Section 2.4).

### Timing normalisation

Response time is recorded at **page level**. Per-task time is approximated as

```
per_task_sec = page_duration_sec / tasks_per_page
```

Page sizes differ by pool type (1–3 tasks / page); this normalisation is required to make durations comparable across pool types.

### Key EDA findings (full list in `results/eda_key_findings.md`)

| # | Finding | Value |
|---|---|---|
| 1 | Dataset size | 4.12 M answers · 14 452 workers · 5 projects |
| 2 | Regular-pool gold accuracy | ~89 % |
| 3 | Rehabilitation-pool accuracy | ~82 % |
| 4 | Training-pool accuracy | ~66 % |
| 5 | Unanimous tasks | 70.7 % |
| 6 | One-shot workers | 11.4 % |
| 7 | Power workers (≥ 100 answers) | 7.5 % |
| 8 | Rehabilitation effect | +1.24 pp (Wilcoxon p = 0.002) |
| 9 | Skippers vs. non-skippers | −3.5 pp accuracy gap |
| 10 | Speed–accuracy relationship | 87.7 – 91.0 % across deciles (confounded by task difficulty) |

---

## Stage 2 — Feature Engineering & Supervised Baselines (`02_Feature_Engineering_and_Baseline.ipynb`)

### Worker feature inventory

The `build_worker_features` function (in `src/features.py`) produces a full **~45-column** inventory grouped into five categories. A **compact 16-feature modelling subset** (`MODELLING_FEATURES` in the same module) is selected for the supervised stage to keep the linear models well-conditioned.

| Group | Sample columns (16-subset shown in **bold**) |
|---|---|
| Activity | **n_answers**, **n_projects**, n_pools, **active_days**, span_days, **skip_rate**, tasks_per_page_mean, **overlap_mean**, overlap_std, price_mean, share_gold_tasks, share_rehab_pools |
| Gold history | **n_gold**, **gold_acc**, gold_acc_std, gold_longest_success_streak, gold_longest_error_streak, **gold_recent5_acc**, **gold_recent10_acc**, gold_first5_acc, gold_class_gap, **gold_learning_delta** |
| Regular-task proxy | n_regular, **reg_agreement_proxy**, reg_agreement_std, regular_answer_entropy, regular_answer_mode_share |
| Speed | **per_task_sec_median**, per_task_sec_mean, **per_task_sec_std**, fast_task_share, slow_task_share, mean_hour, weekend_share |
| Behaviour | **answer_entropy**, **answer_mode_share**, **main_project_share**, pct_label_1 |
| Derived rates | answers_per_active_day, answers_per_span_day, gold_share_among_answers |

### Temporal split design (no leakage)

For each worker, the chronologically ordered answer list is divided into a **history window** (first 80 % of their answers) and a **holdout window** (last 20 %). All features are constructed from the history window only; both targets are defined on the holdout window only. Workers absent from either window are dropped.

```
each worker's timeline:
|------------- history window (80%) --------------|--- holdout (20%) ---|
       └─ features built here                                  └─ targets measured here
```

The per-worker split (rather than a single global calendar split) avoids systematically under-sampling features for workers who joined late in the ~1-month observation window. Calendar-level seasonal effects are negligible over this horizon.

### Target variables

* **Regression target**: `future_gold_acc` — worker's mean gold-task correctness in the holdout window.
* **Classification target**: `future_high_quality` = 1 if `future_gold_acc ≥ 0.85` AND `future_n_gold ≥ 5`, else 0. The threshold is calibrated to match the platform's operational definition of acceptable gold performance.

Both targets are reported in parallel; the continuous target acts as a robustness check on the binary one.

### Classification models

Five models are compared under 5-fold **stratified** cross-validation:

| # | Model | Notes |
|---|---|---|
| 1 | Logistic regression on `gold_acc` only | Single-feature sanity baseline |
| 2 | Logistic regression on `reg_agreement_proxy` only | Cheap-proxy sanity baseline |
| 3 | Logistic regression — all features | L2, balanced class weights |
| 4 | Random forest | 400 trees, depth 6, leaf 3, balanced |
| 5 | LightGBM | 400 estimators, lr = 0.05, 31 leaves, balanced |

The LightGBM baseline is additionally tuned via `GridSearchCV` over

```
n_estimators ∈ {200, 400}
learning_rate ∈ {0.03, 0.05, 0.1}
num_leaves ∈ {15, 31, 63}
min_child_samples ∈ {5, 10, 20}
```

with ROC-AUC as the scoring metric and 5-fold stratified CV. The refit estimator is retained as `tuned_lgbm`.

### Regression models

Three regressors mirror the classification setup: ridge, random-forest regressor, LightGBM regressor. Metrics: R², MAE under 5-fold cross-validation.

### Reported classification metrics

* **ROC-AUC** — primary ranking metric (threshold-free, invariant to class prior).
* **PR-AUC (average precision)** — minority-class focus; the positive class is the minority under the 0.85 threshold.
* **F1** — retained for comparability with prior crowdsourcing-quality work.

### Feature attribution

**Permutation importance** is used rather than the random forest's impurity-based `feature_importances_`. The latter is biased towards high-cardinality features (`n_answers`, `lifetime_h`); permutation importance measures the drop in held-out ROC-AUC when each feature is independently shuffled, with 15 repetitions per feature for error bars.

The attribution is computed on a 30 % worker-level held-out split — not on training-set splits — so the reported importance is genuinely out-of-sample. This combination (permutation, held-out, repeated) is the standard recommendation in scikit-learn's current documentation.

---

## Stage 3 — Aggregation, Composite Score, Tiers, Segmentation, Simulation (`03_Advanced_Models.ipynb`)

### Aggregation models

Both probabilistic models are implemented **from scratch** in fully vectorised numpy. See `src/aggregation.py`.

#### Dawid–Skene (1979)

Each worker `w` is modelled by a `K × K` confusion matrix `π_w` where `π_w[j, l] = P(emit l | true is j)`. EM alternates between:

* **E-step**: posterior over the true label of task `t` given current `π` and class priors `ρ`,

  ```
  T[t, j] ∝ ρ[j] · ∏_{(w, a)} π_w[j, a]
  ```

* **M-step**: re-estimate per-worker confusion matrices and priors as the posterior-weighted observation counts.

Convergence: change in observed-data log-likelihood below `1e-6`, max 100 iterations. Initialisation: majority-vote soft assignments.

Worker-level reliability score: mean diagonal of the confusion matrix.

#### MACE (Hovy et al., 2013)

Each worker `w` is described by a spam probability `σ_w` and a marginal answer distribution `θ_w`. Observation model:

```
P(a | j, w) = σ_w · θ_w[a]  +  (1 − σ_w) · 1[a = j]
```

EM as for Dawid–Skene; competence reported as `1 − σ_w`, clipped to `[0.01, 0.99]` for numerical stability.

#### Implementation note (vectorisation)

The expensive E- and M-step accumulations over the annotation table are performed with `np.add.at` broadcasted writes — there is no Python-level loop over annotation rows. On the full 4.12 M-row dataset one iteration takes roughly half a second on a laptop CPU; convergence in 10–30 iterations gives end-to-end wall times under a minute.

### Evaluation of aggregation models

Both models are evaluated against **two** references:

* **Gold-task ground truth** (the external criterion, Section 2.3.5).
* **Platform `task_ans` on regular tasks** with the circularity caveat made explicit. The divergence between the two references is itself one of the findings: majority vote dominates against the platform labels (by construction), while probabilistic models dominate against gold (the honest criterion).

Reported metrics: accuracy, Cohen's κ, macro-F1, and Spearman ρ between Dawid–Skene and MACE worker scores.

### Composite worker quality score

A **ridge regression** learns weights for three quality signals:

| Predictor | Source |
|---|---|
| `agreement_rate` | Worker's agreement with the platform majority on regular tasks (cheap but biased) |
| `ds_score` | Dawid–Skene diagonal (model-based, structurally distinct) |
| `mace_score` | MACE competence `1 − σ_w` (model-based, sparse parameterisation) |

The ridge is fit on the **rated** set (workers with ≥ 40 gold observations) and validated on a **worker-disjoint** held-out cohort. Raw ridge predictions are winsorised at the 1 % / 99 % percentiles and Min-Max scaled to **[0, 100]** to produce the final published composite score for every worker.

This replaces the hand-weighted equal-share composite used in earlier drafts; the learned weights typically place the heaviest emphasis on the model-based signals while keeping `agreement_rate` as a useful low-cost anchor for workers with few gold observations.

### Confidence tiers (task-level)

Tasks (not workers) are partitioned into four tiers based on worker agreement and Dawid–Skene posterior confidence:

| Tier | Rule | Operational interpretation |
|---|---|---|
| **Confident** | unanimity AND `ds_conf ≥ 0.90` | Accept as-is |
| **Likely correct** | `agreement ≥ 0.67` AND `ds_conf ≥ 0.70` | Accept |
| **Borderline** | `agreement ≥ 0.50` | Light review |
| **Contested** | otherwise | Higher overlap / expert review |

The operational value is measured by **error concentration**: if the Contested tier captures a disproportionate share of disagreements at a small share of total tasks, the tiering provides actionable triage. The 70.7 % unanimous-task share (EDA finding #5) puts a floor under the Confident tier.

### Worker segmentation

**Feature set.** Clustering uses five features: `quality_score`, `log(1 + n_answers)`, `answer_entropy`, `per_task_sec_median`, `pct_label_1`. Features are standardised before clustering.

**Algorithms compared.** k-means, Gaussian mixture models, and agglomerative clustering — three different inductive biases (spherical hard / ellipsoidal soft / hierarchical).

**Method / K sweep.** Each algorithm is run for `K ∈ {3, 4, 5, 6}`. Both **silhouette** and **Calinski–Harabasz** are reported, since the two metrics often disagree on the optimal K and we want the agreement (or lack thereof) to be visible.

**Operational K.** `K = 3` is fixed for the final segmentation based on internal-metric agreement and on the resulting segments mapping cleanly to distinct platform actions.

**Segment naming (deterministic, no analyst judgement).** Cluster centroids in the (score, activity) plane map to:

| Centroid in (score, n_answers) | Segment |
|---|---|
| score ≥ 75 AND n_answers ≥ 100 | **Reliable veteran** |
| score ≥ 75 AND n_answers < 100 | **Promising newcomer** |
| score < 65 (any activity) | **Low quality** |
| otherwise | **Average worker** |

The rule is deterministic given the centroids; re-running the clustering must produce the same named segments up to cluster permutation.

### Robustness checks for the segmentation

Two checks are reported (Section 2.12 / 3.6 of the thesis):

1. **Stability under alternative K.** The Adjusted Rand Index (ARI) between the base K = 3 assignment and `K ∈ {3, 5}` quantifies how much of the segment structure survives perturbations of the cluster count. ARI ≈ 1.0 against the same K confirms determinism; an ARI above ~0.5 against K = 5 indicates a robust underlying structure that is merely re-partitioned at finer resolution.
2. **Behaviour-only re-clustering.** k-means is re-fit on the four behaviour features alone, with `quality_score` removed from the input. If the behaviour-only clusters still stratify by `quality_score` (mean-score spread between top and bottom cluster), the segmentation is grounded in genuine behavioural differences rather than tautologically driven by the composite-score feature.

### Adaptive-overlap simulation

Each task is assigned an overlap that depends on the **worst** worker segment that touched it:

| Worst segment on task | Overlap assigned |
|---|---|
| Reliable veteran | 1 |
| Promising newcomer | 2 |
| Average worker | 2 |
| Low quality | 3 |
| Unknown (no segment) | 2 |

**Baseline.** Uniform overlap = 3 (platform default).
**Cost metric.** Total collected answers (= task count × average overlap).
**Quality metric.** Fraction of aggregated labels matching the platform reference.

**Scope of the claim.** This is a scenario analysis, not a causal evaluation. It assumes worker behaviour does not change under a policy change. The reported savings are a **plausible upper bound** under a static counterfactual; a causal estimate would require an A / B test against live traffic.

---

## Limitations

1. **No randomised control for the rehabilitation analysis.** Pre/post comparison conflates a genuine treatment effect with regression to the mean; survivors are over-represented.
2. **Circular evaluation on regular tasks.** `task_ans` is the platform majority vote, not an independent label; the convention is to evaluate aggregation models against gold tasks.
3. **Single platform.** All data comes from one e-commerce annotation pipeline; generalisability to other platforms is not established.
4. **Sample vs. full dataset.** Notebooks ship a 100 K-row sample for reproducibility; a few distributional statistics (especially in the low-volume tail) differ from full-dataset values.
5. **Adaptive overlap simulation is counterfactual.** No live A / B test; reported savings are an upper bound under static-behaviour assumptions.
