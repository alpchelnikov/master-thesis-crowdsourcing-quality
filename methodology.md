# Methodology

**Thesis:** Evaluation of Performer Quality and Segmentation of Results in Crowdsourcing Systems  
**Author:** Aleksandr Pchelnikov · HSE FCS MNAD, 2024–26

---

## Overview

The study analyses a large-scale industrial crowdsourcing dataset with the goal of
(1) characterising worker quality, (2) building predictive models for future performance,
and (3) segmenting workers into interpretable quality tiers.

The pipeline consists of three sequential stages, each implemented in a dedicated notebook:

```
01_EDA  →  02_Feature_Engineering_Baseline  →  03_Advanced_Models
```

---

## Stage 1 — Exploratory Data Analysis (`01_EDA.ipynb`)

### Data

Each row in the dataset is one worker's answer to one task.
The hierarchy is: **answer → task → page → pool → project**.

| Pool type | Role | Tasks inside |
|---|---|---|
| `pool_type = 0` | Regular (paid) production work | Regular + gold tasks |
| `pool_type = 1` | Rehabilitation — triggered after ≥ 2 errors in last 10 gold tasks | Gold tasks only |
| `pool_type = 3` | Training — onboarding before entering a project | Training tasks only |

### Ground truth and correctness

**Gold tasks** (`task_type = 1`) carry a verified ground-truth label (`task_ans`) independent
of worker votes. They are the primary quality signal throughout this study.

**Important caveat:** On regular tasks (`task_type = 0`), `task_ans` reflects the platform's
own majority-vote aggregation — not an independent label. Comparing a new aggregation model
against `task_ans` on regular tasks creates a **circular evaluation**. All model comparisons
in this thesis use gold tasks or held-out workers as the external criterion.

### Exclusion: `user_ans = 3`

Value 3 in `user_ans` is a platform service flag for malformed task pages. It is **not** a
valid annotation class and is excluded from all correctness and distribution computations
before any analysis begins.

### Timing

Response time is recorded at the **page level**. Per-task time is approximated as:

```
per_task_sec = page_duration_sec / tasks_per_page
```

Page sizes differ by pool type (1–3 tasks/page), so raw page duration is not
comparable across pool types without this normalisation.

### Key EDA findings

| # | Finding | Value |
|---|---|---|
| 1 | Dataset size | 4.12 M answers, 14 452 workers, 5 projects |
| 2 | Regular pool gold accuracy | ~89% |
| 3 | Rehabilitation pool accuracy | ~82% |
| 4 | Training pool accuracy | ~66% |
| 5 | Unanimous tasks (all workers agree) | 70.7% |
| 6 | One-shot workers | 11.4% |
| 7 | Power workers (100+ answers) | 7.5% |
| 8 | Rehabilitation effect | +1.24 pp (Wilcoxon p = 0.002) |
| 9 | Skippers vs non-skippers accuracy gap | −3.5 pp |
| 10 | Accuracy range by speed decile | 87.7%–91.0% (confounded by task difficulty) |

---

## Stage 2 — Feature Engineering & Baseline Models (`02_Feature_Engineering_Baseline.ipynb`)

### Feature construction

Worker-level features are grouped into six categories:

| Group | Features |
|---|---|
| Activity | `n_answers`, `n_projects`, `n_pools`, `lifetime_h` |
| Accuracy | `gold_acc`, `n_gold`, `reg_acc`, `n_regular`, `agreement_rate` |
| Timing | `med_duration`, `std_duration`, `p10_duration`, `p90_duration` |
| Diversity | `answer_entropy`, `skip_rate` |
| Platform flags | `has_regular`, `has_rehab`, `has_training` |
| Pricing | `mean_price`, `max_price` |

Total: **21 features** for **14 452 workers**.

### Temporal validation design

To prevent **data leakage**, features and prediction target are computed on
non-overlapping time windows:

```
|---- feature window (early days) ----|-- gap --|-- target window (later days) --|
```

- Features are built from a worker's activity in the **feature window**.
- The target variable is the worker's **gold accuracy in the target window**.
- Workers absent from either window are excluded.

This ensures the model cannot directly observe the outcome it is predicting.

### Target variable

`future_gold_acc` — mean gold-task correctness in the target window.
This is a regression target; for classification experiments a threshold is applied.

### Baseline classifiers

Three sklearn classifiers trained as baselines:

| Model | Notes |
|---|---|
| Logistic Regression | Linear baseline |
| Decision Tree | Non-linear, interpretable |
| Random Forest | Ensemble, typically strongest baseline |

Evaluation metric: <!-- TODO: fill in after notebook 02 outputs are confirmed (AUC-ROC / F1 / MAE) -->

---

## Stage 3 — Advanced Models & Segmentation (`03_Advanced_Models.ipynb`)

### Aggregation models

Both models implemented **from scratch** (no external crowdsourcing libraries).

#### Dawid-Skene (1979)

An EM algorithm that jointly estimates:
- A **confusion matrix** per worker (K × K, where K = number of classes)
- **Posterior label probabilities** per task

Worker quality is summarised as the **mean diagonal** of their confusion matrix.

#### MACE (Hovy et al., 2013)

Models each worker as a mixture of:
- A *spammer* who answers uniformly at random
- A *competent* annotator who answers from a per-worker distribution

The spam probability is the worker's quality penalty;
competence = 1 − spam_probability.

### Evaluation of aggregation models

All aggregation models are compared against **gold-task accuracy** as the
external criterion — avoiding the circularity of using platform majority vote.

<!-- TODO: add comparison table (Dawid-Skene vs MACE vs Majority Vote) after notebook 03 is run -->

### Composite worker score

An ML-optimised composite score combines three quality signals:

```
composite = w1 * gold_acc_scaled
          + w2 * ds_quality_scaled
          + w3 * mace_competence_scaled
```

All components are Min-Max scaled to [0, 1] before combining.
Default weights: equal (1/3 each).

### Worker segmentation (clustering)

Three algorithms compared at K = 4 clusters:

| Algorithm | Notes |
|---|---|
| KMeans | Hard assignment, fast |
| GMM | Soft assignment, probabilistic |
| Agglomerative | Hierarchical, no centroid assumption |

Optimal K selected by silhouette score analysis.

Cluster interpretation focuses on:
1. High accuracy + moderate entropy → reliable annotators
2. Low accuracy + low entropy → potential spammers
3. Low accuracy + high entropy → genuinely struggling workers
4. <!-- TODO: describe 4th cluster after notebook 03 is finalised -->

### Adaptive overlap simulation

Simulates the cost-quality trade-off under a policy that:
- Routes easy tasks (high worker agreement) to **fewer workers** (overlap = 1–2)
- Routes contested tasks to **more workers** (overlap = 3–5)

Baseline: uniform overlap = 3 (platform default).

<!-- TODO: add simulation results (cost reduction % vs accuracy loss) after notebook 03 -->

---

## Limitations

1. **Rehabilitation confound.** Pre/post accuracy comparison conflates genuine
   rehabilitation effect with regression to the mean. No randomised control group.

2. **Circular evaluation on regular tasks.** `task_ans` on non-gold tasks is a
   majority-vote proxy — all aggregation model comparisons must use gold tasks.

3. **Single platform.** All data comes from one e-commerce annotation pipeline.
   Generalisability to other domains is not established.

4. **Sample vs full dataset.** Notebooks run on a 100 K-row sample by default.
   Some distributions differ from full-dataset values (noted inline where relevant).
