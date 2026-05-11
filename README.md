# Evaluation of Performer Quality and Segmentation of Results in Crowdsourcing Systems

**Master's Thesis** · HSE FCS — Master of Data Science (MNAD) · 2024–2026
**Author:** Aleksandr Pchelnikov
**Supervisor:** Armen Beklaryan, Associate Professor, HSE FCS

---

## Overview

This repository contains the research code for a master's thesis investigating how to evaluate and predict worker quality in large-scale crowdsourcing platforms, using data from a real Russian e-commerce annotation pipeline.

The work addresses three interconnected problems:

1. **Worker quality estimation** — measuring labeller reliability with gold (control) tasks and probabilistic aggregation models (Dawid–Skene, MACE) implemented from scratch.
2. **Quality prediction** — training supervised models to predict a worker's future gold accuracy from behavioural features, under a leakage-free per-worker chronological split.
3. **Result segmentation** — clustering workers into operationally interpretable segments, validating the segmentation against alternative cluster counts (ARI) and against a behaviour-only re-clustering, then simulating an adaptive overlap policy.

The dataset covers **4.12 million annotation answers** from **14 452 workers** across **5 projects** over approximately one month of production activity.

---

## Research Questions

**RQ1.** Which worker-level features best predict future gold-task accuracy in a production crowdsourcing system, and does the engineered feature representation add information above what historical gold accuracy alone provides?

**RQ2.** How do probabilistic aggregation models (Dawid–Skene, MACE) compare to majority voting when evaluated against independent gold-task ground truth, and how do their worker-level reliability scores correlate with direct gold accuracy?

**RQ3.** What worker segments emerge from behavioural data when clustering is driven by a composite quality score, and how do these segments map to operationally meaningful platform interventions (in particular, adaptive overlap)?

---

## Repository Structure

```
master-thesis-crowdsourcing-quality/
│
├── notebooks/
│   ├── 01_EDA.ipynb                                  # Exploratory data analysis (13 EDA findings)
│   ├── 02_Feature_Engineering_and_Baseline.ipynb     # Features + supervised baselines + tuning
│   └── 03_Advanced_Models.ipynb                      # DS, MACE, ridge composite, clustering, simulation
│
├── src/                              # Canonical implementations (imported by every notebook)
│   ├── data_prep.py                  #   CSV loading + uniform pre-processing
│   ├── features.py                   #   build_worker_features, build_temporal_split
│   ├── aggregation.py                #   vectorised Dawid–Skene, MACE (np.add.at, ~1 s/iter on 4.12 M)
│   ├── scoring.py                    #   ridge composite score, 4-tier task-level confidence
│   ├── segmentation.py               #   k-means/GMM/agglomerative, ARI, CH, robustness checks
│   └── plots.py                      #   shared matplotlib helpers and colour palette
│
├── data/
│   ├── README.md                     # Dataset schema and access note
│   └── sample_data.csv               # 100 K-row reproducibility sample
│
├── artifacts/                        # Serialised intermediate outputs from notebooks
│   ├── README.md
│   ├── worker_features.csv           #   full inventory (~45 columns)
│   ├── worker_features_history.csv   #   history-window features
│   ├── worker_model_data.csv         #   features + supervised targets
│   ├── ds_worker_quality.csv         #   Dawid–Skene diagonal score
│   └── mace_worker_competence.csv    #   MACE competence (1 − σ)
│
├── docs/
│   ├── data_description.md
│   ├── methodology.md
│   └── reproducibility.md
│
├── figures/                          # Exported figures for the thesis text
├── results/
│   └── eda_key_findings.md
│
├── requirements.txt
├── LICENSE
└── README.md
```

> Raw production data is not included; see [`data/README.md`](data/README.md).

---

## Dataset

| Property | Value |
|---|---|
| Total answers | 4.12 million rows |
| Sample used in notebooks | 100 000 rows (`sample_data.csv`) |
| Workers | 14 452 unique |
| Projects | 5 (IDs: 575, 576, 577, 578, 581) |
| Time span | ~1 month |
| Task structure | Binary labelling (labels 1 and 2; service flag 3 excluded; label 4 is rare and dropped where binary models are fit) |

### Key domain concepts

| Column / Term | Meaning |
|---|---|
| `pool_type = 0` | Regular (paid) pool — production annotation work |
| `pool_type = 1` | Rehabilitation pool — triggered when a worker makes ≥ 2 errors in their last 10 gold tasks |
| `pool_type = 3` | Training pool — onboarding before entering a project |
| `task_type = 0` | Regular task |
| `task_type = 1` | Gold (control) task — known answer, used for real-time quality monitoring |
| `user_ans = 3` | Service flag for malformed task pages — **excluded from all correctness calculations** |
| `overlap` | Number of independent workers assigned to the same task |
| `tasks_per_page` | Tasks shown per screen; timing data is page-level, not task-level |

---

## Notebooks Overview

### `01_EDA.ipynb` — Exploratory Data Analysis

Establishes the empirical context against which all subsequent modelling is interpreted. Thirteen findings drive the methodology in the rest of the thesis.

| # | Finding | Value |
|---|---|---|
| 1 | Dataset scale | 4.12 M answers · 14 452 workers · 5 projects |
| 2 | One-shot workers (single answer) | 11.4 % |
| 3 | Power workers (≥ 100 answers) | 7.5 % |
| 4 | Class-1 share across projects | 58 – 73 % (trivial baseline) |
| 5 | Regular-pool gold accuracy | ~89 % |
| 6 | Rehabilitation-pool accuracy | ~82 % |
| 7 | Training-pool accuracy | ~66 % |
| 8 | Rehabilitation effect (pre/post) | +1.24 pp (Wilcoxon p = 0.002) |
| 9 | Skippers vs. non-skippers accuracy gap | −3.5 pp |
| 10 | Fast-decile accuracy (~5 s / task) | 91.0 % |
| 11 | Slow-decile accuracy (~79 s / task) | 87.7 % |
| 12 | Unanimous regular tasks | 70.7 % |
| 13 | Binary-class error structure | 96–98 % of errors are class 1 ↔ class 2 swaps |

> **`task_ans` caveat.** On gold tasks `task_ans` is an independent verified label. On regular tasks it is the platform's own majority-vote aggregation — comparing a new aggregation model against `task_ans` on regular tasks is circular. Gold tasks are the external criterion used throughout.

### `02_Feature_Engineering_and_Baseline.ipynb` — Features and Supervised Baselines

Constructs the full ~45-column worker feature inventory across five groups (activity, gold history, regular-task proxy, speed, behaviour) and selects a 16-feature compact subset for modelling. A leakage-free **per-worker chronological 80 / 20 split** divides each worker's timeline into a feature window (early 80 %) and a target window (later 20 %); two parallel targets are defined on the target window:

* `future_gold_acc` — continuous mean gold accuracy in the target window;
* `future_high_quality` — binary indicator (`future_gold_acc ≥ 0.85` AND `future_n_gold ≥ 5`).

**Five classification models** are compared under 5-fold stratified cross-validation:

| # | Model | Role |
|---|---|---|
| 1 | Logistic regression on `gold_acc` only | Single-feature sanity baseline |
| 2 | Logistic regression on `reg_agreement_proxy` only | Cheap proxy baseline |
| 3 | Logistic regression on all features | Linear baseline (L2, balanced) |
| 4 | Random forest | Non-linear baseline (400 trees, depth 6) |
| 5 | LightGBM | Production-grade gradient boosting |

The gradient-boosting model is additionally tuned with `GridSearchCV` over n_estimators × learning_rate × num_leaves × min_child_samples (5-fold stratified, ROC-AUC scoring), producing `tuned_lgbm`.

**Three regression models** (ridge, random-forest regressor, LightGBM regressor) are trained on `future_gold_acc` for comparability with the classification setup. Metrics: ROC-AUC, PR-AUC, F1 for classification; R², MAE for regression.

**Feature attribution** uses permutation importance (15 repeats, ROC-AUC scoring) on a 30 % worker-level held-out split — chosen over the random forest's impurity-based `feature_importances_` because the latter is biased towards high-cardinality features. The two attributions produce qualitatively different top-10 lists.

### `03_Advanced_Models.ipynb` — Aggregation, Composite Score, Tiers, Segmentation, Simulation

**Aggregation models.** `Dawid–Skene` and `MACE` are implemented from scratch in fully vectorised numpy (`np.add.at` writes in both E- and M-steps). One EM iteration on the 4.12 M-row dataset runs in well under a second on a laptop CPU. Predictions are evaluated against both gold-task ground truth and the platform's regular-task labels; the divergence between the two evaluations is itself one of the findings (circular-evaluation pathology).

**Composite worker score.** A ridge regression learns weights for three quality signals — `agreement_rate`, Dawid–Skene diagonal, MACE competence — trained on workers with at least 40 gold observations (the *rated* set), validated on a worker-disjoint held-out cohort. Raw ridge predictions are winsorised at 1 % / 99 % and Min-Max scaled to a 0–100 published score.

**Confidence tiers.** Tasks (not workers) are partitioned into four tiers based on worker agreement and Dawid–Skene posterior confidence:

| Tier | Rule | Use |
|---|---|---|
| Confident | unanimity AND `ds_conf ≥ 0.90` | accept |
| Likely correct | `agreement ≥ 0.67` AND `ds_conf ≥ 0.70` | accept |
| Borderline | `agreement ≥ 0.50` | review |
| Contested | otherwise | re-route / expert review |

**Worker segmentation.** Clustering compares **three algorithms** (k-means, GMM, agglomerative) across **K ∈ {3, 4, 5, 6}** under **both** silhouette and Calinski–Harabasz scores. K = 3 is fixed for the operational segmentation. Two robustness checks: (a) ARI between the K = 3 assignment and K ∈ {3, 5} alternatives quantifies stability under alternative cluster counts; (b) re-running k-means on behaviour-only features (without `quality_score`) and checking whether the resulting clusters still stratify by quality rules out the concern that the segmentation is tautologically driven by the composite score.

Segments are named deterministically from cluster centroids in (score, activity) space:

* high score (≥ 75) + high activity (≥ 100 answers) → **Reliable veteran**
* high score + low activity → **Promising newcomer**
* low score (< 65) → **Low quality**
* otherwise → **Average worker**

**Adaptive overlap simulation.** A segment-to-overlap mapping is simulated against the platform's uniform baseline; the cost metric is total collected answers, the quality metric is fraction of aggregated labels matching the platform reference. Reported as a *plausible upper bound* under a static counterfactual, not as a causal estimate (no A / B test on live traffic).

---

## Methodological Notes

**Gold accuracy vs. agreement rate.** Gold accuracy is the primary quality criterion. Agreement rate on regular tasks is a proxy *feature*, never a target — `task_ans` on regular tasks is the platform's own majority vote, so using it as ground truth would be circular.

**Per-worker chronological split (no leakage).** All worker features are built on the first 80 % of each worker's chronologically-ordered answers; both targets are defined on the remaining 20 %. Workers absent from either window are excluded.

**`user_ans = 3` exclusion.** Service flag for malformed pages, dropped before any correctness computation.

**Page-to-task timing.** Recorded at page level; per-task time is approximated as `page_duration_sec / tasks_per_page`.

**Code lives in `src/`.** Every algorithm above has a single canonical implementation under `src/`; the notebooks import from `src/` and the in-notebook cells reproduce the same logic in expanded form for the thesis text. Editing `src/aggregation.py` is sufficient to change the corresponding cell behaviour in notebook 03.

---

## Reproducibility

### Requirements

Python 3.10+ is recommended. Install dependencies:

```bash
pip install -r requirements.txt
```

### Running the notebooks

```bash
jupyter notebook notebooks/01_EDA.ipynb
jupyter notebook notebooks/02_Feature_Engineering_and_Baseline.ipynb
jupyter notebook notebooks/03_Advanced_Models.ipynb
```

Notebooks look for `data.csv` next to the notebook or in `data/`; the 100 K-row `sample_data.csv` shipped with this repository is sufficient to reproduce every visualisation and most numerical results.

On the full 4.12 M-row dataset, notebooks 02 and 03 may require 8–16 GB RAM; the vectorised aggregation EM in `src/aggregation.py` keeps wall time on the full dataset to roughly one minute end-to-end.

See [`docs/reproducibility.md`](docs/reproducibility.md) for the full step-by-step procedure.

---

## Limitations

* **No randomised control for the rehabilitation analysis.** Pre/post comparison conflates a genuine treatment effect with regression to the mean; workers who fail to recover leave the platform and create survivorship bias.
* **`task_ans` circularity on regular tasks.** The platform majority vote is not an independent label; aggregation-model comparisons against it are reported with the caveat made explicit.
* **Single platform.** All data comes from one e-commerce annotation platform; generalisability to other task types or platforms is not established.
* **Sample vs. full dataset.** The sample preserves project proportions but a few distributional statistics (especially for the long tail of low-volume workers) differ from full-dataset values; these are noted inline in the EDA.
* **Adaptive-overlap simulation is a static counterfactual.** No causal evaluation; the reported savings are a plausible upper bound assuming worker behaviour does not drift under a policy change.

---

## Thesis Context

| Field | Value |
|---|---|
| Program | Master of Data Science (MNAD), HSE Faculty of Computer Science |
| Academic year | 2024–2026 |
| Thesis title | Evaluation of Performer Quality and Segmentation of Results in Crowdsourcing Systems |
| Author | Aleksandr Pchelnikov |
| Supervisor | Armen Beklaryan, Associate Professor, HSE FCS |

---

## Citation

```
Pchelnikov, A. (2026). Evaluation of Performer Quality and Segmentation of Results
in Crowdsourcing Systems. Master's Thesis, HSE Faculty of Computer Science,
Moscow, Russia.
```

---

## License

Code in this repository is released under the MIT License. See [`LICENSE`](LICENSE) for details. The dataset is proprietary and not included in this repository.
