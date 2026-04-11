# Evaluation of Performer Quality and Segmentation of Results in Crowdsourcing Systems

**Master's Thesis** · HSE FCS — Master of Data Science (MNAD) · 2024–2026  
**Author:** Aleksandr Pchelnikov  
**Supervisor:** Armen Beklaryan, Associate Professor, HSE FCS

---

## Overview

This repository contains the full research code for a master's thesis investigating how to evaluate and predict worker quality in large-scale crowdsourcing platforms, using data from a real Russian e-commerce annotation pipeline.

The work addresses three interconnected problems:
1. **Worker quality estimation** — measuring labeller reliability using gold (control) tasks and aggregation models;
2. **Quality prediction** — training supervised models to predict a worker's future gold accuracy from behavioural features, enabling proactive quality management;
3. **Result segmentation** — clustering workers into interpretable quality segments and simulating adaptive overlap strategies.

The dataset covers **4.12 million annotation answers** from **14 452 workers** across **5 projects** over approximately one month of production activity.

---

## Research Questions

**RQ1.** Which worker-level features best predict gold-task accuracy in a production crowdsourcing system?  
**RQ2.** How do probabilistic aggregation models (Dawid-Skene, MACE) compare to majority voting when evaluated against independent gold-task ground truth?  
**RQ3.** What worker segments emerge from behavioural data, and how do they relate to quality and platform interventions (rehabilitation)?

---

## Repository Structure

```
master-thesis-crowdsourcing-quality/
│
├── notebooks/
│   ├── 01_EDA.ipynb                          # Exploratory data analysis (18 sections)
│   ├── 02_Feature_Engineering_Baseline.ipynb # Feature construction + baseline models
│   └── 03_Advanced_Models.ipynb              # Dawid-Skene, MACE, clustering, simulation
│
├── data/
│   └── README.md                             # Dataset description and access note
│
├── figures/                                  # [TODO: export key plots from notebooks]
│
├── requirements.txt                          # Python dependencies
├── .gitignore
├── LICENSE
└── README.md
```

> **Note:** Raw data files are not included in this repository. See [`data/README.md`](data/README.md) for the dataset schema and access instructions.

---

## Dataset

The dataset is drawn from a proprietary crowdsourcing platform used for product catalogue annotation (binary labelling tasks).

| Property | Value |
|---|---|
| Total answers | 4.12 million rows |
| Sample used in notebooks | 100 000 rows |
| Workers | 14 452 unique |
| Projects | 5 |
| Time span | ~1 month |
| Task structure | Binary labelling (labels 1 and 2) |

**Key domain concepts:**

| Column / Term | Meaning |
|---|---|
| `pool_type = 0` | Regular (paid) pool — production annotation work |
| `pool_type = 1` | Rehabilitation pool — triggered when a worker makes ≥ 2 errors in the last 10 gold tasks |
| `pool_type = 3` | Training pool — onboarding exercises before entering a project |
| `task_type = 0` | Regular task |
| `task_type = 1` | Gold (control) task — known answer, used for real-time quality monitoring |
| `user_ans = 3` | Service flag for malformed tasks — **excluded from all correctness calculations** |
| `overlap` | Number of workers assigned to the same task |
| `tasks_per_page` | Tasks shown per screen; timing data is page-level, not task-level |

---

## Notebooks Overview

### `01_EDA.ipynb` — Exploratory Data Analysis

Covers 18 analytical sections including: data structure validation, page timing model, per-project breakdown, worker activity distribution, response time analysis, answer distribution, correctness by pool type, pricing analysis, temporal patterns, speed vs quality, task difficulty, skip behaviour, overlap dynamics, worker-level feature correlation, quality heterogeneity, and rehabilitation effectiveness.

**Key empirical findings:**

| Finding | Value |
|---|---|
| Regular pool gold accuracy | ~89% |
| Rehabilitation pool accuracy | ~82% |
| Training pool accuracy | ~66% |
| Unanimous tasks (all workers agree) | 70.7% |
| One-shot workers (single answer) | 11.4% |
| "Power workers" (100+ answers) | 7.5% |
| Rehabilitation effect | +1.24 pp (Wilcoxon p = 0.002) |
| Skippers vs non-skippers accuracy gap | −3.5 pp |
| Accuracy at fast speed (~5 s/task) | 91.0% |
| Accuracy at slow speed (~79 s/task) | 87.7% |

> **Important caveat:** `task_ans` on regular tasks reflects the platform's majority vote — using it as ground truth for aggregation model evaluation creates a circular comparison. Gold tasks (`task_type = 1`) are the independent ground truth used throughout this work.

---

### `02_Feature_Engineering_Baseline.ipynb` — Feature Construction & Baseline Models

Constructs ~20 worker-level behavioural features grouped into six categories (activity, accuracy, timing, diversity, rehabilitation history, pricing sensitivity) using a **temporal validation design** — features are derived from early days of the observation window, and the model target (future gold accuracy) is measured on later days, eliminating data leakage.

Three baseline classifiers are trained to predict future gold accuracy:

- Logistic Regression
- Decision Tree
- Random Forest

<!-- TODO: add final baseline metric table (AUC / F1 / accuracy) once notebook outputs are confirmed -->

---

### `03_Advanced_Models.ipynb` — Advanced Models & Segmentation

Implements from scratch:

- **Dawid-Skene** — EM-based probabilistic model estimating per-worker confusion matrices and inferred true labels;
- **MACE** (Multi-Annotator Competence Estimation) — models worker spamming probability alongside a competence score;
- **ML-optimised composite worker score** — combines Dawid-Skene and MACE outputs with behavioural features;
- **Worker clustering** — KMeans, GMM, and Agglomerative clustering (K=4), yielding interpretable quality segments;
- **Adaptive overlap simulation** — models cost-quality trade-off when dynamically routing tasks to high-quality workers.

<!-- TODO: add aggregation model comparison table (Dawid-Skene vs MACE vs majority vote vs gold accuracy) -->

---

## Methodological Notes

**Why gold accuracy, not agreement rate?**  
Gold accuracy (correctness on tasks with known answers) is the primary quality metric throughout this work. Agreement rate on regular tasks is a proxy metric only and is treated as such. This distinction matters especially when evaluating aggregation models: `task_ans` on regular tasks reflects the platform's own majority vote, and comparing a new aggregation model against it is circular.

**Temporal validation design**  
To prevent data leakage in supervised models, all features are constructed from a worker's activity in an earlier time window, and the prediction target (future gold accuracy) is measured in a later window with no overlap.

**`user_ans = 3` exclusion**  
Value 3 in `user_ans` is a platform service flag for malformed task pages — it is not a valid annotation class and is excluded from all correctness and distribution analyses.

**Timing normalisation**  
Response time data is recorded at page level. Per-task time is approximated as `page_duration_sec / tasks_per_page` to enable fair cross-pool comparison.

---

## Reproducibility

### Requirements

Python 3.10+ is recommended. Install dependencies:

```bash
pip install -r requirements.txt
```

### Running the notebooks

Run notebooks in order:

```bash
jupyter notebook notebooks/01_EDA.ipynb
jupyter notebook notebooks/02_Feature_Engineering_Baseline.ipynb
jupyter notebook notebooks/03_Advanced_Models.ipynb
```

The notebooks expect a file named `data.csv` (or `sample_data.csv`) in the project root or a `data/` subdirectory. See [`data/README.md`](data/README.md) for the expected schema.

> On the full 4.12 M-row dataset, notebooks 02 and 03 may require 8–16 GB RAM. The 100 K sample provided is sufficient to reproduce all visualisations and model outputs.

---

## Limitations

- **No randomised control for rehabilitation analysis.** The pre/post comparison confounds genuine rehabilitation effect with regression to the mean. Workers who quit after rehab are excluded (survivorship bias).
- **`task_ans` circularity on regular tasks.** The platform majority vote is used as ground truth on regular tasks; direct comparison of aggregation models against this metric is circular. Results should be interpreted relative to gold-task ground truth.
- **Single platform.** All data comes from one e-commerce annotation platform. Generalisability to other task types or platforms is not established.
- **Sample vs full dataset.** The public-facing notebooks use a 100 K-row sample. Some distributional statistics differ from full-dataset values (noted in the EDA where relevant).

---

## Thesis Context

| Field | Value |
|---|---|
| Program | Master of Data Science (MNAD), HSE Faculty of Computer Science |
| Academic year | 2024–2026 |
| Thesis title | Evaluation of Performer Quality and Segmentation of Results in Crowdsourcing Systems |
| Author | Aleksandr Pchelnikov |
| Supervisor | Armen Beklaryan, Associate Professor, HSE FCS |
| Defense status | <!-- TODO: update when known --> In progress |

---

## Citation

If you reference this work, please cite it as:

```
Pchelnikov, A. (2026). Evaluation of Performer Quality and Segmentation of Results
in Crowdsourcing Systems. Master's Thesis, HSE Faculty of Computer Science,
Moscow, Russia.
```

---

## License

Code in this repository is released under the MIT License. See [`LICENSE`](LICENSE) for details.  
The dataset is proprietary and is not included in this repository.
