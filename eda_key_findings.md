# EDA Key Findings

Source: `notebooks/01_EDA.ipynb`, Section 18 · Full dataset (4.12 M rows)

| # | Finding | Value | Implication |
|---|---|---|---|
| 1 | Dataset scale | 4.12 M answers, 14 452 workers, 5 projects, ~1 month | Sufficient statistical power for modelling |
| 2 | Activity distribution | 11.4% one-shot workers; 7.5% power workers (100+ answers) produce most volume | Quality must be estimated from models, not raw counts |
| 3 | Task structure | 96–98% of errors are class-1 ↔ class-2 swaps | Effectively binary classification |
| 4 | Class imbalance | 58–73% label 1 across projects | Trivial "always predict 1" achieves 58–73% accuracy |
| 5 | Regular pool accuracy | ~89% | Baseline for aggregation models to improve upon |
| 6 | Rehabilitation pool accuracy | ~82% | Expected — workers enter rehab after errors |
| 7 | Training pool accuracy | ~66% | Expected — mistakes are part of learning |
| 8 | Rehabilitation effect | +1.24 pp (Wilcoxon p = 0.002); 54% of workers improve | Real but modest; regression-to-mean confound present |
| 9 | Speed vs accuracy | 91.0% at ~5 s/task → 87.7% at ~79 s/task | Confounded by task difficulty, not a speed–quality trade-off |
| 10 | Overlap = 5 bias | Workers answering overlap-5 tasks face harder items | Selection bias; not a failure of redundancy |
| 11 | Unanimous tasks | 70.7% of tasks are unanimous | Aggregation challenge lies in the 29% borderline cases |
| 12 | Skippers accuracy | −3.5 pp vs non-skippers | Skipping signals disengagement, not conscientiousness |
| 13 | Gold accuracy as proxy | Strong correlation with agreement rate | Gold tasks are a valid quality signal for modelling |
