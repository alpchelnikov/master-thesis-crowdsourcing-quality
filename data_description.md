# Data Description

> The raw dataset is proprietary and is **not distributed** with this repository.
> This file documents its structure for reproducibility and peer review.
> A 100 000-row sample (`sample_data.csv`) is included for code review purposes.

---

## Source

Data was collected from a production crowdsourcing platform used for binary
product-catalogue annotation (e-commerce domain). The observation window
covers approximately **one calendar month** of platform activity.

## Scale

| Dimension | Value |
|---|---|
| Total annotation rows | ~4.12 million |
| Unique workers (`ozon_id`) | 14 452 |
| Unique projects (`project_id`) | 5 (IDs: 575, 576, 577, 578, 581) |
| Unique pools (`pool_id`) | varies per project |
| Unique tasks (`task_id`) | varies per project |

---

## Schema

Each row represents **one worker's answer to one task**.

| Column | Type | Description |
|---|---|---|
| `task_id` | int | Unique task identifier |
| `pool_id` | int | Pool the task belongs to |
| `project_id` | int | Project (5 total: 575, 576, 577, 578, 581) |
| `ozon_id` | int | Worker identifier |
| `page_id` | str | Page shown to the worker (groups 1–3 tasks) |
| `price` | float | Payment per task in roubles (0 = unpaid / internal) |
| `created_at` | datetime | Timestamp when page was opened |
| `finished_at` | datetime | Timestamp when page was submitted (NaN if abandoned) |
| `skipped` | bool | Whether worker skipped this specific task |
| `task_type` | int | 0 = regular, 1 = gold (control), 2 = training |
| `pool_type` | int | 0 = regular, 1 = rehabilitation, 3 = training |
| `task_ans` | float | Ground-truth or majority-vote label (see note below) |
| `user_ans` | float | Worker's answer (see valid values below) |
| `tasks_per_page` | int | Tasks per page (1–3 depending on pool type) |
| `overlap` | int | Workers assigned to the same task |

---

## Domain-critical notes

### Pool types

| Value | Name | Description |
|---|---|---|
| 0 | Regular | Production annotation; workers are paid; contains both regular and gold tasks |
| 1 | Rehabilitation | Triggered when worker makes ≥ 2 errors in last 10 gold tasks; gold tasks only |
| 3 | Training | Onboarding pool before entering a project; training tasks only |

### Task types

| Value | Name | Description |
|---|---|---|
| 0 | Regular | Actual labelling work; `task_ans` = platform majority vote |
| 1 | Gold (control) | Known answer verified by platform; `task_ans` = true label |
| 2 | Training | Used only in training pools; `task_ans` = true label |

### `task_ans` — two different meanings

`task_ans` has **different semantics** depending on `task_type`:

- **Gold tasks (`task_type = 1`):** `task_ans` is an **independent verified label** — true ground truth.
  This is the correct external criterion for evaluating worker and model quality.

- **Regular tasks (`task_type = 0`):** `task_ans` is the **platform's majority-vote aggregation**.
  Using this as ground truth when benchmarking aggregation models creates a **circular comparison**
  and inflates performance estimates. Avoided throughout this thesis.

### `user_ans` — valid values and exclusion

| Value | Meaning |
|---|---|
| 1 | Valid annotation class (most common) |
| 2 | Valid annotation class |
| 3 | **Service flag — malformed page; not a valid class. MUST be excluded.** |
| 4 | Valid annotation class (rare) |
| NaN | Task was skipped |

All correctness analyses exclude rows where `user_ans == 3`.

### Timing

Timestamps are at the **page level**, not the task level.
Per-task duration is approximated as:

```
per_task_sec = (finished_at - created_at).total_seconds() / tasks_per_page
```

Raw page duration is **not comparable** across pool types without this normalisation,
because pool types differ in their page size (1–3 tasks/page).

---

## Pool type × task type constraints

These constraints are verified in notebook 01_EDA Section 2:

| Pool type | Contains task types |
|---|---|
| Regular (0) | Regular (0) + Gold (1) — gold tasks are ~4.2% of regular-pool answers |
| Rehabilitation (1) | Gold (1) only |
| Training (3) | Training (2) only |

---

## Missing values

| Column | Missing (full dataset) | Notes |
|---|---|---|
| `finished_at` | ~7 189 rows (0.17%) | Pages never submitted; no duration computable |
| `task_ans` | ~6 218 rows (0.15%) | Tasks without available ground truth |
| `user_ans` | ~170 693 rows (4.14%) | Skipped tasks; regular pool is the primary driver (4.5% skip rate) |

---

## Class balance (regular pool, gold tasks)

Annotation classes are 1 and 2 (binary task). Class 1 is majority across all projects:

| Project | Class 1 share | Class 2 share |
|---|---|---|
| 575 | ~65% | ~35% |
| 576 | ~60% | ~40% |
| 577 | ~73% | ~27% |
| 578 | ~58% | ~42% |
| 581 | ~63% | ~37% |

A trivial "always predict class 1" baseline achieves 58–73% accuracy — important
context when interpreting worker and model performance.
