# Dataset Description

The raw data files are **not included** in this repository. They contain proprietary annotation data from a production crowdsourcing platform and cannot be redistributed.

The notebooks in `notebooks/` are designed to run against a CSV file with the schema described below. A 100 000-row sample (`sample_data.csv`) is provided for development and review purposes.

---

## Schema

Each row represents **one worker's answer to one task**.

| Column | Type | Description |
|---|---|---|
| `task_id` | int | Unique task identifier |
| `pool_id` | int | Pool the task belongs to |
| `project_id` | int | Project the pool belongs to (5 projects total) |
| `ozon_id` | int | Worker identifier |
| `page_id` | str | Page shown to the worker (may group multiple tasks) |
| `price` | float | Payment amount for the task in roubles (0 for unpaid/internal tasks) |
| `created_at` | datetime | Timestamp when the page was opened |
| `finished_at` | datetime | Timestamp when the page was submitted |
| `skipped` | bool | Whether the worker skipped this task |
| `task_type` | int | `0` = regular task, `1` = gold (control) task, `2` = training task |
| `pool_type` | int | `0` = regular pool, `1` = rehabilitation pool, `3` = training pool |
| `task_ans` | float | Ground-truth label (available for gold and training tasks; on regular tasks reflects platform majority vote) |
| `user_ans` | float | Worker's answer. `1` and `2` are valid annotation classes. **`3` is a service flag for malformed tasks — exclude from analysis.** `NaN` means the task was skipped. |
| `tasks_per_page` | int | Number of tasks on this page. **Timing is per page, not per task** — divide page duration by this value for per-task timing. |
| `overlap` | int | Number of workers assigned to the same task |

---

## Key domain notes

**Pool types:**
- `pool_type = 0` — Regular production work. Workers are paid per task. Contains both regular tasks and injected gold (control) tasks (~4.2 % of answers).
- `pool_type = 1` — Rehabilitation pool. Triggered automatically when a worker makes ≥ 2 errors in their last 10 gold tasks. Contains only gold tasks.
- `pool_type = 3` — Training pool. Workers complete training tasks before entering a project. Contains only training tasks.

**Task types and ground truth:**
- `task_type = 1` (gold tasks) provide independent ground truth (`task_ans` is verified by the platform).
- `task_type = 0` (regular tasks): `task_ans` reflects the platform's own majority vote aggregation. Using this as ground truth when evaluating aggregation models is circular — avoid it.

**Timing:**
- `finished_at - created_at` gives page-level duration.
- Per-task time = `page_duration_sec / tasks_per_page`.
- `finished_at` is missing for ~7 189 rows (pages never submitted).

---

## Full dataset statistics

| Metric | Value |
|---|---|
| Total rows | ~4.12 million |
| Unique workers | 14 452 |
| Unique projects | 5 |
| Time span | ~1 month |
| Missing `user_ans` (skipped) | 4.14% |
| Missing `task_ans` | 0.15% |

---

## Loading the data

```python
import pandas as pd

df = pd.read_csv("data.csv", index_col=0)
df["created_at"]  = pd.to_datetime(df["created_at"])
df["finished_at"] = pd.to_datetime(df["finished_at"])

# Exclude service flag
df = df[df["user_ans"] != 3]
```
