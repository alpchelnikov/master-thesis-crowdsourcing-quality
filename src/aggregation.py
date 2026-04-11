# src/aggregation.py
"""
Answer aggregation models for crowdsourcing quality evaluation.

Implements Dawid-Skene and MACE from scratch, as described in notebook 03.
Both models are implemented as simple classes with fit() / predict() methods
to keep the interface familiar for scikit-learn users.
"""

import numpy as np
import pandas as pd
from typing import Optional


# ── Majority Vote (baseline) ──────────────────────────────────────────────────

def majority_vote(
    annotations: pd.DataFrame,
    worker_col: str = "ozon_id",
    task_col: str = "task_id",
    answer_col: str = "user_ans",
) -> pd.Series:
    """
    Simple majority vote aggregation.

    Parameters
    ----------
    annotations : pd.DataFrame
        Long-format annotation table (one row per worker-task answer).
    worker_col, task_col, answer_col : str
        Column names.

    Returns
    -------
    pd.Series
        Most frequent answer per task, indexed by task_id.
    """
    return (
        annotations
        .groupby(task_col)[answer_col]
        .agg(lambda x: x.mode().iloc[0])
    )


# ── Dawid-Skene ───────────────────────────────────────────────────────────────

class DawidSkene:
    """
    Dawid-Skene (1979) EM model for label aggregation.

    Each worker is modelled by a confusion matrix.  The EM algorithm
    alternates between:
      E-step: estimate the probability of each true label given annotations
      M-step: update worker confusion matrices and class priors

    Implemented from scratch as in notebook 03.

    Parameters
    ----------
    n_classes : int
        Number of annotation classes (default 2 for binary tasks).
    max_iter : int
        Maximum EM iterations.
    tol : float
        Convergence tolerance on the log-likelihood.
    """

    def __init__(self, n_classes: int = 2, max_iter: int = 100, tol: float = 1e-6):
        self.n_classes = n_classes
        self.max_iter = max_iter
        self.tol = tol

        # Fitted attributes
        self.class_priors_: Optional[np.ndarray] = None       # (n_classes,)
        self.error_rates_: Optional[dict] = None              # worker -> (n_classes, n_classes)
        self.T_: Optional[np.ndarray] = None                  # (n_tasks, n_classes) posterior

    def fit(
        self,
        annotations: pd.DataFrame,
        worker_col: str = "ozon_id",
        task_col: str = "task_id",
        answer_col: str = "user_ans",
    ) -> "DawidSkene":
        """
        Fit the model using EM on the annotation table.

        Parameters
        ----------
        annotations : pd.DataFrame
            Long-format table with at least worker_col, task_col, answer_col.
            Answers are expected to be integers starting from 1.
        """
        # Map answers to 0-based integer indices
        ann = annotations[[worker_col, task_col, answer_col]].dropna().copy()
        ann[answer_col] = ann[answer_col].astype(int) - 1   # 1,2 -> 0,1

        tasks = ann[task_col].unique()
        workers = ann[worker_col].unique()
        n_tasks = len(tasks)
        n_workers = len(workers)

        task_idx = {t: i for i, t in enumerate(tasks)}
        worker_idx = {w: i for i, w in enumerate(workers)}

        ann["_t"] = ann[task_col].map(task_idx)
        ann["_w"] = ann[worker_col].map(worker_idx)
        ann["_a"] = ann[answer_col]

        K = self.n_classes

        # Initialise T with majority vote
        counts = np.zeros((n_tasks, K))
        for _, row in ann.iterrows():
            counts[int(row["_t"]), int(row["_a"])] += 1
        T = counts / (counts.sum(axis=1, keepdims=True) + 1e-10)

        log_likelihood_prev = -np.inf

        for iteration in range(self.max_iter):
            # ── M-step ────────────────────────────────────────────────────────
            priors = T.mean(axis=0)                             # (K,)

            # Worker confusion matrices: pi[w, j, l] = P(annotate l | true j)
            pi = np.zeros((n_workers, K, K)) + 1e-10
            for _, row in ann.iterrows():
                w, t, a = int(row["_w"]), int(row["_t"]), int(row["_a"])
                pi[w] += T[t, :, None] * (np.arange(K) == a)
            pi /= pi.sum(axis=2, keepdims=True)

            # ── E-step ────────────────────────────────────────────────────────
            log_T = np.log(priors + 1e-10)[None, :]            # (1, K)
            for _, row in ann.iterrows():
                w, t, a = int(row["_w"]), int(row["_t"]), int(row["_a"])
                log_T[0] += np.log(pi[w, :, a] + 1e-10)

            # Vectorised E-step
            log_T = np.tile(np.log(priors + 1e-10), (n_tasks, 1))
            for _, row in ann.iterrows():
                w, t, a = int(row["_w"]), int(row["_t"]), int(row["_a"])
                log_T[t] += np.log(pi[w, :, a] + 1e-10)

            # Normalise
            log_T -= log_T.max(axis=1, keepdims=True)
            T = np.exp(log_T)
            T /= T.sum(axis=1, keepdims=True)

            # ── Convergence check ─────────────────────────────────────────────
            log_likelihood = (T * log_T).sum()
            if abs(log_likelihood - log_likelihood_prev) < self.tol:
                break
            log_likelihood_prev = log_likelihood

        self.class_priors_ = priors
        self.error_rates_ = {
            workers[i]: pi[i] for i in range(n_workers)
        }
        self.T_ = T
        self._tasks = tasks
        self._task_idx = task_idx

        return self

    def predict(self) -> pd.Series:
        """
        Return the MAP label estimate for each task (1-indexed to match raw data).

        Returns
        -------
        pd.Series
            Predicted label per task, indexed by task_id.
        """
        if self.T_ is None:
            raise RuntimeError("Call fit() before predict().")
        labels = self.T_.argmax(axis=1) + 1   # back to 1-based
        return pd.Series(labels, index=self._tasks, name="ds_label")

    def worker_quality(self) -> pd.Series:
        """
        Compute a scalar quality score per worker as the mean diagonal of
        their confusion matrix (average correctness across classes).

        Returns
        -------
        pd.Series
            Quality score in [0, 1], indexed by ozon_id.
        """
        if self.error_rates_ is None:
            raise RuntimeError("Call fit() before worker_quality().")
        scores = {
            w: np.diag(pi).mean()
            for w, pi in self.error_rates_.items()
        }
        return pd.Series(scores, name="ds_quality")


# ── MACE ─────────────────────────────────────────────────────────────────────

class MACE:
    """
    Multi-Annotator Competence Estimation (Hovy et al., 2013).

    Models each worker as either a *spammer* (answers randomly) or a *competent*
    annotator (answers from a per-worker distribution).  The mixing weight is
    the worker's spam probability.

    Implemented from scratch as in notebook 03.

    Parameters
    ----------
    n_classes : int
        Number of annotation classes.
    max_iter : int
        Maximum EM iterations.
    tol : float
        Convergence tolerance.
    """

    def __init__(self, n_classes: int = 2, max_iter: int = 100, tol: float = 1e-6):
        self.n_classes = n_classes
        self.max_iter = max_iter
        self.tol = tol

        self.spam_probs_: Optional[pd.Series] = None    # P(spam) per worker
        self.competence_: Optional[pd.Series] = None    # 1 - spam_prob
        self.T_: Optional[np.ndarray] = None            # task posteriors

    def fit(
        self,
        annotations: pd.DataFrame,
        worker_col: str = "ozon_id",
        task_col: str = "task_id",
        answer_col: str = "user_ans",
    ) -> "MACE":
        """
        Fit MACE via EM.

        Parameters
        ----------
        annotations : pd.DataFrame
            Long-format annotation table.  Answers should be integers ≥ 1.
        """
        ann = annotations[[worker_col, task_col, answer_col]].dropna().copy()
        ann[answer_col] = ann[answer_col].astype(int) - 1

        tasks = ann[task_col].unique()
        workers = ann[worker_col].unique()
        n_tasks = len(tasks)
        n_workers = len(workers)
        K = self.n_classes

        task_idx = {t: i for i, t in enumerate(tasks)}
        worker_idx = {w: i for i, w in enumerate(workers)}

        ann["_t"] = ann[task_col].map(task_idx)
        ann["_w"] = ann[worker_col].map(worker_idx)
        ann["_a"] = ann[answer_col]

        # Initialise
        spam = np.full(n_workers, 0.5)       # P(spam) per worker
        theta = np.ones((n_workers, K)) / K  # competent answer distribution
        T = np.ones((n_tasks, K)) / K        # task label posteriors

        log_likelihood_prev = -np.inf

        for _ in range(self.max_iter):
            # ── E-step ────────────────────────────────────────────────────────
            # Recompute T from current parameters
            log_T = np.zeros((n_tasks, K))
            for _, row in ann.iterrows():
                w, t, a = int(row["_w"]), int(row["_t"]), int(row["_a"])
                # P(a | spam) = 1/K;  P(a | not spam) = theta[w, a]
                p_obs = spam[w] / K + (1 - spam[w]) * theta[w, a]
                log_T[t] += np.log(p_obs + 1e-10)

            log_T -= log_T.max(axis=1, keepdims=True)
            T = np.exp(log_T)
            T /= T.sum(axis=1, keepdims=True)

            # ── M-step ────────────────────────────────────────────────────────
            # Update theta: competent answer distribution per worker
            new_theta = np.zeros((n_workers, K)) + 1e-10
            for _, row in ann.iterrows():
                w, t, a = int(row["_w"]), int(row["_t"]), int(row["_a"])
                new_theta[w, a] += T[t, a]
            theta = new_theta / new_theta.sum(axis=1, keepdims=True)

            # Update spam: fraction of annotation weight explained by spam
            new_spam = np.zeros(n_workers)
            spam_denom = np.zeros(n_workers)
            for _, row in ann.iterrows():
                w, t, a = int(row["_w"]), int(row["_t"]), int(row["_a"])
                p_spam_part = spam[w] / K
                p_comp_part = (1 - spam[w]) * theta[w, a]
                total = p_spam_part + p_comp_part + 1e-10
                new_spam[w] += p_spam_part / total
                spam_denom[w] += 1
            spam = new_spam / (spam_denom + 1e-10)
            spam = np.clip(spam, 0.01, 0.99)

            # ── Convergence ───────────────────────────────────────────────────
            log_likelihood = (T * log_T).sum()
            if abs(log_likelihood - log_likelihood_prev) < self.tol:
                break
            log_likelihood_prev = log_likelihood

        self.spam_probs_ = pd.Series(spam, index=workers, name="spam_prob")
        self.competence_ = (1 - self.spam_probs_).rename("mace_competence")
        self.T_ = T
        self._tasks = tasks

        return self

    def predict(self) -> pd.Series:
        """
        MAP label estimate per task (1-indexed).

        Returns
        -------
        pd.Series
            Predicted label per task, indexed by task_id.
        """
        if self.T_ is None:
            raise RuntimeError("Call fit() before predict().")
        labels = self.T_.argmax(axis=1) + 1
        return pd.Series(labels, index=self._tasks, name="mace_label")

    def worker_quality(self) -> pd.Series:
        """
        Return competence score (1 - spam probability) per worker.

        Returns
        -------
        pd.Series
            Competence in [0, 1], indexed by ozon_id.
        """
        if self.competence_ is None:
            raise RuntimeError("Call fit() before worker_quality().")
        return self.competence_
