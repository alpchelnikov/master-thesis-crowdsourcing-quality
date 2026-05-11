# src/aggregation.py
"""
Answer-aggregation models for the crowdsourcing quality thesis.

Three models are provided:

  * ``majority_vote``                – non-parametric baseline.
  * ``DawidSkene`` (1979)            – EM with per-worker K×K confusion matrices.
  * ``MACE`` (Hovy et al., 2013)     – EM with per-worker (spam, θ) parameterisation.

Both EM models are **fully vectorised numpy**: the expensive E- and M-step
accumulations over the annotation table are expressed as `np.add.at`
broadcasted writes, which means a single EM iteration on the full
4.12 M-annotation dataset runs in well under a second on a laptop CPU.

The implementations here are the authoritative versions used in
``notebooks/03_Advanced_Models.ipynb``; the notebook imports the classes
from this module so that the two stay in sync.

References
----------
Dawid, A. P. and Skene, A. M. (1979). *Maximum likelihood estimation of
observer error-rates using the EM algorithm.* Applied Statistics 28(1),
20–28.

Hovy, D., Berg-Kirkpatrick, T., Vaswani, A. and Hovy, E. (2013).
*Learning whom to trust with MACE.* In Proceedings of NAACL-HLT.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

import numpy as np
import pandas as pd


# ── Majority Vote (non-EM baseline) ──────────────────────────────────────────

def majority_vote(
    annotations: pd.DataFrame,
    worker_col: str = "ozon_id",
    task_col: str = "task_id",
    answer_col: str = "user_ans",
) -> pd.Series:
    """
    Most-frequent-answer aggregation per task.

    Ties are broken in favour of the lower label (the behaviour of
    ``pd.Series.mode().iloc[0]``).  This is the reference implementation
    used by the studied platform; it is included here as the floor against
    which the EM-based aggregators are compared.

    Parameters
    ----------
    annotations : pd.DataFrame
        Long-format table with one row per (worker, task, answer) triple.
    worker_col, task_col, answer_col : str
        Column names.  ``answer_col`` must be 1-indexed integer labels.

    Returns
    -------
    pd.Series
        Majority label per task, indexed by task_id.
    """
    return (
        annotations
        .groupby(task_col)[answer_col]
        .agg(lambda x: x.mode().iloc[0])
        .rename("mv_label")
    )


# ── shared helpers for the two EM models ─────────────────────────────────────

@dataclass
class _EncodedAnnotations:
    """Integer-encoded annotation arrays ready for vectorised EM."""

    n_tasks: int
    n_workers: int
    task_idx: np.ndarray         # (n_ann,) task index per annotation
    worker_idx: np.ndarray       # (n_ann,) worker index per annotation
    answer_idx: np.ndarray       # (n_ann,) 0-based answer index
    unique_tasks: np.ndarray
    unique_workers: np.ndarray


def _encode_annotations(
    annotations: pd.DataFrame,
    worker_col: str,
    task_col: str,
    answer_col: str,
    n_classes: int,
) -> _EncodedAnnotations:
    """
    Drop rows with missing answers, map ids to dense integers, and validate
    that all answers fall within ``[1, n_classes]``.

    The studied platform uses 1-indexed labels.  Service-flag and out-of-range
    values must be filtered upstream (see ``data_prep.load_data`` and the
    ``user_ans`` exclusions described in Section 2.4 of the thesis).
    """
    ann = annotations[[worker_col, task_col, answer_col]].dropna()
    if len(ann) == 0:
        raise ValueError("No valid annotations to fit on.")

    answers_int = ann[answer_col].astype(np.int64).values
    if answers_int.min() < 1 or answers_int.max() > n_classes:
        bad = sorted(set(answers_int[(answers_int < 1) | (answers_int > n_classes)]))
        raise ValueError(
            f"Answer labels out of range for n_classes={n_classes}: found "
            f"{bad}. Expected integers in [1, {n_classes}]. "
            f"Filter the input (e.g. user_ans ∈ {{1, …, {n_classes}}}) before "
            f"fitting; the platform service flag user_ans=3 in particular "
            f"must be excluded, see data_prep.load_data()."
        )

    unique_tasks = np.asarray(ann[task_col].unique())
    unique_workers = np.asarray(ann[worker_col].unique())

    task_map = {t: i for i, t in enumerate(unique_tasks)}
    worker_map = {w: i for i, w in enumerate(unique_workers)}

    task_idx = np.fromiter(
        (task_map[t] for t in ann[task_col].values),
        dtype=np.int64, count=len(ann),
    )
    worker_idx = np.fromiter(
        (worker_map[w] for w in ann[worker_col].values),
        dtype=np.int64, count=len(ann),
    )
    answer_idx = answers_int - 1  # → 0-indexed for array math

    return _EncodedAnnotations(
        n_tasks=len(unique_tasks),
        n_workers=len(unique_workers),
        task_idx=task_idx,
        worker_idx=worker_idx,
        answer_idx=answer_idx,
        unique_tasks=unique_tasks,
        unique_workers=unique_workers,
    )


# ── Dawid–Skene ──────────────────────────────────────────────────────────────

class DawidSkene:
    """
    Dawid–Skene (1979) EM aggregator with per-worker confusion matrices.

    Each worker ``w`` is modelled by a K×K confusion matrix ``π_w`` where
    ``π_w[j, l] = P(worker w emits label l | true label is j)``.  True
    labels are latent; the EM algorithm alternates between estimating the
    posterior distribution over true labels (E-step) and re-estimating the
    confusion matrices and class priors (M-step).

    Implementation note
    -------------------
    The E- and M-step accumulations are vectorised.  Concretely:

      * the M-step builds ``cm[w, j, l]`` from a single ``np.add.at`` write
        per latent class (loop of length ``K``, not over annotations);
      * the E-step builds ``log_T[t, j]`` from a single ``np.add.at`` write
        over the entire annotation array.

    Both steps therefore run in O(|annotations|·K) without a Python-level
    loop over individual rows.  On the full 4.12 M-row dataset one
    iteration takes ≈ 0.4 s on a modern laptop CPU; the model converges
    in 10–30 iterations.

    Parameters
    ----------
    n_classes : int
        Number of annotation classes (default 2 for binary labelling).
    max_iter : int
        Maximum number of EM iterations.
    tol : float
        Convergence tolerance on the observed-data log-likelihood.
    """

    def __init__(self, n_classes: int = 2, max_iter: int = 100, tol: float = 1e-6):
        self.n_classes = n_classes
        self.max_iter = max_iter
        self.tol = tol

        # Fitted attributes (populated by fit)
        self.class_priors_: Optional[np.ndarray] = None      # (K,)
        self.confusion_matrices_: Optional[np.ndarray] = None  # (W, K, K)
        self.task_posteriors_: Optional[np.ndarray] = None    # (N, K)
        self.n_iter_: Optional[int] = None
        self.log_likelihood_: Optional[float] = None
        self._unique_tasks: Optional[np.ndarray] = None
        self._unique_workers: Optional[np.ndarray] = None

    def fit(
        self,
        annotations: pd.DataFrame,
        worker_col: str = "ozon_id",
        task_col: str = "task_id",
        answer_col: str = "user_ans",
    ) -> "DawidSkene":
        """
        Fit the model on a long-format annotation table.

        Parameters
        ----------
        annotations : pd.DataFrame
            Long-format table containing at minimum ``worker_col``,
            ``task_col`` and ``answer_col``.  Answers are expected to be
            1-indexed positive integers.
        """
        enc = _encode_annotations(annotations, worker_col, task_col, answer_col, self.n_classes)
        N, W, K = enc.n_tasks, enc.n_workers, self.n_classes
        ti, wi, ai = enc.task_idx, enc.worker_idx, enc.answer_idx

        # ── initialise T with majority-vote soft assignments ─────────────
        T = np.zeros((N, K))
        np.add.at(T, (ti, ai), 1.0)
        T /= T.sum(axis=1, keepdims=True) + 1e-12

        log_likelihood_prev = -np.inf
        n_iter_done = self.max_iter

        priors = T.mean(axis=0)
        cm = np.full((W, K, K), 1.0 / K)

        for it in range(self.max_iter):
            # ── M-step: priors and per-worker confusion matrices ────────
            priors = T.mean(axis=0)

            cm = np.full((W, K, K), 1e-10)
            # Build cm[w, j, ai] += T[ti, j] for each latent class j.
            # The inner loop is over K (not over annotations), so this
            # stays vectorised on the |annotations| dimension.
            for j in range(K):
                np.add.at(cm[:, j, :], (wi, ai), T[ti, j])
            cm /= cm.sum(axis=2, keepdims=True) + 1e-12

            # ── E-step: posterior over true labels per task ──────────────
            log_cm = np.log(cm + 1e-10)
            # contrib[n, j] = log P(observed answer ai[n] | true label j, worker wi[n])
            contrib = log_cm[wi, :, ai]                              # (n_ann, K)
            log_T = np.tile(np.log(priors + 1e-10), (N, 1))
            np.add.at(log_T, ti, contrib)

            # observed-data log-likelihood via log-sum-exp
            mx = log_T.max(axis=1, keepdims=True)
            log_likelihood = float(
                (mx.squeeze(1) + np.log(np.exp(log_T - mx).sum(axis=1) + 1e-12)).sum()
            )

            T = np.exp(log_T - mx)
            T /= T.sum(axis=1, keepdims=True) + 1e-12

            if abs(log_likelihood - log_likelihood_prev) < self.tol:
                n_iter_done = it + 1
                log_likelihood_prev = log_likelihood
                break
            log_likelihood_prev = log_likelihood

        self.class_priors_ = priors
        self.confusion_matrices_ = cm
        self.task_posteriors_ = T
        self.n_iter_ = n_iter_done
        self.log_likelihood_ = log_likelihood_prev
        self._unique_tasks = enc.unique_tasks
        self._unique_workers = enc.unique_workers
        return self

    def predict(self) -> pd.Series:
        """MAP label estimate per task (1-indexed, matching the raw data)."""
        if self.task_posteriors_ is None:
            raise RuntimeError("Call fit() before predict().")
        labels = self.task_posteriors_.argmax(axis=1) + 1
        return pd.Series(labels, index=self._unique_tasks, name="ds_label")

    def task_confidence(self) -> pd.Series:
        """Maximum posterior probability per task (used for confidence tiers)."""
        if self.task_posteriors_ is None:
            raise RuntimeError("Call fit() before task_confidence().")
        return pd.Series(
            self.task_posteriors_.max(axis=1),
            index=self._unique_tasks, name="ds_conf",
        )

    def worker_quality(self) -> pd.Series:
        """
        Scalar quality per worker = mean diagonal of the confusion matrix.

        This equals average per-class correctness, which is the natural
        scalar summary of a worker's confusion matrix under approximately
        balanced class priors.
        """
        if self.confusion_matrices_ is None:
            raise RuntimeError("Call fit() before worker_quality().")
        diag = np.einsum("wjj->w", self.confusion_matrices_) / self.n_classes
        return pd.Series(diag, index=self._unique_workers, name="ds_score")


# ── MACE ─────────────────────────────────────────────────────────────────────

class MACE:
    """
    Multi-Annotator Competence Estimation (Hovy et al., 2013).

    Each worker ``w`` is described by

      * a spam probability ``σ_w ∈ (0, 1)``,
      * a marginal answer distribution ``θ_w`` over the K classes.

    The observation model for a worker producing answer ``a`` on a task
    whose true label is ``j`` is

        P(a | j, w) = σ_w · θ_w[a]  +  (1 − σ_w) · 1[a = j]

    i.e. with probability ``σ_w`` the worker is spamming and emits from
    ``θ_w``; otherwise the worker reveals the true label.

    Worker competence is reported as ``1 − σ_w``, clipped to
    ``[0.01, 0.99]`` for numerical stability.

    Implementation note
    -------------------
    Like ``DawidSkene``, all E- and M-step accumulations are vectorised:
    the responsibility ``r_spam`` is computed in a single broadcasted
    expression over the entire annotation array, and the per-worker
    aggregations use ``np.add.at`` writes.  One iteration on the full
    4.12 M-row dataset runs in under a second on a laptop CPU.

    Parameters
    ----------
    n_classes : int
        Number of annotation classes.
    max_iter : int
        Maximum EM iterations.
    tol : float
        Convergence tolerance on the observed-data log-likelihood.
    """

    def __init__(self, n_classes: int = 2, max_iter: int = 100, tol: float = 1e-6):
        self.n_classes = n_classes
        self.max_iter = max_iter
        self.tol = tol

        # Fitted attributes
        self.spam_probs_: Optional[np.ndarray] = None        # (W,)
        self.theta_: Optional[np.ndarray] = None             # (W, K)
        self.task_posteriors_: Optional[np.ndarray] = None   # (N, K)
        self.n_iter_: Optional[int] = None
        self.log_likelihood_: Optional[float] = None
        self._unique_tasks: Optional[np.ndarray] = None
        self._unique_workers: Optional[np.ndarray] = None

    def fit(
        self,
        annotations: pd.DataFrame,
        worker_col: str = "ozon_id",
        task_col: str = "task_id",
        answer_col: str = "user_ans",
    ) -> "MACE":
        """Fit MACE via EM. Parameters mirror :class:`DawidSkene`."""
        enc = _encode_annotations(annotations, worker_col, task_col, answer_col, self.n_classes)
        N, W, K = enc.n_tasks, enc.n_workers, self.n_classes
        ti, wi, ai = enc.task_idx, enc.worker_idx, enc.answer_idx

        spam = np.full(W, 0.5)
        theta = np.ones((W, K)) / K
        T = np.ones((N, K)) / K

        log_likelihood_prev = -np.inf
        n_iter_done = self.max_iter

        for it in range(self.max_iter):
            # ── E-step ───────────────────────────────────────────────────
            # spam_part[n]  = σ_{w(n)} · θ_{w(n)}[a(n)]    -> shape (n_ann, 1)
            # comp_part[n,j]= (1 − σ_{w(n)}) · 1[a(n) = j]  -> shape (n_ann, K)
            spam_part = (spam[wi] * theta[wi, ai])[:, None]
            indicator_aj = (np.arange(K)[None, :] == ai[:, None]).astype(float)
            comp_part = (1.0 - spam[wi])[:, None] * indicator_aj
            obs_lik = spam_part + comp_part                              # (n_ann, K)

            log_obs = np.log(obs_lik + 1e-12)
            log_T = np.zeros((N, K))
            np.add.at(log_T, ti, log_obs)

            mx = log_T.max(axis=1, keepdims=True)
            log_likelihood = float(
                (mx.squeeze(1) + np.log(np.exp(log_T - mx).sum(axis=1) + 1e-12)).sum()
            )

            T = np.exp(log_T - mx)
            T /= T.sum(axis=1, keepdims=True) + 1e-12

            # ── M-step ───────────────────────────────────────────────────
            # Posterior responsibility that an observation was spam
            weighted_obs = (T[ti] * obs_lik).sum(axis=1) + 1e-12
            r_spam = (T[ti] * spam_part).sum(axis=1) / weighted_obs       # (n_ann,)

            spam_num = np.zeros(W)
            spam_den = np.zeros(W)
            np.add.at(spam_num, wi, r_spam)
            np.add.at(spam_den, wi, 1.0)
            spam = np.clip(spam_num / (spam_den + 1e-12), 0.01, 0.99)

            new_theta = np.full((W, K), 1e-10)
            np.add.at(new_theta, (wi, ai), r_spam)
            theta = new_theta / (new_theta.sum(axis=1, keepdims=True) + 1e-12)

            if abs(log_likelihood - log_likelihood_prev) < self.tol:
                n_iter_done = it + 1
                log_likelihood_prev = log_likelihood
                break
            log_likelihood_prev = log_likelihood

        self.spam_probs_ = spam
        self.theta_ = theta
        self.task_posteriors_ = T
        self.n_iter_ = n_iter_done
        self.log_likelihood_ = log_likelihood_prev
        self._unique_tasks = enc.unique_tasks
        self._unique_workers = enc.unique_workers
        return self

    def predict(self) -> pd.Series:
        """MAP label estimate per task (1-indexed)."""
        if self.task_posteriors_ is None:
            raise RuntimeError("Call fit() before predict().")
        labels = self.task_posteriors_.argmax(axis=1) + 1
        return pd.Series(labels, index=self._unique_tasks, name="mace_label")

    def task_confidence(self) -> pd.Series:
        """Maximum posterior probability per task."""
        if self.task_posteriors_ is None:
            raise RuntimeError("Call fit() before task_confidence().")
        return pd.Series(
            self.task_posteriors_.max(axis=1),
            index=self._unique_tasks, name="mace_conf",
        )

    def worker_quality(self) -> pd.Series:
        """Per-worker competence score (1 − spam probability)."""
        if self.spam_probs_ is None:
            raise RuntimeError("Call fit() before worker_quality().")
        return pd.Series(
            1.0 - self.spam_probs_,
            index=self._unique_workers, name="mace_score",
        )
