"""
evaluation.py
=============

Three-step statistical algorithm for selecting the best-performing model
among k >= 3 candidate machine learning models evaluated under repeated
Monte Carlo Cross-Validation (MCCV).

Pipeline
--------
Step 1 — Temporal Stability
    For each model, a paired t-test compares Test-split vs. Validation-split
    scores across N MCCV iterations. Models with no significant train/val
    drift (p >= alpha) are retained as M_temporal. If fewer than
    `min_stable_models` survive, the pipeline falls back to the top-k models
    ranked by mean validation score.

Step 2 — Global Comparison (Friedman test)
    A Friedman test is run on the validation-split scores of M_temporal
    (= M_candidate) across all MCCV iterations. If the omnibus test fails to
    reject H0 (p >= alpha), the models are statistically equivalent and the
    one with the highest mean validation score is returned directly
    (early return — no post-hoc needed).

Step 3 — Pairwise Selection (Conover post-hoc, compare-to-best)
    If Step 2 rejects H0, the top-ranked model is compared against every
    other candidate using the Conover post-hoc test (Conover & Iman, 1979),
    a rank-based test appropriate as a follow-up to Friedman. A Bonferroni
    correction for k-1 compare-to-best comparisons (alpha / (k-1)) is
    applied, analogous to a Dunnett-style "compare all to control" scheme
    rather than full all-pairs Bonferroni. The highest-scoring model that is
    NOT significantly worse than the top-ranked model is returned as M_best.

References
----------
Friedman, M. (1937). The use of ranks to avoid the assumption of normality
    implicit in the analysis of variance. Journal of the American
    Statistical Association, 32(200), 675-701.

Conover, W. J., & Iman, R. L. (1979). On multiple-comparisons procedures.
    Technical Report LA-7677-MS, Los Alamos Scientific Laboratory.

Demsar, J. (2006). Statistical comparisons of classifiers over multiple
    data sets. Journal of Machine Learning Research, 7, 1-30.

Minimum sample size (k)
------------------------
k >= 3 is a hard requirement. The Friedman test is degenerate at k = 2
(equivalent to a sign test on ranks and redundant with Step 1's paired
t-test), and a post-hoc procedure is not meaningful with only one
comparison. k >= 5 is recommended for stable rank estimates and a
correction (alpha / (k-1)) that is not overly conservative.
"""

from __future__ import annotations

import warnings
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Sequence

import numpy as np
import pandas as pd
from scipy import stats

warnings.filterwarnings("ignore")

MIN_MODELS_HARD = 3
MIN_MODELS_RECOMMENDED = 5


class InsufficientModelsError(ValueError):
    """Raised when fewer than MIN_MODELS_HARD models are supplied."""


@dataclass
class SelectionResult:
    """Container for the full output of the selection pipeline."""

    m_best: str
    m_temporal: List[str]
    fallback_used: bool
    friedman_stat: float
    friedman_p: float
    early_return: bool
    ranked_candidates: List[str]
    conover_pvalues: Dict[str, float] = field(default_factory=dict)
    table_temporal_stability: Optional[pd.DataFrame] = None
    table_selection_summary: Optional[pd.DataFrame] = None


def paired_t_test(x: np.ndarray, y: np.ndarray):
    """Paired t-test between two equal-length arrays (Test vs. Val scores)."""
    return stats.ttest_rel(x, y)


def friedman_test(data_matrix: np.ndarray):
    """Friedman test across >=2 groups. data_matrix: shape (k, N)."""
    return stats.friedmanchisquare(*data_matrix)


def conover_posthoc(rank_matrix: np.ndarray, top_idx: int, other_idx: int,
                     n_iterations: int, k: int) -> float:
    """
    Conover post-hoc pairwise test (Conover & Iman, 1979), used as a
    follow-up to a significant Friedman test. Compares the rank sums of
    two models (columns top_idx and other_idx) in rank_matrix, shape
    (n_iterations, k).
    """
    a1 = np.sum(rank_matrix ** 2)
    c1 = n_iterations * k * (k + 1) ** 2 / 4
    r_top = np.sum(rank_matrix[:, top_idx])
    r_other = np.sum(rank_matrix[:, other_idx])
    denom = np.sqrt((2 * n_iterations * (a1 - c1)) / ((n_iterations - 1) * (k - 1)))
    t_stat = np.abs(r_top - r_other) / denom
    df = (n_iterations - 1) * (k - 1)
    return 2 * (1 - stats.t.cdf(t_stat, df))


def validate_models(models: Sequence[str]) -> None:
    """Enforce the minimum-k requirement described in the module docstring."""
    k = len(models)
    if k < MIN_MODELS_HARD:
        raise InsufficientModelsError(
            f"At least {MIN_MODELS_HARD} models are required for this "
            f"pipeline (Friedman + post-hoc are degenerate below k=3). "
            f"Received k={k}: {list(models)}."
        )
    if k < MIN_MODELS_RECOMMENDED:
        warnings.warn(
            f"k={k} models supplied. k >= {MIN_MODELS_RECOMMENDED} is "
            f"recommended for stable rank estimates and a non-conservative "
            f"compare-to-best correction (alpha / (k-1)).",
            stacklevel=2,
        )


def build_tradeoff_table(model_data: Dict[str, Dict[str, Dict[str, np.ndarray]]],
                          models: Sequence[str],
                          metrics: Sequence[str],
                          metric_labels: Optional[Dict[str, str]] = None) -> pd.DataFrame:
    """
    Table 1 — Test/Validation tradeoff per model per metric.

    model_data[model]['test'][metric] / model_data[model]['val'][metric]
    must be equal-length arrays of per-iteration scores.
    """
    metric_labels = metric_labels or {m: m for m in metrics}
    rows = []
    for model in models:
        row = {"Model": model}
        for metric in metrics:
            label = metric_labels.get(metric, metric)
            test_vals = np.asarray(model_data[model]["test"][metric])
            val_vals = np.asarray(model_data[model]["val"][metric])
            delta_pct = float(np.mean(test_vals - val_vals) * 100)
            _, p_val = paired_t_test(test_vals, val_vals)
            row[f"Delta_{label}_pct"] = round(delta_pct, 4)
            row[f"PValue_{label}"] = p_val
        rows.append(row)
    return pd.DataFrame(rows)


def select_best_model(
    model_data: Dict[str, Dict[str, Dict[str, np.ndarray]]],
    models: Sequence[str],
    metric: str = "AUC",
    n_iterations: int = 100,
    alpha: float = 0.05,
    min_stable_models: int = 3,
) -> SelectionResult:
    validate_models(models)
    models = list(models)

    mean_val_all = {m: float(np.mean(model_data[m]["val"][metric])) for m in models}

    # ---------- Step 1: Temporal Stability ----------
    m_temporal = []
    for model in models:
        _, p_val = paired_t_test(model_data[model]["test"][metric],
                                  model_data[model]["val"][metric])
        if p_val >= alpha:
            m_temporal.append(model)

    fallback_used = False
    if len(m_temporal) < min_stable_models:
        fallback_used = True
        m_temporal = sorted(models, key=lambda x: mean_val_all[x], reverse=True)[
            : max(min_stable_models, MIN_MODELS_HARD)
        ]

    # ---------- Step 2: Global Comparison (Friedman) ----------
    m_candidate = m_temporal
    friedman_data = [model_data[m]["val"][metric] for m in m_candidate]

    if len(friedman_data) > 1:
        friedman_stat, friedman_p = friedman_test(np.array(friedman_data))
    else:
        friedman_stat, friedman_p = np.nan, 1.0

    mean_val = {m: mean_val_all[m] for m in m_candidate}
    ranked_candidates = sorted(m_candidate, key=lambda x: mean_val[x], reverse=True)
    k = len(ranked_candidates)

    conover_pvalues: Dict[str, float] = {}
    early_return = False

    if friedman_p >= alpha and not np.isnan(friedman_stat):
        # Fail to reject H0: models are statistically equivalent.
        m_best = ranked_candidates[0]
        early_return = True
    else:
        # ---------- Step 3: Pairwise Selection (Conover post-hoc) ----------
        if k == 1:
            m_best = ranked_candidates[0]
        else:
            top_model = ranked_candidates[0]
            all_val_scores = [model_data[m]["val"][metric] for m in ranked_candidates]
            rank_matrix = np.zeros((n_iterations, k))
            for i in range(n_iterations):
                iter_vals = [all_val_scores[j][i] for j in range(k)]
                rank_matrix[i] = stats.rankdata(iter_vals)

            alpha_adj = alpha / (k - 1)
            top_idx = 0
            survivors = [top_model]
            for idx, model in enumerate(ranked_candidates[1:], 1):
                p_val = conover_posthoc(rank_matrix, top_idx, idx, n_iterations, k)
                conover_pvalues[f"{top_model}_vs_{model}"] = p_val
                if p_val >= alpha_adj:
                    survivors.append(model)

            m_best = max(survivors, key=lambda m: mean_val[m])

    table_selection_summary = _build_summary_table(
        m_temporal, fallback_used, friedman_stat, friedman_p,
        early_return, conover_pvalues, alpha, m_best,
    )

    return SelectionResult(
        m_best=m_best,
        m_temporal=m_temporal,
        fallback_used=fallback_used,
        friedman_stat=friedman_stat,
        friedman_p=friedman_p,
        early_return=early_return,
        ranked_candidates=ranked_candidates,
        conover_pvalues=conover_pvalues,
        table_selection_summary=table_selection_summary,
    )


def _build_summary_table(m_temporal, fallback_used, friedman_stat, friedman_p,
                          early_return, conover_pvalues, alpha, m_best) -> pd.DataFrame:
    rows = [
        {"Item": "Step 1: Temporal Stability", "Value": ""},
        {"Item": "Models retained (M_temporal)", "Value": str(m_temporal)},
        {"Item": "Fallback triggered (< min_stable_models)", "Value": str(fallback_used)},
        {"Item": "", "Value": ""},
        {"Item": "Step 2: Friedman test among M_candidate", "Value": ""},
        {"Item": "Chi-square", "Value": round(friedman_stat, 4) if not np.isnan(friedman_stat) else "NA"},
        {"Item": "P-value", "Value": f"{friedman_p:.4e}{'*' if friedman_p < alpha else ''}"},
        {"Item": "Early return (models equivalent)", "Value": str(early_return)},
        {"Item": "", "Value": ""},
        {"Item": "Step 3: Conover post-hoc test", "Value": "N/A (early return)" if early_return else ""},
    ]
    for label, p_val in conover_pvalues.items():
        rows.append({"Item": f"P-Value {label}", "Value": f"{p_val:.4e}"})
    rows += [{"Item": "", "Value": ""}, {"Item": "M_best", "Value": m_best}]
    return pd.DataFrame(rows)
