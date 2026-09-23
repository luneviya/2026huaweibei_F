"""Shared metrics used by every candidate scheme."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from scipy.stats import spearmanr


@dataclass(frozen=True)
class MultiOutputMetrics:
    standardized_rmse: np.ndarray
    standardized_mae: np.ndarray
    spearman: np.ndarray

    @property
    def mean_standardized_rmse(self) -> float:
        return float(np.mean(self.standardized_rmse))

    @property
    def mean_spearman(self) -> float:
        return float(np.nanmean(self.spearman))


def multioutput_metrics(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    *,
    training_scale: np.ndarray,
) -> MultiOutputMetrics:
    """Calculate per-task metrics using scales estimated from training data."""

    observed = np.asarray(y_true, dtype=float)
    predicted = np.asarray(y_pred, dtype=float)
    scale = np.asarray(training_scale, dtype=float)
    if observed.shape != predicted.shape or observed.ndim != 2:
        raise ValueError("y_true and y_pred must have the same two-dimensional shape.")
    if scale.shape != (observed.shape[1],) or np.any(scale <= 0):
        raise ValueError("training_scale must be positive and match the number of tasks.")

    error = predicted - observed
    rmse = np.sqrt(np.mean(error**2, axis=0)) / scale
    mae = np.mean(np.abs(error), axis=0) / scale
    rank_correlation = np.array(
        [spearmanr(observed[:, j], predicted[:, j]).statistic for j in range(observed.shape[1])]
    )
    return MultiOutputMetrics(rmse, mae, rank_correlation)


def top_k_regret(
    true_objective: np.ndarray,
    predicted_objective: np.ndarray,
    *,
    fraction: float = 0.10,
) -> float:
    """Regret of the best true mixture among the model's predicted top-k set."""

    true_values = np.asarray(true_objective, dtype=float).reshape(-1)
    predicted_values = np.asarray(predicted_objective, dtype=float).reshape(-1)
    if true_values.shape != predicted_values.shape or true_values.size == 0:
        raise ValueError("Objective arrays must be non-empty and have the same shape.")
    if not 0 < fraction <= 1:
        raise ValueError("fraction must lie in (0, 1].")

    k = max(1, int(np.ceil(fraction * true_values.size)))
    selected = np.argsort(predicted_values)[:k]
    return float(np.min(true_values[selected]) - np.min(true_values))
