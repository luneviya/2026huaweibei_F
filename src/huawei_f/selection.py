"""Deterministic one-standard-error model selection."""

from __future__ import annotations

import pandas as pd


REQUIRED_COLUMNS = {
    "scheme",
    "cv_mean",
    "cv_se",
    "top_k_regret",
    "mean_spearman",
    "q_stability",
    "complexity",
}


def select_unique_winner(summary: pd.DataFrame) -> tuple[str, pd.DataFrame]:
    """Return one winner and the one-standard-error candidate table."""

    missing = REQUIRED_COLUMNS.difference(summary.columns)
    if missing:
        raise ValueError(f"Missing selection columns: {sorted(missing)}")
    if summary.empty or summary["scheme"].duplicated().any():
        raise ValueError("Summary must contain one unique row per scheme.")

    best_index = summary["cv_mean"].idxmin()
    threshold = float(summary.loc[best_index, "cv_mean"] + summary.loc[best_index, "cv_se"])
    candidates = summary.loc[summary["cv_mean"] <= threshold].copy()
    candidates = candidates.sort_values(
        by=["top_k_regret", "mean_spearman", "q_stability", "complexity", "scheme"],
        ascending=[True, False, False, True, True],
        kind="mergesort",
    )
    return str(candidates.iloc[0]["scheme"]), candidates.reset_index(drop=True)
