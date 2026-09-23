"""Immutable descriptions of the four Q1 candidate pipelines."""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class SchemeSpec:
    scheme_id: str
    name: str
    quality_model: str
    loss_model: str


SCHEMES: tuple[SchemeSpec, ...] = (
    SchemeSpec("S1", "critic_sparse_scheffe", "critic_robust", "sparse_quadratic_scheffe"),
    SchemeSpec("S2", "latent_ilr_elasticnet", "robust_latent", "ilr_multitask_elasticnet"),
    SchemeSpec("S3", "rank_xgboost", "robust_rank_aggregation", "xgboost_multioutput"),
    SchemeSpec(
        "S4",
        "bayesian_hierarchical",
        "bayesian_hierarchical_quality",
        "bayesian_mixture_regression",
    ),
)
