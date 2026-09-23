"""Data path resolution and RegMix data-contract checks."""

from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd


DEFAULT_DATA_RELATIVE = Path(
    "第二十三届中国研究生数学建模竞赛 - 中文题目/中文题目/F题/real_attachments"
)


@dataclass(frozen=True)
class RegMixPair:
    """One aligned mixture/loss dataset."""

    mixture: pd.DataFrame
    loss: pd.DataFrame
    mixture_columns: tuple[str, ...]
    loss_columns: tuple[str, ...]


def repository_root() -> Path:
    return Path(__file__).resolve().parents[2]


def resolve_data_root(override: str | Path | None = None) -> Path:
    """Resolve the local real_attachments directory without hard-coding a user path."""

    candidate = override or os.getenv("HUAWEI_F_DATA_ROOT")
    if candidate:
        root = Path(candidate).expanduser().resolve()
    else:
        root = (repository_root().parent / DEFAULT_DATA_RELATIVE).resolve()

    if not root.is_dir():
        raise FileNotFoundError(
            f"Data root does not exist: {root}. Set HUAWEI_F_DATA_ROOT to real_attachments."
        )
    return root


def load_regmix_pair(
    mixture_path: str | Path,
    loss_path: str | Path,
    *,
    sum_tolerance: float = 5e-3,
) -> RegMixPair:
    """Load and validate an aligned mixture/loss CSV pair."""

    mixture = pd.read_csv(mixture_path)
    loss = pd.read_csv(loss_path)

    if len(mixture) != len(loss):
        raise ValueError(f"Row mismatch: mixture={len(mixture)}, loss={len(loss)}")

    if "index" in mixture.columns and "index" in loss.columns:
        if not mixture["index"].equals(loss["index"]):
            raise ValueError("The mixture and loss index columns are not aligned.")

    mixture_columns = tuple(column for column in mixture.columns if column != "index")
    loss_columns = tuple(column for column in loss.columns if column != "index")
    if not mixture_columns or not loss_columns:
        raise ValueError("Mixture or loss columns are empty.")

    x = mixture.loc[:, mixture_columns].to_numpy(dtype=float)
    y = loss.loc[:, loss_columns].to_numpy(dtype=float)
    if not np.isfinite(x).all() or not np.isfinite(y).all():
        raise ValueError("Mixture or loss data contains non-finite values.")
    if (x < -sum_tolerance).any():
        raise ValueError("Mixture proportions contain negative values.")

    row_sums = x.sum(axis=1)
    if not np.allclose(row_sums, 1.0, atol=sum_tolerance, rtol=0.0):
        max_error = float(np.max(np.abs(row_sums - 1.0)))
        raise ValueError(f"Mixture rows do not sum to one; max error={max_error:.3g}")

    # Source proportions are rounded to three decimals. Renormalize once here so
    # all candidate schemes receive exactly compositional rows summing to one.
    mixture = mixture.copy()
    mixture.loc[:, mixture_columns] = x / row_sums[:, None]

    return RegMixPair(
        mixture=mixture,
        loss=loss,
        mixture_columns=mixture_columns,
        loss_columns=loss_columns,
    )
