"""Rebuild lightweight post-hoc Q1 result tables without rerunning model fitting."""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import wilcoxon


REPOSITORY = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPOSITORY / "src"))

from huawei_f.quality import map_quality_to_mixture_domains  # noqa: E402


def main() -> None:
    output = REPOSITORY / "outputs" / "q1_benchmark_20260923"

    observed = pd.read_csv(output / "q_observed_domain_scores.csv")
    mapped_frames = []
    for scheme, group in observed.groupby("scheme"):
        mapped = map_quality_to_mixture_domains(
            group.set_index("quality_domain")["q_score"]
        )
        mapped.insert(0, "scheme", scheme)
        mapped_frames.append(mapped)
    pd.concat(mapped_frames, ignore_index=True).to_csv(
        output / "q_mapped_17_domain_scores.csv", index=False
    )

    folds = pd.read_csv(output / "cv_fold_metrics.csv")
    winner = "S3"
    winner_folds = folds.loc[
        folds["scheme"] == winner,
        ["outer_fold", "mean_standardized_rmse"],
    ].rename(columns={"mean_standardized_rmse": "winner_rmse"})
    rng = np.random.RandomState(20260923)
    rows = []
    for competitor in sorted(set(folds["scheme"]) - {winner}):
        competitor_folds = folds.loc[
            folds["scheme"] == competitor,
            ["outer_fold", "mean_standardized_rmse"],
        ].rename(columns={"mean_standardized_rmse": "competitor_rmse"})
        paired = winner_folds.merge(competitor_folds, on="outer_fold", validate="one_to_one")
        difference = paired["competitor_rmse"].to_numpy() - paired["winner_rmse"].to_numpy()
        bootstrap_means = np.array(
            [np.mean(rng.choice(difference, len(difference), replace=True)) for _ in range(10_000)]
        )
        rows.append(
            {
                "winner": winner,
                "competitor": competitor,
                "mean_rmse_advantage": float(np.mean(difference)),
                "ci95_low": float(np.quantile(bootstrap_means, 0.025)),
                "ci95_high": float(np.quantile(bootstrap_means, 0.975)),
                "wilcoxon_one_sided_p": float(
                    wilcoxon(difference, alternative="greater", zero_method="wilcox").pvalue
                ),
                "winner_better_all_folds": bool(np.all(difference > 0)),
            }
        )
    pairwise = pd.DataFrame(rows)
    pairwise.to_csv(output / "winner_pairwise_cv_tests.csv", index=False)
    print(pairwise.to_string(index=False))


if __name__ == "__main__":
    main()
