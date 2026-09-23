"""Run the frozen Q1 Scheme 3 pipeline only.

Scheme 3 combines robust rank aggregation for the quality score Q with a
multi-output XGBoost model for the 13 loss tasks.
"""

from __future__ import annotations

import argparse
import json
import sys
from dataclasses import asdict
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.model_selection import RepeatedKFold


REPOSITORY = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPOSITORY / "src"))

from huawei_f.benchmark import (  # noqa: E402
    ExperimentSettings,
    _load_pair,
    _make_plots,
    _prediction_record,
    _select_parameters,
    _write_report,
    fit_and_validate_winner,
)
from huawei_f.data import resolve_data_root  # noqa: E402
from huawei_f.quality import (  # noqa: E402
    QUALITY_FIELDS,
    QUALITY_MODEL_FACTORIES,
    QualityPreprocessor,
    bootstrap_domain_stability,
    domain_mean_scores,
    feature_dropout_stability,
    iter_extended_quality_files,
    map_quality_to_mixture_domains,
    read_quality_jsonl,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--output",
        type=Path,
        default=REPOSITORY / "outputs" / "q1_scheme3_run_20260923",
    )
    parser.add_argument("--data-root", type=Path, default=None)
    parser.add_argument("--outer-folds", type=int, default=5)
    parser.add_argument("--outer-repeats", type=int, default=2)
    parser.add_argument("--inner-folds", type=int, default=3)
    parser.add_argument("--bootstrap-repetitions", type=int, default=200)
    parser.add_argument("--feature-dropout-repetitions", type=int, default=10)
    return parser.parse_args()


def run_quality_s3(
    data_root: Path, output_dir: Path, settings: ExperimentSettings
) -> tuple[pd.DataFrame, pd.DataFrame, dict[str, object]]:
    print("[quality] reading A1 quality-signal sample", flush=True)
    quality_root = data_root / "A_data_value"
    raw, domains = read_quality_jsonl(
        quality_root / "slimpajama_quality_signal_sample.jsonl.xz"
    )
    preprocessor = QualityPreprocessor(settings.seed)
    directed = preprocessor.fit_transform(raw)
    model = QUALITY_MODEL_FACTORIES["S3"]().fit(directed)
    scores = model.score(directed)
    sample_means = domain_mean_scores(scores, domains)

    rng = np.random.RandomState(settings.seed)
    stability_indices = []
    for domain in np.unique(domains):
        candidates = np.flatnonzero(domains == domain)
        count = min(3000, len(candidates))
        stability_indices.extend(rng.choice(candidates, count, replace=False).tolist())
    stability_indices = np.asarray(sorted(stability_indices))

    extended_means: dict[str, float] = {}
    extended_counts: dict[str, int] = {}
    for path, domain in iter_extended_quality_files(quality_root):
        extended_raw, _ = read_quality_jsonl(path, domain_override=domain)
        extended_score = model.score(preprocessor.transform(extended_raw))
        extended_means[domain] = float(np.mean(extended_score))
        extended_counts[domain] = len(extended_score)

    final_means = sample_means.copy()
    for domain, score in extended_means.items():
        final_means.loc[domain] = score

    bootstrap_stability = bootstrap_domain_stability(
        scores,
        domains,
        repetitions=settings.bootstrap_repetitions,
        random_state=settings.seed,
    )
    dropout_stability = feature_dropout_stability(
        directed[stability_indices],
        domains[stability_indices],
        "S3",
        repetitions=settings.feature_dropout_repetitions,
        random_state=settings.seed,
    )
    transfer_gap = float(
        np.mean(
            [
                abs(float(sample_means[domain]) - score)
                for domain, score in extended_means.items()
            ]
        )
    )
    diagnostics = pd.DataFrame(
        [
            {
                "scheme": "S3",
                "q_bootstrap_stability": bootstrap_stability,
                "q_feature_dropout_stability": dropout_stability,
                "q_stability": 0.5 * bootstrap_stability + 0.5 * dropout_stability,
                "q_extended_mean_abs_gap": transfer_gap,
                "q_extended_consistency": float(np.clip(1.0 - transfer_gap, 0.0, 1.0)),
            }
        ]
    )
    observed_rows = []
    for domain, score in final_means.items():
        observed_rows.append(
            {
                "scheme": "S3",
                "quality_domain": domain,
                "q_score": float(score),
                "source": "extended" if domain in extended_means else "A1_sample",
                "sample_count": extended_counts.get(domain, int(np.sum(domains == domain))),
            }
        )
    observed = pd.DataFrame(observed_rows)
    mapped = map_quality_to_mixture_domains(final_means)
    mapped.insert(0, "scheme", "S3")
    diagnostics.to_csv(output_dir / "q_quality_diagnostics.csv", index=False)
    observed.to_csv(output_dir / "q_observed_domain_scores.csv", index=False)
    mapped.to_csv(output_dir / "q_mapped_17_domain_scores.csv", index=False)
    joblib.dump(
        {"preprocessor": preprocessor, "model": model, "fields": QUALITY_FIELDS},
        output_dir / "winner_quality_model.joblib",
    )
    print(
        "[quality] S3 fitted: "
        f"stability={diagnostics.iloc[0]['q_stability']:.6f}, "
        f"extended_consistency={diagnostics.iloc[0]['q_extended_consistency']:.6f}",
        flush=True,
    )
    return diagnostics, mapped, {"preprocessor": preprocessor, "model": model}


def run_loss_s3(
    data_root: Path,
    output_dir: Path,
    q_diagnostics: pd.DataFrame,
    settings: ExperimentSettings,
):
    table_dir = data_root / "A_data_value" / "regmix_tables"
    x, y, _, mixture_columns, loss_columns = _load_pair(
        table_dir, "train_mixture_1m.csv", "train_pile_loss_1m.csv"
    )
    outer = RepeatedKFold(
        n_splits=settings.outer_folds,
        n_repeats=settings.outer_repeats,
        random_state=settings.seed,
    )
    fold_rows = []
    tuning_rows = []
    splits = list(outer.split(x))
    print(f"[loss] nested CV for S3: {len(splits)} outer folds", flush=True)
    for fold_number, (train_index, validation_index) in enumerate(splits, start=1):
        parameters, inner_results = _select_parameters(
            "S3",
            x[train_index],
            y[train_index],
            folds=settings.inner_folds,
            random_state=settings.seed + fold_number,
        )
        inner_results.insert(0, "scheme", "S3")
        inner_results.insert(1, "outer_fold", fold_number)
        tuning_rows.append(inner_results)
        from huawei_f.loss_models import create_loss_model

        model = create_loss_model("S3", parameters).fit(x[train_index], y[train_index])
        prediction = model.predict(x[validation_index])
        record = _prediction_record(y[validation_index], prediction, y[train_index])
        record.update(
            {
                "scheme": "S3",
                "outer_fold": fold_number,
                "parameters": json.dumps(parameters, sort_keys=True),
                "complexity": model.complexity(),
            }
        )
        fold_rows.append(record)
        print(
            f"[loss] S3 fold {fold_number}/{len(splits)} "
            f"sRMSE={record['mean_standardized_rmse']:.4f}",
            flush=True,
        )

    folds = pd.DataFrame(fold_rows)
    tuning = pd.concat(tuning_rows, ignore_index=True)
    folds.to_csv(output_dir / "cv_fold_metrics.csv", index=False)
    tuning.to_csv(output_dir / "inner_tuning_results.csv", index=False)
    summary = (
        folds.groupby("scheme")
        .agg(
            cv_mean=("mean_standardized_rmse", "mean"),
            cv_std=("mean_standardized_rmse", "std"),
            standardized_mae=("mean_standardized_mae", "mean"),
            mean_spearman=("mean_spearman", "mean"),
            top_k_regret=("top_k_regret", "mean"),
            worst_task_rmse=("worst_task_standardized_rmse", "mean"),
            complexity=("complexity", "mean"),
            fold_count=("outer_fold", "count"),
        )
        .reset_index()
    )
    summary["cv_se"] = summary["cv_std"] / np.sqrt(summary["fold_count"])
    summary = summary.merge(
        q_diagnostics[
            [
                "scheme",
                "q_stability",
                "q_bootstrap_stability",
                "q_feature_dropout_stability",
                "q_extended_consistency",
            ]
        ],
        on="scheme",
        how="left",
    )
    summary["within_one_se"] = True
    summary["selected"] = True
    summary.to_csv(output_dir / "scheme_comparison.csv", index=False)
    return summary, folds, x, y, mixture_columns, loss_columns


def main() -> None:
    args = parse_args()
    settings = ExperimentSettings(
        outer_folds=args.outer_folds,
        outer_repeats=args.outer_repeats,
        inner_folds=args.inner_folds,
        bootstrap_repetitions=args.bootstrap_repetitions,
        feature_dropout_repetitions=args.feature_dropout_repetitions,
    )
    output_dir = args.output.resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    data_root = resolve_data_root(args.data_root)
    (output_dir / "experiment_settings.json").write_text(
        json.dumps(asdict(settings), ensure_ascii=False, indent=2), encoding="utf-8"
    )

    q_diagnostics, mapped, _ = run_quality_s3(data_root, output_dir, settings)
    summary, _, x_train, y_train, mixture_columns, loss_columns = run_loss_s3(
        data_root, output_dir, q_diagnostics, settings
    )
    model, parameters, holdout, _, cross_scale = fit_and_validate_winner(
        "S3",
        x_train,
        y_train,
        data_root,
        output_dir,
        settings,
        mixture_columns,
        loss_columns,
    )
    _make_plots(summary, mapped, output_dir)
    _write_report(output_dir, "S3", summary, holdout, cross_scale, parameters, settings)
    payload = {
        "winner": "S3",
        "parameters": parameters,
        "holdout_metrics": holdout,
        "settings": asdict(settings),
        "mixture_columns": mixture_columns,
        "loss_columns": loss_columns,
    }
    (output_dir / "winner.json").write_text(
        json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    print(f"[done] scheme=S3; outputs={output_dir}", flush=True)


if __name__ == "__main__":
    main()
