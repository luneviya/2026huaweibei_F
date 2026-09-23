"""End-to-end Q1 four-scheme benchmark and winner selection."""

from __future__ import annotations

import json
import math
import warnings
from dataclasses import asdict, dataclass
from datetime import datetime
from pathlib import Path
from typing import Any

import joblib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from scipy.stats import wilcoxon
from sklearn.exceptions import ConvergenceWarning
from sklearn.model_selection import KFold, RepeatedKFold

from .data import resolve_data_root
from .loss_models import MODEL_FAMILIES, create_loss_model, normalize_compositions
from .metrics import multioutput_metrics, spearman_value, top_k_regret
from .quality import (
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
from .selection import select_unique_winner


warnings.filterwarnings("ignore", category=ConvergenceWarning)


@dataclass(frozen=True)
class ExperimentSettings:
    seed: int = 20260923
    outer_folds: int = 5
    outer_repeats: int = 2
    inner_folds: int = 3
    bootstrap_repetitions: int = 200
    feature_dropout_repetitions: int = 10
    top_k_fraction: float = 0.10


def _load_pair(table_dir: Path, mixture_name: str, loss_name: str):
    mixture_frame = pd.read_csv(table_dir / mixture_name)
    loss_frame = pd.read_csv(table_dir / loss_name)
    if len(mixture_frame) != len(loss_frame):
        raise ValueError(f"Unaligned pair: {mixture_name}, {loss_name}")
    if not mixture_frame["index"].equals(loss_frame["index"]):
        raise ValueError(f"Index mismatch: {mixture_name}, {loss_name}")
    mixture_columns = [column for column in mixture_frame if column != "index"]
    loss_columns = [column for column in loss_frame if column != "index"]
    x = normalize_compositions(mixture_frame[mixture_columns].to_numpy(dtype=float))
    y = loss_frame[loss_columns].to_numpy(dtype=float)
    return x, y, mixture_frame["index"].to_numpy(), mixture_columns, loss_columns


def _prediction_record(y_true: np.ndarray, y_pred: np.ndarray, y_train: np.ndarray) -> dict[str, float]:
    scale = np.std(y_train, axis=0, ddof=1)
    scale = np.where(scale > 1e-12, scale, 1.0)
    metrics = multioutput_metrics(y_true, y_pred, training_scale=scale)
    train_mean = np.mean(y_train, axis=0)
    true_objective = np.mean((y_true - train_mean) / scale, axis=1)
    predicted_objective = np.mean((y_pred - train_mean) / scale, axis=1)
    return {
        "mean_standardized_rmse": metrics.mean_standardized_rmse,
        "mean_standardized_mae": float(np.mean(metrics.standardized_mae)),
        "mean_spearman": metrics.mean_spearman,
        "top_k_regret": top_k_regret(true_objective, predicted_objective, fraction=0.10),
        "worst_task_standardized_rmse": float(np.max(metrics.standardized_rmse)),
    }


def _select_parameters(
    scheme_id: str,
    x: np.ndarray,
    y: np.ndarray,
    *,
    folds: int,
    random_state: int,
) -> tuple[dict[str, Any], pd.DataFrame]:
    splitter = KFold(n_splits=folds, shuffle=True, random_state=random_state)
    rows = []
    for parameter_index, parameters in enumerate(MODEL_FAMILIES[scheme_id].grid):
        fold_scores = []
        for train_index, validation_index in splitter.split(x):
            model = create_loss_model(scheme_id, dict(parameters)).fit(x[train_index], y[train_index])
            prediction = model.predict(x[validation_index])
            fold_scores.append(
                _prediction_record(y[validation_index], prediction, y[train_index])[
                    "mean_standardized_rmse"
                ]
            )
        rows.append(
            {
                "parameter_index": parameter_index,
                "parameters": json.dumps(parameters, sort_keys=True),
                "inner_cv_rmse": float(np.mean(fold_scores)),
            }
        )
    results = pd.DataFrame(rows).sort_values(["inner_cv_rmse", "parameter_index"])
    best_index = int(results.iloc[0]["parameter_index"])
    return dict(MODEL_FAMILIES[scheme_id].grid[best_index]), results


def run_quality_benchmark(
    data_root: Path,
    output_dir: Path,
    settings: ExperimentSettings,
) -> tuple[pd.DataFrame, pd.DataFrame, dict[str, object]]:
    print("[quality] reading A1 quality-signal sample", flush=True)
    quality_root = data_root / "A_data_value"
    raw, domains = read_quality_jsonl(quality_root / "slimpajama_quality_signal_sample.jsonl.xz")
    preprocessor = QualityPreprocessor(settings.seed)
    directed = preprocessor.fit_transform(raw)

    rng = np.random.RandomState(settings.seed)
    stability_indices = []
    for domain in np.unique(domains):
        candidates = np.flatnonzero(domains == domain)
        count = min(3000, len(candidates))
        stability_indices.extend(rng.choice(candidates, count, replace=False).tolist())
    stability_indices = np.asarray(sorted(stability_indices))

    diagnostics = []
    observed_rows = []
    mapped_rows = []
    fitted: dict[str, object] = {"preprocessor": preprocessor}

    for scheme_id in ("S1", "S2", "S3", "S4"):
        print(f"[quality] fitting {scheme_id}", flush=True)
        model = QUALITY_MODEL_FACTORIES[scheme_id]().fit(directed)
        scores = model.score(directed)
        hierarchical = scheme_id == "S4"
        sample_means = domain_mean_scores(
            scores, domains, hierarchical_shrinkage=hierarchical
        )

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
            scheme_id,
            repetitions=settings.feature_dropout_repetitions,
            random_state=settings.seed,
        )
        transfer_gap = float(
            np.mean(
                [abs(float(sample_means[domain]) - score) for domain, score in extended_means.items()]
            )
        )
        q_stability = 0.5 * bootstrap_stability + 0.5 * dropout_stability
        diagnostics.append(
            {
                "scheme": scheme_id,
                "q_bootstrap_stability": bootstrap_stability,
                "q_feature_dropout_stability": dropout_stability,
                "q_stability": q_stability,
                "q_extended_mean_abs_gap": transfer_gap,
                "q_extended_consistency": float(np.clip(1.0 - transfer_gap, 0.0, 1.0)),
            }
        )

        for domain, score in final_means.items():
            observed_rows.append(
                {
                    "scheme": scheme_id,
                    "quality_domain": domain,
                    "q_score": float(score),
                    "source": "extended" if domain in extended_means else "A1_sample",
                    "sample_count": extended_counts.get(domain, int(np.sum(domains == domain))),
                }
            )
        mapped = map_quality_to_mixture_domains(final_means)
        mapped.insert(0, "scheme", scheme_id)
        mapped_rows.append(mapped)
        fitted[f"quality_model_{scheme_id}"] = model

    diagnostics_frame = pd.DataFrame(diagnostics)
    observed_frame = pd.DataFrame(observed_rows)
    mapped_frame = pd.concat(mapped_rows, ignore_index=True)
    diagnostics_frame.to_csv(output_dir / "q_quality_diagnostics.csv", index=False)
    observed_frame.to_csv(output_dir / "q_observed_domain_scores.csv", index=False)
    mapped_frame.to_csv(output_dir / "q_mapped_17_domain_scores.csv", index=False)
    return diagnostics_frame, mapped_frame, fitted


def run_loss_benchmark(
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
    splits = list(outer.split(x))
    fold_rows = []
    tuning_rows = []

    for scheme_id in ("S1", "S2", "S3", "S4"):
        print(f"[loss] nested CV for {scheme_id}: {len(splits)} outer folds", flush=True)
        for fold_number, (train_index, validation_index) in enumerate(splits, start=1):
            parameters, inner_results = _select_parameters(
                scheme_id,
                x[train_index],
                y[train_index],
                folds=settings.inner_folds,
                random_state=settings.seed + fold_number,
            )
            inner_results.insert(0, "scheme", scheme_id)
            inner_results.insert(1, "outer_fold", fold_number)
            tuning_rows.append(inner_results)
            model = create_loss_model(scheme_id, parameters).fit(x[train_index], y[train_index])
            prediction = model.predict(x[validation_index])
            record = _prediction_record(y[validation_index], prediction, y[train_index])
            record.update(
                {
                    "scheme": scheme_id,
                    "outer_fold": fold_number,
                    "parameters": json.dumps(parameters, sort_keys=True),
                    "complexity": model.complexity(),
                }
            )
            fold_rows.append(record)
            print(
                f"[loss] {scheme_id} fold {fold_number}/{len(splits)} "
                f"sRMSE={record['mean_standardized_rmse']:.4f}",
                flush=True,
            )

    folds_frame = pd.DataFrame(fold_rows)
    tuning_frame = pd.concat(tuning_rows, ignore_index=True)
    folds_frame.to_csv(output_dir / "cv_fold_metrics.csv", index=False)
    tuning_frame.to_csv(output_dir / "inner_tuning_results.csv", index=False)

    summary = (
        folds_frame.groupby("scheme")
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
    winner, candidates = select_unique_winner(summary)
    summary["within_one_se"] = summary["scheme"].isin(candidates["scheme"])
    summary["selected"] = summary["scheme"].eq(winner)
    summary = summary.sort_values(["selected", "cv_mean"], ascending=[False, True])
    summary.to_csv(output_dir / "scheme_comparison.csv", index=False)
    winner_folds = folds_frame.loc[
        folds_frame["scheme"] == winner, ["outer_fold", "mean_standardized_rmse"]
    ].rename(columns={"mean_standardized_rmse": "winner_rmse"})
    comparison_rows = []
    rng = np.random.RandomState(settings.seed)
    for competitor in sorted(set(folds_frame["scheme"]) - {winner}):
        competitor_folds = folds_frame.loc[
            folds_frame["scheme"] == competitor,
            ["outer_fold", "mean_standardized_rmse"],
        ].rename(columns={"mean_standardized_rmse": "competitor_rmse"})
        paired = winner_folds.merge(competitor_folds, on="outer_fold", validate="one_to_one")
        difference = paired["competitor_rmse"].to_numpy() - paired["winner_rmse"].to_numpy()
        bootstrap_means = np.array(
            [np.mean(rng.choice(difference, size=len(difference), replace=True)) for _ in range(10_000)]
        )
        test = wilcoxon(difference, alternative="greater", zero_method="wilcox")
        comparison_rows.append(
            {
                "winner": winner,
                "competitor": competitor,
                "mean_rmse_advantage": float(np.mean(difference)),
                "ci95_low": float(np.quantile(bootstrap_means, 0.025)),
                "ci95_high": float(np.quantile(bootstrap_means, 0.975)),
                "wilcoxon_one_sided_p": float(test.pvalue),
                "winner_better_all_folds": bool(np.all(difference > 0)),
            }
        )
    pd.DataFrame(comparison_rows).to_csv(output_dir / "winner_pairwise_cv_tests.csv", index=False)
    return winner, summary, folds_frame, x, y, mixture_columns, loss_columns


def fit_and_validate_winner(
    winner: str,
    x_train: np.ndarray,
    y_train: np.ndarray,
    data_root: Path,
    output_dir: Path,
    settings: ExperimentSettings,
    mixture_columns: list[str],
    loss_columns: list[str],
):
    print(f"[winner] tuning {winner} on the full training set", flush=True)
    parameters, full_tuning = _select_parameters(
        winner,
        x_train,
        y_train,
        folds=settings.outer_folds,
        random_state=settings.seed + 10_000,
    )
    full_tuning.to_csv(output_dir / "winner_full_training_tuning.csv", index=False)
    model = create_loss_model(winner, parameters).fit(x_train, y_train)

    table_dir = data_root / "A_data_value" / "regmix_tables"
    x_holdout, y_holdout, holdout_ids, _, _ = _load_pair(
        table_dir, "test_mixture_1m.csv", "test_pile_loss_1m.csv"
    )
    prediction = model.predict(x_holdout)
    holdout_record = _prediction_record(y_holdout, prediction, y_train)

    training_scale = np.std(y_train, axis=0, ddof=1)
    task_metrics = multioutput_metrics(y_holdout, prediction, training_scale=training_scale)
    per_task = pd.DataFrame(
        {
            "task": loss_columns,
            "standardized_rmse": task_metrics.standardized_rmse,
            "standardized_mae": task_metrics.standardized_mae,
            "spearman": task_metrics.spearman,
        }
    )
    per_task.to_csv(output_dir / "winner_holdout_per_task.csv", index=False)

    predictions = pd.DataFrame({"index": holdout_ids})
    for task_index, task in enumerate(loss_columns):
        short_name = task.replace("metric/the_pile_", "").replace("_val_loss", "")
        predictions[f"actual_{short_name}"] = y_holdout[:, task_index]
        predictions[f"predicted_{short_name}"] = prediction[:, task_index]
    predictions.to_csv(output_dir / "winner_holdout_predictions.csv", index=False)

    cross_scale_rows = []
    train_mean = np.mean(y_train, axis=0)
    train_scale = np.std(y_train, axis=0, ddof=1)
    train_scale = np.where(train_scale > 1e-12, train_scale, 1.0)
    for scale_name, mixture_file, loss_file in (
        ("60M", "test_mixture_60m.csv", "test_pile_loss_60m.csv"),
        ("1B", "test_mixture_1B.csv", "test_pile_loss_1B.csv"),
    ):
        x_scale, y_scale, _, _, _ = _load_pair(table_dir, mixture_file, loss_file)
        predicted_scale = model.predict(x_scale)
        actual_standardized = (y_scale - np.mean(y_scale, axis=0)) / np.std(
            y_scale, axis=0, ddof=1
        )
        actual_objective = np.mean(actual_standardized, axis=1)
        predicted_objective = np.mean((predicted_scale - train_mean) / train_scale, axis=1)
        task_rank = [
            spearman_value(y_scale[:, task], predicted_scale[:, task])
            for task in range(y_scale.shape[1])
        ]
        cross_scale_rows.append(
            {
                "scale": scale_name,
                "rows": len(y_scale),
                "aggregate_spearman": spearman_value(actual_objective, predicted_objective),
                "mean_task_spearman": float(np.nanmean(task_rank)),
                "top_k_regret": top_k_regret(
                    actual_objective,
                    predicted_objective,
                    fraction=settings.top_k_fraction,
                ),
            }
        )
    cross_scale = pd.DataFrame(cross_scale_rows)
    cross_scale.to_csv(output_dir / "winner_cross_scale_rank_transfer.csv", index=False)
    joblib.dump(model, output_dir / "winner_loss_model.joblib")
    return model, parameters, holdout_record, per_task, cross_scale


def _make_plots(
    summary: pd.DataFrame,
    mapped_quality: pd.DataFrame,
    output_dir: Path,
) -> None:
    sns.set_theme(style="whitegrid")
    comparison = summary.sort_values("cv_mean")
    plt.figure(figsize=(7.2, 4.4))
    plt.errorbar(
        comparison["scheme"],
        comparison["cv_mean"],
        yerr=comparison["cv_se"],
        fmt="o",
        color="#205493",
        capsize=5,
        markersize=7,
    )
    plt.ylabel("Nested-CV mean standardized RMSE")
    plt.xlabel("Candidate scheme")
    plt.tight_layout()
    plt.savefig(output_dir / "scheme_cv_comparison.png", dpi=220)
    plt.close()

    quality_matrix = mapped_quality.pivot(
        index="mixture_domain", columns="scheme", values="q_score"
    )
    plt.figure(figsize=(7.2, 7.0))
    sns.heatmap(quality_matrix, annot=True, fmt=".3f", cmap="YlGnBu", cbar_kws={"label": "Q"})
    plt.xlabel("Candidate scheme")
    plt.ylabel("RegMix training domain")
    plt.tight_layout()
    plt.savefig(output_dir / "mapped_domain_quality_heatmap.png", dpi=220)
    plt.close()


def _write_report(
    output_dir: Path,
    winner: str,
    summary: pd.DataFrame,
    holdout: dict[str, float],
    cross_scale: pd.DataFrame,
    parameters: dict[str, Any],
    settings: ExperimentSettings,
) -> None:
    selected = summary.loc[summary["scheme"] == winner].iloc[0]
    report = [
        "# 问题一四方案实测结果",
        "",
        f"运行时间：{datetime.now().isoformat(timespec='seconds')}",
        "",
        f"唯一入选方案：**{winner}**",
        "",
        "## 选择依据",
        "",
        f"- 嵌套交叉验证平均标准化 RMSE：{selected['cv_mean']:.6f} ± {selected['cv_se']:.6f}（标准误）",
        f"- 交叉验证平均 Spearman：{selected['mean_spearman']:.6f}",
        f"- 交叉验证 Top-k regret：{selected['top_k_regret']:.6f}",
        f"- Q 稳定性：{selected['q_stability']:.6f}",
        "- 选择过程只使用 A4/A5；A6/A7 在冠军确定后才打开验证。",
        "",
        "## A6/A7 独立验证",
        "",
        f"- 平均标准化 RMSE：{holdout['mean_standardized_rmse']:.6f}",
        f"- 平均标准化 MAE：{holdout['mean_standardized_mae']:.6f}",
        f"- 平均 Spearman：{holdout['mean_spearman']:.6f}",
        f"- Top-k regret：{holdout['top_k_regret']:.6f}",
        "",
        "## 冠军超参数",
        "",
        f"```json\n{json.dumps(parameters, ensure_ascii=False, indent=2, sort_keys=True)}\n```",
        "",
        "## 跨规模排序迁移",
        "",
        cross_scale.to_markdown(index=False, floatfmt=".6f"),
        "",
        "## 冻结说明",
        "",
        "从本次结果开始，问题一的质量信号方向、分位数变换、领域映射、配比表示、冠军模型形式、随机种子和评价口径均冻结。后续问题只允许增加 N、D、C 等新变量并估计新增参数。",
        "",
        "## 实验设置",
        "",
        f"```json\n{json.dumps(asdict(settings), ensure_ascii=False, indent=2)}\n```",
    ]
    (output_dir / "experiment_report.md").write_text("\n".join(report), encoding="utf-8")


def run_full_benchmark(
    output_dir: str | Path,
    *,
    data_root: str | Path | None = None,
    settings: ExperimentSettings | None = None,
) -> dict[str, Any]:
    settings = settings or ExperimentSettings()
    output_path = Path(output_dir).resolve()
    output_path.mkdir(parents=True, exist_ok=True)
    resolved_data_root = resolve_data_root(data_root)
    (output_path / "experiment_settings.json").write_text(
        json.dumps(asdict(settings), ensure_ascii=False, indent=2), encoding="utf-8"
    )

    q_diagnostics, mapped_quality, fitted_quality = run_quality_benchmark(
        resolved_data_root, output_path, settings
    )
    winner, summary, folds, x_train, y_train, mixture_columns, loss_columns = run_loss_benchmark(
        resolved_data_root, output_path, q_diagnostics, settings
    )
    model, parameters, holdout, per_task, cross_scale = fit_and_validate_winner(
        winner,
        x_train,
        y_train,
        resolved_data_root,
        output_path,
        settings,
        mixture_columns,
        loss_columns,
    )
    joblib.dump(
        {
            "preprocessor": fitted_quality["preprocessor"],
            "model": fitted_quality[f"quality_model_{winner}"],
            "fields": QUALITY_FIELDS,
        },
        output_path / "winner_quality_model.joblib",
    )
    _make_plots(summary, mapped_quality, output_path)
    _write_report(output_path, winner, summary, holdout, cross_scale, parameters, settings)

    winner_payload = {
        "winner": winner,
        "parameters": parameters,
        "holdout_metrics": holdout,
        "settings": asdict(settings),
        "mixture_columns": mixture_columns,
        "loss_columns": loss_columns,
    }
    (output_path / "winner.json").write_text(
        json.dumps(winner_payload, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    print(f"[done] winner={winner}; outputs={output_path}", flush=True)
    return winner_payload
