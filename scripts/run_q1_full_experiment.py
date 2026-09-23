"""Run all four Q1 schemes, select one winner, and freeze its artifacts."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path


REPOSITORY = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPOSITORY / "src"))

from huawei_f.benchmark import ExperimentSettings, run_full_benchmark  # noqa: E402


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--output",
        type=Path,
        default=REPOSITORY / "outputs" / "q1_benchmark_20260923",
    )
    parser.add_argument("--data-root", type=Path, default=None)
    parser.add_argument("--outer-folds", type=int, default=5)
    parser.add_argument("--outer-repeats", type=int, default=2)
    parser.add_argument("--inner-folds", type=int, default=3)
    parser.add_argument("--bootstrap-repetitions", type=int, default=200)
    parser.add_argument("--feature-dropout-repetitions", type=int, default=10)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    settings = ExperimentSettings(
        outer_folds=args.outer_folds,
        outer_repeats=args.outer_repeats,
        inner_folds=args.inner_folds,
        bootstrap_repetitions=args.bootstrap_repetitions,
        feature_dropout_repetitions=args.feature_dropout_repetitions,
    )
    run_full_benchmark(args.output, data_root=args.data_root, settings=settings)


if __name__ == "__main__":
    main()
