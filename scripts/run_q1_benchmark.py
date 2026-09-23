"""Entry point for listing schemes and validating the Q1 data contract."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT / "src"))

from huawei_f.config import load_toml  # noqa: E402
from huawei_f.data import load_regmix_pair, resolve_data_root  # noqa: E402
from huawei_f.schemes import SCHEMES  # noqa: E402


def list_schemes() -> None:
    for scheme in SCHEMES:
        print(
            f"{scheme.scheme_id}: {scheme.name} | "
            f"Q={scheme.quality_model} | Loss={scheme.loss_model}"
        )


def validate_data(config_path: Path) -> None:
    config = load_toml(config_path)
    root = resolve_data_root()
    table_dir = root / config["data"]["regmix_subdir"]
    pairs = {
        "train": (
            table_dir / config["data"]["train_mixture"],
            table_dir / config["data"]["train_loss"],
        ),
        "holdout": (
            table_dir / config["data"]["holdout_mixture"],
            table_dir / config["data"]["holdout_loss"],
        ),
    }
    print(f"data_root={root}")
    for label, (mixture_path, loss_path) in pairs.items():
        pair = load_regmix_pair(mixture_path, loss_path)
        print(
            f"{label}: rows={len(pair.mixture)}, "
            f"mixture_domains={len(pair.mixture_columns)}, "
            f"loss_tasks={len(pair.loss_columns)}"
        )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--config",
        type=Path,
        default=REPO_ROOT / "configs" / "q1.toml",
    )
    parser.add_argument("--list-schemes", action="store_true")
    parser.add_argument("--validate-data", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if not args.list_schemes and not args.validate_data:
        raise SystemExit("Choose --list-schemes and/or --validate-data.")
    if args.list_schemes:
        list_schemes()
    if args.validate_data:
        validate_data(args.config)


if __name__ == "__main__":
    main()
