import pandas as pd

from huawei_f.selection import select_unique_winner


def test_one_standard_error_rule_uses_registered_tie_breakers() -> None:
    summary = pd.DataFrame(
        {
            "scheme": ["S1", "S2", "S3", "S4"],
            "cv_mean": [0.50, 0.49, 0.70, 0.80],
            "cv_se": [0.02, 0.03, 0.02, 0.01],
            "top_k_regret": [0.01, 0.02, 0.00, 0.00],
            "mean_spearman": [0.80, 0.90, 0.95, 0.96],
            "q_stability": [0.90, 0.92, 0.95, 0.96],
            "complexity": [1, 2, 3, 4],
        }
    )

    winner, candidates = select_unique_winner(summary)

    assert winner == "S1"
    assert candidates["scheme"].tolist() == ["S1", "S2"]
