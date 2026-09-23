import numpy as np

from huawei_f.metrics import top_k_regret


def test_top_k_regret_is_zero_when_best_item_is_selected() -> None:
    true = np.array([3.0, 1.0, 2.0])
    predicted = np.array([4.0, 0.5, 3.0])

    assert top_k_regret(true, predicted, fraction=1 / 3) == 0.0
