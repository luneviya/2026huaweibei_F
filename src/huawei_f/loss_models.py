"""Four candidate mixture-to-Loss models used in the Q1 benchmark."""

from __future__ import annotations

from dataclasses import dataclass
import inspect
from itertools import combinations
from pathlib import Path
import sys

import numpy as np
from scipy.linalg import helmert
from sklearn.linear_model import BayesianRidge, MultiTaskElasticNet
from sklearn.preprocessing import StandardScaler


LOCAL_DEPENDENCIES = Path(__file__).resolve().parents[2] / ".deps"
if LOCAL_DEPENDENCIES.is_dir() and str(LOCAL_DEPENDENCIES) not in sys.path:
    sys.path.insert(0, str(LOCAL_DEPENDENCIES))


def normalize_compositions(x: np.ndarray) -> np.ndarray:
    values = np.asarray(x, dtype=float)
    if (values < 0).any():
        raise ValueError("Compositions cannot be negative.")
    totals = values.sum(axis=1, keepdims=True)
    if np.any(totals <= 0):
        raise ValueError("Every composition must have positive mass.")
    return values / totals


def quadratic_scheffe_features(x: np.ndarray) -> np.ndarray:
    values = normalize_compositions(x)
    interactions = [values[:, i] * values[:, j] for i, j in combinations(range(values.shape[1]), 2)]
    return np.column_stack([values, *interactions])


def ilr_features(x: np.ndarray, zero_replacement: float = 1e-4) -> np.ndarray:
    values = normalize_compositions(x)
    safe = np.maximum(values, zero_replacement)
    safe /= safe.sum(axis=1, keepdims=True)
    basis = helmert(values.shape[1], full=False)
    return np.log(safe) @ basis.T


class BaseLossModel:
    name = "base"

    def fit(self, x: np.ndarray, y: np.ndarray) -> "BaseLossModel":
        raise NotImplementedError

    def predict(self, x: np.ndarray) -> np.ndarray:
        raise NotImplementedError

    def complexity(self) -> float:
        return np.nan


class _StandardizedTargetMixin:
    y_mean_: np.ndarray
    y_scale_: np.ndarray

    def _fit_target_transform(self, y: np.ndarray) -> np.ndarray:
        values = np.asarray(y, dtype=float)
        self.y_mean_ = np.mean(values, axis=0)
        self.y_scale_ = np.std(values, axis=0, ddof=1)
        self.y_scale_ = np.where(self.y_scale_ > 1e-12, self.y_scale_, 1.0)
        return (values - self.y_mean_) / self.y_scale_

    def _inverse_target_transform(self, y: np.ndarray) -> np.ndarray:
        return np.asarray(y) * self.y_scale_ + self.y_mean_


class SparseScheffeModel(_StandardizedTargetMixin, BaseLossModel):
    name = "sparse_quadratic_scheffe"

    def __init__(
        self,
        *,
        alpha: float = 0.001,
        l1_ratio: float = 0.5,
        random_state: int = 20260923,
    ) -> None:
        self.alpha = alpha
        self.l1_ratio = l1_ratio
        self.random_state = random_state
        self.scaler_: StandardScaler | None = None
        self.model_: MultiTaskElasticNet | None = None

    def fit(self, x: np.ndarray, y: np.ndarray) -> "SparseScheffeModel":
        features = quadratic_scheffe_features(x)
        self.scaler_ = StandardScaler().fit(features)
        standardized_y = self._fit_target_transform(y)
        self.model_ = MultiTaskElasticNet(
            alpha=self.alpha,
            l1_ratio=self.l1_ratio,
            fit_intercept=True,
            max_iter=20_000,
            tol=1e-6,
            random_state=self.random_state,
            selection="cyclic",
        ).fit(self.scaler_.transform(features), standardized_y)
        return self

    def predict(self, x: np.ndarray) -> np.ndarray:
        if self.scaler_ is None or self.model_ is None:
            raise RuntimeError("Model must be fitted first.")
        prediction = self.model_.predict(self.scaler_.transform(quadratic_scheffe_features(x)))
        return self._inverse_target_transform(prediction)

    def complexity(self) -> float:
        if self.model_ is None:
            return np.nan
        return float(np.count_nonzero(np.abs(self.model_.coef_) > 1e-8))


class IlrElasticNetModel(_StandardizedTargetMixin, BaseLossModel):
    name = "ilr_multitask_elasticnet"

    def __init__(
        self,
        *,
        alpha: float = 0.001,
        l1_ratio: float = 0.5,
        zero_replacement: float = 1e-4,
        random_state: int = 20260923,
    ) -> None:
        self.alpha = alpha
        self.l1_ratio = l1_ratio
        self.zero_replacement = zero_replacement
        self.random_state = random_state
        self.scaler_: StandardScaler | None = None
        self.model_: MultiTaskElasticNet | None = None

    def fit(self, x: np.ndarray, y: np.ndarray) -> "IlrElasticNetModel":
        features = ilr_features(x, self.zero_replacement)
        self.scaler_ = StandardScaler().fit(features)
        standardized_y = self._fit_target_transform(y)
        self.model_ = MultiTaskElasticNet(
            alpha=self.alpha,
            l1_ratio=self.l1_ratio,
            fit_intercept=True,
            max_iter=20_000,
            tol=1e-6,
            random_state=self.random_state,
            selection="cyclic",
        ).fit(self.scaler_.transform(features), standardized_y)
        return self

    def predict(self, x: np.ndarray) -> np.ndarray:
        if self.scaler_ is None or self.model_ is None:
            raise RuntimeError("Model must be fitted first.")
        prediction = self.model_.predict(
            self.scaler_.transform(ilr_features(x, self.zero_replacement))
        )
        return self._inverse_target_transform(prediction)

    def complexity(self) -> float:
        if self.model_ is None:
            return np.nan
        return float(np.count_nonzero(np.abs(self.model_.coef_) > 1e-8))


class XGBoostMixtureModel(_StandardizedTargetMixin, BaseLossModel):
    name = "xgboost_multioutput"

    def __init__(
        self,
        *,
        n_estimators: int = 300,
        max_depth: int = 3,
        learning_rate: float = 0.03,
        min_child_weight: float = 3.0,
        subsample: float = 0.85,
        colsample_bytree: float = 0.85,
        reg_lambda: float = 5.0,
        random_state: int = 20260923,
        n_jobs: int = 4,
    ) -> None:
        self.parameters = {
            "n_estimators": n_estimators,
            "max_depth": max_depth,
            "learning_rate": learning_rate,
            "min_child_weight": min_child_weight,
            "subsample": subsample,
            "colsample_bytree": colsample_bytree,
            "reg_lambda": reg_lambda,
            "random_state": random_state,
            "n_jobs": n_jobs,
        }
        self.model_ = None

    @staticmethod
    def _load_xgboost():
        from xgboost import XGBRegressor

        return XGBRegressor

    def fit(self, x: np.ndarray, y: np.ndarray) -> "XGBoostMixtureModel":
        XGBRegressor = self._load_xgboost()
        standardized_y = self._fit_target_transform(y)
        self.model_ = XGBRegressor(
            objective="reg:squarederror",
            tree_method="hist",
            multi_strategy="one_output_per_tree",
            verbosity=0,
            **self.parameters,
        ).fit(normalize_compositions(x), standardized_y)
        return self

    def predict(self, x: np.ndarray) -> np.ndarray:
        if self.model_ is None:
            raise RuntimeError("Model must be fitted first.")
        return self._inverse_target_transform(self.model_.predict(normalize_compositions(x)))

    def complexity(self) -> float:
        if self.model_ is None:
            return np.nan
        dataframe = self.model_.get_booster().trees_to_dataframe()
        return float((dataframe["Feature"] != "Leaf").sum())


class BayesianMixtureModel(_StandardizedTargetMixin, BaseLossModel):
    name = "bayesian_mixture_regression"

    def __init__(
        self,
        *,
        alpha_1: float = 1e-6,
        alpha_2: float = 1e-6,
        lambda_1: float = 1e-6,
        lambda_2: float = 1e-6,
    ) -> None:
        self.priors = {
            "alpha_1": alpha_1,
            "alpha_2": alpha_2,
            "lambda_1": lambda_1,
            "lambda_2": lambda_2,
        }
        self.scaler_: StandardScaler | None = None
        self.models_: list[BayesianRidge] = []

    def fit(self, x: np.ndarray, y: np.ndarray) -> "BayesianMixtureModel":
        features = quadratic_scheffe_features(x)
        self.scaler_ = StandardScaler().fit(features)
        standardized_x = self.scaler_.transform(features)
        standardized_y = self._fit_target_transform(y)
        self.models_ = []
        iteration_argument = (
            {"max_iter": 500}
            if "max_iter" in inspect.signature(BayesianRidge).parameters
            else {"n_iter": 500}
        )
        for task in range(standardized_y.shape[1]):
            model = BayesianRidge(
                tol=1e-6,
                compute_score=True,
                fit_intercept=True,
                **iteration_argument,
                **self.priors,
            ).fit(standardized_x, standardized_y[:, task])
            self.models_.append(model)
        return self

    def predict(self, x: np.ndarray) -> np.ndarray:
        if self.scaler_ is None or not self.models_:
            raise RuntimeError("Model must be fitted first.")
        standardized_x = self.scaler_.transform(quadratic_scheffe_features(x))
        prediction = np.column_stack([model.predict(standardized_x) for model in self.models_])
        return self._inverse_target_transform(prediction)

    def complexity(self) -> float:
        if not self.models_:
            return np.nan
        return float(sum(np.count_nonzero(np.abs(model.coef_) > 1e-8) for model in self.models_))


@dataclass(frozen=True)
class ModelFamily:
    scheme_id: str
    factory: type[BaseLossModel]
    grid: tuple[dict[str, object], ...]


MODEL_FAMILIES = {
    "S1": ModelFamily(
        "S1",
        SparseScheffeModel,
        tuple(
            {"alpha": alpha, "l1_ratio": ratio}
            for alpha in (0.0003, 0.001, 0.003)
            for ratio in (0.25, 0.75)
        ),
    ),
    "S2": ModelFamily(
        "S2",
        IlrElasticNetModel,
        tuple(
            {"alpha": alpha, "l1_ratio": ratio, "zero_replacement": 1e-4}
            for alpha in (0.0003, 0.001, 0.003)
            for ratio in (0.25, 0.75)
        ),
    ),
    "S3": ModelFamily(
        "S3",
        XGBoostMixtureModel,
        (
            {"n_estimators": 250, "max_depth": 2, "learning_rate": 0.03},
            {"n_estimators": 400, "max_depth": 2, "learning_rate": 0.03},
            {"n_estimators": 300, "max_depth": 3, "learning_rate": 0.03},
            {"n_estimators": 250, "max_depth": 4, "learning_rate": 0.02},
        ),
    ),
    "S4": ModelFamily(
        "S4",
        BayesianMixtureModel,
        (
            {"alpha_1": 1e-6, "alpha_2": 1e-6, "lambda_1": 1e-6, "lambda_2": 1e-6},
            {"alpha_1": 1e-3, "alpha_2": 1e-3, "lambda_1": 1e-6, "lambda_2": 1e-6},
            {"alpha_1": 1e-6, "alpha_2": 1e-6, "lambda_1": 1e-3, "lambda_2": 1e-3},
        ),
    ),
}


def create_loss_model(scheme_id: str, parameters: dict[str, object]) -> BaseLossModel:
    family = MODEL_FAMILIES[scheme_id]
    return family.factory(**parameters)
