"""Quality-signal preprocessing and the four candidate Q estimators."""

from __future__ import annotations

import json
import lzma
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

import numpy as np
import pandas as pd
from scipy.stats import spearmanr
from sklearn.decomposition import FactorAnalysis, PCA
from sklearn.preprocessing import QuantileTransformer, StandardScaler

from .metrics import spearman_value


QUALITY_FIELDS = (
    "fineweb_edu",
    "fluency_en",
    "modernbert_cleanliness",
    "modernbert_readability",
    "modernbert_reasoning",
    "modernbert_professionalism",
    "dsir_books",
    "dsir_wiki",
    "dsir_math",
    "qurater",
    "ad_en",
    "rps_doc_word_count",
    "rps_doc_num_sentences",
    "rps_doc_unigram_entropy",
    "rps_doc_frac_unique_words",
    "rps_doc_frac_no_alph_words",
    "rps_doc_frac_chars_top_2gram",
    "rps_doc_frac_chars_top_3gram",
    "rps_lines_uppercase_letter_fraction",
    "rps_lines_ending_with_terminal_punctution_mark",
    "rps_lines_numerical_chars_fraction",
    "rps_doc_mean_word_length",
)

POSITIVE_FIELDS = {
    "fineweb_edu",
    "fluency_en",
    "modernbert_cleanliness",
    "modernbert_readability",
    "modernbert_reasoning",
    "modernbert_professionalism",
    "dsir_books",
    "dsir_wiki",
    "dsir_math",
    "qurater",
    "rps_doc_frac_unique_words",
    "rps_lines_ending_with_terminal_punctution_mark",
}

NEGATIVE_FIELDS = {
    "ad_en",
    "rps_doc_frac_no_alph_words",
    "rps_doc_frac_chars_top_2gram",
    "rps_doc_frac_chars_top_3gram",
    "rps_lines_uppercase_letter_fraction",
}

CENTRAL_FIELDS = set(QUALITY_FIELDS).difference(POSITIVE_FIELDS).difference(NEGATIVE_FIELDS)

FEATURE_GROUPS = {
    "semantic": (
        "fineweb_edu",
        "fluency_en",
        "modernbert_cleanliness",
        "modernbert_readability",
        "modernbert_reasoning",
        "modernbert_professionalism",
        "qurater",
    ),
    "domain_relevance": ("dsir_books", "dsir_wiki", "dsir_math"),
    "hygiene": (
        "ad_en",
        "rps_doc_frac_no_alph_words",
        "rps_doc_frac_chars_top_2gram",
        "rps_doc_frac_chars_top_3gram",
        "rps_lines_uppercase_letter_fraction",
    ),
    "structure": (
        "rps_doc_word_count",
        "rps_doc_num_sentences",
        "rps_lines_ending_with_terminal_punctution_mark",
    ),
    "lexical": (
        "rps_doc_unigram_entropy",
        "rps_doc_frac_unique_words",
        "rps_lines_numerical_chars_fraction",
        "rps_doc_mean_word_length",
    ),
}


def _softmax_expected(value: object) -> float:
    """Compress scalar or list-valued logits to one ordinal scalar."""

    if not isinstance(value, list):
        try:
            return float(value)
        except (TypeError, ValueError):
            return np.nan
    if not value:
        return np.nan
    values = np.asarray(value, dtype=float)
    if values.size == 1:
        return float(values[0])
    shifted = values - np.max(values)
    probabilities = np.exp(shifted)
    probabilities /= probabilities.sum()
    levels = np.linspace(0.0, 1.0, values.size)
    return float(probabilities @ levels)


def read_quality_jsonl(
    path: str | Path,
    *,
    domain_override: str | None = None,
    max_rows: int | None = None,
) -> tuple[np.ndarray, np.ndarray]:
    """Read one compressed quality-signal file without loading document text."""

    columns: list[list[float]] = [[] for _ in QUALITY_FIELDS]
    domains: list[str] = []
    quality_path = Path(path)
    inferred_domain = domain_override or quality_path.name.split("_")[0]
    with lzma.open(quality_path, "rt", encoding="utf-8") as handle:
        for row_number, line in enumerate(handle):
            if max_rows is not None and row_number >= max_rows:
                break
            record = json.loads(line)
            domains.append(str(record.get("_source_domain") or inferred_domain))
            for column, field in zip(columns, QUALITY_FIELDS):
                column.append(_softmax_expected(record.get(field)))
    matrix = np.column_stack([np.asarray(column, dtype=float) for column in columns])
    return matrix, np.asarray(domains, dtype=object)


class QualityPreprocessor:
    """Fit robust empirical transforms and enforce documented desirability directions."""

    def __init__(self, random_state: int = 20260923) -> None:
        self.random_state = random_state
        self.medians_: np.ndarray | None = None
        self.transformer_: QuantileTransformer | None = None

    def fit(self, raw: np.ndarray) -> "QualityPreprocessor":
        values = np.asarray(raw, dtype=float)
        self.medians_ = np.nanmedian(values, axis=0)
        filled = np.where(np.isnan(values), self.medians_, values)
        self.transformer_ = QuantileTransformer(
            n_quantiles=min(1000, len(filled)),
            output_distribution="uniform",
            subsample=min(100_000, len(filled)),
            random_state=self.random_state,
        ).fit(filled)
        return self

    def transform(self, raw: np.ndarray) -> np.ndarray:
        if self.medians_ is None or self.transformer_ is None:
            raise RuntimeError("QualityPreprocessor must be fitted first.")
        values = np.asarray(raw, dtype=float)
        filled = np.where(np.isnan(values), self.medians_, values)
        quantiles = np.clip(self.transformer_.transform(filled), 0.0, 1.0)
        directed = np.empty_like(quantiles)
        for index, field in enumerate(QUALITY_FIELDS):
            if field in POSITIVE_FIELDS:
                directed[:, index] = quantiles[:, index]
            elif field in NEGATIVE_FIELDS:
                directed[:, index] = 1.0 - quantiles[:, index]
            else:
                directed[:, index] = 1.0 - 2.0 * np.abs(quantiles[:, index] - 0.5)
        return np.clip(directed, 0.0, 1.0)

    def fit_transform(self, raw: np.ndarray) -> np.ndarray:
        return self.fit(raw).transform(raw)


class BaseQualityModel:
    name = "base"

    def fit(self, x: np.ndarray) -> "BaseQualityModel":
        raise NotImplementedError

    def score(self, x: np.ndarray) -> np.ndarray:
        raise NotImplementedError


class CriticQualityModel(BaseQualityModel):
    name = "critic_robust"

    def __init__(self, conflict_penalty: float = 0.15) -> None:
        self.conflict_penalty = conflict_penalty
        self.weights_: np.ndarray | None = None

    def fit(self, x: np.ndarray) -> "CriticQualityModel":
        values = np.asarray(x, dtype=float)
        sigma = np.std(values, axis=0, ddof=1)
        correlation = np.nan_to_num(np.corrcoef(values, rowvar=False), nan=0.0)
        information = sigma * np.sum(1.0 - np.abs(correlation), axis=1)
        if information.sum() <= 0:
            information = np.ones(values.shape[1])
        self.weights_ = information / information.sum()
        return self

    def score(self, x: np.ndarray) -> np.ndarray:
        if self.weights_ is None:
            raise RuntimeError("Model must be fitted first.")
        values = np.asarray(x, dtype=float)
        base = values @ self.weights_
        group_scores = []
        available_fields = list(QUALITY_FIELDS[: values.shape[1]])
        for fields in FEATURE_GROUPS.values():
            indices = [available_fields.index(field) for field in fields if field in available_fields]
            if indices:
                group_scores.append(np.mean(values[:, indices], axis=1))
        conflict = np.std(np.column_stack(group_scores), axis=1) if group_scores else 0.0
        return np.clip(base - self.conflict_penalty * conflict, 0.0, 1.0)


class LatentQualityModel(BaseQualityModel):
    name = "robust_latent"

    def __init__(self, max_components: int = 6, random_state: int = 20260923) -> None:
        self.max_components = max_components
        self.random_state = random_state
        self.scaler_: StandardScaler | None = None
        self.pca_: PCA | None = None
        self.weights_: np.ndarray | None = None
        self.reference_: np.ndarray | None = None

    def _raw_score(self, x: np.ndarray) -> np.ndarray:
        if self.scaler_ is None or self.pca_ is None or self.weights_ is None:
            raise RuntimeError("Model must be fitted first.")
        components = self.pca_.transform(self.scaler_.transform(x))
        return components @ self.weights_

    def fit(self, x: np.ndarray) -> "LatentQualityModel":
        values = np.asarray(x, dtype=float)
        self.scaler_ = StandardScaler().fit(values)
        n_components = max(1, min(self.max_components, values.shape[1]))
        self.pca_ = PCA(n_components=n_components, random_state=self.random_state).fit(
            self.scaler_.transform(values)
        )
        components = self.pca_.transform(self.scaler_.transform(values))
        anchor = np.mean(values, axis=1)
        correlations = np.array(
            [spearman_value(components[:, j], anchor) for j in range(n_components)]
        )
        signs = np.where(np.nan_to_num(correlations) >= 0, 1.0, -1.0)
        magnitudes = np.abs(np.nan_to_num(correlations)) * self.pca_.explained_variance_ratio_
        if magnitudes.sum() <= 0:
            magnitudes = self.pca_.explained_variance_ratio_
        self.weights_ = signs * magnitudes / magnitudes.sum()
        raw_score = components @ self.weights_
        self.reference_ = np.sort(raw_score)
        return self

    def score(self, x: np.ndarray) -> np.ndarray:
        if self.reference_ is None:
            raise RuntimeError("Model must be fitted first.")
        raw_score = self._raw_score(np.asarray(x, dtype=float))
        return np.searchsorted(self.reference_, raw_score, side="right") / len(self.reference_)


class RankAggregationQualityModel(BaseQualityModel):
    name = "robust_rank_aggregation"

    def __init__(self, trim_fraction: float = 0.10) -> None:
        self.trim_fraction = trim_fraction

    def fit(self, x: np.ndarray) -> "RankAggregationQualityModel":
        return self

    def score(self, x: np.ndarray) -> np.ndarray:
        values = np.sort(np.asarray(x, dtype=float), axis=1)
        trim = int(np.floor(self.trim_fraction * values.shape[1]))
        if trim > 0 and values.shape[1] - 2 * trim >= 1:
            values = values[:, trim:-trim]
        return np.mean(values, axis=1)


class BayesianHierarchicalQualityModel(BaseQualityModel):
    """One-factor empirical-Bayes quality estimator with ordinal-rank output."""

    name = "bayesian_hierarchical_quality"

    def __init__(self, random_state: int = 20260923) -> None:
        self.random_state = random_state
        self.scaler_: StandardScaler | None = None
        self.factor_: FactorAnalysis | None = None
        self.sign_: float = 1.0
        self.reference_: np.ndarray | None = None

    def fit(self, x: np.ndarray) -> "BayesianHierarchicalQualityModel":
        values = np.asarray(x, dtype=float)
        self.scaler_ = StandardScaler().fit(values)
        self.factor_ = FactorAnalysis(n_components=1, random_state=self.random_state).fit(
            self.scaler_.transform(values)
        )
        latent = self.factor_.transform(self.scaler_.transform(values)).reshape(-1)
        anchor = np.mean(values, axis=1)
        correlation = spearman_value(latent, anchor)
        self.sign_ = 1.0 if np.nan_to_num(correlation) >= 0 else -1.0
        latent *= self.sign_
        self.reference_ = np.sort(latent)
        return self

    def score(self, x: np.ndarray) -> np.ndarray:
        if self.scaler_ is None or self.factor_ is None or self.reference_ is None:
            raise RuntimeError("Model must be fitted first.")
        latent = self.factor_.transform(self.scaler_.transform(x)).reshape(-1) * self.sign_
        return np.searchsorted(self.reference_, latent, side="right") / len(self.reference_)


QUALITY_MODEL_FACTORIES = {
    "S1": CriticQualityModel,
    "S2": LatentQualityModel,
    "S3": RankAggregationQualityModel,
    "S4": BayesianHierarchicalQualityModel,
}


def domain_mean_scores(
    scores: np.ndarray,
    domains: np.ndarray,
    *,
    hierarchical_shrinkage: bool = False,
    prior_strength: float = 100.0,
) -> pd.Series:
    frame = pd.DataFrame({"domain": domains, "score": scores})
    grouped = frame.groupby("domain")["score"].agg(["mean", "count"])
    if hierarchical_shrinkage:
        global_mean = float(frame["score"].mean())
        grouped["mean"] = (
            grouped["count"] * grouped["mean"] + prior_strength * global_mean
        ) / (grouped["count"] + prior_strength)
    return grouped["mean"].sort_index()


def bootstrap_domain_stability(
    scores: np.ndarray,
    domains: np.ndarray,
    *,
    repetitions: int = 200,
    random_state: int = 20260923,
) -> float:
    rng = np.random.RandomState(random_state)
    unique_domains = np.unique(domains)
    indices = {domain: np.flatnonzero(domains == domain) for domain in unique_domains}
    baseline = np.array([np.mean(scores[indices[domain]]) for domain in unique_domains])
    correlations = []
    for _ in range(repetitions):
        sampled_means = []
        for domain in unique_domains:
            domain_indices = indices[domain]
            sampled = rng.choice(domain_indices, size=len(domain_indices), replace=True)
            sampled_means.append(np.mean(scores[sampled]))
        correlations.append(spearman_value(baseline, np.asarray(sampled_means)))
    return float(np.nanmean(correlations))


def feature_dropout_stability(
    x: np.ndarray,
    domains: np.ndarray,
    scheme_id: str,
    *,
    repetitions: int = 10,
    keep_fraction: float = 0.80,
    random_state: int = 20260923,
) -> float:
    rng = np.random.RandomState(random_state)
    factory = QUALITY_MODEL_FACTORIES[scheme_id]
    baseline_model = factory().fit(x)
    baseline_scores = baseline_model.score(x)
    unique_domains = np.unique(domains)
    baseline_means = np.array(
        [np.mean(baseline_scores[domains == domain]) for domain in unique_domains]
    )
    correlations = []
    keep_count = max(3, int(np.ceil(keep_fraction * x.shape[1])))
    for _ in range(repetitions):
        keep = np.sort(rng.choice(x.shape[1], size=keep_count, replace=False))
        perturbed_model = factory().fit(x[:, keep])
        perturbed_scores = perturbed_model.score(x[:, keep])
        perturbed_means = np.array(
            [np.mean(perturbed_scores[domains == domain]) for domain in unique_domains]
        )
        correlations.append(spearman_value(baseline_means, perturbed_means))
    return float(np.nanmean(correlations))


def map_quality_to_mixture_domains(quality_scores: pd.Series) -> pd.DataFrame:
    """Map seven observed quality domains to the 17 RegMix domains."""

    weights: dict[str, dict[str, float]] = {
        "arxiv": {"arxiv": 1.0},
        "github": {"github": 1.0},
        "stackexchange": {"stackexchange": 1.0},
        "wikipedia_en": {"wikipedia": 1.0},
        "gutenberg_pg_19": {"book": 1.0},
        "pile_cc": {"commoncrawl": 1.0},
        "dm_mathematics": {"arxiv": 0.55, "stackexchange": 0.45},
        "freelaw": {"book": 0.55, "arxiv": 0.45},
        "nih_exporter": {"wikipedia": 0.55, "commoncrawl": 0.45},
        "pubmed_central": {"arxiv": 0.80, "wikipedia": 0.20},
        "philpapers": {"arxiv": 0.65, "book": 0.35},
        "enron_emails": {"commoncrawl": 0.80, "stackexchange": 0.20},
        "ubuntu_irc": {"stackexchange": 0.60, "github": 0.40},
        "europarl": {"wikipedia": 0.60, "commoncrawl": 0.40},
        "hackernews": {"stackexchange": 0.55, "commoncrawl": 0.45},
        "pubmed_abstracts": {"arxiv": 0.85, "wikipedia": 0.15},
        "uspto_backgrounds": {"arxiv": 0.60, "book": 0.40},
    }
    direct_domains = {"arxiv", "github", "stackexchange"}
    near_direct_domains = {"wikipedia_en", "gutenberg_pg_19", "pile_cc"}
    rows = []
    for mixture_domain, mapping in weights.items():
        score = sum(weight * float(quality_scores[domain]) for domain, weight in mapping.items())
        if mixture_domain in direct_domains:
            mapping_type = "direct"
        elif mixture_domain in near_direct_domains:
            mapping_type = "near_direct"
        else:
            mapping_type = "inferred"
        rows.append(
            {
                "mixture_domain": mixture_domain,
                "q_score": score,
                "mapping_basis": "+".join(f"{domain}:{weight:.2f}" for domain, weight in mapping.items()),
                "mapping_type": mapping_type,
            }
        )
    return pd.DataFrame(rows).sort_values("mixture_domain").reset_index(drop=True)


def iter_extended_quality_files(quality_root: Path) -> Iterable[tuple[Path, str]]:
    extended = quality_root / "slimpajama_quality_extended"
    for domain in ("arxiv", "github"):
        for path in sorted(extended.glob(f"{domain}_*.jsonl.xz")):
            yield path, domain
