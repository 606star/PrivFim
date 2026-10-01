from __future__ import annotations

import math
from dataclasses import asdict, dataclass

import numpy as np

from .data import VerticalDataset
from .types import Candidate, CandidateEstimate, Itemset


@dataclass(frozen=True)
class EvaluationMetrics:
    precision: float
    recall: float
    f1: float
    jaccard: float
    ncr: float
    mae: float
    rmse: float
    nmae: float
    nrmse: float
    candidate_recall: float

    def to_dict(self) -> dict[str, float]:
        return asdict(self)


def exact_candidate_supports(
    dataset: VerticalDataset,
    candidates: list[Candidate],
) -> dict[Itemset, int]:
    return {
        candidate.itemset: dataset.support(candidate.itemset)
        for candidate in candidates
    }


def evaluate_estimates(
    estimates: list[CandidateEstimate],
    exact_supports: dict[Itemset, int],
    k: int,
    n_rows: int | None = None,
    error_itemsets: set[Itemset] | None = None,
) -> EvaluationMetrics:
    true_order = sorted(exact_supports, key=lambda key: (-exact_supports[key], key))
    predicted_order = [estimate.itemset for estimate in estimates]
    # Do not shrink the true top-k to hide candidates made unestimable by a key cap.
    effective_k = min(k, len(true_order))
    true_top = true_order[:effective_k]
    predicted_top = predicted_order[:k]

    overlap = len(set(true_top) & set(predicted_top))
    precision = overlap / len(predicted_top) if predicted_top else 0.0
    recall = overlap / len(true_top) if true_top else 0.0
    f1 = (
        2.0 * precision * recall / (precision + recall)
        if precision + recall > 0
        else 0.0
    )
    union = len(set(true_top) | set(predicted_top))
    jaccard = overlap / union if union else 0.0

    rank_score = {
        itemset: effective_k - rank for rank, itemset in enumerate(true_top)
    }
    maximum_score = effective_k * (effective_k + 1) / 2.0
    ncr = (
        sum(rank_score.get(itemset, 0) for itemset in predicted_top) / maximum_score
        if maximum_score
        else 0.0
    )

    estimated_by_itemset = {
        estimate.itemset: estimate.estimated_count for estimate in estimates
    }
    error_estimates = (
        estimates
        if error_itemsets is None
        else [estimate for estimate in estimates if estimate.itemset in error_itemsets]
    )
    errors = np.asarray(
        [
            estimate.estimated_count - exact_supports[estimate.itemset]
            for estimate in error_estimates
        ],
        dtype=np.float64,
    )
    mae = float(np.mean(np.abs(errors))) if errors.size else 0.0
    rmse = float(math.sqrt(np.mean(errors**2))) if errors.size else 0.0
    denominator = float(n_rows or max(exact_supports.values(), default=1) or 1)
    candidate_items = set(estimated_by_itemset)
    candidate_recall = (
        len(set(true_top) & candidate_items) / len(true_top) if true_top else 0.0
    )
    return EvaluationMetrics(
        precision=precision,
        recall=recall,
        f1=f1,
        jaccard=jaccard,
        ncr=ncr,
        mae=mae,
        rmse=rmse,
        nmae=mae / denominator,
        nrmse=rmse / denominator,
        candidate_recall=candidate_recall,
    )


def metrics_by_itemset_size(
    estimates: list[CandidateEstimate],
    exact_supports: dict[Itemset, int],
    k: int,
    n_rows: int,
) -> dict[str, dict[str, float]]:
    sizes = sorted({len(itemset) for itemset in exact_supports})
    result = {}
    for size in sizes:
        size_truth = {
            itemset: count
            for itemset, count in exact_supports.items()
            if len(itemset) == size
        }
        size_estimates = [
            estimate for estimate in estimates if len(estimate.itemset) == size
        ]
        if not size_truth or not size_estimates:
            continue
        result[str(size)] = evaluate_estimates(
            size_estimates,
            size_truth,
            k,
            n_rows=n_rows,
        ).to_dict()
    return result


def errors_by_support_band(
    estimates: list[CandidateEstimate],
    exact_supports: dict[Itemset, int],
    n_rows: int,
) -> dict[str, dict[str, float]]:
    bands = {
        "0-1%": (0.0, 0.01),
        "1-5%": (0.01, 0.05),
        "5-20%": (0.05, 0.20),
        "20-100%": (0.20, 1.01),
    }
    result = {}
    for label, (lower, upper) in bands.items():
        errors = np.asarray(
            [
                estimate.estimated_count - exact_supports[estimate.itemset]
                for estimate in estimates
                if lower
                <= exact_supports[estimate.itemset] / max(n_rows, 1)
                < upper
            ],
            dtype=np.float64,
        )
        if errors.size:
            result[label] = {
                "count": int(errors.size),
                "mae": float(np.mean(np.abs(errors))),
                "rmse": float(math.sqrt(np.mean(errors**2))),
                "nrmse": float(math.sqrt(np.mean(errors**2)) / max(n_rows, 1)),
            }
    return result
