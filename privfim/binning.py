from __future__ import annotations

from typing import TypeAlias

import numpy as np

from .types import Item, Itemset, canonical_itemset


# Each attribute has disjoint value bins. Standard Bin uses contiguous ranges;
# BinP protects high-value candidates separately, so tail bins may be noncontiguous.
# Boundaries depend only on public domains, first-round DP histograms, and candidate priors.
BinningPlan: TypeAlias = tuple[tuple[tuple[int, ...], ...], ...]


def bin_values(domain: tuple[int, ...], bin_count: int) -> tuple[tuple[int, ...], ...]:
    """Split the ordered public domain evenly into at most ``bin_count`` bins."""
    if bin_count < 1:
        raise ValueError("bin_count 必须大于 0")
    values = tuple(sorted(int(value) for value in domain))
    if not values:
        raise ValueError("属性公开值域不能为空")
    bucket_count = min(bin_count, len(values))
    buckets: list[list[int]] = [[] for _ in range(bucket_count)]
    for index, value in enumerate(values):
        buckets[index * bucket_count // len(values)].append(value)
    return tuple(tuple(bucket) for bucket in buckets)


def private_quantile_bins(
    domains: tuple[tuple[int, ...], ...],
    noisy_counts: dict[tuple[int, int], float],
    bin_count: int,
) -> BinningPlan:
    """Build contiguous, approximately equal-mass bins from the first-round DP histogram.

    Boundaries use only released Laplace counts and are transcript post-processing.
    If all noisy counts for an attribute clip to zero, use equal-width public bins
    without consulting raw private rows. Each bin contains at least one public value.
    """
    return tuple(
        _private_quantile_values(
            tuple(sorted(int(value) for value in domain)),
            [
                max(float(noisy_counts.get((attr, value), 0.0)), 0.0)
                for value in sorted(int(value) for value in domain)
            ],
            bin_count,
        )
        for attr, domain in enumerate(domains)
    )


def protected_private_quantile_bins(
    domains: tuple[tuple[int, ...], ...],
    noisy_counts: dict[Item, float],
    bin_count: int,
    item_scores: dict[Item, float],
) -> BinningPlan:
    """Protect high-value candidates and bin remaining values by their noisy mass.

    Use at most bin_count bins, reserving at most half for protected single values.
    Compress the long tail using only the DP transcript, public domains, and candidate priors.
    """
    if bin_count < 1:
        raise ValueError("bin_count 必须大于 0")
    plans: list[tuple[tuple[int, ...], ...]] = []
    for attr, domain in enumerate(domains):
        values = tuple(sorted(int(value) for value in domain))
        if not values:
            raise ValueError("属性公开值域不能为空")
        if len(values) <= bin_count:
            plans.append(tuple((value,) for value in values))
            continue
        protected_budget = bin_count // 2
        scored = sorted(
            (
                max(float(item_scores.get((attr, value), 0.0)), 0.0),
                max(float(noisy_counts.get((attr, value), 0.0)), 0.0),
                value,
            )
            for value in values
        )
        scored.sort(key=lambda row: (-row[0], -row[1], row[2]))
        protected = tuple(value for score, _mass, value in scored if score > 0.0)[:protected_budget]
        protected_set = set(protected)
        tail = tuple(value for value in values if value not in protected_set)
        tail_bins = _private_quantile_values(
            tail,
            [max(float(noisy_counts.get((attr, value), 0.0)), 0.0) for value in tail],
            max(1, min(bin_count - len(protected), len(tail))),
        )
        plans.append(tuple(sorted(tuple((value,) for value in protected) + tail_bins, key=lambda bucket: bucket[0])))
    return tuple(plans)


def _private_quantile_values(
    domain: tuple[int, ...], weights: list[float], bin_count: int
) -> tuple[tuple[int, ...], ...]:
    values = tuple(int(value) for value in domain)
    if len(values) != len(weights):
        raise ValueError("值域与私有直方图长度不一致")
    if bin_count < 1:
        raise ValueError("bin_count 必须大于 0")
    if not values:
        raise ValueError("属性公开值域不能为空")
    bucket_count = min(bin_count, len(values))
    if bucket_count == 1:
        return (values,)

    nonnegative = np.maximum(np.asarray(weights, dtype=np.float64), 0.0)
    if float(nonnegative.sum()) <= 0.0:
        return bin_values(values, bucket_count)

    # Choose a prefix closest to the remaining average mass for each bin. max_end
    # reserves a value for every later bin, yielding finer bins around frequent values.
    buckets: list[tuple[int, ...]] = []
    start = 0
    remaining_mass = float(nonnegative.sum())
    for bucket_index in range(bucket_count - 1):
        remaining_bucket_count = bucket_count - bucket_index
        max_end = len(values) - (remaining_bucket_count - 1)
        target_mass = remaining_mass / remaining_bucket_count
        prefix_mass = 0.0
        best_end = start + 1
        best_distance = float("inf")
        for end in range(start + 1, max_end + 1):
            prefix_mass += float(nonnegative[end - 1])
            distance = abs(prefix_mass - target_mass)
            if distance < best_distance:
                best_end = end
                best_distance = distance
            # Mass can only increase; after overshooting, later prefixes cannot be closer.
            if prefix_mass >= target_mass:
                break
        buckets.append(values[start:best_end])
        remaining_mass -= float(nonnegative[start:best_end].sum())
        start = best_end
    buckets.append(values[start:])
    return tuple(buckets)


def _attribute_bins(
    attr: int,
    domain: tuple[int, ...],
    bin_count: int,
    binning: BinningPlan | None,
) -> tuple[tuple[int, ...], ...]:
    if binning is None:
        return bin_values(domain, bin_count)
    if not 0 <= attr < len(binning):
        raise ValueError(f"属性 {attr} 不在分箱计划中")
    buckets = tuple(tuple(int(value) for value in bucket) for bucket in binning[attr])
    expected = tuple(sorted(int(value) for value in domain))
    observed = tuple(value for bucket in buckets for value in bucket)
    if not buckets or tuple(sorted(observed)) != expected or len(set(observed)) != len(observed):
        raise ValueError(f"属性 {attr} 的分箱计划不是公开值域的完整互斥划分")
    return buckets


def value_bin_index(
    value: int,
    domain: tuple[int, ...],
    bin_count: int,
    binning: BinningPlan | None = None,
    attr: int | None = None,
) -> int:
    """Return the bin index containing a public value."""
    buckets = _attribute_bins(
        0 if attr is None else attr, domain, bin_count, binning
    )
    for index, bucket in enumerate(buckets):
        if int(value) in bucket:
            return index
    raise ValueError(f"取值 {value} 不在公开值域中")


def binned_itemset(
    itemset: Itemset,
    domains: tuple[tuple[int, ...], ...],
    bin_count: int,
    binning: BinningPlan | None = None,
) -> Itemset:
    """Map an exact itemset to its public binned representation."""
    return canonical_itemset(
        (attr, value_bin_index(value, domains[attr], bin_count, binning, attr))
        for attr, value in itemset
    )


def bin_membership_values(
    attr: int,
    bin_index: int,
    domains: tuple[tuple[int, ...], ...],
    bin_count: int,
    binning: BinningPlan | None = None,
) -> np.ndarray:
    """Return the original values represented by a public bin key."""
    buckets = _attribute_bins(attr, domains[attr], bin_count, binning)
    if not 0 <= bin_index < len(buckets):
        raise ValueError(f"属性 {attr} 的分箱编号 {bin_index} 不合法")
    return np.asarray(buckets[bin_index], dtype=np.int64)


def within_bin_probability(
    attr: int,
    value: int,
    noisy_counts: dict[tuple[int, int], float],
    domains: tuple[tuple[int, ...], ...],
    bin_count: int,
    binning: BinningPlan | None = None,
) -> float:
    """Estimate ``P(value | public_bin(value))`` from the first-round DP histogram."""
    values = bin_membership_values(
        attr,
        value_bin_index(value, domains[attr], bin_count, binning, attr),
        domains,
        bin_count,
        binning,
    )
    counts = np.asarray(
        [max(float(noisy_counts.get((attr, int(candidate)), 0.0)), 0.0) for candidate in values],
        dtype=np.float64,
    )
    total = float(counts.sum())
    if total <= 0.0:
        return 1.0 / len(values)
    position = int(np.flatnonzero(values == value)[0])
    return float(counts[position] / total)
