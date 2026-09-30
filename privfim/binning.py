from __future__ import annotations

from typing import TypeAlias

import numpy as np

from .types import Item, Itemset, canonical_itemset


# 每个属性对应一组互斥值桶；普通 Bin 使用连续桶，BinP 会把候选高价值
# value 单独保护成桶，因此尾部桶可能不是原始值域上的连续区间。桶边界
# 只由公开值域、第一轮已私有化直方图和候选先验确定。
BinningPlan: TypeAlias = tuple[tuple[tuple[int, ...], ...], ...]


def bin_values(domain: tuple[int, ...], bin_count: int) -> tuple[tuple[int, ...], ...]:
    """按公开离散值域顺序将值均匀分配到至多 ``bin_count`` 个桶。"""
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
    """由第一轮私有直方图构造连续的近似等质量桶。

    边界只读取第一轮已经发布的 Laplace 计数，因此是该轮 transcript 的后处理。
    如果一个属性的全部私有计数都截断为零，则退回公开等宽桶，避免由私有原始行
    补救边界。每个桶至少保留一个公开值，以保证桶映射始终完整。
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
    """保护候选高价值 value，其余 value 使用私有质量分桶。

    总桶数仍不超过 bin_count；最多一半桶用于单值保护，剩余桶压缩长尾。
    输入只来自第一轮 DP transcript、公开值域和候选先验，因此是后处理。
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

    # 逐桶选择最接近当前剩余平均质量的连续前缀。max_end 保证剩余每桶至少
    # 有一个公开值；相较按值编号等宽，这会把高频相邻值放进更细的桶。
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
            # 后续质量只会继续增大；越过目标后不会再次变得更接近。
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
    """返回一个公开值所属的桶编号。"""
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
    """将精确项集映射为相应的公开分箱项集。"""
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
    """返回一个公开分箱键对应的原始值集合。"""
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
    """用第一轮私有直方图估计 ``P(value | public_bin(value))``。"""
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
