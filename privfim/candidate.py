from __future__ import annotations

import math
from itertools import combinations

import numpy as np

from .types import Candidate, Item, canonical_itemset


def select_frequent_singletons(
    noisy_counts: dict[Item, float],
    limit: int,
) -> list[tuple[Item, float]]:
    cleaned = [(item, max(0.0, float(count))) for item, count in noisy_counts.items()]
    cleaned.sort(key=lambda pair: (-pair[1], pair[0]))
    return cleaned[:limit]


def construct_itemset_first_candidates(
    frequent_items: list[tuple[Item, float]],
    n_rows: int,
    candidate_count: int | None,
    min_size: int,
    max_size: int,
) -> list[Candidate]:
    """在第一轮 DP 单项直方图上进行候选优先级消融。

    两种策略复用相同的候选评分和加噪单项统计。本策略优先保留多项集，
    在名额充足时为单项保留一个位置；默认策略则将各长度候选统一排序。
    这里没有直接采集本地联合项集频数，仅测量候选组成的变化。
    """
    all_candidates = construct_svsm_candidates(
        frequent_items=frequent_items,
        n_rows=n_rows,
        candidate_count=None,
        min_size=min_size,
        max_size=max_size,
    )
    if candidate_count is None:
        return sorted(
            all_candidates,
            key=lambda candidate: (
                len(candidate.itemset) == 1,
                -candidate.score,
                candidate.itemset,
            ),
        )
    if candidate_count <= 0:
        return []
    itemsets = [candidate for candidate in all_candidates if len(candidate.itemset) > 1]
    singletons = [candidate for candidate in all_candidates if len(candidate.itemset) == 1]
    # 联合项集优先但保留至少一个单项位置，避免第二阶段完全失去 1-项集。
    itemset_quota = min(len(itemsets), max(candidate_count - 1, 0))
    selected = itemsets[:itemset_quota]
    remaining = candidate_count - len(selected)
    selected.extend(singletons[:remaining])
    remaining = candidate_count - len(selected)
    if remaining:
        selected.extend(itemsets[itemset_quota:itemset_quota + remaining])
    return selected


def construct_svsm_candidates(
    frequent_items: list[tuple[Item, float]],
    n_rows: int,
    candidate_count: int | None,
    min_size: int,
    max_size: int,
) -> list[Candidate]:
    """按 SVSM 的归一化频率乘积选择包含 1-项集的统一 Top 候选。"""
    if not frequent_items:
        return []

    bounded_counts = {
        item: float(np.clip(count, 0.0, n_rows)) for item, count in frequent_items
    }
    max_count = max(bounded_counts.values())
    if max_count <= 0:
        return []

    normalized = {
        item: 0.9 * bounded_counts[item] / max_count for item, _ in frequent_items
    }
    raw_counts = bounded_counts
    items = [item for item, _ in frequent_items]

    candidates: list[Candidate] = []
    upper_size = min(max_size, len(items))
    for size in range(min_size, upper_size + 1):
        for combination in combinations(items, size):
            attrs = [item[0] for item in combination]
            if len(attrs) != len(set(attrs)):
                continue

            itemset = canonical_itemset(combination)
            score = float(np.prod([normalized[item] for item in itemset]))
            relative = [raw_counts[item] / n_rows for item in itemset]
            guessed_count = float(n_rows * np.prod(relative))
            candidates.append(
                Candidate(itemset=itemset, score=score, guessed_count=guessed_count)
            )

    candidates.sort(key=lambda candidate: (-candidate.score, candidate.itemset))
    return candidates if candidate_count is None else candidates[:candidate_count]


def construct_direct_candidates(
    frequent_items: list[tuple[Item, float]],
    n_rows: int,
    min_size: int,
    max_size: int,
) -> list[Candidate]:
    """枚举频繁单项池中的全部合法项集，作为 MAP-S-All 的共同评价域。"""
    if not frequent_items:
        return []
    return construct_svsm_candidates(
        frequent_items=frequent_items,
        n_rows=n_rows,
        candidate_count=sum(
            math.comb(len(frequent_items), size)
            for size in range(min_size, min(max_size, len(frequent_items)) + 1)
        ),
        min_size=min_size,
        max_size=max_size,
    )
