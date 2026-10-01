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
    """Ablate candidate prioritization using the first-round DP item histogram.

    Both policies share candidate scores and noisy item statistics. This policy
    prioritizes joint itemsets and reserves a singleton slot when capacity permits;
    the default ranks all lengths together. No local joint counts are collected.
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
    # Prioritize joint itemsets but reserve a singleton slot for the second round.
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
    """Select top candidates, including singletons, by SVSM normalized-frequency products."""
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
    """Enumerate legal itemsets from the item pool for the shared MAP-S-All evaluation domain."""
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
