"""离线精确真值；不向任何私有挖掘阶段反馈真实计数。"""
from __future__ import annotations

from collections import Counter

import numpy as np

from privfim.data import VerticalDataset
from privfim.types import Item, Itemset


def exact_topk(
    dataset: VerticalDataset,
    k: int,
    max_size: int = 4,
    *,
    allowed_items: set[Item] | None = None,
) -> tuple[dict[Itemset, int], dict]:
    """精确挖掘 1..max_size 项集，按 (-真实支持数, 项集) 排序。

    默认覆盖全部属性值。allowed_items 只用于复现旧的受限真值口径。
    第 k 大单项支持数是全局第 k 大支持数的下界。支持数反单调，
    因而小于此界的单项及其扩展均不可能进入 Top-k；等号不能剪枝。
    ID 集合用 Python 整数位图表示，交集计数为 (left & right).bit_count()。
    """
    if k < 1 or max_size < 1:
        raise ValueError("k 和 max_size 必须为正数")
    counts: dict[Item, int] = {}
    for attr in dataset.attributes:
        values, frequencies = np.unique(dataset.data[:, attr], return_counts=True)
        observed = dict(zip(map(int, values), map(int, frequencies)))
        for value in dataset.domains[attr]:
            item = (int(attr), int(value))
            if allowed_items is None or item in allowed_items:
                counts[item] = observed.get(value, 0)
    ordered_counts = sorted(counts.values(), reverse=True)
    threshold = ordered_counts[k - 1] if len(ordered_counts) >= k else 0
    items = sorted(item for item, count in counts.items() if count >= threshold)
    bitmaps = {
        item: int.from_bytes(
            np.packbits(dataset.data[:, item[0]] == item[1], bitorder="little").tobytes(),
            "little",
        )
        for item in items
    }
    found: list[tuple[Itemset, int]] = []
    visited = 0

    def visit(prefix: Itemset, bitmap: int, start: int) -> None:
        nonlocal visited
        for position in range(start, len(items)):
            item = items[position]
            if prefix and item[0] <= prefix[-1][0]:
                continue
            visited += 1
            joined = bitmap & bitmaps[item]
            count = joined.bit_count()
            if count < threshold:
                continue
            key = prefix + (item,)
            found.append((key, count))
            if len(key) < max_size:
                visit(key, joined, position + 1)

    visit((), (1 << dataset.n_rows) - 1, 0)
    found.sort(key=lambda pair: (-pair[1], pair[0]))
    top = dict(found[:k])
    cutoff = list(top.values())[-1] if top else None
    return top, {
        "universe": "all_attribute_values" if allowed_items is None else "restricted_first_stage_pool",
        "min_itemset_size": 1,
        "max_itemset_size": max_size,
        "tie_break": "support descending, canonical itemset ascending",
        "singleton_count": len(counts),
        "safe_support_floor": threshold,
        "eligible_singletons": len(items),
        "visited_nodes": visited,
        "itemsets_at_or_above_floor": len(found),
        "topk_cutoff_count": cutoff,
        "itemsets_tied_at_cutoff": sum(count == cutoff for _, count in found),
        "topk_size_histogram": dict(sorted(Counter(map(len, top)).items())),
    }


def support_tie_recall(predictions: list[Itemset], truth: dict[Itemset, int], supports: dict[Itemset, int]) -> float:
    """辅助诊断：边界处同支持数的项集可互换，不替代原定义的 F1/NCR。"""
    if not truth:
        return 0.0
    cutoff = min(truth.values())
    required = {key for key, count in truth.items() if count > cutoff}
    predicted = set(predictions)
    tied_slots = len(truth) - len(required)
    tied_hits = sum(supports[key] == cutoff for key in predicted)
    return (len(required & predicted) + min(tied_slots, tied_hits)) / len(truth)
