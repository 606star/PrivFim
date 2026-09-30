"""独立离线精确 Top-k 搜索，支持 k 超过单项值域总数。"""
from __future__ import annotations

import heapq
from collections import Counter
from dataclasses import dataclass

import numpy as np

from privfim.data import VerticalDataset
from privfim.types import Item, Itemset


@dataclass
class _WorstSupport:
    key: Itemset
    count: int

    def __lt__(self, other):
        return (-self.count, self.key) > (-other.count, other.key)


def exact_global_topk(dataset: VerticalDataset, k: int, max_size: int = 4,
                      allowed_items: set[Item] | None = None, *, min_size: int = 1):
    """动态提高支持数下界，绝不剪掉与边界同频的项集。

    真值仅用于离线评价。搜索空间包含全部属性值的合法 min_size..max_size
    项集；allowed_items 只用于对照旧的第一轮单项池评价口径。
    """
    if k < 1 or not 1 <= min_size <= max_size:
        raise ValueError("k 必须为正数，且 1 <= min_size <= max_size")
    counts = {}
    for attr in dataset.attributes:
        values, frequencies = np.unique(dataset.data[:, attr], return_counts=True)
        observed = dict(zip(map(int, values), map(int, frequencies)))
        for value in dataset.domains[attr]:
            item = (int(attr), int(value))
            if allowed_items is None or item in allowed_items:
                counts[item] = observed.get(value, 0)
    # 高频单项优先扩展，尽早找到足够多的候选，提高精确搜索的剪枝下界。
    items = sorted(counts, key=lambda item: (-counts[item], item))
    heap = []

    def retain(key, count):
        entry = _WorstSupport(key, count)
        if len(heap) < k:
            heapq.heappush(heap, entry)
        elif (-count, key) < (-heap[0].count, heap[0].key):
            heapq.heapreplace(heap, entry)

    seeded_pairs = set()
    if min_size == 1:
        for item in items:
            retain((item,), counts[item])
    elif min_size == 2:
        # 用频数最高的有限个不同属性单项形成合法二项集下界。它只是
        # 已验证的支持度下界，不依赖第一轮私有候选，也不会漏剪真值。
        starters = items[:64]
        starter_bitmaps = {
            item: int.from_bytes(np.packbits(dataset.data[:, item[0]] == item[1],
                                              bitorder="little").tobytes(), "little")
            for item in starters
        }
        for i, left in enumerate(starters):
            for right in starters[i + 1:]:
                if left[0] != right[0]:
                    key = tuple(sorted((left, right)))
                    seeded_pairs.add(key)
                    retain(key,
                           (starter_bitmaps[left] & starter_bitmaps[right]).bit_count())

    def floor():
        return heap[0].count if len(heap) == k else 0

    initial_floor = floor()
    items = [item for item in items if counts[item] >= initial_floor]
    bitmaps = {item: int.from_bytes(np.packbits(dataset.data[:, item[0]] == item[1],
                                               bitorder="little").tobytes(), "little")
               for item in items}
    support_histogram = Counter(counts.values()) if min_size == 1 else Counter()
    visited, pruned = len(counts), 0

    def visit(prefix, bitmap, start):
        nonlocal visited, pruned
        used_attrs = {attr for attr, _ in prefix}
        for index in range(start, len(items)):
            item = items[index]
            if item[0] in used_attrs:
                continue
            if counts[item] < floor():
                pruned += len(items) - index
                break
            joined = bitmap & bitmaps[item]
            count = joined.bit_count()
            visited += 1
            if count < floor():
                pruned += 1
                continue
            key = tuple(sorted((*prefix, item)))
            if len(key) >= min_size:
                support_histogram[count] += 1
                if key not in seeded_pairs:
                    retain(key, count)
            if len(key) < max_size:
                visit(key, joined, index + 1)

    if max_size > 1:
        for index, item in enumerate(items):
            if counts[item] < floor():
                break
            visit((item,), bitmaps[item], index + 1)
    top = dict((entry.key, entry.count) for entry in
               sorted(heap, key=lambda entry: (-entry.count, entry.key)))
    cutoff = min(top.values()) if top else None
    return top, {
        "universe": "global" if allowed_items is None else "first_stage_pool",
        "exact": True, "min_size": min_size, "max_size": max_size, "requested_k": k,
        "singleton_count": len(counts), "returned_count": len(top),
        "tie_break": "support descending, canonical itemset ascending",
        "initial_support_floor": initial_floor, "cutoff_support": cutoff,
        "cutoff_tie_count": support_histogram[cutoff] if cutoff is not None else 0,
        "visited_nodes": visited, "pruned_nodes": pruned,
        "topk_size_histogram": dict(sorted(Counter(map(len, top)).items())),
    }
