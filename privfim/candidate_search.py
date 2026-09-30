"""与现有 SVSM 排序严格等价的分支限界搜索，用于独立剪枝消融。"""
from __future__ import annotations

import heapq
from dataclasses import dataclass

import numpy as np

from .types import Candidate, Item, canonical_itemset


@dataclass
class _WorstFirst:
    candidate: Candidate

    def __lt__(self, other):
        # 堆顶为得分最低、同分时字典序最大的候选。
        a, b = self.candidate, other.candidate
        return (-a.score, a.itemset) > (-b.score, b.itemset)


def search_candidates(frequent_items: list[tuple[Item, float]], n_rows: float,
                      candidate_count: int, min_size: int = 1,
                      max_size: int = 4, *, pruning: bool = True):
    """返回候选与搜索计数；只用已发布频数，不访问真实支持数。

    相同的遍历、评分和堆维护用于两个实验臂。每个乘数至多为 0.9，
    后代得分不超过父节点。严格低于阈值才停止扩展，保留同分项。
    """
    stats = {"visited_nodes": 0, "expanded_nodes": 0, "pruned_nodes": 0}
    if not frequent_items or candidate_count <= 0:
        return [], stats
    counts = {item: float(np.clip(count, 0., n_rows)) for item, count in frequent_items}
    maximum = max(counts.values())
    if maximum <= 0:
        return [], stats
    scores = {item: .9 * count / maximum for item, count in counts.items()}
    items = [item for item, _ in frequent_items]
    heap = []

    def visit(prefix, attrs, start):
        for pos in range(start, len(items)):
            item = items[pos]
            if item[0] in attrs:
                continue
            key = canonical_itemset((*prefix, item))
            score = float(np.prod([scores[t] for t in key]))
            stats["visited_nodes"] += 1
            # 浮点裕量避免临界舍入误剪；相等分数继续按字典序比较。
            if pruning and len(heap) == candidate_count and score < heap[0].candidate.score * (1 - 1e-12):
                stats["pruned_nodes"] += 1
                continue
            if len(key) >= min_size:
                candidate = Candidate(key, score, float(n_rows * np.prod([counts[t] / n_rows for t in key])))
                entry = _WorstFirst(candidate)
                if len(heap) < candidate_count:
                    heapq.heappush(heap, entry)
                elif (-score, key) < (-heap[0].candidate.score, heap[0].candidate.itemset):
                    heapq.heapreplace(heap, entry)
            if len(key) < max_size:
                stats["expanded_nodes"] += 1
                visit(key, attrs | {item[0]}, pos + 1)

    visit((), set(), 0)
    result = sorted((e.candidate for e in heap), key=lambda c: (-c.score, c.itemset))
    return result, stats
