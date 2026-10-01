"""Branch-and-bound search equivalent to SVSM ranking, for the pruning ablation."""
from __future__ import annotations

import heapq
from dataclasses import dataclass

import numpy as np

from .types import Candidate, Item, canonical_itemset


@dataclass
class _WorstFirst:
    candidate: Candidate

    def __lt__(self, other):
        # The heap root has the lowest score, breaking ties by greatest lexicographic key.
        a, b = self.candidate, other.candidate
        return (-a.score, a.itemset) > (-b.score, b.itemset)


def search_candidates(frequent_items: list[tuple[Item, float]], n_rows: float,
                      candidate_count: int, min_size: int = 1,
                      max_size: int = 4, *, pruning: bool = True):
    """Return candidates and search counts using released counts, never true supports.

    Both ablation arms use the same traversal, scoring, and heap maintenance.
    Each factor is at most 0.9, so descendants cannot outscore their parent.
    Stop expansion only strictly below the threshold, preserving ties.
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
            # A tolerance avoids pruning from rounding; equal scores use lexicographic order.
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
