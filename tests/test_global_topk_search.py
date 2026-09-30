import numpy as np
import pytest
from itertools import combinations, product

from experiments.global_topk_search import exact_global_topk
from experiments.global_truth import exact_topk
from privfim.data import VerticalDataset


@pytest.mark.parametrize("seed", range(5))
@pytest.mark.parametrize("k", [1, 5, 15, 100, 500])
def test_dynamic_truth_matches_previous_exact_search(seed, k):
    data = np.random.default_rng(seed).integers(0, 3, size=(31, 4))
    ds = VerticalDataset(data, (0, 1, 2, 3), ((0, 1), (2, 3)), ((0, 1, 2),) * 4)
    for max_size in (1, 2, 4):
        expected, _ = exact_topk(ds, k, max_size)
        actual, _ = exact_global_topk(ds, k, max_size)
        assert list(actual.items()) == list(expected.items())


def test_more_results_than_singletons_with_support_ties():
    ds = VerticalDataset(np.zeros((11, 5), dtype=int), tuple(range(5)),
                         ((0, 1, 2), (3, 4)), ((0,),) * 5)
    actual, certificate = exact_global_topk(ds, 20)
    expected, _ = exact_topk(ds, 20)
    assert list(actual.items()) == list(expected.items())
    assert certificate["cutoff_tie_count"] == 30
    assert len(actual) == 20


def test_restricted_search_uses_the_requested_item_pool_only():
    ds = VerticalDataset(np.array([[0, 0], [0, 1], [1, 0]]), (0, 1),
                         ((0,), (1,)), ((0, 1, 2), (0, 1, 2)))
    allowed = {(0, 0), (1, 1), (1, 2)}
    actual, _ = exact_global_topk(ds, 20, allowed_items=allowed)
    expected, _ = exact_topk(ds, 20, allowed_items=allowed)
    assert list(actual.items()) == list(expected.items())


@pytest.mark.parametrize("seed", range(5))
@pytest.mark.parametrize("k", (1, 5, 15, 100))
@pytest.mark.parametrize("min_size", (2, 3, 4))
def test_exact_global_topk_without_short_itemsets_matches_exhaustive(seed, k, min_size):
    data = np.random.default_rng(seed).integers(0, 3, size=(31, 4))
    ds = VerticalDataset(data, (0, 1, 2, 3), ((0, 1), (2, 3)), ((0, 1, 2),) * 4)
    expected = {}
    for size in range(min_size, 5):
        for attrs in combinations(range(4), size):
            for values in product(range(3), repeat=size):
                key = tuple(zip(attrs, values))
                expected[key] = ds.support(key)
    expected = dict(sorted(expected.items(), key=lambda pair: (-pair[1], pair[0]))[:k])
    actual, certificate = exact_global_topk(ds, k, min_size=min_size)
    assert actual == expected
    assert certificate["min_size"] == min_size
